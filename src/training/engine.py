"""Training / inference loops on HF Accelerate (single GPU, 2xT4 DDP, fp16)."""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.config import Config
from dataset.clips import VideoClipDataset, seed_worker
from dataset.features import FeatureClipDataset
from evaluation.metrics import evaluate_clips
from models.clip_classifier import (
    FeatureClipClassifier,
    VideoMAEClassifier,
    VJEPA2Classifier,
    cheat_log_probs,
)
from training.losses import clip_loss
from training.optim import build_optimizer, cosine_schedule


def make_loader(ds: torch.utils.data.Dataset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0, worker_init_fn=seed_worker)


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, acc: Accelerator, keys: list[str],
                  hflip: bool = False) -> dict[str, np.ndarray]:
    """Returns gathered `logits` plus the requested batch `keys`, de-duplicated across processes; with `hflip` also
    `logits_flip`, the mean log-probabilities of each clip and its mirror image (flip TTA). Inputs of shape
    (B, V, T, 3, H, W) hold V views of each clip (`data.eval_passes`); their log-probabilities are averaged."""
    model.eval()
    out: dict[str, list[torch.Tensor]] = {k: [] for k in ["logits", *(["logits_flip"] if hflip else []), *keys]}
    for batch in tqdm(loader, desc="infer" + " (+flip)" * hflip, leave=False, disable=not acc.is_local_main_process):
        inputs = batch["inputs"]
        views = inputs.unbind(1) if inputs.ndim == 6 else [inputs]
        with acc.autocast():
            raw = [model(v).float() for v in views]
            logits = raw[0] if len(raw) == 1 else torch.stack([r.log_softmax(-1) for r in raw]).mean(0)
            flip = []
            if hflip:
                flip = [torch.stack([(r.log_softmax(-1) + model(v.flip(-1)).float().log_softmax(-1)) / 2
                                     for r, v in zip(raw, views, strict=True)]).mean(0)]
        gathered = acc.gather_for_metrics((logits, *flip, *[batch[k] for k in keys]))
        for k, v in zip(out, gathered, strict=True):
            out[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def predictions_frame(res: dict[str, np.ndarray]) -> pd.DataFrame:
    """Turns raw logits into cheat probability, predicted behavior and per-behavior probabilities."""
    logits = torch.from_numpy(res["logits"])
    probs = logits.softmax(-1).numpy()
    df = pd.DataFrame({k: v for k, v in res.items() if not k.startswith("logits")})
    df["p_cheat"] = cheat_log_probs(logits)[1].exp().numpy()
    df["pred_behavior"] = probs.argmax(1)
    return pd.concat([df, pd.DataFrame(probs, columns=[f"p_beh{i}" for i in range(probs.shape[1])])], axis=1)


def build_model(cfg: Config, in_dim: int = 0) -> torch.nn.Module:
    """`model.kind=videomae` / `vjepa2`: video transformers; `features`: head over `in_dim`-wide cached
    pose vectors."""
    if cfg.model.kind == "features":
        return FeatureClipClassifier(cfg.model, in_dim, cfg.data.clip_len // cfg.features.stride)
    if cfg.model.kind == "videomae":
        return VideoMAEClassifier(cfg.model, cfg.data.num_frames, cfg.data.img_size)
    if cfg.model.kind == "vjepa2":
        return VJEPA2Classifier(cfg.model, cfg.data.num_frames, cfg.data.img_size)
    raise ValueError(f"Unknown model.kind '{cfg.model.kind}'")


def train_fold(cfg: Config, fold: int, videos: pd.DataFrame, acc: Accelerator) -> pd.DataFrame:
    """Trains one subject-held-out fold, logs metrics every epoch and returns out-of-fold clip predictions."""
    if cfg.train.eval_last and cfg.train.select != "epoch":
        raise ValueError("train.eval_last skips the per-epoch validation that train.select needs: use select=epoch")
    set_seed(cfg.train.seed + fold)
    fold_dir = cfg.run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if acc.is_main_process:
        # A fold is finished when it has oof.csv and best.pt from the same training: drop leftovers of earlier attempts.
        for stale in ("oof.csv", "oof_flip.csv", "best.pt"):
            (fold_dir / stale).unlink(missing_ok=True)
    va_videos = videos[videos.fold == fold].reset_index(drop=True)
    if cfg.model.kind == "features":
        tr_ds = FeatureClipDataset(videos[videos.fold != fold], cfg, train=True)
        va_ds = FeatureClipDataset(va_videos, cfg, train=False)
    else:
        frames_dir = Path(cfg.paths.cache_dir) / "frames"
        tr_ds = VideoClipDataset(videos[videos.fold != fold], frames_dir, cfg.data, train=True, crop_dir=cfg.crop_dir)
        va_ds = VideoClipDataset(va_videos, frames_dir, cfg.data, train=False, crop_dir=cfg.crop_dir)
    tr_dl = make_loader(tr_ds, cfg.train.batch_size, cfg.data.num_workers, train=True)
    va_dl = make_loader(va_ds, cfg.train.batch_size * 2, cfg.data.num_workers, train=False)

    model = build_model(cfg, tr_ds.dim if cfg.model.kind == "features" else 0)
    optimizer = build_optimizer(model, cfg.train)
    # The EMA copy is taken before `prepare`, which patches `forward` (a later deepcopy would call the live model).
    ema = ModelEmaV3(model, decay=cfg.train.ema, use_warmup=True, device=acc.device) if cfg.train.ema > 0 else None
    model, optimizer, tr_dl, va_dl = acc.prepare(model, optimizer, tr_dl, va_dl)
    steps_per_epoch = math.ceil(len(tr_dl) / cfg.train.grad_accum)
    scheduler = cosine_schedule(optimizer, int(cfg.train.warmup_epochs * steps_per_epoch),
                                cfg.train.epochs * steps_per_epoch)

    acc.print(f"\n=== fold {fold} | train clips {len(tr_ds)} | val clips {len(va_ds)} "
              f"({va_videos.subject.nunique()} subjects) | steps/epoch {steps_per_epoch}")
    header = (f"{'ep':>3} {'tr_loss':>8} {'va_loss':>8} {'va_acc':>7} {'va_f1':>7} {'f1_norm':>7} {'f1_cheat':>8} "
              f"{'auc':>6} {'beh_acc':>7} {'f1@thr':>12} {'vid_f1':>7} {'lr':>9} "
              f"{'time':>6}")
    acc.print(header)
    best, best_df, best_flip, history, bad, step = -float("inf"), pd.DataFrame(), None, [], 0, 0
    flip = cfg.model.kind != "features"  # video models also record flip-TTA predictions (oof_flip.csv)
    # stop_epoch > 0 trains only that many epochs of the `epochs`-long schedule (reproduces a mid-schedule snapshot).
    n_epochs = cfg.train.stop_epoch or cfg.train.epochs
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        tr_loss, n = torch.zeros((), device=acc.device), 0
        for batch in tqdm(tr_dl, desc=f"ep{epoch}", leave=False, disable=not acc.is_local_main_process):
            with acc.accumulate(model):
                loss = clip_loss(model(batch["inputs"]), batch["behavior"], batch["label"], cfg.train)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                if acc.sync_gradients:
                    scheduler.step()
                    step += 1
                    if ema is not None:
                        ema.update(acc.unwrap_model(model), step=step)
                optimizer.zero_grad()
            tr_loss += loss.detach()
            n += 1
        tr_loss = acc.gather(tr_loss[None]).mean().item() / max(n, 1)
        if cfg.train.eval_last and epoch < n_epochs - 1:
            acc.print(f"{epoch:>3} {tr_loss:>8.4f}  (validation and checkpoint after the last epoch only) "
                      f"{scheduler.get_last_lr()[0]:>9.2e} {time.time() - t0:>5.0f}s")
            continue

        eval_model = model if ema is None else ema.module
        res = run_inference(eval_model, va_dl, acc, ["behavior", "label", "video", "start"], hflip=flip)
        meta = {"fold": fold, "video_id": va_videos.video_id.to_numpy()[res["video"]],
                "subject": va_videos.subject.to_numpy()[res["video"]]}
        df = predictions_frame(res).assign(**meta)
        df_flip = predictions_frame(res | {"logits": res["logits_flip"]}).assign(**meta) if flip else None
        va_loss = clip_loss(torch.from_numpy(res["logits"]), torch.from_numpy(res["behavior"]),
                            torch.from_numpy(res["label"]), cfg.train).item()
        m = evaluate_clips(df) | {"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss}
        if flip:
            m_flip = evaluate_clips(df_flip)
            m |= {"auc_flip": m_flip["auc"], "f1_flip": m_flip["f1_best"]}
        history.append(m)
        acc.print(f"{epoch:>3} {tr_loss:>8.4f} {va_loss:>8.4f} {m['acc']:>7.4f} {m['f1_macro']:>7.4f} "
                  f"{m['f1_normal']:>7.4f} {m['f1_cheat']:>8.4f} {m['auc']:>6.3f} {m['beh_acc']:>7.4f} "
                  f"{m['f1_best']:>6.4f}@{m['thr']:.2f} {m['vid_f1']:>7.4f} "
                  f"{scheduler.get_last_lr()[0]:>9.2e} {time.time() - t0:>5.0f}s"
                  + (f" | flip TTA: auc {m['auc_flip']:.3f} f1@thr {m['f1_flip']:.4f}" if flip else ""))
        score = m[cfg.train.select]
        if score > best:
            best, bad, best_df, best_flip = score, 0, df, df_flip
            acc.save(acc.unwrap_model(eval_model).state_dict(), fold_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.train.patience:
                acc.print(f"early stop at epoch {epoch}")
                break
    if acc.is_main_process:
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        if flip:
            best_flip.to_csv(fold_dir / "oof_flip.csv", index=False)
        best_df.to_csv(fold_dir / "oof.csv", index=False)
    acc.print(f"fold {fold} best {cfg.train.select} {abs(best):.4f}")
    acc.free_memory()
    return best_df