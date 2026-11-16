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
from models.clip_classifier import ClipClassifier, FeatureClipClassifier, VideoMAEClassifier, cheat_log_probs
from training.losses import clip_loss
from training.optim import build_optimizer, cosine_schedule


def make_loader(ds: torch.utils.data.Dataset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0, worker_init_fn=seed_worker)


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, acc: Accelerator, keys: list[str],
                  hflip_tta: bool = False) -> dict[str, np.ndarray]:
    """Returns gathered `logits` plus the requested batch `keys`, de-duplicated across processes."""
    model.eval()
    out: dict[str, list[torch.Tensor]] = {k: [] for k in ["logits", *keys]}
    for batch in loader:
        with acc.autocast():
            logits = model(batch["inputs"]).float()
            if hflip_tta:
                logits = (logits.log_softmax(-1) + model(batch["inputs"].flip(-1)).float().log_softmax(-1)) / 2
        gathered = acc.gather_for_metrics((logits, *[batch[k] for k in keys]))
        for k, v in zip(out, gathered, strict=True):
            out[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def predictions_frame(res: dict[str, np.ndarray]) -> pd.DataFrame:
    """Turns raw logits into cheat probability, predicted behavior and per-behavior probabilities."""
    logits = torch.from_numpy(res["logits"])
    probs = logits.softmax(-1).numpy()
    df = pd.DataFrame({k: v for k, v in res.items() if k != "logits"})
    df["p_cheat"] = cheat_log_probs(logits)[1].exp().numpy()
    df["pred_behavior"] = probs.argmax(1)
    return pd.concat([df, pd.DataFrame(probs, columns=[f"p_beh{i}" for i in range(probs.shape[1])])], axis=1)


def build_model(cfg: Config, in_dim: int = 0) -> torch.nn.Module:
    """`model.kind=frames`: end-to-end ClipClassifier; `videomae`: VideoMAEClassifier; `features`: head over
    `in_dim`-wide cached frame vectors."""
    if cfg.model.kind == "features":
        return FeatureClipClassifier(cfg.model, in_dim, cfg.data.clip_len // cfg.features.stride)
    if cfg.model.kind == "videomae":
        return VideoMAEClassifier(cfg.model, cfg.data.num_frames, cfg.data.img_size)
    return ClipClassifier(cfg.model, cfg.data.num_frames, cfg.data.img_size)


def train_fold(cfg: Config, fold: int, videos: pd.DataFrame, acc: Accelerator) -> pd.DataFrame:
    """Trains one subject-held-out fold, logs metrics every epoch and returns out-of-fold clip predictions."""
    set_seed(cfg.train.seed + fold)
    fold_dir = cfg.run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
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
    best, best_df, history, bad, step = -float("inf"), pd.DataFrame(), [], 0, 0
    for epoch in range(cfg.train.epochs):
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

        eval_model = model if ema is None else ema.module
        res = run_inference(eval_model, va_dl, acc, ["behavior", "label", "video", "start"])
        df = predictions_frame(res)
        df = df.assign(fold=fold, video_id=va_videos.video_id.to_numpy()[df.video],
                       subject=va_videos.subject.to_numpy()[df.video])
        va_loss = clip_loss(torch.from_numpy(res["logits"]), torch.from_numpy(res["behavior"]),
                            torch.from_numpy(res["label"]), cfg.train).item()
        m = evaluate_clips(df) | {"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss}
        history.append(m)
        acc.print(f"{epoch:>3} {tr_loss:>8.4f} {va_loss:>8.4f} {m['acc']:>7.4f} {m['f1_macro']:>7.4f} "
                  f"{m['f1_normal']:>7.4f} {m['f1_cheat']:>8.4f} {m['auc']:>6.3f} {m['beh_acc']:>7.4f} "
                  f"{m['f1_best']:>6.4f}@{m['thr']:.2f} {m['vid_f1']:>7.4f} "
                  f"{scheduler.get_last_lr()[0]:>9.2e} {time.time() - t0:>5.0f}s")
        score = m[cfg.train.select]
        if score > best:
            best, bad, best_df = score, 0, df
            acc.save(acc.unwrap_model(eval_model).state_dict(), fold_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.train.patience:
                acc.print(f"early stop at epoch {epoch}")
                break
    if acc.is_main_process:
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        best_df.to_csv(fold_dir / "oof.csv", index=False)
    acc.print(f"fold {fold} best {cfg.train.select} {abs(best):.4f}")
    acc.free_memory()
    return best_df