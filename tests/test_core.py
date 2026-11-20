import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from accelerate import Accelerator

from core.config import load_config
from dataset.clips import SequenceDataset, VideoClipDataset, frame_indices, person_box, roi_boxes
from dataset.features import FeatureClipDataset, FeatureSequenceDataset
from dataset.labels import BEHAVIORS, CHEAT_IDS, IGNORE_BEHAVIOR, window_target
from dataset.onsets import curve_onset, estimate_onsets
from dataset.prepare import build_index, extract_video
from evaluation.gradcam import gradcam
from evaluation.metrics import best_threshold, binary_metrics, evaluate_clips
from evaluation.stack import stack
from features.extract import extract
from features.pose import POSE_DIM, pose_vector
from inference.predict import predict
from models.clip_classifier import LoRALinear, cheat_log_probs
from training.cv import load_videos, pending_folds, run, summarize
from training.engine import build_model, predictions_frame, train_fold
from training.losses import clip_loss
from training.optim import build_optimizer, cosine_schedule


def tiny_cfg(*extra: str):
    return load_config(None, ["model.pretrained=false", "data.num_frames=4",
                              "data.img_size=[64,96]", "data.clips_per_video=2", "data.color_jitter=0.0", *extra])


def test_config_overrides_and_unknown_key():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "debug.yaml", ["train.epochs=7", "data.img_size=[1,2]"])
    assert cfg.train.epochs == 7 and cfg.data.img_size == [1, 2] and cfg.name == "debug"
    with pytest.raises(KeyError):
        load_config(None, ["train.nope=1"])
    cfg = load_config(None, ["train.mixed_precision=no", "model.pretrained=false", "train.lr=3e-4", "train.ema=1"])
    assert cfg.train.mixed_precision == "no" and cfg.model.pretrained is False
    assert cfg.train.lr == 3e-4 and isinstance(cfg.train.ema, float)


def test_behaviors_split_by_label():
    assert len(BEHAVIORS) == 8 and CHEAT_IDS == [4, 5, 6, 7]


def test_frame_indices_cover_clip():
    rng = np.random.default_rng(0)
    idx = frame_indices(30, 30, 8, False, rng)
    assert len(idx) == 8 and idx.min() >= 30 and idx.max() <= 59 and np.all(np.diff(idx) > 0)
    jit = frame_indices(0, 30, 8, True, rng)
    assert jit.min() >= 0 and jit.max() <= 29


def test_metrics():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(np.array([0.1, 0.6, 0.7, 0.9]), y)
    assert m["acc"] == 0.75 and m["auc"] == 1.0
    thr, f1 = best_threshold(np.array([0.1, 0.6, 0.7, 0.9]), y)
    assert f1 == 1.0 and 0.6 < thr <= 0.7


def test_cheat_log_probs_marginalize():
    logits = torch.randn(5, 8)
    log_n, log_c = cheat_log_probs(logits)
    p = logits.softmax(-1)
    assert torch.allclose(log_c.exp(), p[:, 4:].sum(-1), atol=1e-6)
    assert torch.allclose(log_n.exp() + log_c.exp(), torch.ones(5), atol=1e-6)


@pytest.fixture
def fake_cache(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rows = []
    for v in range(4):
        vid = f"s{v // 2}_v{v}"
        d = tmp_path / "frames" / vid
        d.mkdir(parents=True)
        for i in range(65):
            cv2.imwrite(str(d / f"{i:05d}.jpg"), np.full((36, 64, 3), i, np.uint8))
        rows.append({"video_id": vid, "subject": f"s{v // 2}", "behavior_id": v * 2, "label": int(v * 2 >= 4),
                     "n_frames": 65, "fold": v // 2})
    return tmp_path / "frames", pd.DataFrame(rows)


def test_clip_datasets(fake_cache, tmp_path):
    frames_dir, videos = fake_cache
    cfg = tiny_cfg()
    va = VideoClipDataset(videos, frames_dir, cfg.data, train=False)
    assert len(va) == 4 * 2
    item = va[1]
    assert item["inputs"].shape == (4, 3, 64, 96) and item["inputs"].dtype == torch.uint8 and item["start"] == 30
    tr = VideoClipDataset(videos, frames_dir, cfg.data, train=True)
    assert len(tr) == 8 and tr[0]["inputs"].shape == (4, 3, 64, 96)
    seq = tmp_path / "test" / "test_seq_00001"
    seq.mkdir(parents=True)
    for i in range(30):
        cv2.imwrite(str(seq / f"{i + 1:02d}.jpg"), np.zeros((36, 64, 3), np.uint8))
    te = SequenceDataset(tmp_path / "test", cfg.data)
    assert te.ids == ["test_seq_00001"] and te[0]["inputs"].shape == (4, 3, 64, 96)


def test_window_labels_before_onset(fake_cache, tmp_path):
    frames_dir, videos = fake_cache
    cfg = tiny_cfg()
    cheat = videos.assign(onset=30)[videos.label == 1].reset_index(drop=True)
    va = VideoClipDataset(cheat, frames_dir, cfg.data, train=False)
    assert [(va[i]["start"], va[i]["label"], va[i]["behavior"]) for i in range(2)] == [
        (0, 0, IGNORE_BEHAVIOR), (30, 1, cheat.behavior_id[0])]
    assert window_target(cheat.iloc[0], 20, 30) == (cheat.behavior_id[0], 1)
    assert window_target(videos.iloc[0], 0, 30) == (videos.behavior_id[0], videos.label[0])
    pd.DataFrame({"video_id": videos.video_id, "onset": 30}).to_csv(tmp_path / "onsets.csv", index=False)
    videos.to_csv(tmp_path / "videos.csv", index=False)
    cfg = tiny_cfg(f"paths.cache_dir={tmp_path.as_posix()}", f"data.onsets={(tmp_path / 'onsets.csv').as_posix()}")
    assert load_videos(cfg).onset.tolist() == [30] * 4
    pd.DataFrame({"video_id": videos.video_id[:2], "onset": 30}).to_csv(tmp_path / "onsets.csv", index=False)
    with pytest.raises(ValueError, match="no onset"):
        load_videos(cfg)


def test_clip_loss_skips_missing_behavior():
    cfg = tiny_cfg()
    logits = torch.randn(3, len(BEHAVIORS), requires_grad=True)
    beh, lab = torch.tensor([IGNORE_BEHAVIOR, 5, IGNORE_BEHAVIOR]), torch.tensor([0, 1, 0])
    loss = clip_loss(logits, beh, lab, cfg.train)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0].abs().sum() > 0
    all_missing = clip_loss(logits, torch.full((3,), IGNORE_BEHAVIOR), lab, cfg.train)
    assert torch.isfinite(all_missing)


def test_estimate_onsets():
    rows = []
    curves = {"s1_front_a": [0.5] * 7 + [0.9] * 9, "s1_side_a": [0.5] * 16, "s2_front_b": [0.5] * 9 + [0.95] * 7,
              "s3_front_c": [0.4] * 16}
    for vid, curve in curves.items():
        rows += [{"video_id": vid, "start": 30 * s, "label": int(vid != "s3_front_c"), "p_cheat": p}
                 for s, p in enumerate(curve)]
    onsets = estimate_onsets(pd.DataFrame(rows)).set_index("video_id").onset.to_dict()
    assert curve_onset(np.array(curves["s1_front_a"])) == (7, pytest.approx(0.4))
    assert onsets == {"s1_front_a": 210, "s1_side_a": 210, "s2_front_b": 270, "s3_front_c": 240}


@pytest.mark.parametrize("opt", ["adamw", "muon"])
def test_pose_head_train_step(opt):
    cfg = tiny_cfg(f"train.optimizer={opt}", "model.kind=features", "model.hidden=16", "model.temporal_heads=2")
    model = build_model(cfg, POSE_DIM)
    optimizer = build_optimizer(model, cfg.train)
    sched = cosine_schedule(optimizer, 1, 4)
    x = torch.randn(2, cfg.data.clip_len // cfg.features.stride, POSE_DIM)
    before = model.temporal.layers[0].linear1.weight.clone()
    loss = clip_loss(model(x), torch.tensor([0, 5]), torch.tensor([0, 1]), cfg.train)
    loss.backward()
    optimizer.step()
    sched.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.temporal.layers[0].linear1.weight)
    n_muon = sum(isinstance(o, torch.optim.Muon) for o in getattr(optimizer, "optimizers", []))
    assert n_muon == (opt == "muon")


def test_predictions_and_summary(tmp_path):
    res = {"logits": np.random.randn(6, 8).astype(np.float32), "behavior": np.array([0, 1, 4, 5, 2, 6]),
           "label": np.array([0, 0, 1, 1, 0, 1]), "video": np.array([0, 0, 1, 1, 2, 2])}
    df = predictions_frame(res)
    assert {"p_cheat", "pred_behavior", "p_beh7"} <= set(df.columns) and df.p_cheat.between(0, 1).all()
    assert "vid_f1" in evaluate_clips(df)
    for f in range(2):
        (tmp_path / f"fold{f}").mkdir()
        df.assign(video_id=[f"f{f}_{v}" for v in df.video]).to_csv(tmp_path / f"fold{f}" / "oof.csv", index=False)
    cv = summarize(tmp_path)
    assert (tmp_path / "cv.json").exists() and 0 <= cv["f1_macro"] <= 1


def test_pose_vector_cache_and_head(fake_cache, tmp_path):
    kpts = np.stack([np.full(17, 120.0), np.full(17, 60.0), np.full(17, 0.9)], 1)
    v = pose_vector(np.array([100.0, 20.0, 140.0, 100.0]), 0.8, kpts, (360, 640))
    assert v.shape == (POSE_DIM,) and np.allclose(v[:34], 0.0)
    assert np.allclose(v[-5:], [40 / 640, 80 / 360, 120 / 640, 60 / 360, 0.8])
    _, videos = fake_cache
    videos.to_csv(tmp_path / "videos.csv", index=False)
    seq = tmp_path / "data" / "test" / "test_seq_00001"
    seq.mkdir(parents=True)
    for i in range(30):
        cv2.imwrite(str(seq / f"{i + 1:02d}.jpg"), np.zeros((36, 64, 3), np.uint8))
    tmp = tmp_path.as_posix()
    cfg = load_config(None, [f"paths.cache_dir={tmp}", f"paths.data_dir={tmp}/data", "features.backbone=yolo11n-pose",
                             "features.img_size=[64,96]", "features.batch_size=8", "model.kind=features",
                             "model.hidden=16", "model.temporal_heads=2", "data.clips_per_video=2"])
    root = extract(cfg, workers=0, pretrained=False)
    assert np.load(root / "train" / "s0_v0.npy").shape == (33, 1, POSE_DIM)
    assert np.load(root / "test" / "test_seq_00001.npy").shape == (15, 1, POSE_DIM)
    test_only = load_config(None, [f"paths.cache_dir={tmp}/fresh", f"paths.data_dir={tmp}/data",
                                   "features.backbone=yolo11n-pose", "features.img_size=[64,96]"])
    fresh = extract(test_only, workers=0, pretrained=False)
    assert [p.name for p in (fresh / "test").iterdir()] == ["test_seq_00001.npy"]
    assert not any((fresh / "train").iterdir())
    va = FeatureClipDataset(videos, cfg, train=False)
    assert len(va) == 8 and va.dim == POSE_DIM and va[1]["inputs"].shape == (15, POSE_DIM) and va[1]["start"] == 30
    te = FeatureSequenceDataset(cfg)
    assert te.ids == ["test_seq_00001"] and te[0]["inputs"].shape == (15, POSE_DIM)
    model = build_model(cfg, va.dim)
    optimizer = build_optimizer(model, cfg.train)
    before = model.temporal.layers[0].linear1.weight.clone()
    x = torch.stack([FeatureClipDataset(videos, cfg, train=True)[i]["inputs"] for i in range(2)])
    clip_loss(model(x), torch.tensor([0, 5]), torch.tensor([0, 1]), cfg.train).backward()
    optimizer.step()
    assert not torch.equal(before, model.temporal.layers[0].linear1.weight)


def test_videomae_classifier_trains_backbone():
    cfg = tiny_cfg("model.kind=videomae", "data.img_size=[32,48]")
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg.train)
    query = model.backbone.encoder.layer[0].attention.attention.query.weight
    before = query.clone()
    out = model(torch.randint(0, 255, (2, 4, 3, 32, 48), dtype=torch.uint8))
    clip_loss(out, torch.tensor([0, 5]), torch.tensor([0, 1]), cfg.train).backward()
    optimizer.step()
    assert out.shape == (2, 8) and not torch.equal(before, query)

def test_vjepa2_partial_freeze():
    cfg = tiny_cfg("model.kind=vjepa2", "data.img_size=[32,32]", "model.train_blocks=1")
    model = build_model(cfg)
    layers = model.backbone.encoder.layer
    assert not any(p.requires_grad for p in layers[0].parameters())
    assert all(p.requires_grad for p in layers[-1].parameters())
    optimizer = build_optimizer(model, cfg.train)
    last = layers[-1].mlp.fc1.weight
    before = last.clone()
    out = model(torch.randint(0, 255, (2, 4, 3, 32, 32), dtype=torch.uint8))
    clip_loss(out, torch.tensor([0, 5]), torch.tensor([0, 1]), cfg.train).backward()
    optimizer.step()
    assert out.shape == (2, 8) and not torch.equal(before, last) and layers[0].mlp.fc1.weight.grad is None

def test_vjepa2_lora():
    cfg = tiny_cfg("model.kind=vjepa2", "data.img_size=[32,32]", "model.train_blocks=1", "model.lora_rank=4")
    model = build_model(cfg)
    enc = model.backbone.encoder
    trainable = [any(p.requires_grad for n, p in layer.named_parameters() if ".lora_" not in n) for layer in enc.layer]
    assert trainable[-1] and not any(trainable[:-1])
    assert not any(p.requires_grad for p in enc.embeddings.parameters())
    lora = enc.layer[1].attention.query
    assert isinstance(lora, LoRALinear) and not lora.base.weight.requires_grad and lora.lora_a.weight.requires_grad
    model.eval()
    x = torch.randint(0, 255, (2, 4, 3, 32, 32), dtype=torch.uint8)
    lora.lora_b.weight.data.zero_()
    with_lora = model(x)
    assert torch.allclose(with_lora, model.head(model.pooler(model.backbone(
        pixel_values_videos=(x.float() - model.mean) / model.std, skip_predictor=True).last_hidden_state)))
    model.train()
    optimizer = build_optimizer(model, cfg.train)
    lrs = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    last_mlp = enc.layer[-1].mlp.fc1.weight
    assert lrs[id(lora.lora_b.weight)] == cfg.train.lr and lrs[id(last_mlp)] == cfg.train.backbone_lr
    clip_loss(model(x), torch.tensor([0, 5]), torch.tensor([0, 1]), cfg.train).backward()
    optimizer.step()
    assert lora.lora_b.weight.abs().sum() > 0
    mlp = build_model(tiny_cfg("model.kind=vjepa2", "data.img_size=[32,32]", "model.train_blocks=0",
                               "model.lora_rank=4", "model.lora_mlp=true")).backbone.encoder.layer[0]
    assert all(isinstance(m, LoRALinear) for m in (mlp.attention.proj, mlp.mlp.fc1, mlp.mlp.fc2))


def test_person_box_crop(tmp_path):
    pose = np.zeros((3, POSE_DIM), np.float32)
    pose[:, -5:] = [0.2, 0.5, 0.75, 0.5, 0.9]
    x0, y0, x1, y1 = person_box(pose, 1.0, [64, 64])
    assert np.isclose((x1 - x0) * 640, (y1 - y0) * 360) and np.isclose((x0 + x1) / 2, 0.75) and y1 - y0 == 0.5
    assert person_box(np.zeros((3, POSE_DIM), np.float32), 1.2, [64, 64]) is None
    seq = tmp_path / "test" / "test_seq_00001"
    seq.mkdir(parents=True)
    img = np.zeros((36, 64, 3), np.uint8)
    img[:, :32] = 255
    for i in range(30):
        cv2.imwrite(str(seq / f"{i + 1:02d}.jpg"), img)
    (tmp_path / "crop" / "test").mkdir(parents=True)
    np.save(tmp_path / "crop" / "test" / "test_seq_00001.npy", pose[:, None].repeat(5, 0).astype(np.float16))
    cfg = tiny_cfg("data.person_crop=1.0", "data.img_size=[32,32]")
    full = SequenceDataset(tmp_path / "test", cfg.data)[0]["inputs"]
    crop = SequenceDataset(tmp_path / "test", cfg.data, tmp_path / "crop")[0]["inputs"]
    assert crop.shape == full.shape == (4, 3, 32, 32) and full.float().mean() > 100 > 20 > crop.float().mean()

def test_stack_trusts_informative_run(tmp_path):
    rng = np.random.default_rng(0)
    beh = np.repeat(np.arange(8), 20)
    good = np.full((160, 8), 0.02)
    good[np.arange(160), beh] = 0.86
    runs = []
    for name, probs in (("good", good), ("noise", rng.dirichlet(np.ones(8), 160))):
        d = tmp_path / name
        d.mkdir()
        cols = pd.DataFrame(probs, columns=[f"p_beh{i}" for i in range(8)])
        cols.assign(video_id=[f"v{i // 4}" for i in range(160)], start=np.arange(160) % 4 * 30, behavior=beh,
                    label=(beh >= 4).astype(int), pred_behavior=probs.argmax(1),
                    p_cheat=probs[:, 4:].sum(1)).to_csv(d / "oof.csv", index=False)
        cols.assign(id=[f"t{i:03d}" for i in range(160)], p_cheat=probs[:, 4:].sum(1)).to_csv(d / "test_probs.csv",
                                                                                              index=False)
        runs.append(d)
    cv = stack(runs, tmp_path / "st")
    assert cv["coef"]["good"] > abs(cv["coef"]["noise"]) and cv["f1_best"] == 1.0 and cv["n_clips"] == 160
    assert pd.read_csv(tmp_path / "st" / "submission.csv").label.tolist() == (beh >= 4).astype(int).tolist()
    pd.read_csv(runs[1] / "oof.csv").iloc[::2].to_csv(runs[1] / "oof.csv", index=False)
    for name in ("oof", "test_probs"):
        (runs[0] / f"{name}_flip.csv").write_bytes((runs[0] / f"{name}.csv").read_bytes())
    cv = stack([Path(f"{runs[0]}@flip"), runs[1]], tmp_path / "st2")
    assert set(cv["coef"]) == {"good_flip", "noise"} and cv["n_clips"] == 80
    with pytest.raises(ValueError):
        stack([Path(f"{runs[0]}@nope")], tmp_path / "st3")


def test_build_index_folds_are_subject_disjoint(tmp_path):
    for s in range(7):
        for label, behaviors in (("normal", BEHAVIORS[:4]), ("cheat", BEHAVIORS[4:])):
            d = tmp_path / "train" / label / f"subject_{s}"
            d.mkdir(parents=True)
            for b in behaviors:
                for view in ("front", "side"):
                    (d / f"{view}_{b}.MOV").touch()
    df = build_index(tmp_path, n_folds=3, seed=42)
    assert len(df) == 7 * 16 and df.groupby("subject").fold.nunique().max() == 1
    assert df.groupby("subject").view.nunique().min() == 2 and sorted(df.groupby("fold").subject.nunique()) == [2, 2, 3]
    assert (df.groupby("fold").label.mean() == 0.5).all() and df.equals(build_index(tmp_path, n_folds=3, seed=42))

def test_extract_video_matches_test_format(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    for i in range(5):
        w.write(np.full((720, 1280, 3), i * 40, np.uint8))
    w.release()
    assert extract_video(str(path), tmp_path / "out") == 5
    img = cv2.imread(str(tmp_path / "out" / "00000.jpg"))
    assert img.shape == (360, 640, 3)
    assert extract_video(str(path), tmp_path / "out") == 5


@pytest.fixture
def video_cache(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rows = []
    for s in range(2):
        for b in (1, 5):
            for view in ("front", "side"):
                vid = f"s{s}_{view}_{b}"
                d = tmp_path / "frames" / vid
                d.mkdir(parents=True)
                for i in range(65):
                    cv2.imwrite(str(d / f"{i:05d}.jpg"), np.full((36, 64, 3), i, np.uint8))
                rows.append({"video_id": vid, "subject": f"s{s}", "view": view, "behavior_id": b, "label": int(b >= 4),
                             "n_frames": 65, "fold": s})
    videos = pd.DataFrame(rows)
    videos.to_csv(tmp_path / "videos.csv", index=False)
    return tmp_path / "frames", videos


def test_speed_jitter_and_grayscale_clips(video_cache):
    frames_dir, videos = video_cache
    cfg = tiny_cfg("data.speed_jitter=0.2", "data.grayscale=0.5")
    ds = VideoClipDataset(videos, frames_dir, cfg.data, train=True)
    for _ in range(30):
        item = ds[0]
        assert item["inputs"].shape == (4, 3, 64, 96) and 0 <= item["start"] <= 65 - 24
    pose = np.zeros((33, 1, POSE_DIM), np.float16)
    pose[:, 0, -5:] = [0.3, 0.6, 0.5, 0.5, 0.9]
    (frames_dir.parent / "crop" / "train").mkdir(parents=True)
    for v in videos.video_id:
        np.save(frames_dir.parent / "crop" / "train" / f"{v}.npy", pose)
    cfg = tiny_cfg("data.person_crop=1.2")
    va = VideoClipDataset(videos, frames_dir, cfg.data, train=False, crop_dir=frames_dir.parent / "crop")
    assert va[1]["inputs"].shape == (4, 3, 64, 96) and va[1]["start"] == 30


def test_train_fold_ema_clip_selection(video_cache, tmp_path):
    _, videos = video_cache
    tmp = tmp_path.as_posix()
    cfg = tiny_cfg(f"paths.cache_dir={tmp}", f"paths.output_dir={tmp}/out", "data.num_workers=0",
                   "data.speed_jitter=0.2", "train.epochs=2", "train.batch_size=2", "train.ema=0.9",
                   "train.select=auc", "train.mixed_precision=no")
    oof = train_fold(cfg, 0, videos, Accelerator(cpu=True))
    assert len(oof) == 8 and {"auc", "f1_macro", "auc_flip"} <= set(pd.read_csv(cfg.run_dir / "fold0" / "history.csv"))
    assert (cfg.run_dir / "fold0" / "best.pt").exists() and oof.subject.eq("s0").all()
    flip = pd.read_csv(cfg.run_dir / "fold0" / "oof_flip.csv")
    assert flip[["video_id", "start"]].equals(oof[["video_id", "start"]]) and flip.p_cheat.between(0, 1).all()


def test_stop_epoch_keeps_long_schedule(video_cache, tmp_path):
    _, videos = video_cache
    tmp = tmp_path.as_posix()
    cfg = tiny_cfg(f"paths.cache_dir={tmp}", f"paths.output_dir={tmp}/out", "data.num_workers=0", "train.epochs=4",
                   "train.stop_epoch=2", "train.eval_last=true", "train.batch_size=2", "train.select=epoch",
                   "train.patience=99", "train.warmup_epochs=0", "train.mixed_precision=no")
    train_fold(cfg, 0, videos, Accelerator(cpu=True))
    hist = pd.read_csv(cfg.run_dir / "fold0" / "history.csv")
    assert hist.epoch.tolist() == [1] and len(pd.read_csv(cfg.run_dir / "fold0" / "oof.csv")) == 8
    with pytest.raises(ValueError):
        train_fold(tiny_cfg(f"paths.output_dir={tmp}/bad", "train.eval_last=true", "train.select=auc"), 0, videos,
                   Accelerator(cpu=True))


def test_predict_refuses_unfinished_fold(tmp_path):
    cfg = tiny_cfg(f"paths.output_dir={tmp_path.as_posix()}", f"paths.data_dir={tmp_path.as_posix()}")
    (cfg.run_dir / "fold0").mkdir(parents=True)
    torch.save({}, cfg.run_dir / "fold0" / "best.pt")
    with pytest.raises(RuntimeError, match="fold0"):
        predict(cfg)


def seated_pose(n: int) -> np.ndarray:
    """(n, 1, 56) pose cache of one centred person: head 0.3 box heights up, wrists 0.2 down and 0.2 to the sides."""
    pose = np.zeros((n, 1, POSE_DIM), np.float32)
    xy = np.zeros((17, 2), np.float32)
    xy[:5] = [0.0, -0.3]
    xy[9], xy[10] = [0.2, 0.2], [-0.2, 0.2]
    pose[:, 0, :34] = xy.ravel()
    pose[:, 0, 34:51] = 0.9
    pose[:, 0, -5:] = [0.3, 0.6, 0.5, 0.5, 0.9]
    return pose


def test_roi_boxes():
    head, right, left = roi_boxes(seated_pose(5)[:, 0], 0.5)
    assert np.isclose((head[0] + head[2]) / 2, 0.5) and np.isclose((head[1] + head[3]) / 2, 0.5 - 0.3 * 0.6)
    assert np.isclose((head[2] - head[0]) * 640, 2 * (head[3] - head[1]) * 360)
    assert (right[0] + right[2]) / 2 < 0.5 < (left[0] + left[2]) / 2
    assert roi_boxes(np.zeros((3, POSE_DIM), np.float32), 0.5) is None



def test_gradcam_report(video_cache, tmp_path):
    frames_dir, videos = video_cache
    crop = tmp_path / "features" / "crop" / "train"
    crop.mkdir(parents=True)
    for v in videos.video_id:
        np.save(crop / f"{v}.npy", seated_pose(33).astype(np.float16))
    cfg = tiny_cfg("model.kind=videomae", "data.img_size=[32,48]", f"paths.cache_dir={tmp_path.as_posix()}",
                   "data.crop_features=crop", "data.person_crop=1.2", "data.num_workers=0")
    torch.save(build_model(cfg).state_dict(), tmp_path / "best.pt")
    summary = gradcam(cfg, 0, tmp_path / "best.pt", tmp_path / "cam", n_show=2, batch_size=4)
    df = pd.read_csv(tmp_path / "cam" / "gradcam.csv")
    assert len(df) == 8 and df.roi_mass.between(0, 1).all() and (df.roi_area > 0).all()
    assert summary["right"]["n"] + summary["wrong"]["n"] == 8 and (tmp_path / "cam" / "summary.json").exists()


def test_train_then_predict_with_crops(video_cache, tmp_path):
    frames_dir, videos = video_cache
    for split, names in (("train", videos.video_id), ("test", ["test_seq_00001"])):
        (tmp_path / "features" / "crop" / split).mkdir(parents=True)
        for v in names:
            np.save(tmp_path / "features" / "crop" / split / f"{v}.npy", seated_pose(33 if split == "train" else 15))
    seq = tmp_path / "data" / "test" / "test_seq_00001"
    seq.mkdir(parents=True)
    for i in range(30):
        cv2.imwrite(str(seq / f"{i + 1:02d}.jpg"), np.full((36, 64, 3), i, np.uint8))
    tmp = tmp_path.as_posix()
    cfg = tiny_cfg(f"paths.cache_dir={tmp}", f"paths.data_dir={tmp}/data", f"paths.output_dir={tmp}/out",
                   "data.crop_features=crop", "data.person_crop=1.15", "data.num_workers=0", "train.folds=[0]",
                   "train.epochs=3", "train.stop_epoch=1", "train.batch_size=2", "train.select=epoch",
                   "train.patience=99", "train.mixed_precision=no")
    cv = run(cfg)
    probs = predict(cfg)
    sub = pd.read_csv(cfg.run_dir / "submission.csv")
    assert 0 <= cv["thr"] <= 1 and len(pd.read_csv(cfg.run_dir / "oof.csv")) == 8
    assert probs.id.tolist() == sub.id.tolist() == ["test_seq_00001"] and probs.p_cheat.between(0, 1).all()
    assert sub.label.iloc[0] == int(probs.p_cheat.iloc[0] >= cv["thr"])
    cv_flip = json.loads((cfg.run_dir / "cv_flip.json").read_text())
    flip, sub_flip = (pd.read_csv(cfg.run_dir / f"{n}_flip.csv") for n in ("test_probs", "submission"))
    assert len(pd.read_csv(cfg.run_dir / "oof_flip.csv")) == 8 and flip.id.tolist() == ["test_seq_00001"]
    assert sub_flip.label.iloc[0] == int(flip.p_cheat.iloc[0] >= cv_flip["thr"])


def test_resume_skips_finished_folds_of_the_same_recipe(tmp_path):
    cfg = tiny_cfg(f"paths.output_dir={tmp_path.as_posix()}", "train.folds=[0,1,2]", "train.batch_size=8")
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    for fold, files in ((0, ("oof.csv", "best.pt")), (1, ("oof.csv",))):
        (cfg.run_dir / f"fold{fold}").mkdir()
        for name in files:
            (cfg.run_dir / f"fold{fold}" / name).touch()
    assert pending_folds(cfg) == [0, 1, 2]
    split = tiny_cfg(f"paths.output_dir={tmp_path.as_posix()}", "train.folds=[0,1,2]", "train.resume=true",
                     "train.batch_size=4", "train.grad_accum=2", "data.num_workers=7")
    assert pending_folds(split) == [1, 2]
    with pytest.raises(ValueError):
        pending_folds(tiny_cfg(f"paths.output_dir={tmp_path.as_posix()}", "train.resume=true", "train.lr=1e-3"))


def test_predict_refuses_checkpoints_without_matching_oof(tmp_path):
    cfg = tiny_cfg(f"paths.output_dir={tmp_path.as_posix()}", f"paths.data_dir={tmp_path.as_posix()}")
    for fold, files in ((0, ("best.pt", "oof.csv")), (1, ("oof.csv",))):
        (cfg.run_dir / f"fold{fold}").mkdir(parents=True)
        for name in files:
            (cfg.run_dir / f"fold{fold}" / name).touch()
    with pytest.raises(RuntimeError, match="fold1"):
        predict(cfg)