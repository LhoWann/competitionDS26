from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from accelerate import Accelerator

from core.config import load_config
from dataset.clips import SequenceDataset, VideoClipDataset, frame_indices, person_box
from dataset.features import FeatureClipDataset, FeatureSequenceDataset, frame_vectors
from dataset.labels import BEHAVIORS, CHEAT_IDS
from dataset.prepare import extract_video
from evaluation.ensemble import PROB_COLS, ensemble
from evaluation.metrics import best_threshold, binary_metrics, evaluate_clips
from features.extract import extract
from features.pose import POSE_DIM, pose_vector
from models.clip_classifier import ClipClassifier, cheat_log_probs
from training.cv import summarize
from training.engine import build_model, predictions_frame, train_fold
from training.losses import clip_loss
from training.optim import build_optimizer, cosine_schedule


def tiny_cfg(*extra: str):
    return load_config(None, ["model.backbone=resnet18", "model.pretrained=false", "data.num_frames=4",
                              "data.img_size=[64,96]", "data.clips_per_video=2", "data.color_jitter=0.0", *extra])


def test_config_overrides_and_unknown_key():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "debug.yaml", ["train.epochs=7", "data.img_size=[1,2]"])
    assert cfg.train.epochs == 7 and cfg.data.img_size == [1, 2] and cfg.name == "debug"
    with pytest.raises(KeyError):
        load_config(None, ["train.nope=1"])


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


@pytest.mark.parametrize("opt", ["adamw", "muon"])
def test_model_train_step(opt):
    cfg = tiny_cfg(f"train.optimizer={opt}")
    model = ClipClassifier(cfg.model, cfg.data.num_frames, cfg.data.img_size)
    optimizer = build_optimizer(model, cfg.train)
    sched = cosine_schedule(optimizer, 1, 4)
    x = torch.randint(0, 255, (2, 4, 3, 64, 96), dtype=torch.uint8)
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


def test_frame_vectors_pools():
    tokens = np.random.rand(5, 3, 4).astype(np.float16)
    assert frame_vectors(tokens, "cls").shape == (5, 4)
    assert np.allclose(frame_vectors(tokens, "cls_mean")[:, 4:], tokens[:, 1:].astype(np.float32).mean(1), atol=1e-3)
    assert frame_vectors(tokens, "grid").shape == (5, 12)
    with pytest.raises(ValueError):
        frame_vectors(tokens, "nope")


def test_feature_cache_and_head(fake_cache, tmp_path):
    _, videos = fake_cache
    videos.to_csv(tmp_path / "videos.csv", index=False)
    seq = tmp_path / "data" / "test" / "test_seq_00001"
    seq.mkdir(parents=True)
    for i in range(30):
        cv2.imwrite(str(seq / f"{i + 1:02d}.jpg"), np.zeros((36, 64, 3), np.uint8))
    tmp = tmp_path.as_posix()
    cfg = load_config(None, [f"paths.cache_dir={tmp}", f"paths.data_dir={tmp}/data",
                             "features.backbone=vit_tiny_patch16_224", "features.img_size=[32,64]",
                             "features.grid=[1,2]", "features.batch_size=7", "model.kind=features",
                             "model.hidden=16", "model.temporal_heads=2", "data.clips_per_video=2"])
    root = extract(cfg, workers=0, pretrained=False)
    assert np.load(root / "train" / "s0_v0.npy").shape == (33, 3, 192)
    assert np.load(root / "test" / "test_seq_00001.npy").shape == (15, 3, 192)

    va = FeatureClipDataset(videos, cfg, train=False)
    assert len(va) == 8 and va.dim == 384 and va[1]["inputs"].shape == (15, 384) and va[1]["start"] == 30
    te = FeatureSequenceDataset(cfg)
    assert te.ids == ["test_seq_00001"] and te[0]["inputs"].shape == (15, 384)

    model = build_model(cfg, va.dim)
    optimizer = build_optimizer(model, cfg.train)
    before = model.temporal.layers[0].linear1.weight.clone()
    x = torch.stack([FeatureClipDataset(videos, cfg, train=True)[i]["inputs"] for i in range(2)])
    clip_loss(model(x), torch.tensor([0, 5]), torch.tensor([0, 1]), cfg.train).backward()
    optimizer.step()
    assert not torch.equal(before, model.temporal.layers[0].linear1.weight)


def test_pose_vector_and_pose_cache(fake_cache, tmp_path):
    kpts = np.stack([np.full(17, 120.0), np.full(17, 60.0), np.full(17, 0.9)], 1)
    v = pose_vector(np.array([100.0, 20.0, 140.0, 100.0]), 0.8, kpts, (360, 640))
    assert v.shape == (POSE_DIM,) and np.allclose(v[:34], 0.0)
    assert np.allclose(v[-5:], [40 / 640, 80 / 360, 120 / 640, 60 / 360, 0.8])
    _, videos = fake_cache
    videos.to_csv(tmp_path / "videos.csv", index=False)
    cfg = load_config(None, [f"paths.cache_dir={tmp_path.as_posix()}", "paths.data_dir=missing",
                             "features.backbone=yolo11n-pose", "features.img_size=[64,96]", "features.batch_size=8"])
    root = extract(cfg, workers=0, pretrained=False)
    assert np.load(root / "train" / "s0_v0.npy").shape == (33, 1, POSE_DIM)


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

def test_ensemble_picks_informative_run(tmp_path):
    rng = np.random.default_rng(0)
    beh = np.repeat(np.arange(8), 20)
    good = np.full((160, 8), 0.02)
    good[np.arange(160), beh] = 0.86
    runs = []
    for name, probs in (("good", good), ("noise", rng.dirichlet(np.ones(8), 160))):
        d = tmp_path / name
        d.mkdir()
        cols = pd.DataFrame(probs, columns=PROB_COLS)
        cols.assign(video_id=[f"v{i // 4}" for i in range(160)], start=np.arange(160) % 4 * 30, behavior=beh,
                    label=(beh >= 4).astype(int), video=[f"v{i // 4}" for i in range(160)],
                    pred_behavior=probs.argmax(1), p_cheat=probs[:, 4:].sum(1)).to_csv(d / "oof.csv", index=False)
        cols.assign(id=[f"t{i:03d}" for i in range(160)]).to_csv(d / "test_probs.csv", index=False)
        runs.append(d)
    cv = ensemble(runs, tmp_path / "ens", step=0.5)
    assert cv["weights"]["good"] >= 0.5 and cv["f1_best"] == 1.0
    assert pd.read_csv(tmp_path / "ens" / "submission.csv").label.tolist() == (beh >= 4).astype(int).tolist()

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
    assert len(oof) == 8 and {"auc", "f1_macro"} <= set(pd.read_csv(cfg.run_dir / "fold0" / "history.csv"))
    assert (cfg.run_dir / "fold0" / "best.pt").exists() and oof.subject.eq("s0").all()