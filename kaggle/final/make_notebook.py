"""Writes kaggle/final/final.ipynb (+ kernel-metadata.json), the inference notebook that reproduces both final
picks. Run from the repo root: `python kaggle/final/make_notebook.py`, then `kaggle kernels push -p kaggle/final`."""
import json
from pathlib import Path

MD = """# DataSlayer 4.0 - notebook final (inferensi)

Notebook ini membuat `submission.csv` dari model yang sudah dilatih, **tanpa training ulang**:

1. **Fitur per klip test** (`cli.extract`, hanya folder test): kotak orang YOLO11s-pose (untuk crop model video) dan
   keypoint YOLO26s-pose (untuk head pose). Model YOLO dipakai beku dengan bobot COCO (tidak di-fine-tune).
2. **Prediksi per model** (`cli.predict`): V-JEPA 2 ViT-L (2 blok terakhir dilatih), V-JEPA 2 ViT-L dengan LoRA
   rank 16 di q/k/v semua blok, V-JEPA 2 ViT-L dengan LoRA rank 16 di attention + MLP (4 epoch), dan VideoMAE-B di
   crop orang (rata-rata 5 checkpoint fold, versi polos dan flip), serta head pose (5 fold).
3. **Stacking** (`cli.stack`): logistic regression di atas logit `p_cheat` tiap model, dilatih di prediksi
   out-of-fold data train, lalu diterapkan ke tiap klip test.
   - `submission.csv` (#7, public 0.8764): V-JEPA 2 + VideoMAE + head pose, threshold dari OOF (0.44).
   - `submission_lora.csv` (#13, public 0.8738): kelima model, threshold tetap 0.43 (titik optimal F1-macro;
     argmax OOF tidak stabil karena kurva F1 OOF datar di 0.40-0.55).

Kepatuhan aturan: **setiap sequence test diprediksi sendiri-sendiri** (FAQ 27/38/39). Batch hanya untuk kecepatan,
tidak ada statistik, urutan, pengelompokan, atau pseudo-label dari test (FAQ 35). Label train per jendela
(`configs/onsets.csv`) dihitung dari prediksi OOF data train saja (FAQ 40).

Kode lengkap (training, label onset, stacking) ada di dataset `dataslayer-src` (= repo tim). Checkpoint hasil training
(`colab/run_job.sh`, Colab A100) ada di dataset `dataslayer-ckpt-*`; OOF dan config tiap run di `dataslayer-runs`.
Runtime: GPU T4, sekitar 100 menit."""

C1 = """import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

INPUT = Path("/kaggle/input")
COMP = next(INPUT.rglob("sample_submission.csv")).parent
CODE = next(INPUT.rglob("src/core/config.py")).parents[2]
RUNS_IN = next(INPUT.rglob("vjepa2_l_crop_win/config.yaml")).parents[1]
WORK = Path("/kaggle/working")
OUT, CACHE = WORK / "outputs", Path("/tmp/cache")  # noqa: S108 - features stay out of the notebook output
CKPT = {"vjepa2_l_crop_win": "dataslayer-ckpt-vjepa2-l-crop-win",
        "vjepa2_l_lora16_crop_win": "dataslayer-ckpt-vjepa2-l-lora16-crop-win",
        "vjepa2_l_lora16mlp_e4_crop_win": "dataslayer-ckpt-vjepa2-l-lora16mlp-e4-crop-win",
        "videomae_b_crop_win": "dataslayer-ckpt-videomae-b-crop-win"}
RUNS = ["vjepa2_l_crop_win", "vjepa2_l_lora16_crop_win", "vjepa2_l_lora16mlp_e4_crop_win", "videomae_b_crop_win",
        "feat_pose_y26s_win"]
LORA_THR = 0.43
os.environ["PYTHONPATH"] = str(CODE / "src")
os.environ["TQDM_MININTERVAL"] = "30"
PATHS = f"paths.data_dir={COMP} paths.cache_dir={CACHE} paths.output_dir={OUT}"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=WORK)


sh("pip -q install ultralytics")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
print(COMP, CODE, RUNS_IN, sep="\\n")"""

C2 = """# Run folders: OOF + config from dataslayer-runs, fold checkpoints linked from the checkpoint datasets.
for run in RUNS:
    shutil.rmtree(OUT / run, ignore_errors=True)
    shutil.copytree(RUNS_IN / run, OUT / run)
    if run in CKPT:
        # Kaggle mounts datasets at /kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>: search by slug.
        ckpts = sorted(p for p in INPUT.rglob("fold*.pt") if CKPT[run] in p.parts)
        if len(ckpts) != 5:
            raise FileNotFoundError(f"{run}: expected 5 fold checkpoints in dataset {CKPT[run]}, found {len(ckpts)}")
        for pt in ckpts:
            (OUT / run / pt.stem / "best.pt").symlink_to(pt)
    print(run, sorted(p.parent.name for p in (OUT / run).glob("fold*/best.pt")))"""

C3 = """# Frozen YOLO-pose features of the test sequences only (no train videos.csv in this cache).
POSE_CFG = CODE / "configs" / "feat_pose.yaml"
sh(f"python -m cli.extract --config {POSE_CFG} --set {PATHS} features.backbone=yolo11s-pose --workers 4")
sh(f"python -m cli.extract --config {POSE_CFG} --set {PATHS} --workers 4")"""

C4 = """# Test predictions per run: mean of the 5 fold checkpoints (+ flip TTA for the video models).
# T4 has no bf16 -> fp16.
for run in RUNS:
    amp = "no" if run.startswith("feat_") else "fp16"
    sh(f"python -m cli.predict --config {OUT / run / 'config.yaml'} --set {PATHS} train.mixed_precision={amp} "
       "data.num_workers=4")"""

C5 = """# Stacking (logistic regression over the cheat logits, fit on the train OOF): the 3-model stack (#7, OOF
# threshold) -> submission.csv; the 5-model stack with both LoRA runs at the fixed threshold LORA_THR (#13)
# -> submission_lora.csv.
base = f"{OUT}/vjepa2_l_crop_win@flip {OUT}/videomae_b_crop_win@flip {OUT}/feat_pose_y26s_win"
lora = f"{OUT}/vjepa2_l_lora16_crop_win@flip {OUT}/vjepa2_l_lora16mlp_e4_crop_win@flip"
sh(f"python -m cli.stack --runs {base} --out {OUT}/stack_base")
sh(f"python -m cli.stack --runs {base} {lora} --out {OUT}/stack_lora")
probs = pd.read_csv(OUT / "stack_lora" / "test_probs.csv")
probs.assign(label=(probs.p_cheat >= LORA_THR).astype(int))[["id", "label"]].to_csv(
    OUT / "stack_lora" / "submission.csv", index=False)
sample = pd.read_csv(COMP / "sample_submission.csv")
for name, out, ref in [("submission.csv", "stack_base", "stack_win_ckpt_submission.csv"),
                       ("submission_lora.csv", "stack_lora", "stack_best3_t043_submission.csv")]:
    sub = pd.read_csv(OUT / out / "submission.csv").set_index("id").reindex(sample.id).reset_index()
    if sorted(sub.id) != sorted(sample.id):
        raise RuntimeError(f"{name}: ids differ from sample_submission.csv")
    sub.to_csv(WORK / name, index=False)
    r = pd.read_csv(RUNS_IN / "reference" / ref).set_index("id").label.reindex(sub.id).values
    print(f"{name}: {len(sub)} rows | cheat ratio {sub.label.mean():.3f} | "
          f"differs from the A100 reference on {(sub.label.values != r).sum()} clips")"""

cells = [("markdown", MD), ("code", C1), ("code", C2), ("code", C3), ("code", C4), ("code", C5)]
nb = {"cells": [{"cell_type": k, "id": f"c{i}", "metadata": {}, "source": src,
                 **({"outputs": [], "execution_count": None} if k == "code" else {})}
                for i, (k, src) in enumerate(cells)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path("kaggle/final")
out.mkdir(parents=True, exist_ok=True)
(out / "final.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
meta = {"id": "ridwansigma/dataslayer-final", "title": "dataslayer-final", "code_file": "final.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["ridwansigma/dataslayer-src", "ridwansigma/dataslayer-runs",
                            "ridwansigma/dataslayer-ckpt-vjepa2-l-crop-win",
                            "ridwansigma/dataslayer-ckpt-videomae-b-crop-win",
                            "ridwansigma/dataslayer-ckpt-vjepa2-l-lora16-crop-win",
                            "ridwansigma/dataslayer-ckpt-vjepa2-l-lora16mlp-e4-crop-win"],
        "competition_sources": ["data-slayer-4"], "kernel_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
for _, src in cells[1:]:
    for n, line in enumerate(src.splitlines()):
        if len(line) > 120:
            print("LONG", n, len(line))
print("written")