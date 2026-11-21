"""Writes kaggle/final/final.ipynb (+ kernel-metadata.json): the final report notebook (Indonesian) that explains the
method, recomputes the validation numbers from the saved OOF predictions and reproduces the final submission #13.
Run from the repo root: `python kaggle/final/make_notebook.py`, then `kaggle kernels push -p kaggle/final`."""
import json
from pathlib import Path

MD_INTRO = """# Deteksi Kecurangan Ujian dari Klip Video 1 Detik
## Laporan akhir Machine Learning Competition Data Slayer 4.0

**Hasil akhir: submission #13, public leaderboard 0.8738 (F1-macro), nested CV per subjek 0.865.**

Notebook ini adalah laporan akhir sekaligus notebook yang menghasilkan submission final:

1. menjelaskan masalah, data, metode, validasi, dan eksperimen yang sudah dicoba;
2. **menghitung ulang angka validasi** dari prediksi out-of-fold (OOF) yang tersimpan (bagian 5);
3. **menjalankan inferensi test** dengan model yang sudah dilatih (checkpoint `.pt`) dan menulis `submission.csv` yang
   sama persis dengan submission final (bagian 6, diperiksa otomatis di sel terakhir).

**Ringkasan metode.** Setiap klip 1 detik dipotong di sekitar badan peserta (kotak dari YOLO-pose), lalu dinilai oleh
tiga model video berbasis **V-JEPA 2 ViT-L** (fine-tune 2 blok terakhir, LoRA rank 16, LoRA rank 16 + MLP),
**VideoMAE-B**, dan satu model kecil di atas **keypoint kepala/badan YOLO26s-pose**. Model dilatih dengan **label per
jendela**: bagian awal video curang (sebelum aksi dimulai) dianggap normal, sesuai aturan label test. Kelima
probabilitas digabung dengan **stacking logistic regression** yang dilatih di prediksi OOF, lalu diberi threshold 0.43.
Validasi selalu **per subjek** (5 fold), karena subjek test tidak ada di train.

**Daftar isi**

1. Masalah dan data
2. Temuan EDA
3. Metode
4. Validasi
5. Hasil (dihitung ulang di notebook ini)
6. Inferensi test dan `submission.csv`
7. Eksperimen yang dibuang dan pelajaran
8. Reproduksi training, seed, dan kepatuhan aturan

Notebook ini sengaja **tidak menampilkan frame video**, karena dataset berisi wajah peserta (data pribadi yang wajib
dijaga kerahasiaannya)."""

MD_DATA = """## 1. Masalah dan data

- **Input:** klip 1 detik (30 frame JPEG 640x360) seorang peserta ujian. **Output:** 0 = normal, 1 = curang.
- **Metrik:** F1-macro (rata-rata F1 kelas normal dan kelas curang). Leaderboard: 30% public, 70% private.
- **Train:** 26 subjek x 8 perilaku x 2 kamera (depan dan samping) = 416 video, masing-masing sekitar 16 detik.
  Dipotong menjadi jendela 1 detik, jadi ada 6456 jendela OOF.
- **Test:** 1515 klip 1 detik dari subjek **yang tidak ada di train**.

| Normal (0) | Curang (1) |
|---|---|
| Checking_Smartwatch | Using_Smartwatch |
| Looking_Around | Repeated_Peeking |
| Thinking | Looking_Under_the_Desk |
| Writing_Answers | Reading_Hidden_Notes |

**Aturan label test (FAQ 33/41):** setiap klip test diberi label menurut **perilaku yang terlihat di klip itu**,
bukan label videonya. Detik-detik ketika peserta masih duduk atau menulis di dalam video curang berlabel **normal**.
Aturan ini yang paling menentukan desain solusi kami (bagian 3.1)."""

MD_EDA = """## 2. Temuan EDA

1. **Data seimbang:** setiap subjek melakukan kedelapan perilaku di kedua kamera (50% video normal, 50% curang).
2. **Kamera berbeda karakter:** kamera depan iPhone HDR (HEVC 10-bit HLG), kamera samping 8-bit biasa. Warnanya
   berbeda, karena itu dipakai augmentasi grayscale.
3. **Format test:** 640x360, JPEG kualitas 95 hasil OpenCV. Frame video train dibuat ulang dengan proses yang sama
   persis (`cli.prepare`), supaya model tidak belajar perbedaan format.
4. **Subjek test tidak ada di train**, jadi validasi wajib dibagi per subjek (GroupKFold). Pembagian acak per klip
   akan bocor (wajah dan baju yang sama ada di train dan validasi).
5. **Setiap rekaman diawali fase netral.** Rata-rata probabilitas curang OOF per detik datar di detik 0-6 untuk
   **semua** perilaku. Aksinya baru mulai sekitar detik 7-9 (median detik 8) dan umumnya berlangsung sampai akhir
   (grafik di bagian 5.3). Temuan ini berasal dari OOF data train sendiri, bukan dari test.
6. **Pasangan perilaku yang mirip:** Checking vs Using Smartwatch, Looking_Around vs Repeated_Peeking.
   Repeated_Peeking juga **terputus-putus** (menoleh beberapa detik, lalu kembali menulis).
7. Kamera depan dan samping merekam **bersamaan**. Ini hanya dipakai saat training (onset aksi kamera pasangannya),
   tidak pernah untuk menggabungkan prediksi test."""

MD_METHOD = """## 3. Metode

### 3.1 Label per jendela (onset aksi)

Karena label test mengikuti perilaku yang terlihat (bagian 1), melabeli seluruh video curang sebagai curang membuat
model belajar hal yang salah: fase netral di awal video curang terlihat sama dengan video normal. Solusinya:

- Untuk setiap video train diperkirakan **onset**, yaitu detik mulainya aksi, dari kurva probabilitas OOF per detik
  (`cli.onsets`, `dataset/onsets.py`). Baseline = rata-rata detik 0-4; onset = detik pertama (mulai detik 5) yang
  mencapai separuh jarak baseline ke puncak, dibatasi detik 6-10. Kurva yang lemah memakai onset kamera pasangannya,
  dan kalau tidak ada dipakai detik 8. Hasilnya `configs/onsets.csv` (hanya dari data train).
- Jendela yang **titik tengahnya sebelum onset** diberi label normal, dan target perilakunya diabaikan
  (`dataset/labels.py:window_target`). Jendela sesudah onset diberi label perilaku video.

Perubahan ini saja menaikkan public LB dari **0.729 ke 0.876**.

### 3.2 Crop orang

Kotak orang dihitung YOLO11s-pose (bobot COCO, tidak di-fine-tune) **dari klip itu sendiri**. Crop persegi dengan
margin 15% memperbesar tangan, jam tangan, dan kepala sekitar 1.5x.

### 3.3 Model

| Model | Backbone pretrained | Bagian yang dilatih | Input |
|---|---|---|---|
| V-JEPA 2 | ViT-L, Something-Something v2 | 2 blok terakhir + head | 16 frame, crop 256 |
| V-JEPA 2 + LoRA16 | sama | LoRA rank 16 di query/key/value ke-24 blok + head | 16 frame, crop 256 |
| V-JEPA 2 + LoRA16-MLP | sama | LoRA rank 16 di attention (q/k/v + output) dan MLP + head | 16 frame, crop 256 |
| VideoMAE-B | ViT-B, Kinetics-400 | semua lapisan | 16 frame, crop 224 |
| Head pose | YOLO26s-pose beku (COCO) | transformer temporal kecil di atas keypoint | keypoint tiap 2 frame |

Nama bobot Hugging Face ada di tabel resep bagian 5.1. Keypoint head pose relatif terhadap kotak orang.

- V-JEPA 2 (Assran et al. 2025, arXiv 2506.09985) dipilih karena dilatih di video gerakan tangan dan benda
  (Something-Something v2), yang paling mirip dengan aksi menyontek. Model ini menjadi model terkuat.
- LoRA (Hu et al. 2021, arXiv 2106.09685): encoder dibekukan, setiap linear layer mendapat tambahan low-rank `B(A(x))`
  yang dilatih. Cara ini menyesuaikan seluruh 24 blok dengan sedikit parameter, sehingga tidak cepat overfit di 26
  subjek.
- Head pose memakai koordinat relatif terhadap kotak orang, jadi tidak terpengaruh baju, wajah, atau warna kamera.
  Model ini lemah sendirian, tetapi menambah informasi yang berbeda di stacking.

### 3.4 Training

- **Loss:** cross-entropy 8 perilaku (label smoothing 0.1) + loss biner pada `p_cheat` = jumlah peluang 4 perilaku
  curang. Jendela sebelum onset hanya ikut di loss biner.
- **Sampling:** 16 jendela acak per video per epoch. Augmentasi: crop acak 70-100%, kecepatan +-20%, grayscale 10%,
  flip horizontal. Augmentasi yang sama dipakai untuk semua frame satu klip.
- **Optimasi:** AdamW, warmup 1 epoch lalu cosine, EMA bobot. **Jumlah epoch tetap** (bukan memilih epoch terbaik per
  fold, yang membuat OOF terlalu optimistis 0.02-0.04): 2 epoch untuk model video (berhenti di epoch 2 dari jadwal
  4/10), 4 epoch untuk LoRA16-MLP, 7 epoch untuk head pose. Detail tiap run ada di tabel bagian 5.1.
- **Perangkat:** model video di Colab A100 (bf16), head pose di laptop. Checkpoint 5 fold tiap model disimpan sebagai
  dataset Kaggle dan dipakai notebook ini.

### 3.5 Flip TTA

Validasi dan test dihitung dua kali: frame asli dan frame dibalik horizontal. Versi yang dipakai dipilih dari AUC OOF:
untuk keempat model video, versi flip menang tipis (`@flip`). Kamera samping selalu dari sisi yang sama, jadi petunjuk
berarah (jam di tangan kiri, arah mengintip) berubah saat di-flip.

### 3.6 Stacking dan threshold

- Fitur stacking = logit `p_cheat` kelima model untuk satu klip. **Logistic regression** dilatih di OOF gabungan
  (label per jendela), lalu diterapkan apa adanya ke **setiap klip test sendiri-sendiri** (`cli.stack`). Stacking
  mempelajari seberapa besar setiap model bisa dipercaya. Di nested CV hasilnya lebih baik dari rata-rata bobot sama.
- **Threshold 0.43.** Kurva F1 OOF datar di 0.40-0.55 (grafik bagian 5.3), sehingga "threshold F1 terbaik di OOF"
  melompat acak di dalam rentang itu. 0.43 adalah titik di dalam plateau yang juga terpilih dari OOF untuk stack
  3 model sebelumnya. Di nested CV, F1 stack ini pada 0.43 hanya sekitar 0.002 di bawah F1 pada threshold yang dipilih
  per fold (tabel bagian 5.2)."""

MD_VALID = """## 4. Validasi

- **5 fold per subjek:** 26 subjek dibagi 6/5/5/5/5. Kamera depan dan samping satu subjek selalu di fold yang sama.
  Model fold k dilatih di 4 fold lain dan memprediksi fold k. Gabungan prediksinya = **OOF**, dipakai untuk memilih
  model, melatih stacking, dan memilih threshold.
- **Label proksi:** label test tidak tersedia untuk train, jadi OOF dinilai dengan label yang meniru aturan test:
  jendela video curang dianggap curang mulai onset. Dipakai tiga versi: onset dari tabel (`configs/onsets.csv`),
  onset tetap detik 7, dan onset tetap detik 8. Proksi 7s/8s sama untuk semua model, jadi adil untuk membandingkan.
- **Nested CV:** stacking dan threshold juga merupakan langkah pemilihan. Supaya jujur, keduanya dilatih di 4 fold dan
  dinilai di fold ke-5, bergantian. Angka ini yang dipakai untuk memilih submission final.
- Proksi nested selalu sekitar 0.01-0.03 **di bawah** public LB dengan urutan yang sama, jadi keputusan diambil dari
  proksi OOF, bukan dari public LB (sekitar 450 klip, terlalu sedikit untuk membedakan selisih di bawah 0.01)."""

C_SETUP = """import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

# Seed: sama dengan train.seed semua run (pembagian fold dan training). Inferensi sendiri tidak memakai keacakan.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

INPUT = Path("/kaggle/input")
COMP = next(INPUT.rglob("sample_submission.csv")).parent
CODE = next(INPUT.rglob("src/core/config.py")).parents[2]
RUNS_IN = next(INPUT.rglob("vjepa2_l_crop_win/config.yaml")).parents[1]
WORK = Path("/kaggle/working")
OUT, CACHE = WORK / "outputs", Path("/tmp/cache")  # noqa: S108 - fitur test tidak ikut ke output notebook
sys.path.insert(0, str(CODE / "src"))
from evaluation.metrics import best_threshold, binary_metrics  # noqa: E402
from evaluation.stack import KEYS, cheat_logit  # noqa: E402

# Stack final #13: lima model (@flip = versi flip TTA), threshold tetap 0.43.
STACK = ["vjepa2_l_crop_win@flip", "videomae_b_crop_win@flip", "feat_pose_y26s_win", "vjepa2_l_lora16_crop_win@flip",
         "vjepa2_l_lora16mlp_e4_crop_win@flip"]
RUNS = [s.split("@")[0] for s in STACK]
THR = 0.43
CKPT = {"vjepa2_l_crop_win": "dataslayer-ckpt-vjepa2-l-crop-win",
        "vjepa2_l_lora16_crop_win": "dataslayer-ckpt-vjepa2-l-lora16-crop-win",
        "vjepa2_l_lora16mlp_e4_crop_win": "dataslayer-ckpt-vjepa2-l-lora16mlp-e4-crop-win",
        "videomae_b_crop_win": "dataslayer-ckpt-videomae-b-crop-win"}
os.environ["PYTHONPATH"] = str(CODE / "src")
os.environ["TQDM_MININTERVAL"] = "30"
PATHS = f"paths.data_dir={COMP} paths.cache_dir={CACHE} paths.output_dir={OUT}"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=WORK)


sh("pip -q install ultralytics")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
print(COMP, CODE, RUNS_IN, sep="\\n")"""

MD_RESULTS = """## 5. Hasil (dihitung ulang di notebook ini)

Semua angka di bagian ini dihitung dari file OOF di dataset `dataslayer-runs` (prediksi setiap jendela train oleh
model fold yang tidak pernah melihat subjeknya).

### 5.1 Resep training tiap model

Diambil dari `config.yaml` yang disimpan saat training. Sel ini juga memastikan semua run memakai seed 42."""

C_RECIPE = """rows = []
for run in RUNS:
    cfg = yaml.safe_load((RUNS_IN / run / "config.yaml").read_text())
    d, m, t, f = cfg["data"], cfg["model"], cfg["train"], cfg["features"]
    if t["seed"] != SEED:
        raise ValueError(f"{run}: train.seed {t['seed']} != {SEED}")
    if m["kind"] == "features":
        trained, backbone, inp = "head temporal (pose beku)", f["backbone"], f"keypoint tiap {f['stride']} frame"
    elif m.get("lora_rank"):
        trained = f"LoRA r{m['lora_rank']} " + ("attention + MLP" if m.get("lora_mlp") else "q/k/v")
        backbone, inp = m["backbone"], f"{d['num_frames']} frame, crop {d['img_size'][0]}"
    else:
        trained = "semua lapisan" if m["train_blocks"] < 0 else f"{m['train_blocks']} blok terakhir"
        backbone, inp = m["backbone"], f"{d['num_frames']} frame, crop {d['img_size'][0]}"
    epochs = f"{t['stop_epoch']} (jadwal {t['epochs']})" if t["stop_epoch"] else str(t["epochs"])
    rows.append({"run": run, "backbone": backbone, "dilatih": trained, "input": inp, "epoch": epochs,
                 "batch efektif": t["batch_size"] * t["grad_accum"], "lr head": t["lr"],
                 "lr backbone": t["backbone_lr"], "EMA": t["ema"], "presisi": t["mixed_precision"], "seed": t["seed"]})
pd.DataFrame(rows).set_index("run")"""

MD_OOF = """### 5.2 Skor OOF per model dan nested CV stacking

Tabel pertama: setiap model sendiri di OOF (label onset dari tabel). Tabel kedua: **nested CV per fold subjek**,
dinilai dengan proksi onset tetap detik 7 dan 8. Stacking dan threshold dilatih di 4 fold, dinilai di fold ke-5.
"thr dipilih per fold" = threshold F1 terbaik di 4 fold latih; "thr 0.43" = threshold tetap yang dipakai submission."""

C_OOF = """def load_oof(spec: str) -> pd.DataFrame:
    run, _, variant = spec.partition("@")
    return pd.read_csv(RUNS_IN / run / f"oof{'_' + variant if variant else ''}.csv")


oofs = [load_oof(s) for s in STACK]
common = oofs[0][KEYS]
for o in oofs[1:]:
    common = common.merge(o[KEYS], on=KEYS)
oofs = [common.merge(o, on=KEYS).sort_values(KEYS, ignore_index=True) for o in oofs]
base = oofs[0]
y = base.label.to_numpy()
cheat_video = base.label.groupby(base.video_id).transform("max").eq(1)
mid = base.start + 15
proxy = {s: (cheat_video & (mid >= 30 * s)).astype(int).to_numpy() for s in (7, 8)}
print(f"{len(base)} jendela OOF, {base.video_id.nunique()} video, {base.subject.nunique()} subjek, "
      f"{base.fold.nunique()} fold")

rows = []
for spec, o in zip(STACK, oofs):
    p = o.p_cheat.to_numpy()
    rows.append({"model": spec, "AUC": roc_auc_score(y, p), f"F1@{THR}": binary_metrics(p, y, THR)["f1_macro"],
                 "F1 (thr terbaik)": best_threshold(p, y)[1]})
pd.DataFrame(rows).set_index("model").round(4)"""

C_NESTED = """X = np.stack([cheat_logit(o.p_cheat) for o in oofs], 1)


def nested(cols: list[int], thr: float | None) -> tuple[np.ndarray, np.ndarray]:
    prob, pred = np.zeros(len(base)), np.zeros(len(base), int)
    for f in sorted(base.fold.unique()):
        tr, te = (base.fold != f).to_numpy(), (base.fold == f).to_numpy()
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr][:, cols], y[tr])
        t = thr if thr is not None else best_threshold(lr.predict_proba(X[tr][:, cols])[:, 1], y[tr])[0]
        prob[te] = lr.predict_proba(X[te][:, cols])[:, 1]
        pred[te] = prob[te] >= t
    return prob, pred


CANDIDATES = {"V-JEPA 2 saja": [0], "#7: V-JEPA 2 + VideoMAE + pose": [0, 1, 2], "#9: #7 + LoRA16": [0, 1, 2, 3],
              "#13 (final): #9 + LoRA16-MLP": [0, 1, 2, 3, 4]}
rows = []
for name, cols in CANDIDATES.items():
    for rule, thr in (("thr dipilih per fold", None), (f"thr {THR}", THR)):
        prob, pred = nested(cols, thr)
        f1 = {s: f1_score(proxy[s], pred, average="macro") for s in proxy}
        rows.append({"stack": name, "threshold": rule, "F1 proksi 7s": f1[7], "F1 proksi 8s": f1[8],
                     "F1 rata-rata": np.mean(list(f1.values())),
                     "AUC rata-rata": np.mean([roc_auc_score(proxy[s], prob) for s in proxy])})
display(pd.DataFrame(rows).set_index(["stack", "threshold"]).round(4))

full = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
print("Koefisien stacking final (dilatih di seluruh OOF):")
print(pd.Series(full.coef_[0], index=STACK).round(3).to_string(), f"\\nintercept {full.intercept_[0]:.3f}")"""

MD_PLOTS = """### 5.3 Fase netral, plateau threshold, dan kesalahan per perilaku

Kiri: rata-rata probabilitas curang (nested OOF stack final) per detik. Video curang baru naik sekitar detik 7-8, dan
inilah alasan label per jendela. Kanan: F1-macro nested terhadap threshold. Kurvanya datar di sekitar 0.40-0.55,
sehingga threshold tetap 0.43 aman."""

C_PLOTS = """prob13, pred13 = nested(list(range(len(STACK))), THR)
sec = ((base.start + 15) // 30).to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for flag, name in ((True, "video curang"), (False, "video normal")):
    m = cheat_video.to_numpy() == flag
    curve = pd.Series(prob13[m]).groupby(sec[m]).mean()
    ax[0].plot(curve.index, curve.values, marker="o", label=name)
ax[0].axhline(THR, color="gray", ls="--", lw=1, label=f"threshold {THR}")
ax[0].set(xlabel="detik ke- (titik tengah jendela)", ylabel="rata-rata p_cheat", title="Probabilitas curang per detik")
ax[0].legend()
grid = np.round(np.arange(0.2, 0.81, 0.01), 2)
labels = {"proksi 7s": proxy[7], "proksi 8s": proxy[8], "onset tabel": y}
for name, target in labels.items():
    ax[1].plot(grid, [f1_score(target, prob13 >= t, average="macro") for t in grid], label=name)
ax[1].axvline(THR, color="gray", ls="--", lw=1)
ax[1].set(xlabel="threshold", ylabel="F1-macro (nested)", title="F1 terhadap threshold")
ax[1].legend()
plt.tight_layout()
plt.show()

behavior = base.video_id.str.extract(r"_(?:front|side)_(.+)$")[0]
tab = pd.DataFrame({"perilaku": behavior, "label": y, "pred": pred13})
summary = tab.groupby(["perilaku", "label"]).pred.agg(["mean", "size"]).unstack("label")
summary.columns = [f"{'frac. diprediksi curang' if c == 'mean' else 'n jendela'} (label {lab})"
                   for c, lab in summary.columns]
print("Label 1 = jendela curang sesudah onset (fraksi = recall); label 0 = jendela normal (fraksi = false positive).")
summary.round(3)"""

MD_INFER = """## 6. Inferensi test dan `submission.csv`

Langkah-langkah di bawah memakai **model yang sudah dilatih** (tanpa training ulang):

1. Folder run: config dan OOF dari `dataslayer-runs`, checkpoint 5 fold (`fold*.pt`) dari dataset checkpoint.
2. Fitur YOLO-pose beku untuk klip test saja: kotak orang YOLO11s (untuk crop model video) dan keypoint YOLO26s (untuk
   head pose).
3. Prediksi tiap model: rata-rata 5 checkpoint fold, versi asli dan flip. T4 tidak mendukung bf16, jadi dipakai fp16.
4. Stacking: logistic regression di OOF (sama dengan bagian 5.2), diterapkan ke setiap klip test sendiri-sendiri,
   threshold 0.43, lalu `submission.csv`.

**Setiap klip test diprediksi sendiri-sendiri** (FAQ 27/38/39). Batch hanya untuk kecepatan: tidak ada statistik,
urutan, pengelompokan, atau pseudo-label dari test (FAQ 35). Runtime di GPU T4 sekitar 95 menit.

### 6.1 Model terlatih (checkpoint)

| Model | Dataset Kaggle | Isi |
|---|---|---|
| V-JEPA 2 ViT-L | `dataslayer-ckpt-vjepa2-l-crop-win` | `fold0.pt` ... `fold4.pt` |
| V-JEPA 2 + LoRA16 | `dataslayer-ckpt-vjepa2-l-lora16-crop-win` | `fold0.pt` ... `fold4.pt` |
| V-JEPA 2 + LoRA16-MLP | `dataslayer-ckpt-vjepa2-l-lora16mlp-e4-crop-win` | `fold0.pt` ... `fold4.pt` |
| VideoMAE-B | `dataslayer-ckpt-videomae-b-crop-win` | `fold0.pt` ... `fold4.pt` |
| Head pose YOLO26s | `dataslayer-runs` | `feat_pose_y26s_win/fold*/best.pt` |

Kode lengkap (training, label onset, stacking, test) ada di dataset `dataslayer-src`."""

C_LINK = """# Folder run: OOF + config dari dataslayer-runs, checkpoint fold di-link dari dataset checkpoint.
for run in RUNS:
    shutil.rmtree(OUT / run, ignore_errors=True)
    shutil.copytree(RUNS_IN / run, OUT / run)
    if run in CKPT:
        # Kaggle memasang dataset di /kaggle/input/<slug> atau /kaggle/input/datasets/<user>/<slug>: cari lewat slug.
        ckpts = sorted(p for p in INPUT.rglob("fold*.pt") if CKPT[run] in p.parts)
        if len(ckpts) != 5:
            raise FileNotFoundError(f"{run}: expected 5 fold checkpoints in dataset {CKPT[run]}, found {len(ckpts)}")
        for pt in ckpts:
            (OUT / run / pt.stem / "best.pt").symlink_to(pt)
    pts = sorted((OUT / run).glob("fold*/best.pt"))
    print(f"{run}: {len(pts)} checkpoint, {sum(p.stat().st_size for p in pts) / 1e9:.2f} GB")"""

C_EXTRACT = """# Fitur YOLO-pose beku untuk klip test saja (cache ini tidak berisi video train).
POSE_CFG = CODE / "configs" / "feat_pose.yaml"
sh(f"python -m cli.extract --config {POSE_CFG} --set {PATHS} features.backbone=yolo11s-pose --workers 4")
sh(f"python -m cli.extract --config {POSE_CFG} --set {PATHS} --workers 4")"""

C_PREDICT = """# Prediksi test per run: rata-rata 5 checkpoint fold (+ flip TTA model video). T4 tanpa bf16 -> fp16.
for run in RUNS:
    amp = "no" if run.startswith("feat_") else "fp16"
    sh(f"python -m cli.predict --config {OUT / run / 'config.yaml'} --set {PATHS} train.mixed_precision={amp} "
       "data.num_workers=4")"""

C_SUBMIT = """# Stacking 5 model (LR di OOF) -> p_cheat tiap klip test, threshold tetap THR -> submission.csv.
sh(f"python -m cli.stack --runs {' '.join(f'{OUT}/{s}' for s in STACK)} --out {OUT}/stack_final")
probs = pd.read_csv(OUT / "stack_final" / "test_probs.csv")
sample = pd.read_csv(COMP / "sample_submission.csv")
sub = probs.assign(label=(probs.p_cheat >= THR).astype(int))[["id", "label"]].set_index("id").reindex(sample.id)
if sub.label.isna().any():
    raise RuntimeError("submission ids differ from sample_submission.csv")
sub = sub.reset_index().astype({"label": int})
sub.to_csv(WORK / "submission.csv", index=False)
ref = pd.read_csv(RUNS_IN / "reference" / "stack_best3_t043_submission.csv").set_index("id").label.reindex(sub.id)
print(f"submission.csv: {len(sub)} baris, rasio curang {sub.label.mean():.3f}, "
      f"berbeda dari submission final #13 di {(sub.label.to_numpy() != ref.to_numpy()).sum()} klip")
sub.head()"""

MD_LOSERS = """## 7. Eksperimen yang dibuang dan pelajaran

### 7.1 Riwayat submission (public LB)

| # | Isi | Public | Catatan |
|---|---|---|---|
| awal | ConvNeXt / DINOv3 / VideoMAE / V-JEPA 2 / pose, **label per video** | 0.612-0.729 | asumsi label salah |
| 7 | V-JEPA 2 + VideoMAE + pose, **label per jendela**, stacking LR | 0.8764 | lompatan terbesar |
| 8 | #7 dengan onset versi 2 + detik netral dibersihkan | 0.8679 | lebih buruk |
| 9 | #7 + V-JEPA 2 LoRA16 | 0.8738 | |
| 10 | #9 + LoRA16-MLP dengan onset versi 2 | 0.8604 | onset versi 2 merugikan (kedua kalinya) |
| 11 | #9 + LoRA16-MLP 4 epoch, threshold OOF 0.53 | 0.8536 | threshold terlalu tinggi |
| 12 | #7 dengan threshold 0.38 | 0.8745 | menurunkan threshold tidak membantu |
| **13** | **#11 dengan threshold tetap 0.43** | **0.8738** | **final**: nested CV terbaik (0.865) |

**Kenapa #13 dan bukan #7 yang skor public-nya sedikit lebih tinggi?** Selisih public 0.0026 setara sekitar 1 klip
dari sekitar 450 klip public, sedangkan nested CV (seluruh 26 subjek train) memihak #13 dengan selisih sekitar 0.02
(tabel 5.2; bootstrap per subjek terhadap #9: 95% interval di atas 0). Skor akhir dihitung dari 70% klip private,
jadi kami memilih bukti validasi yang lebih kuat. Selain itu, checkpoint model asli #7 tidak tersimpan, sedangkan
#13 dibuat ulang persis oleh notebook ini.

### 7.2 Sudah dicoba lalu dibuang

| Ide | Hasil | Alasan |
|---|---|---|
| CNN per frame (ConvNeXt-T/B, DINOv3 beku) + temporal | AUC 0.60-0.65 (label video) | kalah jauh dari model video |
| Head pose YOLO11s / YOLO26m | lebih lemah / tidak menambah di stacking | YOLO26s dipakai |
| V-JEPA 2 ViT-g, 4 blok dilatih, LoRA rank 32, seed kedua | tidak lebih baik dari ViT-L 2 blok / LoRA16 | dibuang |
| V-JEPA 2 Diving48, VideoMAE SSv2, TimeSformer SSv2 | proksi 0.77-0.82 | lebih lemah dari V-JEPA 2 SSv2 |
| Crop lebih lebar / crop kepala + bahu | proksi turun | konteks tambahan atau crop sempit tidak membantu |
| Surgical fine-tuning (Lee et al. 2022, arXiv 2210.11466) | proksi 0.839 vs LoRA16 0.853 | kalah dari LoRA |
| Onset versi 2 / pembersihan detik netral | LB turun dua kali | detik setelah onset tetap berlabel curang |
| Rata-rata geometrik bobot sama / stacking di 24 log-prob perilaku | nested 0.834 / 0.821 | stacking LR lebih baik |
| Threshold terpisah per kamera | tidak membantu | satu threshold global |
| Memilih epoch terbaik per fold | OOF optimistis 0.02-0.04 | diganti jumlah epoch tetap |

### 7.3 Pelajaran

- **Baca ulang aturan label.** Mengganti label per video dengan label per jendela menaikkan LB 0.729 -> 0.876, jauh
  lebih besar dari semua perubahan arsitektur.
- **Validasi per subjek + proksi yang meniru aturan test** bisa dipercaya (selalu sedikit di bawah LB, urutan sama).
- **Kenaikan di bawah 0.01 tidak terlihat di public LB** (~450 klip). Jangan memilih final hanya dari public LB.
- **Threshold penting di test walau kurva OOF datar:** stack yang sama mendapat 0.8536 di threshold 0.53 dan 0.8738 di
  0.43.
- **Sisa kesalahan terbesar ada di Repeated_Peeking** (sering tertukar dengan Looking_Around, tabel 5.3). Perilakunya
  terputus-putus, jadi sebagian detik di dalamnya kemungkinan berlabel normal di test."""

MD_REPRO = """## 8. Reproduksi training, seed, dan kepatuhan aturan

### 8.1 Training ulang (tidak dijalankan di notebook ini)

Semua perintah dari root kode (`dataslayer-src`), dengan `src` di `PYTHONPATH`:

```bash
python -m cli.prepare --config configs/feat_pose.yaml --workers 6        # frame gaya test + fold per subjek
python -m cli.extract --config configs/feat_pose.yaml --set features.backbone=yolo11s-pose   # kotak orang
python -m cli.extract --config configs/feat_pose.yaml                    # keypoint YOLO26s
python -m cli.train --config configs/feat_pose.yaml && python -m cli.predict --config configs/feat_pose.yaml
python -m cli.onsets --oof <oof.csv model label-video> --out configs/onsets.csv   # onset dari OOF train
# Model video di GPU A100 (Colab), checkpoint di-upload ke dataset Kaggle:
bash colab/run_job.sh --ckpt configs/colab_vjepa2_win.yaml configs/colab_videomae_win.yaml \\
    configs/colab_vjepa2_lora16_win.yaml configs/colab_vjepa2_lora16mlp_e4_win.yaml
python -m cli.stack --runs outputs/vjepa2_l_crop_win@flip outputs/videomae_b_crop_win@flip outputs/feat_pose_y26s_win \\
    outputs/vjepa2_l_lora16_crop_win@flip outputs/vjepa2_l_lora16mlp_e4_crop_win@flip --out outputs/stack_best3
```

### 8.2 Seed dan determinisme

- Seed 42 dipakai untuk pembagian fold dan training (`train.seed`, fold k memakai seed 42 + k); diperiksa di sel 5.1.
- Inferensi tidak memakai keacakan (tanpa augmentasi acak, flip bersifat tetap). Satu-satunya sumber perbedaan adalah
  presisi GPU: model dilatih di A100 (bf16) dan notebook ini berjalan di T4 (fp16). Untuk submission final, selisihnya
  0 dari 1515 klip (sel terakhir bagian 6).

### 8.3 Kepatuhan aturan

| Aturan | Cara dipatuhi |
|---|---|
| Tanpa data eksternal; pretrained boleh | hanya bobot pretrained publik (Hugging Face, YOLO COCO) |
| Tanpa VLM, API pihak ketiga, atau prediksi zero-shot | semua model dilatih di data train dan dijalankan lokal |
| Setiap klip test diprediksi sendiri (FAQ 27/38/39) | crop, fitur, prediksi, dan stacking per klip |
| Tanpa statistik test, pseudo-label, transductive (FAQ 35/39) | stacking dan threshold dari OOF train saja |
| Anotasi onset train boleh, tanpa informasi test (FAQ 30/40) | onset dari kurva OOF train (`cli.onsets`) |
| Tanpa prediksi manual | semua label test berasal dari model |"""

cells = [("markdown", MD_INTRO), ("markdown", MD_DATA), ("markdown", MD_EDA), ("markdown", MD_METHOD),
         ("markdown", MD_VALID), ("code", C_SETUP), ("markdown", MD_RESULTS), ("code", C_RECIPE),
         ("markdown", MD_OOF), ("code", C_OOF), ("code", C_NESTED), ("markdown", MD_PLOTS), ("code", C_PLOTS),
         ("markdown", MD_INFER), ("code", C_LINK), ("code", C_EXTRACT), ("code", C_PREDICT), ("code", C_SUBMIT),
         ("markdown", MD_LOSERS), ("markdown", MD_REPRO)]
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
for _, src in cells:
    for n, line in enumerate(src.splitlines()):
        if len(line) > 120:
            print("LONG", n, len(line), line[:60])
print("written")