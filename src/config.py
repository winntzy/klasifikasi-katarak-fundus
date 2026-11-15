"""Konfigurasi proyek: path, hyperparameter, dan pengaturan eksperimen.

Path dataset dapat diubah lewat variabel lingkungan ODIR_DATASET_PATH.
Struktur yang diharapkan di dalam folder dataset:

    <DATASET_PATH>/Images/      citra fundus (*.jpg)
    <DATASET_PATH>/data.xlsx    file anotasi pasien
"""
import os
from pathlib import Path

# --- Path ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = Path(os.environ.get("ODIR_DATASET_PATH", ROOT / "data" / "ODIR-5K"))
IMG_DIR = DATASET_PATH / "Images"
ANNO_FILE = DATASET_PATH / "data.xlsx"

PROCESSED_DIR = ROOT / "data" / "processed"   # cache citra setelah CLAHE (tidak di-commit)
MODELS_DIR = ROOT / "models"                  # bobot model (tidak di-commit)
RESULTS_DIR = ROOT / "results"                # metrik dan riwayat training (CSV)
SPLITS_DIR = RESULTS_DIR / "splits"           # daftar citra per split (CSV)
IMAGES_DIR = ROOT / "images"                  # gambar untuk README

# --- Umum ---------------------------------------------------------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Split (per pasien) -------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- CLAHE --------------------------------------------------------------------
CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)

# --- Balancing (hanya data latih) ---------------------------------------------
# Citra katarak diduplikasi N_DUP kali, sehingga jumlahnya menjadi (1 + N_DUP) kali lipat.
N_DUP = 3

# --- Model --------------------------------------------------------------------
MODEL_NAMES = ("ResNet50", "EfficientNetB0")
WEIGHTS = "imagenet"      # bobot awal backbone
DROPOUT = 0.4
L2_REG = 1e-4

# --- Training dua fase --------------------------------------------------------
# Fase 1: backbone dibekukan, hanya head yang dilatih.
LR = 1e-4
EPOCHS_HEAD = 15
# Fase 2: fine-tuning N_UNFREEZE layer terakhir backbone.
FINE_TUNE_LR = 1e-5
EPOCHS_FINE_TUNE = 25
N_UNFREEZE = 30
# Early stopping memantau val_auc.
EARLY_STOP_PATIENCE = 12
