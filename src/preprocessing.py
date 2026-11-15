"""Preprocessing citra fundus: resize, mask fundus, dan CLAHE.

Modul ini tidak membutuhkan TensorFlow. Hasilnya adalah citra RGB uint8.
Normalisasi piksel khusus tiap backbone (preprocess_input) dilakukan
terpisah di src/training.py, sehingga satu cache citra dapat dipakai
oleh kedua model.
"""
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import config


def fundus_mask(img_bgr, threshold: int = 5, kernel_size: int = 9):
    """Mask area fundus (piksel non-hitam), dirapikan dengan closing lalu opening."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def apply_clahe(img_bgr, clip_limit=config.CLIP_LIMIT, tile_grid=config.TILE_GRID):
    """CLAHE dengan mask fundus. Background hitam tidak ikut dikalibrasi.

    Langkah:
      1. Deteksi mask area fundus (piksel non-hitam).
      2. Isi background sementara dengan median L area fundus,
         agar tile CLAHE tidak terdistorsi piksel hitam di tepi.
      3. Terapkan CLAHE pada channel L (ruang warna Lab).
      4. Kembalikan background ke hitam.

    Input : citra BGR (format OpenCV)
    Output: citra RGB uint8 setelah CLAHE
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = fundus_mask(img_bgr)

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel = img_lab[:, :, 0].copy()

    fundus_pixels = l_channel[mask > 0]
    median_l = int(np.median(fundus_pixels)) if len(fundus_pixels) else 128
    l_for_clahe = np.where(mask > 0, l_channel, median_l).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_for_clahe)

    img_lab[:, :, 0] = np.where(mask > 0, l_enhanced, 0)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


def load_clahe_image(img_path, target_size=config.IMG_SIZE):
    """Baca citra, resize ke target_size, lalu terapkan CLAHE.

    Return RGB uint8 berukuran (H, W, 3), atau None jika file gagal dibaca.
    Dengan tile 8x8 pada 224x224, ukuran tile adalah 28x28 piksel.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return apply_clahe(img)


def load_split_images(df, desc: str = "Loading"):
    """Muat dan proses semua citra pada satu split.

    Return (X uint8 [N, H, W, 3], y int32 [N]). Citra yang gagal dibaca dilewati.
    """
    images, labels = [], []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img = load_clahe_image(row["img_path"])
        if img is not None:
            images.append(img)
            labels.append(row["label"])
        else:
            failed += 1
    if failed:
        print(f"  Peringatan: {failed} citra gagal diproses.")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def save_cache(name: str, X, y, cache_dir=config.PROCESSED_DIR) -> None:
    """Simpan citra hasil CLAHE dan label ke cache lokal (.npy)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{name}_X.npy", X)
    np.save(cache_dir / f"{name}_y.npy", y)


def load_cache(name: str, cache_dir=config.PROCESSED_DIR):
    """Muat cache yang dibuat save_cache. Jalankan notebook 01 lebih dulu."""
    cache_dir = Path(cache_dir)
    x_path, y_path = cache_dir / f"{name}_X.npy", cache_dir / f"{name}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Cache '{name}' tidak ditemukan di {cache_dir}. "
            "Jalankan notebooks/01_eda_dan_preprocessing.ipynb terlebih dahulu.")
    return np.load(x_path), np.load(y_path)
