"""Fungsi plot untuk EDA, preprocessing, riwayat training, dan evaluasi.

Modul ini tidak membutuhkan TensorFlow.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from . import config
from .preprocessing import apply_clahe, fundus_mask

COLOR_NORMAL = "#27ae60"
COLOR_CATARACT = "#e74c3c"


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_eda(df_eye, save_path=None):
    """Distribusi kelas, usia, dan jenis kelamin per kelas (sebelum balancing)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EDA Dataset ODIR-5K Per-Citra (sebelum balancing)",
                 fontsize=14, fontweight="bold")

    counts = df_eye["label"].value_counts().sort_index()
    vals = [counts.get(0, 0), counts.get(1, 0)]
    bars = axes[0].bar(["Normal (0)", "Katarak (1)"], vals,
                       color=[COLOR_NORMAL, COLOR_CATARACT], alpha=0.85,
                       edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     str(val), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Distribusi Kelas (sebelum balancing)", fontweight="bold")
    axes[0].set_ylabel("Jumlah Citra")
    axes[0].grid(axis="y", alpha=0.3)

    for lbl, clr, name in [(0, COLOR_NORMAL, "Normal"), (1, COLOR_CATARACT, "Katarak")]:
        axes[1].hist(df_eye[df_eye["label"] == lbl]["age"],
                     bins=20, alpha=0.7, color=clr, label=name)
    axes[1].set_title("Distribusi Usia per Kelas", fontweight="bold")
    axes[1].set_xlabel("Usia (tahun)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    sex_labels = sorted(df_eye["sex"].unique())
    x = np.arange(len(sex_labels))
    w = 0.35
    for offset, lbl, clr, name in [(-w / 2, 1, COLOR_CATARACT, "Katarak"),
                                   (w / 2, 0, COLOR_NORMAL, "Normal")]:
        v = [df_eye[(df_eye["label"] == lbl) & (df_eye["sex"] == s)].shape[0]
             for s in sex_labels]
        axes[2].bar(x + offset, v, w, label=name, color=clr,
                    alpha=0.85, edgecolor="black", linewidth=0.8)
    axes[2].set_title("Distribusi Jenis Kelamin per Kelas", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sex_labels)
    axes[2].set_ylabel("Jumlah")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_clahe_comparison(sample_paths, titles, save_path=None):
    """Citra asli, hasil CLAHE bermask, dan histogram kanal L area fundus."""
    fig, axes = plt.subplots(len(sample_paths), 4, figsize=(18, 4.5 * len(sample_paths)))
    fig.suptitle("CLAHE dengan Mask Fundus", fontsize=13, fontweight="bold")

    for row, (path, ttl) in enumerate(zip(sample_paths, titles)):
        img_bgr = cv2.imread(str(path))
        img_bgr = cv2.resize(img_bgr, config.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_clahe = apply_clahe(img_bgr)

        lab_before = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab_after = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2LAB)

        # Mask untuk histogram memakai parameter yang sama seperti notebook asli
        # (ambang 10, kernel 15), lebih ketat daripada mask di dalam apply_clahe.
        mask = fundus_mask(img_bgr, threshold=10, kernel_size=15)
        l_before = lab_before[:, :, 0][mask > 0]
        l_after = lab_after[:, :, 0][mask > 0]

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"[{ttl}] Asli", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_clahe)
        axes[row, 1].set_title(f"[{ttl}] Setelah CLAHE + Mask", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].hist(l_before, bins=50, color="#3498db", alpha=0.8, density=True)
        axes[row, 2].set_title("Histogram L, sebelum\n(area fundus saja)", fontsize=9)
        axes[row, 2].set_xlabel("Intensitas")
        axes[row, 2].grid(alpha=0.3)

        axes[row, 3].hist(l_after, bins=50, color="#e67e22", alpha=0.8, density=True)
        axes[row, 3].set_title("Histogram L, setelah CLAHE + Mask\n(area fundus saja)", fontsize=9)
        axes[row, 3].set_xlabel("Intensitas")
        axes[row, 3].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_history(history_df, model_name, save_path=None):
    """Kurva loss, accuracy, dan AUC-ROC (train vs validasi).

    history_df: DataFrame riwayat training (lihat training.history_to_df).
    """
    metrics = [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC-ROC")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history_df) + 1)
    boundary = None
    if "phase" in history_df.columns and history_df["phase"].nunique() > 1:
        boundary = int((history_df["phase"] == history_df["phase"].iloc[0]).sum()) + 0.5

    for ax, (m, label) in zip(axes, metrics):
        if m not in history_df.columns:
            continue
        ax.plot(epochs, history_df[m], "b-o", ms=4, lw=1.8, label="Train")
        ax.plot(epochs, history_df[f"val_{m}"], "r-o", ms=4, lw=1.8, label="Validation")
        if boundary is not None:
            ax.axvline(boundary, color="gray", ls="--", lw=1, label="Mulai fine-tuning")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Training History {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_evaluation(y_true, y_prob, cm, threshold, auc, model_name, save_path=None):
    """Confusion matrix dan kurva ROC untuk satu model."""
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluasi Model {model_name}", fontsize=13, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Katarak"], yticklabels=["Normal", "Katarak"],
                ax=axes[0], annot_kws={"size": 16, "weight": "bold"})
    axes[0].set_xlabel("Prediksi", fontsize=12)
    axes[0].set_ylabel("Aktual", fontsize=12)
    axes[0].set_title(f"Confusion Matrix (thr={threshold:.2f})", fontweight="bold")
    axes[0].text(1.0, -0.12, f"TP={tp} TN={tn} FP={fp} FN={fn}",
                 transform=axes[0].transAxes, ha="right", fontsize=9, color="gray")

    fpr, tpr, thrs = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, "b-", lw=2.5, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "r--", lw=1.5, label="Random")
    axes[1].fill_between(fpr, tpr, alpha=0.08, color="blue")
    thr_idx = np.argmin(np.abs(thrs - threshold))
    axes[1].scatter(fpr[thr_idx], tpr[thr_idx], s=80, zorder=5,
                    color="red", label=f"thr={threshold:.2f}")
    axes[1].set_xlabel("False Positive Rate", fontsize=12)
    axes[1].set_ylabel("True Positive Rate", fontsize=12)
    axes[1].set_title("Kurva ROC", fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_roc_comparison(results_by_model, y_true, save_path=None):
    """Kurva ROC beberapa model pada test set yang sama."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, res in results_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
        ax.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {res['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Perbandingan Kurva ROC (test set)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
