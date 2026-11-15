"""Evaluasi model: pemilihan threshold dari val set dan metrik pada test set.

Modul ini tidak membutuhkan TensorFlow. Probabilitas prediksi dihitung
di notebook dengan model.predict().
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from . import config
from .visualization import plot_evaluation


def find_best_threshold(y_true, y_prob, metric="f1"):
    """Cari threshold optimal (0.10 sampai 0.90) berdasarkan metrik pada val set."""
    best_t, best_score = 0.50, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred_t, zero_division=0)
        elif metric == "sensitivity":
            score = recall_score(y_true, y_pred_t, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred_t)
        if score > best_score:
            best_score, best_t = score, t
    return round(best_t, 2), round(best_score, 4)


def evaluate(model_name, y_val, p_val, y_test, p_test, threshold=None,
             plot_path=None, verbose=True):
    """Evaluasi pada test set.

    Jika threshold None, threshold dicari dari val set (maksimum F1).
    Threshold tidak pernah dicari dari test set.
    """
    if threshold is None:
        threshold, thr_f1 = find_best_threshold(y_val, p_val, metric="f1")
        if verbose:
            print(f"  Threshold optimal (dari val set, F1={thr_f1:.4f}): {threshold}")

    auc = roc_auc_score(y_test, p_test)
    y_pred = (p_test >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if verbose:
        print("=" * 62)
        print(f"  HASIL EVALUASI {model_name}  (threshold = {threshold:.2f})")
        print("=" * 62)
        print(f"  {'Accuracy':<24}: {acc:.4f}  ({acc * 100:.2f} %)")
        print(f"  {'Precision':<24}: {prec:.4f}")
        print(f"  {'Sensitivity (Recall)':<24}: {rec:.4f}")
        print(f"  {'Specificity':<24}: {spec:.4f}")
        print(f"  {'F1-Score':<24}: {f1:.4f}")
        print(f"  {'AUC-ROC':<24}: {auc:.4f}")
        print("-" * 62)
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print("=" * 62)
        plot_evaluation(y_test, p_test, cm, threshold, auc, model_name, plot_path)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Katarak"],
                                    zero_division=0))

    return {
        "model": model_name, "threshold": threshold,
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "f1_score": f1, "auc_roc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "cm": cm, "y_pred": y_pred, "y_prob": p_test,
    }


METRIC_COLUMNS = ["model", "threshold", "accuracy", "precision", "recall",
                  "specificity", "f1_score", "auc_roc", "tp", "tn", "fp", "fn"]


def metrics_table(results_by_model) -> pd.DataFrame:
    """Tabel metrik satu baris per model."""
    rows = [{k: r[k] for k in METRIC_COLUMNS} for r in results_by_model.values()]
    df = pd.DataFrame(rows)
    num_cols = ["accuracy", "precision", "recall", "specificity", "f1_score", "auc_roc"]
    df[num_cols] = df[num_cols].round(4)
    return df


def save_results(results_by_model, df_test, results_dir=config.RESULTS_DIR):
    """Simpan metrik (satu CSV gabungan) dan probabilitas prediksi test set per model."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    table = metrics_table(results_by_model)
    table.to_csv(results_dir / "metrics_comparison.csv", index=False)
    for name, r in results_by_model.items():
        pd.DataFrame({
            "filename": df_test["filename"].values,
            "patient_id": df_test["patient_id"].values,
            "y_true": df_test["label"].values,
            "y_prob": np.round(r["y_prob"], 6),
            "y_pred": r["y_pred"],
        }).to_csv(results_dir / f"predictions_{name}.csv", index=False)
    return table
