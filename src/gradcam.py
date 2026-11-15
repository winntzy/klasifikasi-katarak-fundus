"""Grad-CAM untuk backbone yang dibungkus sebagai layer ke-1 pada model."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config


def make_gradcam_heatmap(img_array, model, last_conv_name):
    """Heatmap Grad-CAM (nilai 0 sampai 1) untuk satu citra yang sudah dinormalisasi.

    img_array berukuran (1, H, W, 3), float32, dengan normalisasi yang sama
    seperti saat training.
    """
    backbone = model.layers[1]
    if last_conv_name not in [l.name for l in backbone.layers]:
        raise ValueError(f"Layer '{last_conv_name}' tidak ditemukan di backbone.")

    conv_layer = backbone.get_layer(last_conv_name)
    backbone_ext = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.output],
        name="backbone_ext",
    )

    with tf.GradientTape() as tape:
        conv_outputs, feat = backbone_ext(tf.cast(img_array, tf.float32), training=False)

        # Forward manual melalui head (GAP, Dropout, Dense)
        x = feat
        for layer in model.layers[2:]:
            x = layer(x)
        class_score = x[:, 0]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise ValueError(f"Gradien None untuk layer '{last_conv_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(img_rgb_uint8, heatmap, alpha=0.45):
    """Tumpuk heatmap (colormap JET) di atas citra RGB uint8."""
    h, w = img_rgb_uint8.shape[:2]
    hm_u8 = np.uint8(255 * cv2.resize(heatmap, (w, h)))
    hm_col = cv2.cvtColor(cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb_uint8, 1 - alpha, hm_col, alpha, 0)


def visualize_gradcam(model, X_model, X_display, y_true, results, last_conv_name,
                      model_name, out_dir=config.IMAGES_DIR / "gradcam", n_each=4):
    """Grad-CAM untuk TP, TN, dan FN (maksimal 2 FN) pada test set.

    X_model   : citra test yang sudah dinormalisasi untuk model (float32)
    X_display : citra test CLAHE uint8 (RGB), untuk ditampilkan
    results   : keluaran evaluation.evaluate (memuat y_pred dan y_prob)
    """
    y_pred = results["y_pred"]
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0][:n_each]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0][:n_each]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0][:2]

    groups = []
    if len(tp_idx):
        groups.append(("True Positive (Katarak)", tp_idx))
    if len(tn_idx):
        groups.append(("True Negative (Normal)", tn_idx))
    if len(fn_idx):
        groups.append(("False Negative (Terlewat)", fn_idx))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_title, indices in groups:
        n = len(indices)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f"Grad-CAM {model_name}\n{group_title}", fontsize=12, fontweight="bold")

        for row, idx in enumerate(indices):
            hm = make_gradcam_heatmap(X_model[idx][np.newaxis], model, last_conv_name)
            img = X_display[idx]
            true = "Katarak" if y_true[idx] == 1 else "Normal"
            pred = "Katarak" if y_pred[idx] == 1 else "Normal"

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Citra CLAHE\nAktual: {true}", fontsize=9)
            axes[row, 1].imshow(hm, cmap="jet")
            axes[row, 1].set_title("Heatmap Grad-CAM", fontsize=9)
            axes[row, 2].imshow(overlay_heatmap(img, hm))
            axes[row, 2].set_title(f"Overlay\nPrediksi: {pred} (p={results['y_prob'][idx]:.3f})",
                                   fontsize=9)
            for ax in axes[row]:
                ax.axis("off")

        plt.tight_layout()
        safe_title = group_title.split("(")[0].strip().replace(" ", "_").lower()
        path = out_dir / f"{model_name}_{safe_title}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Grad-CAM disimpan: {path}")
