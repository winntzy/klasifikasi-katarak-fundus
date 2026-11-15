"""Balancing data latih dan training dua fase (head, lalu fine-tuning)."""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from . import config
from .models import MODEL_SPECS, build_model
from .utils import set_seed

AUTOTUNE = tf.data.AUTOTUNE


def prepare_inputs(X_uint8, preprocess_fn):
    """Terapkan normalisasi khusus backbone pada citra uint8 hasil CLAHE.

    Return array float32 baru. Array asli tidak diubah.
    """
    return preprocess_fn(X_uint8.astype(np.float32))


def oversample_minority(X, y, n_dup=config.N_DUP, seed=config.SEED):
    """Duplikasi citra katarak (kelas 1) sebanyak n_dup kali. Hanya untuk data latih."""
    idx_cat = np.where(y == 1)[0]
    idx_nor = np.where(y == 0)[0]
    idx = np.concatenate([idx_nor, np.tile(idx_cat, n_dup)])
    np.random.RandomState(seed).shuffle(idx)
    return X[idx], y[idx]


@tf.function
def augment_fn(image, label):
    """Flip horizontal dan vertikal acak, hanya untuk kelas katarak."""
    def augment_minority(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    image = tf.cond(tf.equal(label, 1),
                    lambda: augment_minority(image),
                    lambda: image)
    return image, label


def make_dataset(X, y, augment_data=False, batch_size=config.BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment_data:
        ds = ds.shuffle(buffer_size=len(X), seed=config.SEED)
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def compile_model(model, lr):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def history_to_df(history, phase: str) -> pd.DataFrame:
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    df.insert(0, "phase", phase)
    return df


def run_training(model_name, X_train, y_train, X_val, y_val,
                 models_dir=config.MODELS_DIR, weights=config.WEIGHTS, verbose=1):
    """Latih satu model dengan dua fase.

    Fase 1: backbone beku, head dilatih (LR=config.LR).
    Fase 2: N_UNFREEZE layer terakhir backbone dibuka (LR=config.FINE_TUNE_LR).

    X_train dan X_val berupa citra uint8 hasil CLAHE. Normalisasi khusus backbone
    dilakukan di sini. Model terbaik (val_auc tertinggi) disimpan di
    models_dir/<model_name>_best.keras.

    Return (model, history_df). history_df memuat kolom phase dan epoch.
    """
    spec = MODEL_SPECS[model_name]
    set_seed(config.SEED)

    X_bal, y_bal = oversample_minority(X_train, y_train)
    nc, nn = int((y_bal == 1).sum()), int((y_bal == 0).sum())
    print(f"[{model_name}] Train setelah oversampling: {len(y_bal)} "
          f"(katarak {nc}, normal {nn}, rasio 1:{nn / nc:.2f})")

    ds_train = make_dataset(prepare_inputs(X_bal, spec.preprocess_input), y_bal,
                            augment_data=True)
    ds_val = make_dataset(prepare_inputs(X_val, spec.preprocess_input), y_val)
    del X_bal

    model, base_model = build_model(model_name, weights=weights)
    trainable = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"[{model_name}] Parameter total: {model.count_params():,} | "
          f"dilatih (head): {trainable:,}")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Callback yang sama dipakai di kedua fase. ModelCheckpoint hanya menyimpan
    # bobot jika val_auc melampaui nilai terbaik dari fase mana pun.
    callbacks = [
        ModelCheckpoint(filepath=str(models_dir / f"{model_name}_best.keras"),
                        monitor="val_auc", mode="max", save_best_only=True, verbose=verbose),
        EarlyStopping(monitor="val_auc", mode="max",
                      patience=config.EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=verbose),
    ]

    # Fase 1: backbone beku
    print(f"[{model_name}] Fase 1: backbone beku, LR={config.LR}, "
          f"maks {config.EPOCHS_HEAD} epoch")
    compile_model(model, config.LR)
    hist_head = model.fit(ds_train, epochs=config.EPOCHS_HEAD,
                          validation_data=ds_val, callbacks=callbacks, verbose=verbose)

    # Fase 2: fine-tuning N_UNFREEZE layer terakhir
    base_model.trainable = True
    for layer in base_model.layers[:-config.N_UNFREEZE]:
        layer.trainable = False
    for layer in base_model.layers[-config.N_UNFREEZE:]:
        layer.trainable = True
    n_train_layers = sum(layer.trainable for layer in base_model.layers)
    print(f"[{model_name}] Fase 2: fine-tuning, layer backbone yang dilatih: {n_train_layers}, "
          f"LR={config.FINE_TUNE_LR}, maks {config.EPOCHS_FINE_TUNE} epoch")
    compile_model(model, config.FINE_TUNE_LR)
    hist_ft = model.fit(ds_train, epochs=config.EPOCHS_FINE_TUNE,
                        validation_data=ds_val, callbacks=callbacks, verbose=verbose)

    history_df = pd.concat([history_to_df(hist_head, "head"),
                            history_to_df(hist_ft, "fine_tune")], ignore_index=True)
    return model, history_df


def free_memory():
    """Bebaskan memori Keras sebelum melatih model berikutnya."""
    tf.keras.backend.clear_session()
    gc.collect()
