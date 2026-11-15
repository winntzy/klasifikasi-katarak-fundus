"""Definisi model: backbone ImageNet yang dibekukan dan head klasifikasi biner."""
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.applications import efficientnet, resnet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from . import config


@dataclass(frozen=True)
class ModelSpec:
    """Perbedaan antar backbone: konstruktor, normalisasi input, dan layer Grad-CAM."""
    name: str
    builder: Callable
    preprocess_input: Callable
    last_conv_layer: str


MODEL_SPECS = {
    # ResNet50: preprocess_input mode caffe (BGR, dikurangi mean ImageNet).
    "ResNet50": ModelSpec(
        name="ResNet50",
        builder=ResNet50,
        preprocess_input=resnet50.preprocess_input,
        last_conv_layer="conv5_block3_out",
    ),
    # EfficientNetB0: normalisasi sudah ada di dalam backbone,
    # sehingga preprocess_input tidak mengubah nilai piksel (rentang [0, 255]).
    "EfficientNetB0": ModelSpec(
        name="EfficientNetB0",
        builder=EfficientNetB0,
        preprocess_input=efficientnet.preprocess_input,
        last_conv_layer="top_conv",
    ),
}


def build_model(model_name, input_shape=(224, 224, 3), weights=config.WEIGHTS):
    """Bangun model: backbone (frozen) -> GAP -> Dropout -> Dense(1, sigmoid, L2).

    Return (model, base_model). Layer ke-1 pada model adalah backbone,
    dan src/gradcam.py bergantung pada susunan ini.
    """
    spec = MODEL_SPECS[model_name]
    base = spec.builder(weights=weights, include_top=False, input_shape=input_shape)
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.DROPOUT)(x)
    outputs = Dense(1, activation="sigmoid",
                    kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG))(x)
    return Model(inputs, outputs, name=model_name), base
