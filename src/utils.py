"""Fungsi bantu umum."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set seed Python, NumPy, dan TensorFlow (jika terpasang)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
