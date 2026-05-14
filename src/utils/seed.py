"""
seed.py — Reproducibility Utilities
======================================

Sets random seeds for all sources of randomness to ensure
reproducible training runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Seeds:
        - Python's built-in random
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA (if available)
        - cuDNN deterministic mode

    Args:
        seed: Integer seed value.

    Note:
        Setting cuDNN to deterministic mode may slightly reduce
        performance but ensures reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")
