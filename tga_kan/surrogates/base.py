"""Surrogate interface. fit/predict/explain. transitions= is accepted by all
but used only by surrogates with a boundary penalty (TGA-KAN, Eq. 8)."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseSurrogate(ABC):
    @abstractmethod
    def fit(self, S: np.ndarray, A: np.ndarray, *, transitions=None) -> "BaseSurrogate":
        ...

    @abstractmethod
    def predict(self, S: np.ndarray) -> np.ndarray:
        ...

    def explain(self, **kwargs) -> dict:
        """Optional: return human-inspectable components. Default empty."""
        return {}

    # -- persistence (mặc định pickle; TGA-KAN override bằng torch.save) -----
    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, **kwargs):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
