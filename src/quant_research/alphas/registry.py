from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import pandas as pd


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    metadata: Dict[str, object]


_ALPHA_REGISTRY: Dict[str, AlphaSpec] = {}


def register_alpha(name: str, **metadata: object) -> Callable[[Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.Series]]:
    def decorator(func: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
        if name in _ALPHA_REGISTRY:
            raise ValueError(f"Alpha already registered: {name}")
        _ALPHA_REGISTRY[name] = AlphaSpec(name=name, func=func, metadata=dict(metadata))
        return func

    return decorator


def get_alpha(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    spec = _ALPHA_REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Unknown alpha: {name}")
    return spec.func


def list_alphas() -> Iterable[str]:
    return list(_ALPHA_REGISTRY.keys())
