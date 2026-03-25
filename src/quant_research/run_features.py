#!/usr/bin/env python3
"""Feature evaluation entrypoint.

Compute IC / ICIR / Sharpe for one or more features by using the feature values
as the alpha signal (identity mapping), aligned with the simplest workflow in
run_experiment.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from quant_research.backtest.engine import evaluate_return_t_p1
    from quant_research.backtest.metrics import sharpe_ratio
    from quant_research.data.loader import load_target_series
    from quant_research.data.handler import FeatureHandler
    from quant_research.features.feature101 import compute_alpha
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from quant_research.backtest.engine import evaluate_return_t_p1
    from quant_research.backtest.metrics import sharpe_ratio
    from quant_research.data.loader import load_target_series
    from quant_research.data.handler import FeatureHandler
    from quant_research.features.feature101 import compute_alpha


@dataclass(frozen=True)
class Config:
    data_path: Path
    hdf_key: str
    target_col: str
    feature_names: Tuple[str, ...]
    output_dir: Path
    t_exec: int
    ic_window: int
    report_top_k: int


def environment_init() -> None:
    np.random.seed(42)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)


def dataset_formatting(readouts: pd.DataFrame) -> pd.DataFrame:
    readouts = readouts.copy()
    readouts["date"] = readouts.index
    return readouts


def forward_return_series(
    target_raw: pd.DataFrame, readouts: pd.DataFrame, t_exec: int
) -> pd.Series:
    tmp = readouts.copy()
    tmp["_ones"] = 1.0
    return evaluate_return_t_p1(target_raw, tmp, t_exec, "_ones")


def compute_ic(feature: pd.Series, forward_returns: pd.Series) -> float:
    aligned = pd.concat(
        {"feature": feature, "forward_return": forward_returns}, axis=1
    ).dropna()
    if aligned.empty:
        return float("nan")
    return float(aligned["feature"].corr(aligned["forward_return"]))


def compute_icir(
    feature: pd.Series, forward_returns: pd.Series, window: int
) -> float:
    aligned = pd.concat(
        {"feature": feature, "forward_return": forward_returns}, axis=1
    ).dropna()
    if aligned.empty or len(aligned) < window:
        return float("nan")
    ic_series = aligned["feature"].rolling(window).corr(aligned["forward_return"])
    ic_series = ic_series.dropna()
    if ic_series.empty:
        return float("nan")
    std = ic_series.std(ddof=0)
    if std == 0:
        return float("nan")
    return float(ic_series.mean() / std)


def compute_strategy_returns(
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    feature_series: pd.Series,
    t_exec: int,
    action_col: str,
) -> pd.Series:
    tmp = readouts.copy()
    tmp[action_col] = feature_series
    return evaluate_return_t_p1(target_raw, tmp, t_exec, action_col)


def evaluate_feature_metrics(
    feature_name: str,
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    forward_returns: pd.Series,
    cfg: Config,
) -> dict:
    if feature_name not in readouts.columns:
        raise ValueError(
            f"Feature {feature_name!r} not found in readouts columns: {list(readouts.columns)}"
        )

    feature_series = readouts[feature_name]
    aligned_feature = feature_series.loc[forward_returns.index]

    ic_value = compute_ic(aligned_feature, forward_returns)
    icir_value = compute_icir(aligned_feature, forward_returns, cfg.ic_window)

    action_col = f"action_{feature_name}"
    strategy_returns = compute_strategy_returns(
        target_raw, readouts, feature_series, cfg.t_exec, action_col
    )
    sharpe_value = sharpe_ratio(strategy_returns)

    return {
        "feature": feature_name,
        "t_exec": cfg.t_exec,
        "ic": ic_value,
        "icir": icir_value,
        "ic_window": cfg.ic_window,
        "sharpe": sharpe_value,
        "samples": int(len(forward_returns.dropna())),
    }


def _normalize_feature_names(feature_names: Iterable[str]) -> Tuple[str, ...]:
    names = [name.strip() for name in feature_names if name.strip()]
    if not names:
        raise ValueError("feature_names is empty.")
    return tuple(names)


def _ensure_alpha_columns(readouts: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Compute alpha columns that are requested but not present in readouts."""
    readouts = readouts.copy()
    for name in feature_names:
        if not name.startswith("alpha_"):
            continue
        if name in readouts.columns:
            continue
        readouts[name] = compute_alpha(name, readouts)
    return readouts


def _plot_report(metrics: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate the image report. "
            "Install it (e.g. `pip install matplotlib`) and rerun."
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics.copy()
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    metrics["abs_ic"] = metrics["ic"].abs()

    def _short_label(name: str) -> str:
        # alpha_001 -> 001 for compact charts
        if name.startswith("alpha_") and len(name) == 9:
            return name[-3:]
        return name

    def _save_bar(df: pd.DataFrame, title: str, sort_col: str, value_col: str, filename: str) -> None:
        df = df.dropna(subset=[sort_col, value_col]).sort_values(sort_col, ascending=False).head(top_k)
        labels = [_short_label(x) for x in df["feature"].tolist()]
        values = df[value_col].tolist()

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(range(len(values)), values, color="#2f6fb0")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    # Save IC-ranked chart first, per request.
    _save_bar(
        metrics,
        f"Top {top_k} |IC| (sorted by |IC|, ic_window={int(metrics['ic_window'].iloc[0])})",
        "abs_ic",
        "ic",
        "report_01_top_abs_ic.png",
    )
    _save_bar(
        metrics,
        f"Top {top_k} Sharpe (t_exec={int(metrics['t_exec'].iloc[0])})",
        "sharpe",
        "sharpe",
        "report_02_top_sharpe.png",
    )
    _save_bar(
        metrics,
        f"Top {top_k} ICIR (ic_window={int(metrics['ic_window'].iloc[0])})",
        "icir",
        "icir",
        "report_03_top_icir.png",
    )

    # Scatter: IC vs Sharpe
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(metrics["ic"], metrics["sharpe"], s=18, alpha=0.7, color="#2f6fb0")
    ax.axvline(0, color="black", linewidth=1, alpha=0.3)
    ax.axhline(0, color="black", linewidth=1, alpha=0.3)
    ax.set_title("IC vs Sharpe (all features)")
    ax.set_xlabel("IC")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "report_04_scatter_ic_sharpe.png", dpi=180)
    plt.close(fig)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    cfg = Config(
        data_path=project_root / "assets/MinutesIdx.h5",
        hdf_key="/d",
        target_col="IC",
        feature_names=(
            *(f"alpha_{i:03d}" for i in range(1, 102)),
        ),
        output_dir=project_root / "output/features_report",
        t_exec=59,
        ic_window=20,
        report_top_k=30,
    )

    environment_init()

    target_raw = load_target_series(cfg.data_path, cfg.hdf_key, cfg.target_col)
    handler = FeatureHandler()
    readouts = handler.build_features(target_raw)
    readouts = dataset_formatting(readouts)
    readouts = _ensure_alpha_columns(readouts, cfg.feature_names)

    forward_returns = forward_return_series(target_raw, readouts, cfg.t_exec)

    rows = []
    for feature_name in _normalize_feature_names(cfg.feature_names):
        rows.append(
            evaluate_feature_metrics(feature_name, target_raw, readouts, forward_returns, cfg)
        )

    metrics = pd.DataFrame(rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(cfg.output_dir / "metrics.csv", index=False)
    _plot_report(metrics, cfg.output_dir, cfg.report_top_k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
