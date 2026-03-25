#!/usr/bin/env python3
"""Grid search for a single feature.

Evaluate one feature across (t_exec, ic_window) grids. The feature values are
used directly as the action (identity mapping), consistent with run_feature.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from quant_research.backtest.engine import evaluate_return_t_p1
    from quant_research.backtest.metrics import sharpe_ratio
    from quant_research.data.loader import load_target_series
    from quant_research.data.handler import FeatureHandler
    from quant_research.portfolio.allocator import equity_curve
    from quant_research.features.feature101 import compute_alpha
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from quant_research.backtest.engine import evaluate_return_t_p1
    from quant_research.backtest.metrics import sharpe_ratio
    from quant_research.data.loader import load_target_series
    from quant_research.data.handler import FeatureHandler
    from quant_research.portfolio.allocator import equity_curve
    from quant_research.features.feature101 import compute_alpha


@dataclass(frozen=True)
class Config:
    data_path: Path
    hdf_key: str
    target_col: str
    feature_name: str
    output_dir: Path
    t_exec_grid: Tuple[int, ...]
    alpha_param_grid: Dict[str, Tuple[int, ...]]
    rank_by: str
    save_detail_top_k: int


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


def _normalize_grid(values: Iterable[int]) -> Tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if not out:
        raise ValueError("Grid values are empty.")
    return out


def _iter_param_grid(param_grid: Dict[str, Tuple[int, ...]]) -> List[Dict[str, int]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = []
    for key in keys:
        raw = param_grid[key]
        if isinstance(raw, (int, np.integer)):
            values.append((int(raw),))
        else:
            values.append(tuple(int(v) for v in raw))

    for vals in values:
        if not vals:
            raise ValueError("alpha_param_grid contains empty value list.")
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _param_signature(params: Dict[str, int]) -> str:
    if not params:
        return "default"
    return "_".join(f"{key}{value}" for key, value in sorted(params.items()))


def _rank_candidates(metrics: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    if rank_by not in metrics.columns:
        raise ValueError(f"rank_by={rank_by!r} not found in metrics columns.")
    return metrics.sort_values(rank_by, ascending=False)


def _save_detail(
    output_dir: Path,
    feature_series: pd.Series,
    forward_returns: pd.Series,
    strategy_returns: pd.Series,
    t_exec: int,
    param_sig: str,
) -> None:
    detail_dir = output_dir / f"{param_sig}_t_exec_{t_exec}"
    detail_dir.mkdir(parents=True, exist_ok=True)
    feature_series.to_csv(detail_dir / "feature.csv", index=True)
    forward_returns.to_csv(detail_dir / "forward_returns.csv", index=True)
    strategy_returns.to_csv(detail_dir / "strategy_returns.csv", index=True)
    equity_curve(strategy_returns).to_csv(detail_dir / "equity_curve.csv", index=True)


def _plot_lines(metrics: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if metrics.empty:
        return

    for param_sig in metrics["param_sig"].dropna().unique():
        subset = metrics[metrics["param_sig"] == param_sig]
        if subset.empty:
            continue
        for metric in ("ic", "sharpe"):
            series = subset.sort_values("t_exec").set_index("t_exec")[metric]
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(series.index, series.values, marker="o")
            ax.set_title(f"{metric} vs t_exec ({param_sig})")
            ax.set_xlabel("t_exec")
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(output_dir / f"line_{metric}_{param_sig}.png", dpi=150)
            plt.close(fig)


def _plot_ranking(metrics: pd.DataFrame, output_dir: Path, rank_by: str, top_k: int = 20) -> None:
    import matplotlib.pyplot as plt

    if metrics.empty or rank_by not in metrics.columns:
        return

    ranked = metrics.sort_values(rank_by, ascending=False).head(top_k).copy()
    if ranked.empty:
        return

    labels = []
    for _, row in ranked.iterrows():
        param_sig = row.get("param_sig", "default")
        labels.append(f"{param_sig}|t{int(row['t_exec'])}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, ranked[rank_by], color="#2f6fb0")
    ax.set_title(f"Top {len(ranked)} by {rank_by}")
    ax.set_ylabel(rank_by)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / f"ranking_{rank_by}.png", dpi=150)
    plt.close(fig)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    cfg = Config(
        data_path=project_root / "assets/MinutesIdx.h5",
        hdf_key="/d",
        target_col="IC",
        feature_name="alpha_007",
        output_dir=project_root / "output/single_feature",
        t_exec_grid=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 239),
        alpha_param_grid={
            "adv_window": (27, 30, 32, 34),
            "delta_window": 6,
            "ts_rank_window": 30,
        },
        rank_by="sharpe",
        save_detail_top_k=3,
    )

    environment_init()

    target_raw = load_target_series(cfg.data_path, cfg.hdf_key, cfg.target_col)
    handler = FeatureHandler()
    readouts = handler.build_features(target_raw)
    readouts = dataset_formatting(readouts)

    if cfg.feature_name not in readouts.columns and not cfg.feature_name.startswith("alpha_"):
        raise ValueError(
            f"Feature {cfg.feature_name!r} not found in readouts columns: {list(readouts.columns)}"
        )

    t_exec_grid = _normalize_grid(cfg.t_exec_grid)
    param_combos = _iter_param_grid(cfg.alpha_param_grid)

    output_dir = cfg.output_dir / f"feature_{cfg.feature_name}" / "grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[dict] = []
    forward_returns_cache: dict[int, pd.Series] = {}
    strategy_returns_cache: dict[tuple[str, int], pd.Series] = {}

    for params in param_combos:
        param_sig = _param_signature(params)
        if cfg.feature_name.startswith("alpha_"):
            feature_series = compute_alpha(cfg.feature_name, readouts, **params)
        else:
            feature_series = readouts[cfg.feature_name]

        for t_exec in t_exec_grid:
            if t_exec not in forward_returns_cache:
                forward_returns_cache[t_exec] = forward_return_series(target_raw, readouts, t_exec)
            forward_returns = forward_returns_cache[t_exec]
            aligned_feature = feature_series.loc[forward_returns.index]
            action_col = f"action_{cfg.feature_name}"
            strategy_returns = compute_strategy_returns(
                target_raw, readouts, feature_series, t_exec, action_col
            )
            strategy_returns_cache[(param_sig, t_exec)] = strategy_returns
            sharpe_value = sharpe_ratio(strategy_returns)

            ic_value = compute_ic(aligned_feature, forward_returns)

            row: Dict[str, Any] = {
                "feature": cfg.feature_name,
                "t_exec": t_exec,
                "ic": ic_value,
                "sharpe": sharpe_value,
                "samples": int(len(forward_returns.dropna())),
                "param_sig": param_sig,
            }
            row.update(params)
            metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)
    ranked = _rank_candidates(metrics, cfg.rank_by)
    _plot_lines(metrics, output_dir)
    _plot_ranking(metrics, output_dir, cfg.rank_by)

    if cfg.save_detail_top_k > 0:
        top_k = ranked.head(cfg.save_detail_top_k)
        for _, row in top_k.iterrows():
            t_exec = int(row["t_exec"])
            params = {
                key: int(row[key])
                for key in cfg.alpha_param_grid.keys()
                if key in row and pd.notna(row[key])
            }
            param_sig = _param_signature(params)
            forward_returns = forward_returns_cache[t_exec]
            strategy_returns = strategy_returns_cache[(param_sig, t_exec)]
            if cfg.feature_name.startswith("alpha_"):
                feature_series = compute_alpha(cfg.feature_name, readouts, **params)
            else:
                feature_series = readouts[cfg.feature_name]
            _save_detail(
                output_dir,
                feature_series,
                forward_returns,
                strategy_returns,
                t_exec,
                param_sig,
            )

    print(f"[ok] grid search saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
