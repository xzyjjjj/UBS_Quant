#!/usr/bin/env python3
"""Visualize feature evaluation results in output/.

Default selection: alpha_001 ~ alpha_010.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Config:
    output_dir: Path
    feature_names: Tuple[str, ...]
    report_dir: Path


def _normalize_feature_names(feature_names: Iterable[str]) -> Tuple[str, ...]:
    names = [name.strip() for name in feature_names if name.strip()]
    if not names:
        raise ValueError("feature_names is empty.")
    return tuple(names)


def _load_metrics(output_dir: Path, feature_name: str) -> pd.DataFrame:
    metrics_path = output_dir / f"feature_{feature_name}" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))
    return pd.read_csv(metrics_path)


def _load_equity_curve(output_dir: Path, feature_name: str) -> pd.Series:
    curve_path = output_dir / f"feature_{feature_name}" / "equity_curve.csv"
    if not curve_path.exists():
        raise FileNotFoundError(str(curve_path))
    curve_df = pd.read_csv(curve_path, index_col=0)
    if curve_df.shape[1] == 0:
        raise ValueError(f"Empty equity curve file: {curve_path}")
    curve = curve_df.iloc[:, 0]
    return curve


def _collect_metrics(cfg: Config) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    missing: List[str] = []
    for feature_name in _normalize_feature_names(cfg.feature_names):
        try:
            df = _load_metrics(cfg.output_dir, feature_name)
            rows.append(df)
        except FileNotFoundError:
            missing.append(feature_name)
    if missing:
        print(f"[warn] metrics missing for: {', '.join(missing)}")
    if not rows:
        raise ValueError("No metrics found for selected features.")
    out = pd.concat(rows, ignore_index=True)
    return out


def _collect_equity_curves(cfg: Config) -> pd.DataFrame:
    curves: List[pd.Series] = []
    missing: List[str] = []
    for feature_name in _normalize_feature_names(cfg.feature_names):
        try:
            curve = _load_equity_curve(cfg.output_dir, feature_name)
            curve.name = feature_name
            curves.append(curve)
        except FileNotFoundError:
            missing.append(feature_name)
    if missing:
        print(f"[warn] equity_curve missing for: {', '.join(missing)}")
    if not curves:
        raise ValueError("No equity curves found for selected features.")
    return pd.concat(curves, axis=1)


def _plot_metrics(metrics: pd.DataFrame, report_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = metrics.copy()
    metrics = metrics.sort_values("feature")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(metrics["feature"], metrics["ic"], color="#2f6fb0")
    axes[0].set_title("IC")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(metrics["feature"], metrics["icir"], color="#3aa66f")
    axes[1].set_title("ICIR")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(metrics["feature"], metrics["sharpe"], color="#d9534f")
    axes[2].set_title("Sharpe")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(report_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)


def _plot_equity_curves(curves: pd.DataFrame, report_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for col in curves.columns:
        ax.plot(curves.index, curves[col], label=col)
    ax.set_title("Equity Curves")
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "equity_curves.png", dpi=150)
    plt.close(fig)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    cfg = Config(
        output_dir=project_root / "output/single_feature",
        feature_names=(
            "alpha_001",
            "alpha_002",
            "alpha_003",
            "alpha_004",
            "alpha_005",
            "alpha_006",
            "alpha_007",
            "alpha_008",
            "alpha_009",
            "alpha_010",
        ),
        report_dir=project_root / "output/feature_report",
    )

    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = _collect_metrics(cfg)
    except ValueError as exc:
        print(f"[error] {exc}")
        print("[hint] Please run: python src/quant_research/run_feature.py")
        return 1
    metrics.to_csv(cfg.report_dir / "metrics_summary.csv", index=False)

    try:
        curves = _collect_equity_curves(cfg)
    except ValueError:
        curves = None

    _plot_metrics(metrics, cfg.report_dir)
    if curves is not None:
        _plot_equity_curves(curves, cfg.report_dir)

    print(f"[ok] report saved to: {cfg.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
