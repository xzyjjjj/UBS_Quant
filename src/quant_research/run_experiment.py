#!/usr/bin/env python3
"""Phase 0 experiment entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from quant_research.backtest.engine import evaluate_return_t_p1
    from quant_research.backtest.metrics import sharpe_ratio
    from quant_research.data.loader import load_target_series
    from quant_research.data.handler import FeatureHandler
    from quant_research.portfolio.allocator import equity_curve
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from quant_research.backtest.engine import evaluate_return_t_p1
    from quant_research.backtest.metrics import sharpe_ratio
    from quant_research.data.loader import load_target_series
    from quant_research.data.handler import FeatureHandler
    from quant_research.portfolio.allocator import equity_curve


@dataclass(frozen=True)
class Config:
    data_path: Path
    hdf_key: str
    target_col: str
    output_dir: Path
    t_exec: int
    exec_grid: Tuple[int, ...]
    exec_mode: str  # "fixed" | "best_sharpe" | "policy"
    exec_policy: Optional[Callable[[pd.DataFrame, pd.DataFrame, Tuple[int, ...]], int]]


def environment_init() -> None:
    np.random.seed(42)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)


def init_instance_by_config(config: Dict[str, Any]) -> Any:
    class_name = config["class"]
    kwargs = config.get("kwargs", {})
    module_path = config.get("module_path")
    if module_path:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
    else:
        cls = globals().get(class_name)
    if cls is None:
        raise ValueError(f"Unknown class: {class_name}")
    return cls(**kwargs)


class Experiment:
    def __init__(self, name: str, output_dir: Path) -> None:
        self.name = name
        self.output_dir = output_dir
        self.recorder = Recorder(output_dir)

    def __enter__(self) -> "Recorder":
        return self.recorder

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class Recorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.params: Dict[str, Any] = {}

    def log_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)


@dataclass
class SimpleDataset:
    data_path: Path
    hdf_key: str
    target_col: str
    handler: FeatureHandler
    target_raw: Optional[pd.DataFrame] = None
    readouts: Optional[pd.DataFrame] = None

    def prepare(self, split: str = "train") -> pd.DataFrame:
        _ = split
        self.target_raw = load_target_series(self.data_path, self.hdf_key, self.target_col)
        readouts = self.handler.build_features(self.target_raw)
        self.readouts = dataset_formatting(readouts)
        return self.readouts


@dataclass
class SimpleModel:
    alpha_col: str = "default_st"
    action_col: str = "action_default_st"
    fitted: bool = False

    def fit(self, dataset: SimpleDataset) -> None:
        _ = dataset
        self.fitted = True


class AlphaRecord:
    def __init__(self, model: SimpleModel, dataset: SimpleDataset, recorder: Recorder) -> None:
        self.model = model
        self.dataset = dataset
        self.recorder = recorder

    def generate(self) -> None:
        if self.dataset.readouts is None:
            raise ValueError("Dataset not prepared.")
        alpha_generation(self.dataset.readouts, self.recorder.output_dir, self.dataset.handler.alpha_list)


def alpha_to_action_identity(readouts: pd.DataFrame, alpha_col: str, action_col: str) -> None:
    readouts[action_col] = readouts[alpha_col]


class DecisionRecord:
    def __init__(
        self,
        model: SimpleModel,
        dataset: SimpleDataset,
        recorder: Recorder,
        exec_grid: Iterable[int],
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.recorder = recorder
        self.exec_grid = exec_grid

    def generate(self) -> pd.DataFrame:
        if self.dataset.readouts is None:
            raise ValueError("Dataset not prepared.")
        if self.dataset.target_raw is None:
            raise ValueError("Dataset not prepared.")
        alpha_to_action_identity(self.dataset.readouts, self.model.alpha_col, self.model.action_col)
        action_generation(self.dataset.readouts, self.recorder.output_dir, [self.model.action_col])
        return_generation(
            self.dataset.target_raw, self.dataset.readouts, self.exec_grid, self.model.action_col
        )
        sharpe_by_exec = sharpe_generation(self.dataset.readouts, self.exec_grid, self.model.action_col)
        if not sharpe_by_exec.empty:
            training_report(sharpe_by_exec, self.recorder.output_dir, self.model.action_col)
        return sharpe_by_exec

class PortAnaRecord:
    def __init__(
        self,
        recorder: Recorder,
        cfg: Config,
        sharpe_by_exec: pd.DataFrame,
        target_raw: pd.DataFrame,
        readouts: pd.DataFrame,
        action_col: str,
    ) -> None:
        self.recorder = recorder
        self.cfg = cfg
        self.sharpe_by_exec = sharpe_by_exec
        self.target_raw = target_raw
        self.readouts = readouts
        self.action_col = action_col

    def generate(self) -> tuple[pd.Series, int]:
        return backtest(self.cfg, self.sharpe_by_exec, self.target_raw, self.readouts, self.action_col)

def dataset_formatting(readouts: pd.DataFrame) -> pd.DataFrame:
    readouts = readouts.copy()
    readouts["date"] = readouts.index
    return readouts


def backtest_config(cfg: Config) -> Iterable[int]:
    return cfg.exec_grid


def data_sample(readouts: pd.DataFrame, output_dir: Path) -> None:
    readouts.head(10).to_csv(output_dir / "data_sample.csv", index=True)


def training_report(sharpe_by_exec: pd.DataFrame, output_dir: Path, action_col: str) -> None:
    sharpe_by_exec.to_csv(output_dir / f"sharpe_by_exec_{action_col}.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(sharpe_by_exec["t_exec"], sharpe_by_exec["sharpe"], c="blue", label="sharpe ratio")
    ax.scatter(sharpe_by_exec["t_exec"], sharpe_by_exec["sharpe"], c="blue")
    ax.set_title(f"time sensitivity of {action_col}")
    ax.set_xlabel("trading time (min)")
    ax.set_ylabel("sharpe ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"time_sensitivity_{action_col}.png", dpi=150)
    plt.close(fig)


def alpha_generation(
    readouts: pd.DataFrame,
    output_dir: Path,
    alpha_cols: Iterable[str],
) -> None:
    readouts[list(alpha_cols)].to_csv(output_dir / "alphas.csv", index=True)


def action_generation(
    readouts: pd.DataFrame,
    output_dir: Path,
    action_cols: Iterable[str],
) -> None:
    readouts[list(action_cols)].to_csv(output_dir / "actions.csv", index=True)


def return_generation(
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    exec_grid: Iterable[int],
    action_col: str,
) -> None:
    for t_exec in exec_grid:
        return_col = f"return_{action_col}_{t_exec}"
        if return_col in readouts.columns:
            continue
        returns = evaluate_return_t_p1(target_raw, readouts, t_exec, action_col)
        readouts[return_col] = returns


def sharpe_generation(
    readouts: pd.DataFrame,
    exec_grid: Iterable[int],
    action_col: str,
) -> pd.DataFrame:
    sharpe_rows = []
    for t_exec in exec_grid:
        return_col = f"return_{action_col}_{t_exec}"
        if return_col not in readouts.columns:
            raise ValueError(f"Missing return column: {return_col}")
        sharpe = sharpe_ratio(readouts[return_col])
        sharpe_rows.append({"t_exec": t_exec, "sharpe": sharpe})
    return pd.DataFrame(sharpe_rows)


def select_exec_time(
    cfg: Config,
    sharpe_by_exec: pd.DataFrame,
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
) -> int:
    if cfg.exec_mode == "fixed":
        return cfg.t_exec
    if cfg.exec_mode == "best_sharpe":
        if sharpe_by_exec.empty:
            raise ValueError("exec_mode=best_sharpe requires non-empty exec_grid.")
        row = sharpe_by_exec.loc[sharpe_by_exec["sharpe"].idxmax()]
        return int(row["t_exec"])
    if cfg.exec_mode == "policy":
        if cfg.exec_policy is None:
            raise ValueError("exec_mode=policy requires exec_policy.")
        return int(cfg.exec_policy(target_raw, readouts, cfg.exec_grid))
    raise ValueError(f"Unknown exec_mode: {cfg.exec_mode}")


def cal_return(
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    t_exec: int,
    action_col: str,
) -> pd.Series:
    return_col = f"return_{action_col}_{t_exec}"
    if return_col not in readouts.columns:
        returns = evaluate_return_t_p1(target_raw, readouts, t_exec, action_col)
        readouts[return_col] = returns
    return readouts[return_col]


def write_equity_curve(return_series: pd.Series, output_dir: Path) -> None:
    equity_curve(return_series).to_csv(output_dir / "equity_curve.csv", index=True)


def plot_equity_curve(
    return_series: pd.Series, output_dir: Path, title: str = "Equity Curve"
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to plot equity curve. "
            "Install it (e.g. `pip install matplotlib`) and rerun."
        ) from e

    equity = equity_curve(return_series)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(equity.index, equity.values, color="#2f6fb0")
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def summarize_performance(return_series: pd.Series, annualization: float = 250.0) -> Dict[str, float]:
    returns = return_series.dropna()
    if returns.empty:
        return {
            "samples": 0.0,
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    equity = equity_curve(returns)
    total_return = float(equity.iloc[-1] - 1.0)
    samples = float(len(returns))
    ann_return = float((1.0 + total_return) ** (annualization / samples) - 1.0)
    return {
        "samples": samples,
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
    }


def print_backtest_report(
    return_series: pd.Series, output_dir: Path, action_col: str, t_exec: int
) -> None:
    stats = summarize_performance(return_series)
    print("\n[report] backtest summary")
    print(f"[report] action_col: {action_col}")
    print(f"[report] t_exec: {t_exec}")
    print(f"[report] samples: {int(stats['samples'])}")
    print(f"[report] total_return: {stats['total_return']:.4f}")
    print(f"[report] annualized_return: {stats['annualized_return']:.4f}")
    print(f"[report] sharpe: {stats['sharpe']:.4f}")
    print(f"[report] max_drawdown: {stats['max_drawdown']:.4f}")
    print(f"[report] equity_curve.csv: {output_dir / 'equity_curve.csv'}")
    print(f"[report] equity_curve.png: {output_dir / 'equity_curve.png'}\n")


def print_saved_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    files = sorted(p for p in output_dir.iterdir() if p.is_file())
    if not files:
        return
    print("[report] saved artifacts:")
    for path in files:
        print(f"[report] - {path}")
    print("")


def backtest(
    cfg: Config,
    sharpe_by_exec: pd.DataFrame,
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    action_col: str,
) -> tuple[pd.Series, int]:
    """Select exec time, compute return series, and write equity curve."""
    chosen_exec = select_exec_time(cfg, sharpe_by_exec, target_raw, readouts)
    return_series = cal_return(target_raw, readouts, chosen_exec, action_col)
    write_equity_curve(return_series, cfg.output_dir)
    plot_equity_curve(
        return_series,
        cfg.output_dir,
        title=f"Equity Curve ({action_col}, t_exec={chosen_exec})",
    )
    return return_series, chosen_exec


def main() -> int:
    cfg = Config(
        data_path=Path("assets/MinutesIdx.h5"),
        hdf_key="/d",
        target_col="IC", # 从数据中抽取目标时间序列
        output_dir=Path("output/phase0"),
        t_exec=59, # 回测执行时刻（分钟）：你在当日的第几分钟下单/换仓
        exec_grid=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 239), # 回测执行时刻网格（多个分钟点），用于遍历计算 Sharpe 的那组时刻。
        exec_mode="fixed", # fixed: 用 t_exec; best_sharpe: 从 exec_grid 选; policy: 外部策略/RL
        exec_policy=None, # 可注入 RL/模型策略: (target_raw, readouts, exec_grid) -> t_exec
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True) # mkdir -p
    environment_init() # 1. 初始化随机种子；2. 调整 pandas 显示格式

    handler_task = {
        "class": "FeatureHandler",
        "module_path": "quant_research.data.handler",
    }
    data_task = {
        "class": "SimpleDataset",
        "kwargs": {
            "data_path": cfg.data_path,
            "hdf_key": cfg.hdf_key,
            "target_col": cfg.target_col,
            "handler": init_instance_by_config(handler_task),
        },
    }
    model_task = {
        "class": "SimpleModel",
        "kwargs": {"alpha_col": "default_st", "action_col": "action_default_st"},
    }

    model = init_instance_by_config(model_task)
    dataset = init_instance_by_config(data_task) 

    example_df = dataset.prepare("train")        # 这一步会把 feature 加好
    data_sample(example_df, cfg.output_dir)      # 输出一份数据的样例

    with Experiment(name="workflow", output_dir=cfg.output_dir) as recorder:
        recorder.log_params(
            t_exec=cfg.t_exec,
            exec_grid=cfg.exec_grid,
            exec_mode=cfg.exec_mode,
            alpha_col=model.alpha_col,
            action_col=model.action_col,
        )
        # 预测层：训练/更新预测模型（当前 SimpleModel 为占位）
        model.fit(dataset)

        # 预测层输出：生成 alpha 并落盘
        ar = AlphaRecord(model, dataset, recorder)
        ar.generate()

        exec_grid = backtest_config(cfg)
        # 决策层：alpha -> action（当前为恒等映射），并完成交易时间筛选与分析
        dr = DecisionRecord(model, dataset, recorder, exec_grid)
        sharpe_by_exec = dr.generate()

        # 决策层：选择执行时刻（fixed / best_sharpe / policy）
        # 回测执行：基于决策结果计算收益并输出净值曲线
        par = PortAnaRecord(
            recorder,
            cfg,
            sharpe_by_exec,
            dataset.target_raw,
            dataset.readouts,
            model.action_col,
        )
        return_series, chosen_exec = par.generate()

        print_backtest_report(
            return_series,
            cfg.output_dir,
            model.action_col,
            chosen_exec,
        )

    # 输出全量读数
    if dataset.readouts is not None:
        dataset.readouts.to_csv(cfg.output_dir / "readouts_full.csv", index=True)
    print_saved_artifacts(cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
