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
    strategy_col: str = "default_st"
    fitted: bool = False

    def fit(self, dataset: SimpleDataset) -> None:
        _ = dataset
        self.fitted = True


class SignalRecord:
    def __init__(self, model: SimpleModel, dataset: SimpleDataset, recorder: Recorder) -> None:
        self.model = model
        self.dataset = dataset
        self.recorder = recorder

    def generate(self) -> None:
        if self.dataset.readouts is None:
            raise ValueError("Dataset not prepared.")
        signal_generation(self.dataset.readouts, self.recorder.output_dir)


class SigAnaRecord:
    def __init__(
        self,
        recorder: Recorder,
        exec_grid: Iterable[int],
        target_raw: pd.DataFrame,
        readouts: pd.DataFrame,
        strategy_col: str,
    ) -> None:
        self.recorder = recorder
        self.exec_grid = exec_grid
        self.target_raw = target_raw
        self.readouts = readouts
        self.strategy_col = strategy_col

    def generate(self) -> pd.DataFrame:
        return_generation(self.target_raw, self.readouts, self.exec_grid, self.strategy_col)
        sharpe_by_exec = sharpe_generation(self.readouts, self.exec_grid, self.strategy_col)
        if not sharpe_by_exec.empty:
            training_report(sharpe_by_exec, self.recorder.output_dir)
        return sharpe_by_exec


class PortAnaRecord:
    def __init__(
        self,
        recorder: Recorder,
        cfg: Config,
        sharpe_by_exec: pd.DataFrame,
        target_raw: pd.DataFrame,
        readouts: pd.DataFrame,
        strategy_col: str,
    ) -> None:
        self.recorder = recorder
        self.cfg = cfg
        self.sharpe_by_exec = sharpe_by_exec
        self.target_raw = target_raw
        self.readouts = readouts
        self.strategy_col = strategy_col

    def generate(self) -> pd.Series:
        return backtest(self.cfg, self.sharpe_by_exec, self.target_raw, self.readouts, self.strategy_col)

def dataset_formatting(readouts: pd.DataFrame) -> pd.DataFrame:
    readouts = readouts.copy()
    readouts["date"] = readouts.index
    return readouts


def backtest_config(cfg: Config) -> Iterable[int]:
    return cfg.exec_grid


def data_sample(readouts: pd.DataFrame, output_dir: Path) -> None:
    readouts.head(10).to_csv(output_dir / "data_sample.csv", index=True)


def training_report(sharpe_by_exec: pd.DataFrame, output_dir: Path) -> None:
    sharpe_by_exec.to_csv(output_dir / "sharpe_by_exec.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(sharpe_by_exec["t_exec"], sharpe_by_exec["sharpe"], c="blue", label="sharpe ratio")
    ax.scatter(sharpe_by_exec["t_exec"], sharpe_by_exec["sharpe"], c="blue")
    ax.set_title("time sensitivity of default_st")
    ax.set_xlabel("trading time (min)")
    ax.set_ylabel("sharpe ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_sensitivity_default_st.png", dpi=150)
    plt.close(fig)


def signal_generation(readouts: pd.DataFrame, output_dir: Path) -> None:
    readouts[["default_st"]].to_csv(output_dir / "signals.csv", index=True)


def return_generation(
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    exec_grid: Iterable[int],
    strategy_col: str,
) -> None:
    for t_exec in exec_grid:
        return_col = f"return_{strategy_col}_{t_exec}"
        if return_col in readouts.columns:
            continue
        returns = evaluate_return_t_p1(target_raw, readouts, t_exec, strategy_col)
        readouts[return_col] = returns


def sharpe_generation(
    readouts: pd.DataFrame,
    exec_grid: Iterable[int],
    strategy_col: str,
) -> pd.DataFrame:
    sharpe_rows = []
    for t_exec in exec_grid:
        return_col = f"return_{strategy_col}_{t_exec}"
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
    strategy_col: str,
) -> pd.Series:
    return_col = f"return_{strategy_col}_{t_exec}"
    if return_col not in readouts.columns:
        returns = evaluate_return_t_p1(target_raw, readouts, t_exec, strategy_col)
        readouts[return_col] = returns
    return readouts[return_col]


def write_equity_curve(return_series: pd.Series, output_dir: Path) -> None:
    equity_curve(return_series).to_csv(output_dir / "equity_curve.csv", index=True)


def backtest(
    cfg: Config,
    sharpe_by_exec: pd.DataFrame,
    target_raw: pd.DataFrame,
    readouts: pd.DataFrame,
    strategy_col: str,
) -> pd.Series:
    """Select exec time, compute return series, and write equity curve."""
    chosen_exec = select_exec_time(cfg, sharpe_by_exec, target_raw, readouts)
    return_series = cal_return(target_raw, readouts, chosen_exec, strategy_col)
    write_equity_curve(return_series, cfg.output_dir)
    return return_series


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
        "kwargs": {"strategy_col": "default_st"},
    }

    model = init_instance_by_config(model_task)
    dataset = init_instance_by_config(data_task)

    example_df = dataset.prepare("train")
    data_sample(example_df, cfg.output_dir)

    with Experiment(name="workflow", output_dir=cfg.output_dir) as recorder:
        recorder.log_params(
            t_exec=cfg.t_exec,
            exec_grid=cfg.exec_grid,
            exec_mode=cfg.exec_mode,
            strategy_col=model.strategy_col,
        )
        model.fit(dataset)

        # 1) 信号生成：输出 signals.csv
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        exec_grid = backtest_config(cfg)
        # 2) 信号分析：生成各执行时刻收益列、计算 Sharpe，输出 sharpe_by_exec.csv 与分析图
        sar = SigAnaRecord(
            recorder, exec_grid, dataset.target_raw, dataset.readouts, model.strategy_col
        )
        sharpe_by_exec = sar.generate()

        # 3) 组合回测：选执行时刻并输出净值曲线 equity_curve.csv
        par = PortAnaRecord(
            recorder,
            cfg,
            sharpe_by_exec,
            dataset.target_raw,
            dataset.readouts,
            model.strategy_col,
        )
        par.generate()

    # 输出全量读数
    if dataset.readouts is not None:
        dataset.readouts.to_csv(cfg.output_dir / "readouts_full.csv", index=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
