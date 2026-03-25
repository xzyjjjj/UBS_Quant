from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


def _signed_power(series: pd.Series, power: int) -> pd.Series:
    return np.sign(series) * np.abs(series) ** power


def _ts_argmax(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).apply(lambda x: float(np.argmax(x)) + 1.0, raw=True)
        )
    return series.rolling(window).apply(lambda x: float(np.argmax(x)) + 1.0, raw=True)


def _ts_argmin(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).apply(lambda x: float(np.argmin(x)) + 1.0, raw=True)
        )
    return series.rolling(window).apply(lambda x: float(np.argmin(x)) + 1.0, raw=True)


def _rank(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=0).rank(pct=True)
    return series.rank(pct=True)


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
        )
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def _delta(series: pd.Series, period: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).diff(period)
    return series.diff(period)


def _delay(series: pd.Series, period: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).shift(period)
    return series.shift(period)


def _ts_sum(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).sum())
    return series.rolling(window).sum()


def _ts_min(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).min())
    return series.rolling(window).min()


def _ts_max(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).max())
    return series.rolling(window).max()


def _covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    if isinstance(x.index, pd.MultiIndex):
        return x.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).cov(y.reindex(s.index))
        )
    return x.rolling(window).cov(y)


def _correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    if isinstance(x.index, pd.MultiIndex):
        return x.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).corr(y.reindex(s.index))
        )
    return x.rolling(window).corr(y)


def _stddev(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).std())
    return series.rolling(window).std()


def _ts_product(series: pd.Series, window: int) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).apply(lambda x: float(np.prod(x)), raw=True)
        )
    return series.rolling(window).apply(lambda x: float(np.prod(x)), raw=True)


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1.0, float(window) + 1.0, dtype=float)
    denom = float(weights.sum())

    def _apply(x: np.ndarray) -> float:
        if len(x) != window:
            return float("nan")
        return float(np.dot(x, weights) / denom)

    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(window).apply(_apply, raw=True)
        )
    return series.rolling(window).apply(_apply, raw=True)


def _scale(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex):
        def _scale_group(s: pd.Series) -> pd.Series:
            denom = s.abs().sum()
            if denom == 0 or np.isnan(denom):
                return s * 0.0
            return s / denom

        return series.groupby(level=0, group_keys=False).apply(_scale_group)

    denom = float(series.abs().sum())
    if denom == 0 or np.isnan(denom):
        return series * 0.0
    return series / denom


def _get_series(readouts: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in readouts.columns:
            return readouts[name]
    raise KeyError(f"Missing columns: {names}")


def _close(readouts: pd.DataFrame) -> pd.Series:
    return _get_series(readouts, "close", "T_last_close")


def _open(readouts: pd.DataFrame) -> pd.Series:
    if "open" in readouts.columns:
        return readouts["open"]
    return _close(readouts)


def _high(readouts: pd.DataFrame) -> pd.Series:
    if "high" in readouts.columns:
        return readouts["high"]
    return _close(readouts)


def _low(readouts: pd.DataFrame) -> pd.Series:
    if "low" in readouts.columns:
        return readouts["low"]
    return _close(readouts)


def _volume(readouts: pd.DataFrame) -> pd.Series:
    if "volume" in readouts.columns:
        return readouts["volume"]
    return pd.Series(np.nan, index=readouts.index)


def _vwap(readouts: pd.DataFrame) -> pd.Series:
    if "vwap" in readouts.columns:
        return readouts["vwap"]
    return _get_series(readouts, "T_vwap", "T_last_close")


def _returns(readouts: pd.DataFrame) -> pd.Series:
    if "returns" in readouts.columns:
        return readouts["returns"]
    return _close(readouts).pct_change()


def _adv20(readouts: pd.DataFrame) -> pd.Series:
    if "adv20" in readouts.columns:
        return readouts["adv20"]
    return _adv(readouts, 20)


def _adv(readouts: pd.DataFrame, window: int) -> pd.Series:
    col = f"adv{int(window)}"
    if col in readouts.columns:
        return readouts[col]
    vol = _volume(readouts)
    if isinstance(vol.index, pd.MultiIndex):
        return vol.groupby(level=1, group_keys=False).apply(lambda s: s.rolling(window).mean())
    return vol.rolling(window).mean()


def _cap(readouts: pd.DataFrame) -> pd.Series:
    candidates = ("cap", "market_cap", "mkt_cap", "marketcap")
    for name in candidates:
        if name in readouts.columns:
            return readouts[name]
    return pd.Series(np.nan, index=readouts.index)


class IndClass:
    industry = "industry"
    sector = "sector"
    subindustry = "subindustry"


def _indneutralize(series: pd.Series, readouts: pd.DataFrame, cls: Any) -> pd.Series:
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.MultiIndex):
        return series
    cls_name = str(cls)
    if cls_name not in readouts.columns:
        return series
    groups = readouts[cls_name].reindex(series.index)
    tmp = pd.DataFrame({"x": series, "g": groups})
    demeaned = tmp["x"] - tmp.groupby([tmp.index.get_level_values(0), "g"])["x"].transform("mean")
    return demeaned


def _to_int_window(value: Union[int, float]) -> int:
    out = int(round(float(value)))
    return max(out, 1)


def _signed_power_any(x: Union[pd.Series, float], power: Union[pd.Series, float]) -> Union[pd.Series, float]:
    return np.sign(x) * (np.abs(x) ** power)


def _where(cond: Union[pd.Series, bool], x: Any, y: Any) -> Any:
    if not isinstance(cond, pd.Series):
        return x if bool(cond) else y
    index = cond.index
    x_aligned = x.reindex(index) if isinstance(x, pd.Series) else x
    y_aligned = y.reindex(index) if isinstance(y, pd.Series) else y
    return pd.Series(np.where(cond.values, x_aligned, y_aligned), index=index)


def _min_any(*args: Any) -> Any:
    if not args:
        raise ValueError("min() requires at least one argument")
    out = args[0]
    for nxt in args[1:]:
        out = np.minimum(out, nxt)
    return out


def _max_any(*args: Any) -> Any:
    if not args:
        raise ValueError("max() requires at least one argument")
    out = args[0]
    for nxt in args[1:]:
        out = np.maximum(out, nxt)
    return out


def alpha_001(
    readouts: pd.DataFrame,
    std_window: int = 20,
    signed_power: int = 2,
    ts_argmax_window: int = 5,
) -> pd.Series:
    close = _close(readouts)
    returns = close.pct_change()
    stddev = _stddev(returns, std_window)
    base = stddev.where(returns < 0, close)
    signed = _signed_power(base, signed_power)
    argmax = _ts_argmax(signed, ts_argmax_window)
    ranked = _rank(argmax)
    return ranked - 0.5


def alpha_002(
    readouts: pd.DataFrame,
    delta_window: int = 2,
    corr_window: int = 6,
) -> pd.Series:
    volume = _volume(readouts)
    close = _close(readouts)
    open_ = _open(readouts)
    lhs = _rank(_delta(np.log(volume), delta_window))
    rhs = _rank((close - open_) / open_)
    return -1.0 * _correlation(lhs, rhs, corr_window)


def alpha_003(readouts: pd.DataFrame, corr_window: int = 10) -> pd.Series:
    open_ = _open(readouts)
    volume = _volume(readouts)
    return -1.0 * _correlation(_rank(open_), _rank(volume), corr_window)


def alpha_004(readouts: pd.DataFrame, ts_rank_window: int = 9) -> pd.Series:
    low = _low(readouts)
    return -1.0 * _ts_rank(_rank(low), ts_rank_window)


def alpha_005(readouts: pd.DataFrame, vwap_window: int = 10) -> pd.Series:
    open_ = _open(readouts)
    close = _close(readouts)
    vwap = _vwap(readouts)
    avg_vwap = _ts_sum(vwap, vwap_window) / float(vwap_window)
    left = _rank(open_ - avg_vwap)
    right = -1.0 * np.abs(_rank(close - vwap))
    return left * right

def alpha_006(readouts: pd.DataFrame, corr_window: int = 10) -> pd.Series:
    open_ = _open(readouts)
    volume = _volume(readouts)
    return -1.0 * _correlation(open_, volume, corr_window)


def alpha_007(
    readouts: pd.DataFrame,
    adv_window: int = 20,
    delta_window: int = 7,
    ts_rank_window: int = 60,
) -> pd.Series:
    close = _close(readouts)
    volume = _volume(readouts)
    adv20 = _adv(readouts, adv_window)
    delta_close = _delta(close, delta_window)
    ts_rank_abs = _ts_rank(np.abs(delta_close), ts_rank_window)
    signal = -1.0 * ts_rank_abs * np.sign(delta_close)
    return pd.Series(np.where(volume > adv20, signal, -1.0), index=readouts.index)


def alpha_008(
    readouts: pd.DataFrame,
    sum_window: int = 5,
    delay_window: int = 10,
) -> pd.Series:
    open_ = _open(readouts)
    returns = _returns(readouts)
    sum_open = _ts_sum(open_, sum_window)
    sum_returns = _ts_sum(returns, sum_window)
    prod = sum_open * sum_returns
    return -1.0 * _rank(prod - _delay(prod, delay_window))


def alpha_009(readouts: pd.DataFrame, delta_window: int = 1, ts_window: int = 5) -> pd.Series:
    close = _close(readouts)
    delta_close = _delta(close, delta_window)
    min_delta = _ts_min(delta_close, ts_window)
    max_delta = _ts_max(delta_close, ts_window)
    out = np.where(min_delta > 0, delta_close, np.where(max_delta < 0, delta_close, -1.0 * delta_close))
    return pd.Series(out, index=readouts.index)


def alpha_010(readouts: pd.DataFrame, delta_window: int = 1, ts_window: int = 4) -> pd.Series:
    close = _close(readouts)
    delta_close = _delta(close, delta_window)
    min_delta = _ts_min(delta_close, ts_window)
    max_delta = _ts_max(delta_close, ts_window)
    out = np.where(min_delta > 0, delta_close, np.where(max_delta < 0, delta_close, -1.0 * delta_close))
    return _rank(pd.Series(out, index=readouts.index))


_ALPHA_FORMULAS: Optional[Dict[int, str]] = None
_ALPHA_AST: Dict[int, Any] = {}


def _load_alpha_formulas_from_markdown(path: Path) -> Dict[int, str]:
    text = path.read_text(encoding="utf-8")
    import re

    header = re.compile(r"(?m)^###\s*Alpha#(\d+)\s*$")
    codeblock = re.compile(r"```text\n(.*?)\n```", re.S)

    out: Dict[int, str] = {}
    matches = list(header.finditer(text))
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        cb = codeblock.search(chunk)
        if cb is None:
            continue
        out[n] = " ".join(cb.group(1).strip().split())
    return out


def _tokenize(formula: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < len(formula) and formula[i + 1].isdigit()):
            j = i + 1
            while j < len(formula) and (formula[j].isdigit() or formula[j] == "."):
                j += 1
            tokens.append(("num", formula[i:j]))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < len(formula) and (formula[j].isalnum() or formula[j] == "_"):
                j += 1
            tokens.append(("ident", formula[i:j]))
            i = j
            continue
        # multi-char operators
        if formula.startswith((">=", "<=", "==", "!=", "&&", "||"), i):
            tokens.append(("op", formula[i : i + 2]))
            i += 2
            continue
        if ch in "+-*/^<>":
            tokens.append(("op", ch))
            i += 1
            continue
        if ch in "(),?:.":
            tokens.append(("punc", ch))
            i += 1
            continue
        raise ValueError(f"Unexpected character {ch!r} in formula: {formula}")
    tokens.append(("eof", ""))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self._toks = tokens
        self._i = 0

    def _peek(self) -> tuple[str, str]:
        return self._toks[self._i]

    def _accept(self, typ: str, val: Optional[str] = None) -> bool:
        t, v = self._peek()
        if t != typ:
            return False
        if val is not None and v != val:
            return False
        self._i += 1
        return True

    def _expect(self, typ: str, val: Optional[str] = None) -> tuple[str, str]:
        if not self._accept(typ, val):
            raise ValueError(f"Expected {typ} {val or ''} but got {self._peek()}")
        return self._toks[self._i - 1]

    def parse(self) -> Any:
        node = self._parse_ternary()
        self._expect("eof")
        return node

    def _parse_ternary(self) -> Any:
        cond = self._parse_or()
        if self._accept("punc", "?"):
            on_true = self._parse_ternary()
            self._expect("punc", ":")
            on_false = self._parse_ternary()
            return ("tern", cond, on_true, on_false)
        return cond

    def _parse_or(self) -> Any:
        node = self._parse_and()
        while self._accept("op", "||"):
            rhs = self._parse_and()
            node = ("bin", "||", node, rhs)
        return node

    def _parse_and(self) -> Any:
        node = self._parse_cmp()
        while self._accept("op", "&&"):
            rhs = self._parse_cmp()
            node = ("bin", "&&", node, rhs)
        return node

    def _parse_cmp(self) -> Any:
        node = self._parse_add()
        while True:
            t, v = self._peek()
            if t == "op" and v in ("<", "<=", ">", ">=", "==", "!="):
                self._i += 1
                rhs = self._parse_add()
                node = ("bin", v, node, rhs)
            else:
                break
        return node

    def _parse_add(self) -> Any:
        node = self._parse_mul()
        while True:
            if self._accept("op", "+"):
                rhs = self._parse_mul()
                node = ("bin", "+", node, rhs)
            elif self._accept("op", "-"):
                rhs = self._parse_mul()
                node = ("bin", "-", node, rhs)
            else:
                break
        return node

    def _parse_mul(self) -> Any:
        node = self._parse_pow()
        while True:
            if self._accept("op", "*"):
                rhs = self._parse_pow()
                node = ("bin", "*", node, rhs)
            elif self._accept("op", "/"):
                rhs = self._parse_pow()
                node = ("bin", "/", node, rhs)
            else:
                break
        return node

    def _parse_pow(self) -> Any:
        node = self._parse_unary()
        if self._accept("op", "^"):
            rhs = self._parse_pow()
            return ("bin", "^", node, rhs)
        return node

    def _parse_unary(self) -> Any:
        if self._accept("op", "-"):
            return ("unary", "-", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Any:
        t, v = self._peek()
        if self._accept("punc", "("):
            node = self._parse_ternary()
            self._expect("punc", ")")
            return node
        if self._accept("num"):
            return ("num", float(v))
        if self._accept("ident"):
            node: Any = ("var", v)
            while self._accept("punc", "."):
                _, attr = self._expect("ident")
                node = ("attr", node, attr)
            if self._accept("punc", "("):
                args: list[Any] = []
                if not self._accept("punc", ")"):
                    args.append(self._parse_ternary())
                    while self._accept("punc", ","):
                        args.append(self._parse_ternary())
                    self._expect("punc", ")")
                return ("call", node, args)
            return node
        raise ValueError(f"Unexpected token {self._peek()}")


def _get_formula(alpha_num: int) -> str:
    global _ALPHA_FORMULAS
    if _ALPHA_FORMULAS is None:
        md_path = Path(__file__).with_suffix(".md")
        if not md_path.exists():
            raise FileNotFoundError(f"Missing alpha markdown file: {md_path}")
        _ALPHA_FORMULAS = _load_alpha_formulas_from_markdown(md_path)
    formula = _ALPHA_FORMULAS.get(alpha_num)
    if not formula:
        raise KeyError(f"Missing Alpha#{alpha_num} formula in markdown.")
    return formula


def _eval_alpha_formula(alpha_num: int, readouts: pd.DataFrame) -> pd.Series:
    ast = _ALPHA_AST.get(alpha_num)
    if ast is None:
        ast = _Parser(_tokenize(_get_formula(alpha_num))).parse()
        _ALPHA_AST[alpha_num] = ast

    env: Dict[str, Any] = {
        # series inputs
        "open": _open(readouts),
        "high": _high(readouts),
        "low": _low(readouts),
        "close": _close(readouts),
        "vwap": _vwap(readouts),
        "volume": _volume(readouts),
        "returns": _returns(readouts),
        "cap": _cap(readouts),
        # adv aliases used by formulas
        "adv5": _adv(readouts, 5),
        "adv10": _adv(readouts, 10),
        "adv15": _adv(readouts, 15),
        "adv20": _adv(readouts, 20),
        "adv30": _adv(readouts, 30),
        "adv40": _adv(readouts, 40),
        "adv50": _adv(readouts, 50),
        "adv60": _adv(readouts, 60),
        "adv81": _adv(readouts, 81),
        "adv120": _adv(readouts, 120),
        "adv150": _adv(readouts, 150),
        "adv180": _adv(readouts, 180),
        # objects
        "IndClass": IndClass,
    }

    def _as_bool(x: Any) -> Any:
        if isinstance(x, pd.Series):
            return x.astype(bool)
        return bool(x)

    def _eval(node: Any) -> Any:
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "var":
            return env[node[1]]
        if kind == "attr":
            base = _eval(node[1])
            return getattr(base, node[2])
        if kind == "unary":
            op = node[1]
            val = _eval(node[2])
            if op == "-":
                return -val
            raise ValueError(f"Unsupported unary op: {op}")
        if kind == "bin":
            op = node[1]
            left = _eval(node[2])
            right = _eval(node[3])
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "^":
                return left ** right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "&&":
                return _as_bool(left) & _as_bool(right)
            if op == "||":
                return _as_bool(left) | _as_bool(right)
            raise ValueError(f"Unsupported binary op: {op}")
        if kind == "tern":
            cond = _eval(node[1])
            on_true = _eval(node[2])
            on_false = _eval(node[3])
            return _where(cond, on_true, on_false)
        if kind == "call":
            fn = _eval(node[1])
            args = [_eval(a) for a in node[2]]
            if callable(fn):
                return fn(*args)
            raise ValueError(f"Attempted to call non-callable: {fn}")
        raise ValueError(f"Unknown AST node: {node}")

    # Inject callable operators into env (late-bound so they can close over readouts).
    env.update(
        {
            "rank": _rank,
            "scale": _scale,
            "ts_rank": lambda s, w: _ts_rank(s, _to_int_window(w)),
            "Ts_Rank": lambda s, w: _ts_rank(s, _to_int_window(w)),
            "ts_argmax": lambda s, w: _ts_argmax(s, _to_int_window(w)),
            "Ts_ArgMax": lambda s, w: _ts_argmax(s, _to_int_window(w)),
            "ts_argmin": lambda s, w: _ts_argmin(s, _to_int_window(w)),
            "Ts_ArgMin": lambda s, w: _ts_argmin(s, _to_int_window(w)),
            "delta": lambda s, p: _delta(s, _to_int_window(p)),
            "delay": lambda s, p: _delay(s, _to_int_window(p)),
            "sum": lambda s, w: _ts_sum(s, _to_int_window(w)),
            "ts_min": lambda s, w: _ts_min(s, _to_int_window(w)),
            "ts_max": lambda s, w: _ts_max(s, _to_int_window(w)),
            "stddev": lambda s, w: _stddev(s, _to_int_window(w)),
            "correlation": lambda a, b, w: _correlation(a, b, _to_int_window(w)),
            "covariance": lambda a, b, w: _covariance(a, b, _to_int_window(w)),
            "product": lambda s, w: _ts_product(s, _to_int_window(w)),
            "decay_linear": lambda s, w: _decay_linear(s, _to_int_window(w)),
            "SignedPower": _signed_power_any,
            "abs": np.abs,
            "sign": np.sign,
            "Sign": np.sign,
            "log": np.log,
            "Log": np.log,
            "min": _min_any,
            "max": _max_any,
            "IndNeutralize": lambda s, cls: _indneutralize(s, readouts, cls),
            "indneutralize": lambda s, cls: _indneutralize(s, readouts, cls),
        }
    )

    return _eval(ast)


def _make_alpha(alpha_num: int) -> Callable[[pd.DataFrame], pd.Series]:
    def _alpha(readouts: pd.DataFrame) -> pd.Series:
        return _eval_alpha_formula(alpha_num, readouts)

    _alpha.__name__ = f"alpha_{alpha_num:03d}"
    return _alpha


for _n in range(11, 102):
    globals()[f"alpha_{_n:03d}"] = _make_alpha(_n)


ALPHA_PARAM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "alpha_001": {"std_window": 20, "signed_power": 2, "ts_argmax_window": 5},
    "alpha_002": {"delta_window": 2, "corr_window": 6},
    "alpha_003": {"corr_window": 10},
    "alpha_004": {"ts_rank_window": 9},
    "alpha_005": {"vwap_window": 10},
    "alpha_006": {"corr_window": 10},
    "alpha_007": {"adv_window": 20, "delta_window": 7, "ts_rank_window": 60},
    "alpha_008": {"sum_window": 5, "delay_window": 10},
    "alpha_009": {"delta_window": 1, "ts_window": 5},
    "alpha_010": {"delta_window": 1, "ts_window": 4},
}


def compute_alpha(name: str, readouts: pd.DataFrame, **overrides: Any) -> pd.Series:
    """Compute alpha with centralized parameter defaults.

    Parameter precedence:
    1) ALPHA_PARAM_DEFAULTS (project-wide defaults)
    2) overrides passed here
    3) function defaults (only used if a key is missing above)
    """
    fn = globals().get(name)
    if fn is None:
        raise ValueError(f"Unknown alpha function: {name}")
    params = dict(ALPHA_PARAM_DEFAULTS.get(name, {}))
    params.update(overrides)
    return fn(readouts, **params)


def feature_101(readouts: pd.DataFrame) -> pd.Series:
    return alpha_001(readouts)
