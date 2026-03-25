#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def norm_rate(v: float | None) -> float | None:
    if v is None:
        return None
    return v / 100.0 if abs(v) > 3.0 else v


def norm_sigma(v: float | None) -> float | None:
    if v is None:
        return None
    return v / 100.0 if v > 3.0 else v


def norm_type(x: str) -> str | None:
    s = str(x or "").strip().lower()
    if "call" in s or s in {"c", "1"}:
        return "call"
    if "put" in s or s in {"p", "2"}:
        return "put"
    return None


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def psi(hist_a: list[int], hist_b: list[int]) -> float:
    eps = 1e-9
    sa = sum(hist_a)
    sb = sum(hist_b)
    if sa == 0 or sb == 0:
        return float("nan")
    out = 0.0
    for a, b in zip(hist_a, hist_b):
        pa = max(a / sa, eps)
        pb = max(b / sb, eps)
        out += (pa - pb) * math.log(pa / pb)
    return out


def bucket_ttm(t: float) -> str:
    if t <= 7 / 365:
        return "<=1w"
    if t <= 30 / 365:
        return "1w-1m"
    if t <= 90 / 365:
        return "1m-3m"
    if t <= 180 / 365:
        return "3m-6m"
    return ">6m"


def bucket_mny(m: float) -> str:
    if m < 0.9:
        return "deep_otm"
    if m < 0.97:
        return "otm"
    if m <= 1.03:
        return "atm"
    if m <= 1.1:
        return "itm"
    return "deep_itm"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Check distribution shift across train/val/test date split")
    p.add_argument("--data-csv", default=str(root / "output" / "server_bundle_week5" / "data" / "processed" / "jpm_options_final.csv"))
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--out-dir", default=str(root / "output" / "week8_distribution_check"))
    return p.parse_args()


def split_thresholds(path: Path, tr: float, va: float) -> tuple[str, str, int]:
    dates = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            d = str(row.get("date", ""))
            if d:
                dates.add(d)
    ds = sorted(dates)
    n = len(ds)
    i1 = max(1, int(n * tr))
    i2 = max(i1 + 1, int(n * (tr + va)))
    if i2 >= n:
        i2 = n - 1
    return ds[i1 - 1], ds[i2 - 1], n


def split_key(d: str, train_end: str, val_end: str) -> str:
    if d <= train_end:
        return "train"
    if d <= val_end:
        return "val"
    return "test"


def main() -> int:
    args = parse_args()
    data_csv = Path(args.data_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_end, val_end, n_dates = split_thresholds(data_csv, float(args.train_ratio), float(args.val_ratio))

    num = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    cnt_type = {"train": Counter(), "val": Counter(), "test": Counter()}
    cnt_ttm = {"train": Counter(), "val": Counter(), "test": Counter()}
    cnt_mny = {"train": Counter(), "val": Counter(), "test": Counter()}
    counts = Counter()

    with data_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            d = str(row.get("date", ""))
            if not d:
                continue
            k = split_key(d, train_end, val_end)
            counts[k] += 1

            px = safe_float(row.get("market_price"))
            s0 = safe_float(row.get("S0"))
            strike = safe_float(row.get("strike_price"))
            t = safe_float(row.get("T_years"))
            sigma = norm_sigma(safe_float(row.get("sigma")))
            r = norm_rate(safe_float(row.get("r")))
            q = norm_rate(safe_float(row.get("q")))
            typ = norm_type(row.get("option_type", ""))

            if typ is not None:
                cnt_type[k][typ] += 1
            if t is not None and t >= 0:
                cnt_ttm[k][bucket_ttm(t)] += 1
            if s0 is not None and strike is not None and s0 > 0 and strike > 0:
                m = s0 / strike
                cnt_mny[k][bucket_mny(m)] += 1
                num[k]["moneyness"].append(m)
                num[k]["log_moneyness"].append(math.log(max(m, 1e-12)))

            if px is not None:
                num[k]["market_price"].append(px)
            if sigma is not None:
                num[k]["sigma"].append(sigma)
            if t is not None:
                num[k]["T_years"].append(t)
            if r is not None:
                num[k]["r"].append(r)
            if q is not None:
                num[k]["q"].append(q)

    numeric_features = ["market_price", "sigma", "T_years", "moneyness", "log_moneyness", "r", "q"]
    split_order = ["train", "val", "test"]

    # numeric summary
    rows = []
    for s in split_order:
        for feat in numeric_features:
            xs = num[s][feat]
            if not xs:
                continue
            rows.append(
                {
                    "split": s,
                    "feature": feat,
                    "n": len(xs),
                    "mean": sum(xs) / len(xs),
                    "p10": quantile(xs, 0.1),
                    "p50": quantile(xs, 0.5),
                    "p90": quantile(xs, 0.9),
                }
            )

    num_csv = out_dir / "distribution_numeric_summary.csv"
    with num_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # categorical ratios
    def write_ratio(counter_map: dict[str, Counter], name: str) -> Path:
        keys = sorted({k for s in split_order for k in counter_map[s].keys()})
        p = out_dir / f"distribution_{name}_ratio.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "n_total"] + keys)
            for s in split_order:
                tot = sum(counter_map[s].values())
                row = [s, tot]
                for k in keys:
                    row.append((counter_map[s][k] / tot) if tot > 0 else float("nan"))
                w.writerow(row)
        return p

    type_csv = write_ratio(cnt_type, "option_type")
    ttm_csv = write_ratio(cnt_ttm, "ttm_bucket")
    mny_csv = write_ratio(cnt_mny, "moneyness_bucket")

    # simple PSI diagnostics for numeric features train vs test
    psi_rows = []
    for feat in numeric_features:
        tr = num["train"][feat]
        te = num["test"][feat]
        if len(tr) < 100 or len(te) < 100:
            continue
        # bins by train deciles
        cuts = [quantile(tr, q / 10.0) for q in range(11)]
        # ensure monotonic bins
        for i in range(1, len(cuts)):
            if cuts[i] < cuts[i - 1]:
                cuts[i] = cuts[i - 1]

        def hist(xs: list[float]) -> list[int]:
            h = [0] * 10
            for v in xs:
                bi = 9
                for i in range(10):
                    lo, hi = cuts[i], cuts[i + 1]
                    if i < 9:
                        if lo <= v < hi:
                            bi = i
                            break
                    else:
                        if lo <= v <= hi:
                            bi = i
                            break
                h[bi] += 1
            return h

        h_tr = hist(tr)
        h_te = hist(te)
        pv = psi(h_tr, h_te)
        if pv < 0.1:
            level = "low"
        elif pv < 0.25:
            level = "medium"
        else:
            level = "high"
        psi_rows.append({"feature": feat, "psi_train_vs_test": pv, "shift_level": level})

    psi_rows.sort(key=lambda r: r["psi_train_vs_test"], reverse=True)
    psi_csv = out_dir / "distribution_psi_train_vs_test.csv"
    with psi_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(psi_rows[0].keys()))
        w.writeheader()
        w.writerows(psi_rows)

    # markdown report
    md = out_dir / "q3_distribution_shift_supplement.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Q3 补充报告：训练/验证/测试分布偏移检查\n\n")
        f.write(f"- 日期切分节点：train_end=`{train_end}`，val_end=`{val_end}`\n")
        f.write(f"- 交易日总数：{n_dates}\n")
        f.write(f"- 样本数：train={counts['train']}, val={counts['val']}, test={counts['test']}\n\n")

        f.write("## 1) Numeric 特征偏移（PSI，train vs test）\n\n")
        f.write("| feature | PSI | level |\n")
        f.write("|---|---:|---|\n")
        for r in psi_rows:
            f.write(f"| {r['feature']} | {r['psi_train_vs_test']:.4f} | {r['shift_level']} |\n")

        f.write("\nPSI 判读：`<0.1 低偏移`，`0.1~0.25 中等偏移`，`>0.25 高偏移`。\n\n")

        f.write("## 2) 结论\n\n")
        high = [r for r in psi_rows if r["shift_level"] == "high"]
        med = [r for r in psi_rows if r["shift_level"] == "medium"]
        if high:
            f.write(f"- 发现高偏移特征：{', '.join(x['feature'] for x in high)}。\n")
        if med:
            f.write(f"- 发现中等偏移特征：{', '.join(x['feature'] for x in med)}。\n")
        if (not high) and (not med):
            f.write("- 本次检查未发现明显分布偏移。\n")
        f.write("- 建议：对偏移高的特征做分层重加权，或分桶建模，并单独报告 OOS 表现。\n\n")

        f.write("## 3) 结果文件\n\n")
        f.write(f"- `{num_csv}`\n")
        f.write(f"- `{type_csv}`\n")
        f.write(f"- `{ttm_csv}`\n")
        f.write(f"- `{mny_csv}`\n")
        f.write(f"- `{psi_csv}`\n")

    print("[OK] distribution shift check done")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] report={md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
