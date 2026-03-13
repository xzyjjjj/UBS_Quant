#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


FEATURES = [
    "log_moneyness",
    "ttm_years",
    "sigma",
    "r",
    "q",
    "is_call",
    "sqrt_ttm",
    "moneyness",
]


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def to_rate(v: float | None) -> float | None:
    if v is None:
        return None
    if abs(v) > 3.0:
        return v / 100.0
    return v


def to_sigma(v: float | None) -> float | None:
    if v is None:
        return None
    if v > 3.0:
        return v / 100.0
    return v


def norm_option_type(x: Any) -> str | None:
    s = str(x or "").strip().lower()
    if not s:
        return None
    if "call" in s or s in {"c", "1"}:
        return "call"
    if "put" in s or s in {"p", "2"}:
        return "put"
    return None


def ttm_bucket(t: float) -> str:
    if t <= 7 / 365:
        return "<=1w"
    if t <= 30 / 365:
        return "1w-1m"
    if t <= 90 / 365:
        return "1m-3m"
    if t <= 180 / 365:
        return "3m-6m"
    return ">6m"


def moneyness_bucket(m: float) -> str:
    if m < 0.9:
        return "deep_otm"
    if m < 0.97:
        return "otm"
    if m <= 1.03:
        return "atm"
    if m <= 1.1:
        return "itm"
    return "deep_itm"


def load_model(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    model = d["model"]
    if model.get("kind") != "mlp":
        raise ValueError(f"Unsupported model kind: {model.get('kind')}")
    return {
        "features": d.get("features", FEATURES),
        "mean": [float(x) for x in d["normalization"]["mean"]],
        "std": [float(x) for x in d["normalization"]["std"]],
        "w1": [[float(z) for z in row] for row in model["w1"]],
        "b1": [float(x) for x in model["b1"]],
        "w2": [float(x) for x in model["w2"]],
        "b2": float(model["b2"]),
    }


def predict_one(xn: list[float], model: dict[str, Any]) -> float:
    w1 = model["w1"]
    b1 = model["b1"]
    w2 = model["w2"]
    b2 = model["b2"]
    hidden = len(b1)
    z = [0.0] * hidden
    for j in range(hidden):
        s = b1[j]
        wj = w1[j]
        for i, v in enumerate(xn):
            s += wj[i] * v
        z[j] = s if s > 0 else 0.0
    y = b2
    for j in range(hidden):
        y += w2[j] * z[j]
    return y


def normalize(x: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(x[i] - mean[i]) / std[i] for i in range(len(x))]


def build_feature(row: dict[str, str]) -> tuple[list[float], float, str, float, float] | None:
    s0 = safe_float(row.get("S0"))
    k = safe_float(row.get("strike_price"))
    t = safe_float(row.get("T_years"))
    sigma = to_sigma(safe_float(row.get("sigma")))
    r = to_rate(safe_float(row.get("r")))
    q = to_rate(safe_float(row.get("q")))
    y = safe_float(row.get("market_price"))
    opt = norm_option_type(row.get("option_type"))
    if None in (s0, k, t, sigma, r, q, y) or opt is None:
        return None
    if s0 <= 0 or k <= 0 or t < 0:
        return None
    m = s0 / k
    x = [
        math.log(max(m, 1e-12)),
        t,
        sigma,
        r,
        q,
        1.0 if opt == "call" else 0.0,
        math.sqrt(max(t, 0.0)),
        m,
    ]
    return x, float(y), opt, float(t), float(m)


def mae_rmse(y: list[float], p: list[float]) -> tuple[float, float]:
    if not y:
        return float("nan"), float("nan")
    ae = 0.0
    se = 0.0
    for yt, yp in zip(y, p):
        d = yp - yt
        ae += abs(d)
        se += d * d
    n = len(y)
    return ae / n, math.sqrt(se / n)


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["empty"])
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Week6 MLP explainability: permutation importance + PDP + subgroup error.")
    p.add_argument("--model-json", default=str(root / "models" / "mlp_direct.json"))
    p.add_argument("--data-csv", default=str(root / "output" / "server_bundle_week5" / "data" / "processed" / "jpm_options_final.csv"))
    p.add_argument("--max-rows", type=int, default=50000, help="Max sampled rows for explainability")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perm-repeats", type=int, default=5)
    p.add_argument("--pdp-bins", type=int, default=12)
    p.add_argument("--out-dir", default=str(root / "output" / "week6_explainability"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    model = load_model(Path(args.model_json))
    mean = model["mean"]
    std = model["std"]

    X_raw: list[list[float]] = []
    y: list[float] = []
    opt_types: list[str] = []
    ttm_list: list[float] = []
    mny_list: list[float] = []

    # Reservoir sample for uniform row-level sampling.
    seen = 0
    cap = int(args.max_rows)
    with Path(args.data_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = build_feature(row)
            if rec is None:
                continue
            x, yy, opt, t, m = rec
            seen += 1
            if len(y) < cap:
                X_raw.append(x)
                y.append(yy)
                opt_types.append(opt)
                ttm_list.append(t)
                mny_list.append(m)
            else:
                j = rnd.randint(0, seen - 1)
                if j < cap:
                    X_raw[j] = x
                    y[j] = yy
                    opt_types[j] = opt
                    ttm_list[j] = t
                    mny_list[j] = m

    X = [normalize(x, mean, std) for x in X_raw]
    pred = [predict_one(xn, model) for xn in X]
    base_mae, base_rmse = mae_rmse(y, pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) overall
    overall_rows = [
        {
            "n_rows": len(y),
            "base_mae": base_mae,
            "base_rmse": base_rmse,
            "y_p50": quantile(y, 0.5),
            "y_p90": quantile(y, 0.9),
            "pred_p50": quantile(pred, 0.5),
            "pred_p90": quantile(pred, 0.9),
        }
    ]
    write_csv(out_dir / "mlp_explain_overall.csv", overall_rows)

    # 2) permutation importance
    perm_rows: list[dict[str, Any]] = []
    for fi, fname in enumerate(FEATURES):
        maes: list[float] = []
        rmses: list[float] = []
        for rep in range(int(args.perm_repeats)):
            idx = list(range(len(X)))
            rnd.shuffle(idx)
            p_perm: list[float] = []
            for i, xn in enumerate(X):
                x2 = xn.copy()
                x2[fi] = X[idx[i]][fi]
                p_perm.append(predict_one(x2, model))
            m1, m2 = mae_rmse(y, p_perm)
            maes.append(m1)
            rmses.append(m2)
        perm_rows.append(
            {
                "feature": fname,
                "base_mae": base_mae,
                "perm_mae_mean": sum(maes) / len(maes),
                "delta_mae": (sum(maes) / len(maes)) - base_mae,
                "base_rmse": base_rmse,
                "perm_rmse_mean": sum(rmses) / len(rmses),
                "delta_rmse": (sum(rmses) / len(rmses)) - base_rmse,
            }
        )
    perm_rows.sort(key=lambda r: r["delta_mae"], reverse=True)
    write_csv(out_dir / "mlp_explain_permutation_importance.csv", perm_rows)

    # 3) subgroup errors
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y": [], "p": []})
    for i in range(len(y)):
        by_group[f"option_type={opt_types[i]}"]["y"].append(y[i])
        by_group[f"option_type={opt_types[i]}"]["p"].append(pred[i])
        tb = ttm_bucket(ttm_list[i])
        by_group[f"ttm={tb}"]["y"].append(y[i])
        by_group[f"ttm={tb}"]["p"].append(pred[i])
        mb = moneyness_bucket(mny_list[i])
        by_group[f"moneyness={mb}"]["y"].append(y[i])
        by_group[f"moneyness={mb}"]["p"].append(pred[i])

    grp_rows: list[dict[str, Any]] = []
    for g, d in sorted(by_group.items()):
        m1, m2 = mae_rmse(d["y"], d["p"])
        grp_rows.append(
            {
                "group": g,
                "n": len(d["y"]),
                "mae": m1,
                "rmse": m2,
                "mae_minus_overall": m1 - base_mae,
                "rmse_minus_overall": m2 - base_rmse,
            }
        )
    grp_rows.sort(key=lambda r: r["mae"], reverse=True)
    write_csv(out_dir / "mlp_explain_group_errors.csv", grp_rows)

    # 4) PDP tables for key features
    pdp_targets = ["sigma", "ttm_years", "moneyness", "r", "q", "log_moneyness"]
    pdp_rows: list[dict[str, Any]] = []
    for fname in pdp_targets:
        fi = FEATURES.index(fname)
        vals = [x[fi] for x in X_raw]
        if not vals:
            continue
        mn = min(vals)
        mx = max(vals)
        if mx - mn < 1e-12:
            continue
        bins = int(args.pdp_bins)
        edges = [mn + (mx - mn) * i / bins for i in range(bins + 1)]
        y_bins: list[list[float]] = [[] for _ in range(bins)]
        p_bins: list[list[float]] = [[] for _ in range(bins)]
        for i, v in enumerate(vals):
            bi = bins - 1
            for k in range(bins):
                if edges[k] <= v < edges[k + 1]:
                    bi = k
                    break
            y_bins[bi].append(y[i])
            p_bins[bi].append(pred[i])
        for k in range(bins):
            yc = y_bins[k]
            pc = p_bins[k]
            if not yc:
                continue
            pdp_rows.append(
                {
                    "feature": fname,
                    "bin_left": edges[k],
                    "bin_right": edges[k + 1],
                    "n": len(yc),
                    "avg_market_price": sum(yc) / len(yc),
                    "avg_model_pred": sum(pc) / len(pc),
                    "avg_residual": (sum(pc) / len(pc)) - (sum(yc) / len(yc)),
                }
            )
    write_csv(out_dir / "mlp_explain_pdp_bins.csv", pdp_rows)

    # 5) markdown summary
    top = perm_rows[:5]
    md = out_dir / "mlp_explain_summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Week6 MLP Explainability Summary\n\n")
        f.write(f"- model: `{args.model_json}`\n")
        f.write(f"- data: `{args.data_csv}`\n")
        f.write(f"- sampled rows: **{len(y)}**\n")
        f.write(f"- base MAE/RMSE: **{base_mae:.6f} / {base_rmse:.6f}**\n\n")
        f.write("## Top Permutation Importance (by delta MAE)\n\n")
        f.write("| feature | delta MAE | delta RMSE |\n")
        f.write("|---|---:|---:|\n")
        for r in top:
            f.write(f"| {r['feature']} | {r['delta_mae']:.6f} | {r['delta_rmse']:.6f} |\n")
        f.write("\n## Output Files\n\n")
        f.write(f"- `{out_dir / 'mlp_explain_overall.csv'}`\n")
        f.write(f"- `{out_dir / 'mlp_explain_permutation_importance.csv'}`\n")
        f.write(f"- `{out_dir / 'mlp_explain_group_errors.csv'}`\n")
        f.write(f"- `{out_dir / 'mlp_explain_pdp_bins.csv'}`\n")

    print("[OK] week6 MLP explainability done.")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] n_rows={len(y)} base_mae={base_mae:.6f} base_rmse={base_rmse:.6f}")
    print(f"[INFO] top_feature={top[0]['feature'] if top else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
