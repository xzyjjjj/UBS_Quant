#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


BUCKET_DEEP_ITM = "deep_itm"
BUCKET_ATM = "atm"
BUCKET_DEEP_OTM = "deep_otm"


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def norm_rate(v: float | None) -> float:
    if v is None:
        return 0.0
    return v / 100.0 if abs(v) > 3.0 else v


def norm_sigma(v: float | None) -> float:
    if v is None:
        return 0.0
    return v / 100.0 if v > 3.0 else v


def norm_type(x: str) -> str | None:
    s = str(x or "").strip().lower()
    if "call" in s or s in {"c", "1"}:
        return "call"
    if "put" in s or s in {"p", "2"}:
        return "put"
    return None


def moneyness_bucket(mny: float, opt: str) -> str:
    if opt == "call":
        if mny >= 1.15:
            return BUCKET_DEEP_ITM
        if mny <= 0.85:
            return BUCKET_DEEP_OTM
        return BUCKET_ATM
    # put option uses opposite ITM/OTM direction versus moneyness.
    if mny <= 0.85:
        return BUCKET_DEEP_ITM
    if mny >= 1.15:
        return BUCKET_DEEP_OTM
    return BUCKET_ATM


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(*, is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0.0:
        return max(0.0, (s - k) if is_call else (k - s))
    sigma = max(sigma, 1e-12)
    sqrt_t = math.sqrt(max(t, 1e-12))
    d1 = (math.log(max(s, 1e-12) / max(k, 1e-12)) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df_r = math.exp(-r * t)
    df_q = math.exp(-q * t)
    if is_call:
        return s * df_q * _norm_cdf(d1) - k * df_r * _norm_cdf(d2)
    return k * df_r * _norm_cdf(-d2) - s * df_q * _norm_cdf(-d1)


def load_model(model_json: Path) -> dict[str, Any]:
    d = json.loads(model_json.read_text(encoding="utf-8"))
    if str(d.get("model_name")) != "bsm_mlp_residual":
        raise ValueError("This script expects bsm_mlp_residual model.")
    m = d["model"]["residual_model"]
    return {
        "mean": [float(x) for x in d["normalization"]["mean"]],
        "std": [float(x) for x in d["normalization"]["std"]],
        "w1": [[float(z) for z in row] for row in m["w1"]],
        "b1": [float(x) for x in m["b1"]],
        "w2": [float(x) for x in m["w2"]],
        "b2": float(m["b2"]),
    }


def mlp_predict(xn: list[float], model: dict[str, Any]) -> float:
    z = []
    for j, row in enumerate(model["w1"]):
        s = model["b1"][j]
        for a, b in zip(row, xn):
            s += a * b
        z.append(s if s > 0 else 0.0)
    y = model["b2"]
    for w, a in zip(model["w2"], z):
        y += w * a
    return y


def model_price(*, model: dict[str, Any], s0: float, k: float, t: float, sigma: float, r: float, q: float, opt: str) -> float:
    mny = s0 / k
    x = [
        math.log(max(mny, 1e-12)),
        t,
        sigma,
        r,
        q,
        1.0 if opt == "call" else 0.0,
        math.sqrt(max(t, 0.0)),
        mny,
    ]
    xn = [(x[i] - model["mean"][i]) / model["std"][i] for i in range(len(x))]
    base = bs_price(is_call=(opt == "call"), s=s0, k=k, t=t, r=r, q=q, sigma=max(sigma, 1e-12))
    res = mlp_predict(xn, model)
    return base + res


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description=(
            "Week7+ extreme scenario tests with moneyness buckets and joint shocks: "
            "sigma +50%, rate +2%, sigma+rate"
        )
    )
    p.add_argument("--data-csv", default=str(root / "output" / "server_bundle_week5" / "data" / "processed" / "jpm_options_final.csv"))
    p.add_argument("--model-json", default=str(root / "models" / "bsm_mlp_residual.json"))
    p.add_argument("--max-rows", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(root / "output" / "week7_extreme_tests"))
    p.add_argument(
        "--sigma-shock-grid",
        default="-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1.0",
        help="Comma-separated sigma relative shocks. Example: -0.5 means sigma*(1-0.5).",
    )
    p.add_argument(
        "--r-shock-grid",
        default="-0.02,-0.01,0.0,0.01,0.02",
        help="Comma-separated absolute rate shocks. Example: 0.02 means r+2%.",
    )
    return p.parse_args()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


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


def summarize_scenario(
    *,
    scenario: str,
    bucket: str,
    px: list[float],
    base_px: list[float],
    pct_changes: list[float],
) -> dict[str, Any]:
    base_mean = mean(base_px)
    px_mean = mean(px)
    return {
        "scenario": scenario,
        "moneyness_bucket": bucket,
        "n": len(px),
        "price_mean": px_mean,
        "price_p50": quantile(px, 0.5),
        "price_p90": quantile(px, 0.9),
        "delta_vs_base_mean_pct": ((px_mean - base_mean) / max(abs(base_mean), 1e-12)) * 100.0,
        "delta_vs_base_median_pct": quantile(pct_changes, 0.5) * 100.0,
    }


def main() -> int:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    model = load_model(Path(args.model_json))

    groups: dict[str, list[tuple[float, float, float, float, float, float, str]]] = {
        BUCKET_DEEP_ITM: [],
        BUCKET_ATM: [],
        BUCKET_DEEP_OTM: [],
    }
    seen: dict[str, int] = {k: 0 for k in groups.keys()}
    cap_per_bucket = max(1, int(args.max_rows) // 3)
    sigma_shocks = []
    for x in str(args.sigma_shock_grid).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            sigma_shocks.append(float(x))
        except Exception:
            pass
    sigma_shocks = sorted(set(sigma_shocks))
    r_shocks = []
    for x in str(args.r_shock_grid).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            r_shocks.append(float(x))
        except Exception:
            pass
    r_shocks = sorted(set(r_shocks))

    with Path(args.data_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s0 = safe_float(row.get("S0"))
            k = safe_float(row.get("strike_price"))
            t = safe_float(row.get("T_years"))
            sigma = norm_sigma(safe_float(row.get("sigma")))
            r = norm_rate(safe_float(row.get("r")))
            q = norm_rate(safe_float(row.get("q")))
            opt = norm_type(row.get("option_type", ""))
            if None in (s0, k, t) or opt is None:
                continue
            if s0 <= 0 or k <= 0 or t <= 0:
                continue
            mny = s0 / k
            # Keep realistic trade corridor but cover deep ITM/OTM and ATM.
            if mny < 0.60 or mny > 1.40:
                continue
            bucket = moneyness_bucket(mny, opt)
            rec = (float(s0), float(k), float(t), float(sigma), float(r), float(q), opt)
            seen[bucket] += 1
            arr = groups[bucket]
            if len(arr) < cap_per_bucket:
                arr.append(rec)
            else:
                j = rnd.randint(0, seen[bucket] - 1)
                if j < cap_per_bucket:
                    arr[j] = rec

    rows: list[dict[str, Any]] = []
    for bucket, samples in groups.items():
        base_px: list[float] = []
        sig_up_px: list[float] = []
        r_up_px: list[float] = []
        joint_px: list[float] = []
        sig_pct: list[float] = []
        r_pct: list[float] = []
        joint_pct: list[float] = []

        for s0, k, t, sigma, r, q, opt in samples:
            base_sigma = max(sigma, 1e-4)
            p0 = model_price(model=model, s0=s0, k=k, t=t, sigma=base_sigma, r=r, q=q, opt=opt)
            p_sig = model_price(model=model, s0=s0, k=k, t=t, sigma=max(base_sigma * 1.5, 1e-4), r=r, q=q, opt=opt)
            p_r = model_price(model=model, s0=s0, k=k, t=t, sigma=base_sigma, r=r + 0.02, q=q, opt=opt)
            p_joint = model_price(model=model, s0=s0, k=k, t=t, sigma=max(base_sigma * 1.5, 1e-4), r=r + 0.02, q=q, opt=opt)

            base_px.append(p0)
            sig_up_px.append(p_sig)
            r_up_px.append(p_r)
            joint_px.append(p_joint)
            sig_pct.append((p_sig - p0) / max(abs(p0), 1e-8))
            r_pct.append((p_r - p0) / max(abs(p0), 1e-8))
            joint_pct.append((p_joint - p0) / max(abs(p0), 1e-8))

        rows.append(
            {
                "scenario": "base",
                "moneyness_bucket": bucket,
                "n": len(base_px),
                "price_mean": mean(base_px),
                "price_p50": quantile(base_px, 0.5),
                "price_p90": quantile(base_px, 0.9),
                "delta_vs_base_mean_pct": 0.0,
                "delta_vs_base_median_pct": 0.0,
            }
        )
        rows.append(summarize_scenario(scenario="sigma_plus_50pct", bucket=bucket, px=sig_up_px, base_px=base_px, pct_changes=sig_pct))
        rows.append(summarize_scenario(scenario="rate_plus_2pct", bucket=bucket, px=r_up_px, base_px=base_px, pct_changes=r_pct))
        rows.append(summarize_scenario(scenario="sigma_plus_50pct_and_rate_plus_2pct", bucket=bucket, px=joint_px, base_px=base_px, pct_changes=joint_pct))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "week7_extreme_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "week7_extreme_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Week7 Extreme Scenario Summary (Stratified)\n\n")
        f.write("Buckets: deep ITM (m>=1.15), ATM (0.85<m<1.15), deep OTM (m<=0.85).\\n\\n")
        f.write("| bucket | scenario | n | mean price | p50 | p90 | mean change vs base | median change vs base |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['moneyness_bucket']} | {r['scenario']} | {r['n']} | {r['price_mean']:.6f} | {r['price_p50']:.6f} | "
                f"{r['price_p90']:.6f} | {r['delta_vs_base_mean_pct']:.4f}% | {r['delta_vs_base_median_pct']:.4f}% |\\n"
            )

    by_bucket: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        b = str(r["moneyness_bucket"])
        s = str(r["scenario"])
        by_bucket.setdefault(b, {})[s] = r

    joint_rows: list[dict[str, Any]] = []
    for b, d in by_bucket.items():
        base = d.get("base")
        sig = d.get("sigma_plus_50pct")
        rate = d.get("rate_plus_2pct")
        joint = d.get("sigma_plus_50pct_and_rate_plus_2pct")
        if not all((base, sig, rate, joint)):
            continue
        add_mean = float(sig["delta_vs_base_mean_pct"]) + float(rate["delta_vs_base_mean_pct"])
        add_median = float(sig["delta_vs_base_median_pct"]) + float(rate["delta_vs_base_median_pct"])
        joint_rows.append(
            {
                "moneyness_bucket": b,
                "base_price_mean": float(base["price_mean"]),
                "joint_change_mean_pct": float(joint["delta_vs_base_mean_pct"]),
                "additive_est_mean_pct": add_mean,
                "interaction_mean_pctpt": float(joint["delta_vs_base_mean_pct"]) - add_mean,
                "joint_change_median_pct": float(joint["delta_vs_base_median_pct"]),
                "additive_est_median_pct": add_median,
                "interaction_median_pctpt": float(joint["delta_vs_base_median_pct"]) - add_median,
            }
        )

    joint_csv = out_dir / "week7_joint_impact_summary.csv"
    with joint_csv.open("w", encoding="utf-8", newline="") as f:
        if joint_rows:
            w = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
            w.writeheader()
            w.writerows(joint_rows)

    joint_md = out_dir / "week7_joint_impact_summary.md"
    with joint_md.open("w", encoding="utf-8") as f:
        f.write("# Week7 Joint Shock Impact (Vol + Rate)\n\n")
        f.write("Interaction > 0 means joint shock is stronger than single-factor sum; interaction < 0 means weaker.\n\n")
        f.write("| bucket | joint mean change | additive mean est | interaction (mean, pct-pt) | joint median change | additive median est | interaction (median, pct-pt) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in joint_rows:
            f.write(
                f"| {r['moneyness_bucket']} | {r['joint_change_mean_pct']:.4f}% | {r['additive_est_mean_pct']:.4f}% | "
                f"{r['interaction_mean_pctpt']:.4f} | {r['joint_change_median_pct']:.4f}% | "
                f"{r['additive_est_median_pct']:.4f}% | {r['interaction_median_pctpt']:.4f} |\n"
            )

    sigma_curve_rows: list[dict[str, Any]] = []
    for bucket, samples in groups.items():
        base_px: list[float] = []
        for s0, k, t, sigma, r, q, opt in samples:
            p0 = model_price(model=model, s0=s0, k=k, t=t, sigma=max(sigma, 1e-4), r=r, q=q, opt=opt)
            base_px.append(p0)
        base_mean = mean(base_px)
        for shock in sigma_shocks:
            px: list[float] = []
            for s0, k, t, sigma, r, q, opt in samples:
                sigma_used = max(sigma * (1.0 + shock), 1e-4)
                p = model_price(model=model, s0=s0, k=k, t=t, sigma=sigma_used, r=r, q=q, opt=opt)
                px.append(p)
            px_mean = mean(px)
            sigma_curve_rows.append(
                {
                    "moneyness_bucket": bucket,
                    "sigma_shock_pct": shock * 100.0,
                    "base_price_mean": base_mean,
                    "scenario_price_mean": px_mean,
                    "mean_shift_pct_vs_base": ((px_mean - base_mean) / max(abs(base_mean), 1e-12)) * 100.0,
                }
            )

    sigma_curve_csv = out_dir / "week7_sigma_mean_shift_curve.csv"
    with sigma_curve_csv.open("w", encoding="utf-8", newline="") as f:
        if sigma_curve_rows:
            w = csv.DictWriter(f, fieldnames=list(sigma_curve_rows[0].keys()))
            w.writeheader()
            w.writerows(sigma_curve_rows)

    sigma_curve_png = out_dir / "week7_sigma_mean_shift_curve.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for bucket in (BUCKET_DEEP_ITM, BUCKET_ATM, BUCKET_DEEP_OTM):
            rows_b = [r for r in sigma_curve_rows if r["moneyness_bucket"] == bucket]
            rows_b = sorted(rows_b, key=lambda x: float(x["sigma_shock_pct"]))
            xs = [float(r["sigma_shock_pct"]) for r in rows_b]
            ys = [float(r["mean_shift_pct_vs_base"]) for r in rows_b]
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=bucket)
        plt.axhline(0.0, color="gray", linewidth=1.0)
        plt.xlabel("Sigma shock (%)")
        plt.ylabel("Mean price shift vs base (%)")
        plt.title("Mean Shift vs Sigma Shock (by moneyness bucket)")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(sigma_curve_png, dpi=150)
        plt.close()
    except Exception:
        pass

    r_curve_rows: list[dict[str, Any]] = []
    for bucket, samples in groups.items():
        base_px: list[float] = []
        for s0, k, t, sigma, r, q, opt in samples:
            p0 = model_price(model=model, s0=s0, k=k, t=t, sigma=max(sigma, 1e-4), r=r, q=q, opt=opt)
            base_px.append(p0)
        base_mean = mean(base_px)
        for r_shock in r_shocks:
            px: list[float] = []
            for s0, k, t, sigma, r, q, opt in samples:
                p = model_price(model=model, s0=s0, k=k, t=t, sigma=max(sigma, 1e-4), r=r + r_shock, q=q, opt=opt)
                px.append(p)
            px_mean = mean(px)
            r_curve_rows.append(
                {
                    "moneyness_bucket": bucket,
                    "r_shock_pctpt": r_shock * 100.0,
                    "base_price_mean": base_mean,
                    "scenario_price_mean": px_mean,
                    "mean_shift_pct_vs_base": ((px_mean - base_mean) / max(abs(base_mean), 1e-12)) * 100.0,
                }
            )

    r_curve_csv = out_dir / "week7_r_mean_shift_curve.csv"
    with r_curve_csv.open("w", encoding="utf-8", newline="") as f:
        if r_curve_rows:
            w = csv.DictWriter(f, fieldnames=list(r_curve_rows[0].keys()))
            w.writeheader()
            w.writerows(r_curve_rows)

    r_curve_png = out_dir / "week7_r_mean_shift_curve.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for bucket in (BUCKET_DEEP_ITM, BUCKET_ATM, BUCKET_DEEP_OTM):
            rows_b = [r for r in r_curve_rows if r["moneyness_bucket"] == bucket]
            rows_b = sorted(rows_b, key=lambda x: float(x["r_shock_pctpt"]))
            xs = [float(r["r_shock_pctpt"]) for r in rows_b]
            ys = [float(r["mean_shift_pct_vs_base"]) for r in rows_b]
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=bucket)
        plt.axhline(0.0, color="gray", linewidth=1.0)
        plt.xlabel("Rate shock (pct-pt)")
        plt.ylabel("Mean price shift vs base (%)")
        plt.title("Mean Shift vs Rate Shock (by moneyness bucket)")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(r_curve_png, dpi=150)
        plt.close()
    except Exception:
        pass

    sigma_r_joint_rows: list[dict[str, Any]] = []
    for bucket, samples in groups.items():
        base_px: list[float] = []
        for s0, k, t, sigma, r, q, opt in samples:
            p0 = model_price(model=model, s0=s0, k=k, t=t, sigma=max(sigma, 1e-4), r=r, q=q, opt=opt)
            base_px.append(p0)
        base_mean = mean(base_px)
        for r_shock in r_shocks:
            for sigma_shock in sigma_shocks:
                px: list[float] = []
                for s0, k, t, sigma, r, q, opt in samples:
                    sigma_used = max(sigma * (1.0 + sigma_shock), 1e-4)
                    p = model_price(model=model, s0=s0, k=k, t=t, sigma=sigma_used, r=r + r_shock, q=q, opt=opt)
                    px.append(p)
                px_mean = mean(px)
                sigma_r_joint_rows.append(
                    {
                        "moneyness_bucket": bucket,
                        "sigma_shock_pct": sigma_shock * 100.0,
                        "r_shock_pctpt": r_shock * 100.0,
                        "base_price_mean": base_mean,
                        "scenario_price_mean": px_mean,
                        "mean_shift_pct_vs_base": ((px_mean - base_mean) / max(abs(base_mean), 1e-12)) * 100.0,
                    }
                )

    sigma_r_joint_csv = out_dir / "week7_sigma_r_joint_mean_shift_curve.csv"
    with sigma_r_joint_csv.open("w", encoding="utf-8", newline="") as f:
        if sigma_r_joint_rows:
            w = csv.DictWriter(f, fieldnames=list(sigma_r_joint_rows[0].keys()))
            w.writeheader()
            w.writerows(sigma_r_joint_rows)

    sigma_r_joint_png = out_dir / "week7_sigma_r_joint_mean_shift_curve.png"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        bucket_order = (BUCKET_DEEP_ITM, BUCKET_ATM, BUCKET_DEEP_OTM)
        for i, bucket in enumerate(bucket_order):
            ax = axes[i]
            rows_b = [r for r in sigma_r_joint_rows if r["moneyness_bucket"] == bucket]
            for r_shock in r_shocks:
                rows_br = [r for r in rows_b if abs(float(r["r_shock_pctpt"]) - r_shock * 100.0) < 1e-12]
                rows_br = sorted(rows_br, key=lambda x: float(x["sigma_shock_pct"]))
                xs = [float(r["sigma_shock_pct"]) for r in rows_br]
                ys = [float(r["mean_shift_pct_vs_base"]) for r in rows_br]
                ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"r{r_shock*100:+.1f}pp")
            ax.axhline(0.0, color="gray", linewidth=1.0)
            ax.set_title(bucket)
            ax.set_xlabel("Sigma shock (%)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Mean price shift vs base (%)")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
        fig.suptitle("Joint Shock Mean Shift (Sigma + Rate)")
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(sigma_r_joint_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    sigma_r_joint_heatmap_png = out_dir / "week7_sigma_r_joint_mean_shift_heatmap.png"
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
        bucket_order = (BUCKET_DEEP_ITM, BUCKET_ATM, BUCKET_DEEP_OTM)
        sigma_vals = [x * 100.0 for x in sigma_shocks]
        r_vals = [x * 100.0 for x in r_shocks]

        v_all = [float(r["mean_shift_pct_vs_base"]) for r in sigma_r_joint_rows]
        vmin = min(v_all) if v_all else -1.0
        vmax = max(v_all) if v_all else 1.0

        for i, bucket in enumerate(bucket_order):
            ax = axes[i]
            z = np.full((len(r_vals), len(sigma_vals)), np.nan, dtype=float)
            rows_b = [r for r in sigma_r_joint_rows if r["moneyness_bucket"] == bucket]
            for rr in rows_b:
                sx = float(rr["sigma_shock_pct"])
                ry = float(rr["r_shock_pctpt"])
                v = float(rr["mean_shift_pct_vs_base"])
                ix = sigma_vals.index(sx)
                iy = r_vals.index(ry)
                z[iy, ix] = v

            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[min(sigma_vals), max(sigma_vals), min(r_vals), max(r_vals)],
                cmap="RdYlBu_r",
            )
            ax.set_title(bucket)
            ax.set_xlabel("Sigma shock (%)")
            ax.grid(False)
            if i == 0:
                ax.set_ylabel("Rate shock (pct-pt)")

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
        cbar.set_label("Mean price shift vs base (%)")
        fig.suptitle("Joint Shock Heatmap (Sigma + Rate)")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(sigma_r_joint_heatmap_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    print("[OK] week7 extreme tests complete.")
    print(f"[INFO] n_samples_total={sum(int(r['n']) for r in rows if r['scenario'] == 'base')}")
    print(f"[INFO] out_csv={csv_path}")
    print(f"[INFO] out_md={md_path}")
    print(f"[INFO] out_joint_csv={joint_csv}")
    print(f"[INFO] out_joint_md={joint_md}")
    print(f"[INFO] out_sigma_curve_csv={sigma_curve_csv}")
    print(f"[INFO] out_sigma_curve_png={sigma_curve_png}")
    print(f"[INFO] out_r_curve_csv={r_curve_csv}")
    print(f"[INFO] out_r_curve_png={r_curve_png}")
    print(f"[INFO] out_sigma_r_joint_csv={sigma_r_joint_csv}")
    print(f"[INFO] out_sigma_r_joint_png={sigma_r_joint_png}")
    print(f"[INFO] out_sigma_r_joint_heatmap_png={sigma_r_joint_heatmap_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
