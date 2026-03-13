#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return float("nan")
    y_bar = sum(y_true) / len(y_true)
    ss_tot = sum((y - y_bar) ** 2 for y in y_true)
    ss_res = sum((yp - yt) ** 2 for yt, yp in zip(y_true, y_pred))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae_rmse(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    if not y_true:
        return float("nan"), float("nan")
    ae = 0.0
    se = 0.0
    n = len(y_true)
    for yt, yp in zip(y_true, y_pred):
        d = yp - yt
        ae += abs(d)
        se += d * d
    return ae / n, math.sqrt(se / n)


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
    p = argparse.ArgumentParser(description="Week6 requirements runner: hyper-search + MAE/RMSE/R2 + explainability.")
    p.add_argument("--data-csv", default=str(root / "output" / "server_bundle_week5" / "data" / "processed" / "jpm_options_final.csv"))
    p.add_argument("--max-train", type=int, default=60000)
    p.add_argument("--max-val", type=int, default=20000)
    p.add_argument("--max-test", type=int, default=20000)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perm-repeats", type=int, default=4)
    p.add_argument("--local-k", type=int, default=8, help="Top-K worst test samples for local explanation")
    p.add_argument("--out-dir", default=str(root / "output" / "week6_requirements"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    week5_src = root / "output" / "server_bundle_week5" / "src"
    if str(week5_src) not in sys.path:
        sys.path.insert(0, str(week5_src))

    # Reuse week5 pure-python implementations.
    from week5_models import FEATURE_NAMES, MLPRegressor, load_split_datasets  # type: ignore

    train, val, test, split_info = load_split_datasets(
        Path(args.data_csv),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        max_train=int(args.max_train),
        max_val=int(args.max_val),
        max_test=int(args.max_test),
    )
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError("Empty split after loading dataset.")

    # 1) Hyper-parameter search
    grid_hidden = [16, 24, 32]
    grid_lr = [0.0015, 0.0025]
    grid_epochs = [6, 10]
    batch_size = 128
    search_rows: list[dict[str, Any]] = []
    best = None
    best_key = None

    trial_id = 0
    for h in grid_hidden:
        for lr in grid_lr:
            for ep in grid_epochs:
                trial_id += 1
                m = MLPRegressor(
                    in_dim=len(FEATURE_NAMES),
                    hidden=h,
                    lr=lr,
                    epochs=ep,
                    batch_size=batch_size,
                    seed=int(args.seed) + trial_id,
                )
                m.fit(train.X, train.y)
                p_val = m.predict(val.X)
                val_mae, val_rmse = mae_rmse(val.y, p_val)
                val_r2 = r2_score(val.y, p_val)
                row = {
                    "trial_id": trial_id,
                    "hidden": h,
                    "lr": lr,
                    "epochs": ep,
                    "batch_size": batch_size,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                }
                search_rows.append(row)
                key = val_rmse
                if best is None or key < best_key:
                    best = m
                    best_key = key

    search_rows.sort(key=lambda r: r["val_rmse"])

    # 2) Final metrics (MAE/RMSE/R2)
    assert best is not None
    p_train = best.predict(train.X)
    p_val = best.predict(val.X)
    p_test = best.predict(test.X)
    tr_mae, tr_rmse = mae_rmse(train.y, p_train)
    va_mae, va_rmse = mae_rmse(val.y, p_val)
    te_mae, te_rmse = mae_rmse(test.y, p_test)
    tr_r2 = r2_score(train.y, p_train)
    va_r2 = r2_score(val.y, p_val)
    te_r2 = r2_score(test.y, p_test)

    metrics_rows = [
        {"split": "train", "n": len(train.y), "mae": tr_mae, "rmse": tr_rmse, "r2": tr_r2},
        {"split": "val", "n": len(val.y), "mae": va_mae, "rmse": va_rmse, "r2": va_r2},
        {"split": "test", "n": len(test.y), "mae": te_mae, "rmse": te_rmse, "r2": te_r2},
    ]

    # 3) Explainability: permutation importance (global)
    rnd = random.Random(int(args.seed))
    perm_rows: list[dict[str, Any]] = []
    base_mae, base_rmse = te_mae, te_rmse
    base_r2 = te_r2
    X = test.X
    y = test.y
    for fi, fname in enumerate(FEATURE_NAMES):
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []
        for _ in range(int(args.perm_repeats)):
            idx = list(range(len(X)))
            rnd.shuffle(idx)
            p_perm: list[float] = []
            for i, x in enumerate(X):
                x2 = x.copy()
                x2[fi] = X[idx[i]][fi]
                p_perm.append(best.predict([x2])[0])
            m1, m2 = mae_rmse(y, p_perm)
            r2v = r2_score(y, p_perm)
            maes.append(m1)
            rmses.append(m2)
            r2s.append(r2v)
        pm = sum(maes) / len(maes)
        pr = sum(rmses) / len(rmses)
        p2 = sum(r2s) / len(r2s)
        perm_rows.append(
            {
                "feature": fname,
                "base_mae": base_mae,
                "perm_mae": pm,
                "delta_mae": pm - base_mae,
                "base_rmse": base_rmse,
                "perm_rmse": pr,
                "delta_rmse": pr - base_rmse,
                "base_r2": base_r2,
                "perm_r2": p2,
                "delta_r2": p2 - base_r2,
            }
        )
    perm_rows.sort(key=lambda r: r["delta_mae"], reverse=True)

    # 4) Explainability: SHAP/LIME style local proxy (occlusion-to-mean)
    # mean in normalized space is close to 0 for each feature.
    local_rows: list[dict[str, Any]] = []
    abs_err = [abs(p - t) for p, t in zip(p_test, test.y)]
    top_idx = sorted(range(len(abs_err)), key=lambda i: abs_err[i], reverse=True)[: int(args.local_k)]
    for rank, i in enumerate(top_idx, start=1):
        x = test.X[i]
        y_hat = p_test[i]
        y_true = test.y[i]
        base = y_hat
        contrib = []
        for fi, fname in enumerate(FEATURE_NAMES):
            x2 = x.copy()
            x2[fi] = 0.0
            y2 = best.predict([x2])[0]
            contrib.append((fname, base - y2))
        contrib.sort(key=lambda t: abs(t[1]), reverse=True)
        row = {
            "rank_by_abs_error": rank,
            "sample_index": i,
            "y_true": y_true,
            "y_pred": y_hat,
            "abs_error": abs(y_hat - y_true),
            "top1_feature": contrib[0][0],
            "top1_contrib": contrib[0][1],
            "top2_feature": contrib[1][0],
            "top2_contrib": contrib[1][1],
            "top3_feature": contrib[2][0],
            "top3_contrib": contrib[2][1],
        }
        local_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "week6_hyperparam_search.csv", search_rows)
    write_csv(out_dir / "week6_metrics_mae_rmse_r2.csv", metrics_rows)
    write_csv(out_dir / "week6_explain_global_permutation.csv", perm_rows)
    write_csv(out_dir / "week6_explain_local_proxy.csv", local_rows)
    (out_dir / "week6_split_info.json").write_text(json.dumps(split_info, ensure_ascii=True, indent=2), encoding="utf-8")

    # 5) concise report (table-first)
    best_cfg = search_rows[0]
    md = out_dir / "week6_report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Week6 Report (Concise)\n\n")
        f.write("## Requirement Checklist\n\n")
        f.write("| Item | Status | Evidence |\n")
        f.write("|---|---|---|\n")
        f.write("| Hyper-parameter search | Done | `week6_hyperparam_search.csv` |\n")
        f.write("| MAE/RMSE/R2 | Done | `week6_metrics_mae_rmse_r2.csv` |\n")
        f.write("| SHAP/LIME explainability | Done* | `week6_explain_global_permutation.csv`, `week6_explain_local_proxy.csv` |\n")
        f.write("\n")
        f.write("> *Current env uses dependency-free SHAP/LIME-style proxy (permutation global + local occlusion). If `shap/lime` libs are installed, can upgrade to exact package outputs.\n\n")

        f.write("## Best Hyper-Parameters (by val RMSE)\n\n")
        f.write("| hidden | lr | epochs | batch_size | val_mae | val_rmse | val_r2 |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {best_cfg['hidden']} | {best_cfg['lr']:.4f} | {best_cfg['epochs']} | {best_cfg['batch_size']} | "
            f"{best_cfg['val_mae']:.6f} | {best_cfg['val_rmse']:.6f} | {best_cfg['val_r2']:.6f} |\n\n"
        )

        f.write("## Core Metrics\n\n")
        f.write("| split | n | MAE | RMSE | R2 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in metrics_rows:
            f.write(f"| {r['split']} | {r['n']} | {r['mae']:.6f} | {r['rmse']:.6f} | {r['r2']:.6f} |\n")
        f.write("\n")

        f.write("## Top Global Drivers (Permutation)\n\n")
        f.write("| feature | delta_MAE | delta_RMSE | delta_R2 |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in perm_rows[:5]:
            f.write(f"| {r['feature']} | {r['delta_mae']:.6f} | {r['delta_rmse']:.6f} | {r['delta_r2']:.6f} |\n")
        f.write("\n")

        f.write("## Notes\n\n")
        f.write("- Data split: 70/15/15 by date (anti look-ahead).\n")
        f.write("- If you provide server full-data outputs, this report can be refreshed with your exact production run.\n")

    print("[OK] week6 requirements run complete.")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] report={md}")
    print(f"[INFO] best_hidden={best_cfg['hidden']} best_lr={best_cfg['lr']} best_epochs={best_cfg['epochs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
