#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def std_name(x: str) -> str:
    s = (x or '').strip().lower()
    if s == 't_years':
        return 'ttm_years'
    return s


def norm(vals: list[float]) -> list[float]:
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return [0.0 for _ in vals]
    return [(v - mn) / (mx - mn) for v in vals]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    psi_csv = root / 'output' / 'week8_distribution_check' / 'distribution_psi_train_vs_test.csv'
    imp_csv = root / 'output' / 'week6_explainability' / 'mlp_explain_permutation_importance.csv'
    out_dir = root / 'output' / 'week8_distribution_check'
    out_dir.mkdir(parents=True, exist_ok=True)

    psi_rows = read_csv(psi_csv)
    imp_rows = read_csv(imp_csv)

    psi_map = {std_name(r['feature']): float(r['psi_train_vs_test']) for r in psi_rows}
    delta_map = {std_name(r['feature']): float(r['delta_mae']) for r in imp_rows}

    feats = sorted(set(psi_map.keys()) | set(delta_map.keys()))
    psi_vals = [psi_map.get(f, 0.0) for f in feats]
    d_vals = [delta_map.get(f, 0.0) for f in feats]
    psi_n = norm(psi_vals)
    d_n = norm(d_vals)

    merged = []
    for i, f in enumerate(feats):
        p = psi_vals[i]
        d = d_vals[i]
        score = psi_n[i] * d_n[i]
        if p >= 0.25 and d >= 2.0:
            level = 'critical'
        elif p >= 0.1 and d >= 1.0:
            level = 'watch'
        else:
            level = 'low'
        merged.append(
            {
                'feature': f,
                'psi': p,
                'delta_mae': d,
                'psi_norm': psi_n[i],
                'delta_norm': d_n[i],
                'joint_score': score,
                'joint_level': level,
            }
        )

    merged.sort(key=lambda r: r['joint_score'], reverse=True)

    out_csv = out_dir / 'psi_sensitivity_joint.csv'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)

    out_md = out_dir / 'psi_sensitivity_joint.md'
    with out_md.open('w', encoding='utf-8') as f:
        f.write('# PSI × Sensitivity Joint Analysis\n\n')
        f.write('| feature | PSI | delta_MAE | joint_score | level |\n')
        f.write('|---|---:|---:|---:|---|\n')
        for r in merged:
            f.write(f"| {r['feature']} | {r['psi']:.4f} | {r['delta_mae']:.4f} | {r['joint_score']:.4f} | {r['joint_level']} |\\n")

    print('[OK] joint report generated')
    print(f'[INFO] out_csv={out_csv}')
    print(f'[INFO] out_md={out_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
