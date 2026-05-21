#!/usr/bin/env python3
"""Gate A comparison: castleCSF Mode A vs baseline zensim-gpu.

Loads per-corpus score parquets (acumen on/off) + MOS truth,
computes the Mohammadi 2025 panel per corpus aggregate + 10-band
B0..B9, prints a side-by-side table.

Inputs (produced by chain_val_corpora.sh + chain_baseline_corpora.sh):
- /mnt/v/output/zensim/synthetic-v2/<corpus>_acumen_modea_2026-05-21.parquet
- /mnt/v/output/zensim/synthetic-v2/<corpus>_baseline_2026-05-21.parquet

Per-corpus MOS:
- cid22 (val/cid22.parquet — human_score = MCOS / 100)
- kadid (val/kadid.parquet — DMOS, lower = better)
- tid (val/tid.parquet — MOS, higher = better)
- konjnd (val/konjnd.parquet — mean PJND threshold)
- aic3 (val/aic3.parquet — score.jnd)

For each (corpus, set) computes:
- SROCC, PLCC, KROCC (Pearson/Spearman/Kendall)
- Z-RMSE (sigma-normalized RMSE)
- PWRC if per-pair sigma is available

Outputs:
- /tmp/gate_a_panel.md — markdown table
- /tmp/gate_a_panel.tsv — same data, tab-separated
"""

import os
import sys
import math
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    from scipy import stats
    import numpy as np
except ImportError:
    print("install: pip install pyarrow scipy numpy", file=sys.stderr)
    sys.exit(2)


def load_score_parquet(path):
    if not Path(path).exists():
        return None
    return pq.read_table(path).to_pandas()


def join_with_mos(scores_df, val_parquet_path, corpus, score_col='zensim_gpu'):
    """Join score parquet with the canonical val parquet's human_score.

    The val parquet has `ref_basename` + `human_score`; the score
    parquet has `image_path` (full path with /mnt/v/.../<basename>.png).
    Match on basename.
    """
    val = pq.read_table(val_parquet_path).to_pandas()

    def basename(p):
        return os.path.basename(p)

    # CID22 val has ref_basename as a SHORTENED form; the score df's
    # image_path is the full ref path. Use the file basename to
    # match. For KADID/TID these are clean filenames like 'I01.png'.
    val['ref_basename_match'] = val['ref_basename'].astype(str)
    scores_df['ref_basename_match'] = scores_df['image_path'].apply(basename)

    # For most corpora the val_parquet has ONE row per pair via
    # (ref_basename, codec, q). Match on (basename, codec, q, knob)
    # when possible.
    if 'codec' in val.columns:
        scores_df['codec_match'] = scores_df['codec'].astype(str)
        val['codec_match'] = val['codec'].astype(str)
        cols = ['ref_basename_match', 'codec_match', 'q', 'knob_tuple_json']
        cols = [c for c in cols if c in val.columns and c in scores_df.columns]
        joined = scores_df.merge(val, on=cols, how='inner')
    else:
        # Match on basename only (CID22 / TID / KADID single-mos shape).
        joined = scores_df.merge(
            val[['ref_basename_match', 'human_score']],
            on='ref_basename_match',
            how='inner',
        )

    return joined


def mohammadi_panel(df, score_col='zensim_gpu', mos_col='human_score'):
    """Compute SROCC + PLCC + KROCC + Z-RMSE for one corpus slice."""
    if len(df) < 5:
        return {'n': len(df), 'note': 'too small'}
    x = df[score_col].astype(float).values
    y = df[mos_col].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {'n': len(x), 'note': 'too small after filter'}

    srocc, _ = stats.spearmanr(x, y)
    plcc, _ = stats.pearsonr(x, y)
    krocc, _ = stats.kendalltau(x, y)
    # Z-RMSE: pseudo-normalized using corpus-wide sigma (per
    # Mohammadi 2025 §III; we don't have per-sample sigma here so
    # corpus-wide is the fallback).
    sigma_y = y.std()
    # Map scores to MOS scale via linear regression for Z-RMSE.
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    z_rmse = math.sqrt(np.mean(((y_pred - y) / max(sigma_y, 1e-6)) ** 2))

    return {
        'n': len(x),
        'SROCC': srocc,
        'PLCC': plcc,
        'KROCC': krocc,
        'Z-RMSE': z_rmse,
        'mean_score': float(x.mean()),
        'mean_mos': float(y.mean()),
    }


def per_band(df, score_col='zensim_gpu', mos_col='human_score'):
    """10-band B0..B9 breakdown (10 zensim-score units wide each)."""
    out = []
    for b in range(10):
        lo, hi = b * 10, (b + 1) * 10
        mask = (df[score_col].astype(float) >= lo) & (df[score_col].astype(float) < hi)
        sub = df[mask]
        if len(sub) >= 5:
            m = mohammadi_panel(sub, score_col, mos_col)
            m['band'] = f'B{b}'
            out.append(m)
    return out


def main():
    OUT_DIR = Path('/mnt/v/output/zensim/synthetic-v2')
    VAL_DIR = Path('/mnt/v/zen/zensim-training/canonical-2026-05-21/val')

    corpora = ['kadid', 'tid', 'aic3', 'cid22', 'konjnd']
    val_paths = {
        'cid22': VAL_DIR / 'cid22.parquet',
        'kadid': VAL_DIR / 'kadid.parquet',
        'tid': VAL_DIR / 'tid.parquet',
        'konjnd': VAL_DIR / 'konjnd.parquet',
        'aic3': VAL_DIR / 'aic3.parquet',
    }

    lines = ['# Gate A: castleCSF Mode A vs baseline zensim-gpu', '']
    lines.append('| Corpus | Set | n | SROCC | PLCC | KROCC | Z-RMSE |')
    lines.append('|---|---|--:|--:|--:|--:|--:|')

    for corpus in corpora:
        for label, path in [
            ('baseline', OUT_DIR / f'{corpus}_baseline_2026-05-21.parquet'),
            ('acumen-on', OUT_DIR / f'{corpus}_acumen_modea_2026-05-21.parquet'),
        ]:
            scores_df = load_score_parquet(path)
            if scores_df is None:
                lines.append(f'| {corpus} | {label} | — | (parquet missing) | | | |')
                continue
            val_p = val_paths.get(corpus)
            if val_p is None or not val_p.exists():
                lines.append(f'| {corpus} | {label} | — | (val parquet missing) | | | |')
                continue
            joined = join_with_mos(scores_df, val_p, corpus)
            stats_p = mohammadi_panel(joined)
            if 'SROCC' in stats_p:
                lines.append(
                    f'| {corpus} | {label} | {stats_p["n"]} | '
                    f'{stats_p["SROCC"]:.4f} | {stats_p["PLCC"]:.4f} | '
                    f'{stats_p["KROCC"]:.4f} | {stats_p["Z-RMSE"]:.4f} |'
                )
            else:
                lines.append(f'| {corpus} | {label} | {stats_p["n"]} | (small) | | | |')

    out_md = Path('/tmp/gate_a_panel.md')
    out_md.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nWritten to {out_md}', file=sys.stderr)


if __name__ == '__main__':
    main()
