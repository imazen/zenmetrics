"""HTML and visual display utilities for zenmetrics in Jupyter Notebooks."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Metadata for each known metric: (higher_is_better, typical_min, typical_max, unit, description)
METRIC_META: Dict[str, Tuple[bool, float, float, str, str]] = {
    "ssim2": (True, 0.0, 100.0, "/ 100", "SSIMULACRA 2 visual quality score (100 is identical)"),
    "ssimulacra2": (True, 0.0, 100.0, "/ 100", "SSIMULACRA 2 visual quality score (100 is identical)"),
    "ssim2-gpu": (True, 0.0, 100.0, "/ 100", "SSIMULACRA 2 (GPU accelerated)"),
    "butteraugli": (False, 0.0, 5.0, "pnorm", "Butteraugli psychovisual distance (lower is better, <0.5 indistinguishable)"),
    "butteraugli-gpu": (False, 0.0, 5.0, "pnorm", "Butteraugli psychovisual distance (GPU)"),
    "dssim": (False, 0.0, 0.1, "dist", "Structural dissimilarity (lower is better, 0 is identical)"),
    "dssim-gpu": (False, 0.0, 0.1, "dist", "Structural dissimilarity (GPU)"),
    "ssim": (True, 0.0, 1.0, "", "Structural Similarity Index (1.0 is identical)"),
    "ssim-libvmaf": (True, 0.0, 1.0, "", "SSIM (libvmaf compatible)"),
    "msssim": (True, 0.0, 1.0, "", "Multi-Scale SSIM (1.0 is identical)"),
    "msssim-libvmaf": (True, 0.0, 1.0, "", "Multi-Scale SSIM (libvmaf compatible)"),
    "psnr": (True, 20.0, 60.0, "dB", "Peak Signal-to-Noise Ratio (higher is better)"),
    "psnr-y": (True, 20.0, 60.0, "dB", "Peak Signal-to-Noise Ratio (Luma BT.709)"),
    "psnr-y601": (True, 20.0, 60.0, "dB", "Peak Signal-to-Noise Ratio (Luma BT.601)"),
    "psnr-y-studio601": (True, 20.0, 60.0, "dB", "Peak Signal-to-Noise Ratio (Studio BT.601)"),
    "psnr-y-libvmaf": (True, 20.0, 60.0, "dB", "Peak Signal-to-Noise Ratio (libvmaf)"),
    "cvvdp": (True, 0.0, 10.0, "JOD", "Color Visual Difference Predictor (higher is better, >8 near identical)"),
    "cvvdp-gpu": (True, 0.0, 10.0, "JOD", "Color Visual Difference Predictor (GPU)"),
    "iwssim": (True, 0.0, 1.0, "", "Information-theoretic Weighted SSIM (1.0 is identical)"),
    "iwssim-gpu": (True, 0.0, 1.0, "", "Information-theoretic Weighted SSIM (GPU)"),
    "iwssim-piq": (True, 0.0, 1.0, "", "Information-theoretic Weighted SSIM (PIQ compatible)"),
    "gmsd": (False, 0.0, 0.3, "", "Gradient Magnitude Similarity Deviation (lower is better)"),
    "mdsi": (False, 0.0, 0.5, "", "Mean Deviation Similarity Index (lower is better)"),
    "ms-gmsd": (False, 0.0, 0.3, "", "Multi-Scale GMSD (lower is better)"),
    "ms-gmsdc": (False, 0.0, 0.3, "", "Multi-Scale GMSD with Chrominance (lower is better)"),
    "fsim": (True, 0.0, 1.0, "", "Feature Similarity Index (1.0 is identical)"),
    "fsim-y": (True, 0.0, 1.0, "", "Feature Similarity Index (Luma)"),
    "haarpsi": (True, 0.0, 1.0, "", "Haar Wavelet-based Perceptual Similarity (1.0 is identical)"),
    "haarpsi-y": (True, 0.0, 1.0, "", "HaarPSI (Luma)"),
    "vif": (True, 0.0, 1.0, "", "Visual Information Fidelity (higher is better)"),
    "vifvec": (True, 0.0, 1.0, "", "Visual Information Fidelity (Vector)"),
    "vmaf": (True, 0.0, 100.0, "/ 100", "Video Multi-Method Assessment Fusion (100 is identical)"),
    "vmaf-neg": (True, 0.0, 100.0, "/ 100", "VMAF NEG mode"),
    "vmaf-4k": (True, 0.0, 100.0, "/ 100", "VMAF 4K model"),
    "vsi": (True, 0.0, 1.0, "", "Visual Saliency-induced Index (1.0 is identical)"),
    "nlpd": (False, 0.0, 0.5, "", "Normalized Laplacian Pyramid Distance (lower is better)"),
    "psnrhvs": (True, 20.0, 60.0, "dB", "PSNR with Human Visual System CSF (higher is better)"),
    "mad": (False, 0.0, 100.0, "", "Most Apparent Distortion (lower is better)"),
    "mdctpsnr": (True, 20.0, 60.0, "dB", "MDCT-based PSNR (higher is better)"),
    "zensim": (True, 0.0, 100.0, "/ 100", "ZenSIM Learned Perceptual Quality Score"),
}


def is_higher_better(metric: str) -> bool:
    """Return True if higher values indicate higher visual quality."""
    m = metric.lower().replace("_", "-")
    meta = METRIC_META.get(m)
    if meta:
        return meta[0]
    # Heuristic for unknown metrics: distances / deviations are lower is better
    if any(k in m for k in ("dist", "loss", "diff", "dev", "err", "gmsd", "mdsi", "nlpd", "mad", "butter")):
        return False
    return True


def get_metric_quality_rating(metric: str, value: float) -> Tuple[str, str, float]:
    """Return (label, color_class, normalized_pct [0-100]).

    color_class: 'zm-excellent' | 'zm-good' | 'zm-fair' | 'zm-poor'
    """
    m = metric.lower().replace("_", "-")
    higher = is_higher_better(m)

    if m in ("ssim2", "ssimulacra2", "ssim2-gpu", "zensim", "vmaf"):
        pct = max(0.0, min(100.0, value))
        if value >= 85.0:
            return ("Excellent", "zm-excellent", pct)
        elif value >= 70.0:
            return ("Good", "zm-good", pct)
        elif value >= 50.0:
            return ("Fair", "zm-fair", pct)
        else:
            return ("Poor", "zm-poor", pct)

    if "butteraugli" in m or m == "butter":
        # 0.0 is perfect, 0.5 near imperceptible, 1.0 subtle, 2.0+ noticeable
        pct = max(0.0, min(100.0, (1.0 - min(value, 3.0) / 3.0) * 100.0))
        if value < 0.5:
            return ("Imperceptible (<0.5)", "zm-excellent", pct)
        elif value < 1.0:
            return ("Near-Lossless (<1.0)", "zm-good", pct)
        elif value < 2.0:
            return ("Noticeable (1-2)", "zm-fair", pct)
        else:
            return ("Distorted (>2.0)", "zm-poor", pct)

    if m in ("dssim", "dssim-gpu"):
        pct = max(0.0, min(100.0, (1.0 - min(value, 0.05) / 0.05) * 100.0))
        if value < 0.005:
            return ("Excellent (<0.005)", "zm-excellent", pct)
        elif value < 0.02:
            return ("Good (<0.02)", "zm-good", pct)
        elif value < 0.05:
            return ("Fair (<0.05)", "zm-fair", pct)
        else:
            return ("Poor (>0.05)", "zm-poor", pct)

    if "psnr" in m:
        pct = max(0.0, min(100.0, (min(max(value, 20.0), 50.0) - 20.0) / 30.0 * 100.0))
        if value >= 40.0:
            return ("Excellent (>40 dB)", "zm-excellent", pct)
        elif value >= 35.0:
            return ("Good (35-40 dB)", "zm-good", pct)
        elif value >= 30.0:
            return ("Fair (30-35 dB)", "zm-fair", pct)
        else:
            return ("Poor (<30 dB)", "zm-poor", pct)

    if "cvvdp" in m:
        pct = max(0.0, min(100.0, (min(max(value, 0.0), 10.0) / 10.0) * 100.0))
        if value >= 8.5:
            return ("Excellent (>8.5 JOD)", "zm-excellent", pct)
        elif value >= 6.5:
            return ("Good (6.5-8.5 JOD)", "zm-good", pct)
        elif value >= 4.0:
            return ("Fair (4-6.5 JOD)", "zm-fair", pct)
        else:
            return ("Poor (<4 JOD)", "zm-poor", pct)

    if m in ("ssim", "msssim", "iwssim", "fsim", "haarpsi", "vsi"):
        pct = max(0.0, min(100.0, value * 100.0))
        if value >= 0.98:
            return ("Excellent (>0.98)", "zm-excellent", pct)
        elif value >= 0.95:
            return ("Good (0.95-0.98)", "zm-good", pct)
        elif value >= 0.90:
            return ("Fair (0.90-0.95)", "zm-fair", pct)
        else:
            return ("Poor (<0.90)", "zm-poor", pct)

    # Generic fallback
    if higher:
        return ("Score", "zm-good", 75.0)
    else:
        return ("Score", "zm-good", 75.0)


BASE_STYLES = """
<style>
.zm-card {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--jp-layout-color1, #ffffff);
    color: var(--jp-ui-font-color1, #1e293b);
    border: 1px solid var(--jp-border-color1, #e2e8f0);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    max-width: 680px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
@media (prefers-color-scheme: dark) {
    .zm-card {
        background: #1e1e2e;
        color: #f1f5f9;
        border-color: #334155;
    }
}
.zm-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.zm-badge-brand {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: #eef2ff;
    color: #4f46e5;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
@media (prefers-color-scheme: dark) {
    .zm-badge-brand {
        background: #312e81;
        color: #c7d2fe;
    }
}
.zm-badge-rating {
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
}
.zm-excellent { background: #dcfce7; color: #15803d; }
.zm-good { background: #e0f2fe; color: #0369a1; }
.zm-fair { background: #fef3c7; color: #b45309; }
.zm-poor { background: #fee2e2; color: #b91c1c; }

@media (prefers-color-scheme: dark) {
    .zm-excellent { background: #064e3b; color: #6ee7b7; }
    .zm-good { background: #0c4a6e; color: #7dd3fc; }
    .zm-fair { background: #78350f; color: #fde68a; }
    .zm-poor { background: #7f1d1d; color: #fca5a5; }
}

.zm-score-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
}
.zm-score-val {
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}
.zm-score-unit {
    font-size: 14px;
    color: #64748b;
    font-weight: 500;
}
.zm-meter-bg {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 12px;
}
@media (prefers-color-scheme: dark) {
    .zm-meter-bg { background: #334155; }
}
.zm-meter-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.3s ease;
}
.zm-desc {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 10px;
}
.zm-subscores {
    margin-top: 10px;
    border-top: 1px dashed var(--jp-border-color1, #e2e8f0);
    padding-top: 8px;
    font-size: 12px;
}
.zm-subscore-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 6px 12px;
    margin-top: 4px;
}
.zm-subscore-item {
    display: flex;
    justify-content: space-between;
    color: #64748b;
}
.zm-subscore-item strong {
    color: inherit;
}
.zm-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
}
.zm-table th {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 2px solid var(--jp-border-color1, #cbd5e1);
    font-weight: 600;
    color: #475569;
}
@media (prefers-color-scheme: dark) {
    .zm-table th { color: #94a3b8; border-color: #475569; }
}
.zm-table td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--jp-border-color1, #f1f5f9);
}
@media (prefers-color-scheme: dark) {
    .zm-table td { border-color: #334155; }
}
.zm-table tr:hover td {
    background: rgba(99, 102, 241, 0.04);
}
.zm-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 10px;
}
.zm-stat-box {
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}
.zm-stat-box .title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
}
.zm-stat-box .val {
    font-size: 20px;
    font-weight: 700;
    margin-top: 2px;
}
</style>
"""


def render_score_html(result: Any) -> str:
    """Render a ScoreResult object as an interactive HTML badge."""
    metric = getattr(result, "metric", "score")
    value = float(getattr(result, "value", 0.0))
    scores: Dict[str, float] = getattr(result, "scores", {})

    meta = METRIC_META.get(metric.lower().replace("_", "-"))
    unit = meta[3] if meta else ""
    desc = meta[4] if meta else "Image Quality Metric"

    rating_label, rating_cls, fill_pct = get_metric_quality_rating(metric, value)

    # Meter fill color
    meter_color = {
        "zm-excellent": "#10b981",
        "zm-good": "#0284c7",
        "zm-fair": "#f59e0b",
        "zm-poor": "#ef4444",
    }.get(rating_cls, "#6366f1")

    # Format value
    if abs(value) >= 100:
        val_str = f"{value:.2f}"
    elif abs(value) >= 1:
        val_str = f"{value:.4f}"
    else:
        val_str = f"{value:.6f}"

    # Build subscores if more than 1
    subscores_html = ""
    if len(scores) > 1:
        items = "".join(
            f'<div class="zm-subscore-item"><span>{html.escape(k)}</span><strong>{v:.4f}</strong></div>'
            for k, v in scores.items()
        )
        subscores_html = f"""
        <details class="zm-subscores">
            <summary style="cursor: pointer; color: #6366f1; font-weight: 600;">Details ({len(scores)} sub-metrics)</summary>
            <div class="zm-subscore-grid">{items}</div>
        </details>
        """

    return f"""{BASE_STYLES}
    <div class="zm-card">
        <div class="zm-header">
            <span class="zm-badge-brand">zenmetrics &bull; {html.escape(metric.upper())}</span>
            <span class="zm-badge-rating {rating_cls}">{html.escape(rating_label)}</span>
        </div>
        <div class="zm-score-row">
            <span class="zm-score-val">{val_str}</span>
            <span class="zm-score-unit">{html.escape(unit)}</span>
        </div>
        <div class="zm-meter-bg">
            <div class="zm-meter-fill" style="width: {fill_pct:.1f}%; background: {meter_color};"></div>
        </div>
        <div class="zm-desc">{html.escape(desc)}</div>
        {subscores_html}
    </div>
    """


def render_compare_html(rows: List[Dict[str, Any]], metrics: List[str]) -> str:
    """Render multi-variant comparison results as a rich HTML table."""
    if not rows:
        return "<p>No comparison data.</p>"

    # Find best value for each metric column
    best_vals: Dict[str, float] = {}
    for m in metrics:
        higher = is_higher_better(m)
        vals = [r[m] for r in rows if m in r and isinstance(r[m], (int, float))]
        if vals:
            best_vals[m] = max(vals) if higher else min(vals)

    headers = "".join(
        f'<th>{html.escape(m)} <span style="font-size: 10px; color: #94a3b8;">{"(▲)" if is_higher_better(m) else "(▼)"}</span></th>'
        for m in metrics
    )

    tbody_rows = []
    for r in rows:
        variant = str(r.get("variant", ""))
        row_tds = [f"<td><strong>{html.escape(variant)}</strong></td>"]
        for m in metrics:
            val = r.get(m)
            if val is None:
                row_tds.append("<td>-</td>")
            elif isinstance(val, (int, float)):
                is_best = abs(val - best_vals.get(m, float("nan"))) < 1e-7
                best_badge = ' style="font-weight: 700; color: #10b981;"' if is_best else ""
                val_str = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
                star = " ★" if is_best else ""
                row_tds.append(f"<td{best_badge}>{val_str}{star}</td>")
            else:
                row_tds.append(f"<td>{html.escape(str(val))}</td>")
        tbody_rows.append(f"<tr>{''.join(row_tds)}</tr>")

    table_body = "\n".join(tbody_rows)

    return f"""{BASE_STYLES}
    <div class="zm-card" style="max-width: 800px;">
        <div class="zm-header">
            <span class="zm-badge-brand">zenmetrics &bull; Multi-Variant Comparison</span>
            <span style="font-size: 12px; color: #64748b;">{len(rows)} variants evaluated</span>
        </div>
        <table class="zm-table">
            <thead>
                <tr>
                    <th>Variant</th>
                    {headers}
                </tr>
            </thead>
            <tbody>
                {table_body}
            </tbody>
        </table>
        <div class="zm-desc" style="margin-top: 10px;">▲ Higher is better &bull; ▼ Lower is better &bull; ★ Best in column</div>
    </div>
    """


def render_folder_html(folder_res: Dict[str, Any]) -> str:
    """Render folder-to-folder evaluation summary in Jupyter."""
    metric = folder_res.get("metric", "metric")
    total = folder_res.get("total_pairs", 0)
    success = folder_res.get("successful_pairs", 0)
    failed = folder_res.get("failed_pairs", 0)
    summary = folder_res.get("summary", {})
    results = folder_res.get("results", [])

    # Main metric stats
    m_stat = summary.get(metric, {})
    mean_val = m_stat.get("mean", 0.0)
    min_val = m_stat.get("min", 0.0)
    max_val = m_stat.get("max", 0.0)

    # Preview up to 10 rows
    preview_rows = results[:10]
    tbody = []
    for p in preview_rows:
        ref_name = Path(p.get("reference", "")).name
        dist_name = Path(p.get("distorted", "")).name
        val = p.get("value")
        err = p.get("error")
        if err:
            status = f'<span style="color: #ef4444;">Error: {html.escape(err)}</span>'
        else:
            status = f'<strong style="color: #6366f1;">{val:.4f}</strong>'
        tbody.append(f"<tr><td>{html.escape(ref_name)}</td><td>{html.escape(dist_name)}</td><td>{status}</td></tr>")

    more_hint = f'<tr><td colspan="3" style="text-align: center; color: #64748b; font-size: 11px;">... and {total - 10} more pairs (see .results or .to_dataframe())</td></tr>' if total > 10 else ""

    return f"""{BASE_STYLES}
    <div class="zm-card" style="max-width: 780px;">
        <div class="zm-header">
            <span class="zm-badge-brand">zenmetrics &bull; Folder Evaluation ({html.escape(metric.upper())})</span>
            <span class="zm-badge-rating zm-good">{success}/{total} Pairs Evaluated</span>
        </div>
        <div class="zm-stat-grid">
            <div class="zm-stat-box">
                <div class="title">Mean Score</div>
                <div class="val" style="color: #6366f1;">{mean_val:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title">Min Score</div>
                <div class="val">{min_val:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title">Max Score</div>
                <div class="val">{max_val:.4f}</div>
            </div>
        </div>
        <table class="zm-table" style="margin-top: 14px;">
            <thead>
                <tr>
                    <th>Reference</th>
                    <th>Distorted</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(tbody)}
                {more_hint}
            </tbody>
        </table>
    </div>
    """


def render_panel_stats_html(stats: Any) -> str:
    """Render IQA evaluation benchmark statistics panel (ITU-T standard)."""
    srocc = getattr(stats, "srocc", 0.0)
    plcc = getattr(stats, "plcc", 0.0)
    krocc = getattr(stats, "krocc", 0.0)
    outlier = getattr(stats, "outlier_ratio", 0.0)
    pwrc = getattr(stats, "pwrc_auc", 0.0)
    z_rmse = getattr(stats, "z_rmse", 0.0)
    n = getattr(stats, "n", 0)

    return f"""{BASE_STYLES}
    <div class="zm-card" style="max-width: 680px;">
        <div class="zm-header">
            <span class="zm-badge-brand">zenmetrics &bull; IQA Benchmark Panel</span>
            <span style="font-size: 12px; color: #64748b;">N = {n} subjective MOS pairs</span>
        </div>
        <div class="zm-stat-grid" style="grid-template-columns: repeat(3, 1fr);">
            <div class="zm-stat-box">
                <div class="title" title="Spearman Rank Order Correlation">SROCC (Monotonicity)</div>
                <div class="val" style="color: #10b981;">{srocc:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title" title="Pearson Linear Correlation after logistic rescaling">PLCC (Linearity)</div>
                <div class="val" style="color: #0284c7;">{plcc:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title" title="Kendall Rank Correlation">KROCC (Rank)</div>
                <div class="val">{krocc:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title" title="Pairwise Ranking Consistency AUC">PWRC-AUC</div>
                <div class="val" style="color: #6366f1;">{pwrc:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title" title="Outlier Ratio (< 2 sigma)">Outlier Ratio</div>
                <div class="val">{outlier:.4f}</div>
            </div>
            <div class="zm-stat-box">
                <div class="title" title="Z-Score Root Mean Square Error">Z-RMSE</div>
                <div class="val">{z_rmse:.4f}</div>
            </div>
        </div>
        <div class="zm-desc" style="margin-top: 12px;">
            ITU-T P.1401 recommended evaluation metrics for subjective image quality correlation.
        </div>
    </div>
    """
