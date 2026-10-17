"""Statistical assessment metrics from zenmetrics-stats.

Implements standard ITU-T P.1401 subjective IQA evaluation metrics:
SROCC, PLCC, Kendall tau, Outlier Ratio, PWRC-AUC, Z-RMSE, and 4-parameter logistic curve fitting.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union
from . import _zenmetrics
from .display import render_panel_stats_html

NumberSeq = Union[Sequence[float], Any]


def _to_float_list(v: NumberSeq) -> list[float]:
    if hasattr(v, "tolist"):
        return [float(x) for x in v.tolist()]
    return [float(x) for x in v]


class PanelStatsResult:
    """Rich container for ITU-T benchmark statistics with Jupyter visualization."""

    def __init__(self, raw: _zenmetrics.PanelStats):
        self._raw = raw
        self.srocc: float = raw.srocc
        self.plcc: float = raw.plcc
        self.krocc: float = raw.krocc
        self.outlier_ratio: float = raw.outlier_ratio
        self.pwrc_auc: float = raw.pwrc_auc
        self.z_rmse: float = raw.z_rmse
        self.n: int = int(raw.n)

    def to_dict(self) -> Dict[str, float]:
        return {
            "srocc": self.srocc,
            "plcc": self.plcc,
            "krocc": self.krocc,
            "outlier_ratio": self.outlier_ratio,
            "pwrc_auc": self.pwrc_auc,
            "z_rmse": self.z_rmse,
            "n": float(self.n),
        }

    def __repr__(self) -> str:
        return (
            f"PanelStats(srocc={self.srocc:.4f}, plcc={self.plcc:.4f}, "
            f"krocc={self.krocc:.4f}, outlier_ratio={self.outlier_ratio:.4f}, "
            f"pwrc={self.pwrc_auc:.4f}, z_rmse={self.z_rmse:.4f}, n={self.n})"
        )

    def _repr_html_(self) -> str:
        return render_panel_stats_html(self)


def spearman(a: NumberSeq, b: NumberSeq) -> float:
    """Compute Spearman Rank Order Correlation Coefficient (SROCC)."""
    return _zenmetrics.stats_spearman(_to_float_list(a), _to_float_list(b))


def pearson(a: NumberSeq, b: NumberSeq) -> float:
    """Compute Pearson Linear Correlation Coefficient (PLCC)."""
    return _zenmetrics.stats_pearson(_to_float_list(a), _to_float_list(b))


def kendall_tau(a: NumberSeq, b: NumberSeq) -> float:
    """Compute Kendall Rank Correlation Coefficient (KROCC)."""
    return _zenmetrics.stats_kendall_tau(_to_float_list(a), _to_float_list(b))


def outlier_ratio(predictions: NumberSeq, targets: NumberSeq) -> float:
    """Compute Outlier Ratio (fraction of predictions beyond 2 standard deviations)."""
    return _zenmetrics.stats_outlier_ratio(_to_float_list(predictions), _to_float_list(targets))


def pwrc_auc(predictions: NumberSeq, targets: NumberSeq) -> float:
    """Compute Pairwise Ranking Consistency Area Under Curve (PWRC-AUC)."""
    return _zenmetrics.stats_pwrc_auc(_to_float_list(predictions), _to_float_list(targets))


def z_rmse(predictions: NumberSeq, targets: NumberSeq) -> float:
    """Compute Root Mean Square Error on Z-scored values."""
    return _zenmetrics.stats_z_rmse(_to_float_list(predictions), _to_float_list(targets))


def rescale_logistic(predictions: NumberSeq, targets: NumberSeq) -> list[float]:
    """Fit a 4-parameter monotonic logistic function mapping predictions to targets."""
    return _zenmetrics.stats_rescale_logistic(_to_float_list(predictions), _to_float_list(targets))


def compute_panel(predictions: NumberSeq, targets: NumberSeq) -> PanelStatsResult:
    """Compute full standard IQA evaluation panel (SROCC, PLCC, KROCC, PWRC, Outlier Ratio, Z-RMSE)."""
    raw = _zenmetrics.stats_compute_panel(_to_float_list(predictions), _to_float_list(targets))
    return PanelStatsResult(raw)
