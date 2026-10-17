"""zenmetrics - High-performance Image Quality Assessment in Rust with Pythonic ergonomics.

Provides zero-overhead file scoring, seamless in-memory NumPy/PIL/PyTorch integration,
multi-variant rate-distortion grids, directory batch evaluation, and standard ITU-T
subjective IQA statistical panel evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from . import _zenmetrics
from . import stats
from .display import (
    is_higher_better,
    render_compare_html,
    render_folder_html,
    render_score_html,
)
from .image import is_batch_image, is_path_like, normalize_image, unbatch_images

__version__ = "0.1.0"
__all__ = [
    "score",
    "compare",
    "score_folder",
    "list_metrics",
    "list_models",
    "Metric",
    "create_metric",
    "ScoreResult",
    "BatchScoreResult",
    "CompareResult",
    "FolderScoreResult",
    "stats",
]


class ScoreResult:
    """Quality assessment result wrapping primary score and sub-metric channels."""

    def __init__(self, metric: str, value: float, scores: Dict[str, float]):
        self.metric: str = metric
        self.value: float = float(value)
        self.scores: Dict[str, float] = scores

    @classmethod
    def _from_raw(cls, raw: _zenmetrics.ScoreResult) -> ScoreResult:
        return cls(metric=raw.metric, value=raw.value, scores=dict(raw.scores))

    def __float__(self) -> float:
        return self.value

    def __int__(self) -> int:
        return int(self.value)

    def item(self) -> float:
        """Return score value as a standard Python float (PyTorch/PyIQA convention)."""
        return self.value

    def numpy(self) -> Any:
        """Return score value as a numpy float."""
        import numpy as np

        return np.float64(self.value)

    def __repr__(self) -> str:
        return f"ScoreResult(metric='{self.metric}', value={self.value:.6f}, scores={self.scores})"

    def __getitem__(self, key: str) -> float:
        return self.scores[key]

    def __contains__(self, key: str) -> bool:
        return key in self.scores

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.scores.get(key, default)

    def keys(self):
        return self.scores.keys()

    def values(self):
        return self.scores.values()

    def items(self):
        return self.scores.items()

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "value": self.value, "scores": self.scores}

    @property
    def alpha_max_delta(self) -> Optional[float]:
        return self.scores.get("alpha_max_delta")

    @property
    def alpha_psnr(self) -> Optional[float]:
        return self.scores.get("alpha_psnr")

    @property
    def alpha_mae(self) -> Optional[float]:
        return self.scores.get("alpha_mae")

    @property
    def alpha_pixels_differing(self) -> Optional[float]:
        return self.scores.get("alpha_pixels_differing")

    @property
    def alpha_stats(self) -> Optional[Dict[str, float]]:
        if "alpha_max_delta" in self.scores:
            return {
                "max_delta": self.scores["alpha_max_delta"],
                "pixels_differing": self.scores.get("alpha_pixels_differing", 0.0),
                "mae": self.scores.get("alpha_mae", 0.0),
                "psnr": self.scores.get("alpha_psnr", 0.0),
            }
        return None

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)

    def __add__(self, other: Any) -> float:
        return self.value + (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __radd__(self, other: Any) -> float:
        return (float(other) if isinstance(other, (int, float, ScoreResult)) else other) + self.value

    def __sub__(self, other: Any) -> float:
        return self.value - (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __rsub__(self, other: Any) -> float:
        return (float(other) if isinstance(other, (int, float, ScoreResult)) else other) - self.value

    def __mul__(self, other: Any) -> float:
        return self.value * (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __rmul__(self, other: Any) -> float:
        return (float(other) if isinstance(other, (int, float, ScoreResult)) else other) * self.value

    def __truediv__(self, other: Any) -> float:
        return self.value / (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __rtruediv__(self, other: Any) -> float:
        return (float(other) if isinstance(other, (int, float, ScoreResult)) else other) / self.value

    # Rich comparisons allow `if score > 80:` or sorting directly
    def __lt__(self, other: Any) -> bool:
        return self.value < (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __le__(self, other: Any) -> bool:
        return self.value <= (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __gt__(self, other: Any) -> bool:
        return self.value > (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __ge__(self, other: Any) -> bool:
        return self.value >= (float(other) if isinstance(other, (int, float, ScoreResult)) else other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float, ScoreResult)):
            return abs(self.value - float(other)) < 1e-9
        return False

    def _repr_html_(self) -> str:
        return render_score_html(self)


class BatchScoreResult:
    """Result of batched image quality assessment over multiple image pairs."""

    def __init__(self, metric: str, scores: List[ScoreResult]):
        self.metric: str = metric
        self.scores: List[ScoreResult] = scores

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> ScoreResult:
        return self.scores[idx]

    def __iter__(self):
        return iter(self.scores)

    @property
    def values(self) -> List[float]:
        return [s.value for s in self.scores]

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def item(self) -> float:
        """Return the batch mean score as a float (matching PyTorch/PyIQA convention)."""
        return self.mean()

    def __float__(self) -> float:
        return self.mean()

    def __format__(self, format_spec: str) -> str:
        return self.mean().__format__(format_spec)

    def to_tensor(self) -> Any:
        """Convert scores to a 1D PyTorch tensor."""
        import torch

        return torch.tensor(self.values, dtype=torch.float32)

    def to_numpy(self) -> Any:
        """Convert scores to a 1D NumPy array."""
        import numpy as np

        return np.array(self.values, dtype=np.float64)

    def __repr__(self) -> str:
        preview = [round(v, 4) for v in self.values[:5]]
        suffix = "..." if len(self.values) > 5 else ""
        return (
            f"BatchScoreResult(metric='{self.metric}', count={len(self.scores)}, "
            f"mean={self.mean():.4f}, values={preview}{suffix})"
        )

    def _repr_html_(self) -> str:
        # Show mean score card with preview of individual batch items
        mean_score = ScoreResult(self.metric, self.mean(), {self.metric: self.mean()})
        return render_score_html(mean_score)


class CompareResult:
    """Result of multi-variant rate-distortion comparison across models or codecs."""

    def __init__(self, rows: List[Dict[str, Any]], metrics: List[str]):
        self.rows: List[Dict[str, Any]] = rows
        self.metrics: List[str] = metrics

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]

    def __repr__(self) -> str:
        return f"CompareResult({len(self.rows)} variants, metrics={self.metrics})"

    def to_dict(self) -> List[Dict[str, Any]]:
        return list(self.rows)

    def to_dataframe(self) -> Any:
        """Convert results to a pandas DataFrame (or polars if pandas is missing)."""
        try:
            import pandas as pd

            return pd.DataFrame(self.rows)
        except ImportError:
            pass

        try:
            import polars as pl

            return pl.DataFrame(self.rows)
        except ImportError:
            raise ImportError(
                "Neither 'pandas' nor 'polars' is installed. "
                "Use .to_dict() or install pandas via `pip install pandas`."
            )

    to_pandas = to_dataframe

    def to_polars(self) -> Any:
        """Convert results directly to a polars DataFrame."""
        import polars as pl

        return pl.DataFrame(self.rows)

    def _repr_html_(self) -> str:
        return render_compare_html(self.rows, self.metrics)


class FolderScoreResult:
    """Result of batch directory evaluation with summary statistics."""

    def __init__(self, raw: Dict[str, Any]):
        self.metric: str = raw.get("metric", "")
        self.total_pairs: int = raw.get("total_pairs", 0)
        self.successful_pairs: int = raw.get("successful_pairs", 0)
        self.failed_pairs: int = raw.get("failed_pairs", 0)
        self.summary: Dict[str, Any] = raw.get("summary", {})
        self.results: List[Dict[str, Any]] = raw.get("results", [])

    def __repr__(self) -> str:
        return (
            f"FolderScoreResult(metric='{self.metric}', "
            f"pairs={self.successful_pairs}/{self.total_pairs})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "total_pairs": self.total_pairs,
            "successful_pairs": self.successful_pairs,
            "failed_pairs": self.failed_pairs,
            "summary": self.summary,
            "results": self.results,
        }

    def _flatten_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for r in self.results:
            row: Dict[str, Any] = {
                "reference": r.get("reference"),
                "distorted": r.get("distorted"),
                "value": r.get("value"),
            }
            if "scores" in r and isinstance(r["scores"], dict):
                row.update(r["scores"])
            if "error" in r:
                row["error"] = r["error"]
            rows.append(row)
        return rows

    def to_dataframe(self) -> Any:
        """Convert per-pair results to a pandas DataFrame (or polars if pandas is missing)."""
        rows = self._flatten_rows()
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            try:
                import polars as pl

                return pl.DataFrame(rows)
            except ImportError:
                raise ImportError(
                    "Neither 'pandas' nor 'polars' is installed. "
                    "Use .results or install pandas."
                )

    to_pandas = to_dataframe

    def to_polars(self) -> Any:
        """Convert per-pair results directly to a polars DataFrame."""
        import polars as pl

        return pl.DataFrame(self._flatten_rows())

    def _repr_html_(self) -> str:
        return render_folder_html(self.to_dict())


def list_metrics() -> List[str]:
    """Return a list of all available quality metric names."""
    return _zenmetrics.list_metrics()


def _load_image_buffer(
    img: Any,
) -> Tuple[str, bytes, int, int]:
    """Load image returning ('rgb'|'rgba', bytes, w, h)."""
    norm = normalize_image(img)
    if norm[0] in ("rgb", "rgba"):
        kind, bytes_data, w, h = norm
        return kind, bytes_data, w, h

    # If it was a file path, load using PIL
    file_path = norm[1]
    try:
        from PIL import Image

        with Image.open(file_path) as im:
            if im.mode in ("RGBA", "LA", "PA"):
                rgba = im.convert("RGBA")
                return "rgba", rgba.tobytes(), rgba.width, rgba.height
            rgb = im.convert("RGB")
            return "rgb", rgb.tobytes(), rgb.width, rgb.height
    except Exception as e:
        raise ValueError(f"Failed to load image from file '{file_path}': {e}") from e


def _rgb_to_rgba_bytes(rgb_bytes: bytes, w: int, h: int) -> bytes:
    import numpy as np

    arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h, w, 3))
    rgba = np.dstack([arr, np.full((h, w, 1), 255, dtype=np.uint8)])
    return np.ascontiguousarray(rgba).tobytes()


def score(
    reference: Any,
    distorted: Any,
    metric: str = "ssim2",
    display_model: Optional[str] = None,
    luma_ingress: Optional[str] = None,
    gpu_runtime: Optional[str] = None,
    alpha_policy: str = "opaque",
    alpha_bg: Tuple[int, int, int] = (255, 255, 255),
) -> Union[ScoreResult, BatchScoreResult]:
    """Compute perceptual quality metric between reference and distorted images.

    Supports single images or batches (4D tensors/arrays or lists of images).

    Parameters
    ----------
    reference : str, Path, PIL.Image, np.ndarray, or torch.Tensor
        Reference (ground truth) image or image batch.
    distorted : str, Path, PIL.Image, np.ndarray, or torch.Tensor
        Distorted (evaluated) image or image batch.
    metric : str, default "ssim2"
        Metric to evaluate. Run `zm.list_metrics()` for available metrics.
        Supported metrics include: 'ssim2', 'butteraugli', 'psnr', 'cvvdp',
        'dssim', 'iwssim', 'ms-ssim', 'gmsd', 'haarpsi', 'vif', 'vmaf', 'zensim', etc.
    display_model : str, optional
        Target display geometry/calibration (e.g. "standard_4k", "standard_fhd").
        Required for CVVDP.
    luma_ingress : str, optional
        Luma conversion ingress mode ("house", "rec601", "rec709", etc.).
    gpu_runtime : str, optional
        GPU backend acceleration ("auto", "cuda", "vulkan", "wgpu", "cpu").
    alpha_policy : str, default "opaque"
        Policy for alpha channels when present:
        - "opaque": Require 100% opacity, erroring if non-opaque pixels are present.
        - "noise": Deterministic linear-light noise background (matching zensim).
        - "white": Solid white background.
        - "black": Solid black background.
        - "min_wb": Worst score of white and black mattes.
        - "discard": Strip alpha unconditionally.
    alpha_bg : tuple of (int, int, int), default (255, 255, 255)
        Background color for "composite" alpha policy.

    Returns
    -------
    ScoreResult or BatchScoreResult
        Score object supporting float conversion, dictionary access, and rich Jupyter display.
    """
    # Batched inputs: if either input is a batch (e.g. 4D array/tensor with B > 1, or list of images)
    if is_batch_image(reference) or is_batch_image(distorted):
        ref_items = unbatch_images(reference)
        dist_items = unbatch_images(distorted)
        if len(ref_items) != len(dist_items):
            raise ValueError(
                f"Batch size mismatch: reference has {len(ref_items)} images, "
                f"distorted has {len(dist_items)} images"
            )
        batch_scores = [
            score(
                r,
                d,
                metric=metric,
                display_model=display_model,
                luma_ingress=luma_ingress,
                gpu_runtime=gpu_runtime,
                alpha_policy=alpha_policy,
                alpha_bg=alpha_bg,
            )
            for r, d in zip(ref_items, dist_items)
        ]
        return BatchScoreResult(metric=metric, scores=batch_scores)

    # Fast path: both inputs are filesystem paths -> decode directly in Rust
    if is_path_like(reference) and is_path_like(distorted):
        raw = _zenmetrics.score_files(
            str(reference),
            str(distorted),
            metric,
            alpha_policy,
            alpha_bg,
            display_model,
            luma_ingress,
            gpu_runtime,
        )
        return ScoreResult._from_raw(raw)

    # In-memory arrays / PIL images / mixed
    ref_kind, ref_bytes, rw, rh = _load_image_buffer(reference)
    dist_kind, dist_bytes, dw, dh = _load_image_buffer(distorted)

    if rw != dw or rh != dh:
        raise ValueError(
            f"Image dimension mismatch: reference is {rw}x{rh}, distorted is {dw}x{dh}"
        )

    if ref_kind == "rgba" or dist_kind == "rgba":
        if ref_kind == "rgb":
            ref_bytes = _rgb_to_rgba_bytes(ref_bytes, rw, rh)
        if dist_kind == "rgb":
            dist_bytes = _rgb_to_rgba_bytes(dist_bytes, dw, dh)
        raw = _zenmetrics.score_rgba(
            ref_bytes,
            dist_bytes,
            rw,
            rh,
            metric,
            alpha_policy,
            alpha_bg,
            display_model,
            luma_ingress,
            gpu_runtime,
        )
    else:
        raw = _zenmetrics.score_rgb8(
            ref_bytes,
            dist_bytes,
            rw,
            rh,
            metric,
            display_model,
            luma_ingress,
            gpu_runtime,
        )
    return ScoreResult._from_raw(raw)


def compare(
    reference: Any,
    variants: Union[Dict[str, Any], Sequence[Any]],
    metrics: Union[str, Sequence[str]] = ("ssim2", "psnr", "butteraugli"),
    display_model: Optional[str] = None,
) -> CompareResult:
    """Compare reference against multiple image variants across multiple metrics.

    Parameters
    ----------
    reference : str, Path, PIL.Image, np.ndarray, or torch.Tensor
        Reference (ground truth) image.
    variants : dict of {name: img} or sequence of images/paths
        Distorted variants to compare.
    metrics : sequence of str or str
        Metrics to evaluate across all variants.

    Returns
    -------
    CompareResult
        Tabular comparison container with DataFrame export and HTML table display.
    """
    if isinstance(metrics, str):
        metric_list = [metrics]
    else:
        metric_list = list(metrics)

    # Normalize variants to list of (label, image)
    if isinstance(variants, dict):
        variant_pairs = list(variants.items())
    elif isinstance(variants, (list, tuple)):
        variant_pairs = []
        for v in variants:
            label = Path(v).stem if is_path_like(v) else f"variant_{len(variant_pairs)}"
            variant_pairs.append((label, v))
    else:
        raise TypeError(f"Expected dict or list for variants, got {type(variants)!r}")

    # Fast path: reference and all variants are file paths -> call Rust compare_files directly
    all_paths = is_path_like(reference) and all(is_path_like(v) for _, v in variant_pairs)
    if all_paths:
        var_paths = [str(v) for _, v in variant_pairs]
        raw_rows = _zenmetrics.compare_files(
            str(reference),
            var_paths,
            metric_list,
            display_model,
        )
        # Add labels to rows
        for i, row in enumerate(raw_rows):
            row["variant"] = variant_pairs[i][0]
        return CompareResult(raw_rows, metric_list)

    # Mixed or in-memory arrays
    rows = []
    for label, var_img in variant_pairs:
        row: Dict[str, Any] = {"variant": label}
        for m in metric_list:
            try:
                s = score(reference, var_img, metric=m, display_model=display_model)
                row[m] = s.value
            except Exception as e:
                row[m] = float("nan")
                row[f"{m}_error"] = str(e)
        rows.append(row)

    return CompareResult(rows, metric_list)


def score_folder(
    reference_dir: Union[str, Path, os.PathLike],
    distorted_dir: Union[str, Path, os.PathLike],
    metric: str = "ssim2",
    recursive: bool = True,
    display_model: Optional[str] = None,
) -> FolderScoreResult:
    """Evaluate quality metric between two directories of corresponding images.

    Automatically matches images between directories by relative path, stem,
    or unique basename across file formats (.png, .jpg, .webp, .jxl, etc.).

    Parameters
    ----------
    reference_dir : str or Path
        Directory containing reference images.
    distorted_dir : str or Path
        Directory containing distorted images.
    metric : str, default "ssim2"
        Metric to evaluate.
    recursive : bool, default True
        Whether to scan subdirectories recursively.
    display_model : str, optional
        Target display model.

    Returns
    -------
    FolderScoreResult
        Summary statistics, list of per-image scores, DataFrame export, and Jupyter dashboard.
    """
    raw = _zenmetrics.score_folder(
        str(reference_dir),
        str(distorted_dir),
        metric,
        recursive,
        display_model,
    )
    return FolderScoreResult(raw)


class Metric:
    """Callable Metric object for PyIQA-style workflow."""

    def __init__(
        self,
        name: str = "ssim2",
        device: Optional[Any] = None,
        display_model: Optional[str] = None,
        luma_ingress: Optional[str] = None,
        gpu_runtime: Optional[str] = None,
        alpha_policy: str = "opaque",
        alpha_bg: Tuple[int, int, int] = (255, 255, 255),
        **kwargs,
    ):
        self.name: str = name
        self.display_model: Optional[str] = display_model
        self.luma_ingress: Optional[str] = luma_ingress
        self.alpha_policy: str = alpha_policy
        self.alpha_bg: Tuple[int, int, int] = alpha_bg

        # Resolve device to gpu_runtime
        if device is not None:
            dev_str = str(device).lower()
            if "cuda" in dev_str:
                self.gpu_runtime = "cuda"
            elif "cpu" in dev_str:
                self.gpu_runtime = "cpu"
            else:
                self.gpu_runtime = gpu_runtime or dev_str
            self._device = dev_str
        else:
            self.gpu_runtime = gpu_runtime
            self._device = gpu_runtime or "cpu"

    def __call__(self, reference: Any, distorted: Any) -> Union[ScoreResult, BatchScoreResult]:
        return score(
            reference,
            distorted,
            metric=self.name,
            display_model=self.display_model,
            luma_ingress=self.luma_ingress,
            gpu_runtime=self.gpu_runtime,
            alpha_policy=self.alpha_policy,
            alpha_bg=self.alpha_bg,
        )

    @property
    def higher_better(self) -> bool:
        return is_higher_better(self.name)

    @property
    def lower_better(self) -> bool:
        """PyIQA compatibility: True if lower metric values denote better visual quality."""
        return not self.higher_better

    @property
    def metric_name(self) -> str:
        return self.name

    @property
    def device(self) -> str:
        return self._device

    def to(self, device: Any) -> Metric:
        """Move metric evaluation target to the specified device (PyTorch/PyIQA style)."""
        dev_str = str(device).lower()
        if "cuda" in dev_str:
            self.gpu_runtime = "cuda"
        elif "cpu" in dev_str:
            self.gpu_runtime = "cpu"
        else:
            self.gpu_runtime = dev_str
        self._device = dev_str
        return self

    def cuda(self) -> Metric:
        """Switch metric to CUDA GPU runtime."""
        return self.to("cuda")

    def cpu(self) -> Metric:
        """Switch metric to CPU runtime."""
        return self.to("cpu")

    def __repr__(self) -> str:
        return f"Metric('{self.name}', device='{self.device}')"


def create_metric(name: str = "ssim2", **kwargs) -> Metric:
    """Create a callable Metric instance (PyIQA-compatible API)."""
    return Metric(name, **kwargs)


# PyIQA compatibility alias
list_models = list_metrics
