"""Image normalization utilities for zenmetrics.

Handles seamless conversion from file paths, PIL Images, NumPy arrays,
and PyTorch tensors into raw RGB8 buffers for zero-copy/minimal-copy Rust scoring.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple, Union

PathLike = Union[str, os.PathLike, Path]


def is_path_like(obj: Any) -> bool:
    """Return True if obj is a filesystem path string or PathLike."""
    return isinstance(obj, (str, Path, os.PathLike))


def is_batch_image(obj: Any) -> bool:
    """Return True if obj represents a batch of images (e.g. 4D tensor/array or list/tuple)."""
    if isinstance(obj, (list, tuple)):
        return True
    if hasattr(obj, "ndim") and hasattr(obj, "shape"):
        return obj.ndim == 4 and obj.shape[0] > 1
    return False


def unbatch_images(obj: Any) -> list[Any]:
    """Split a batched image object into a list of individual image inputs."""
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if hasattr(obj, "ndim") and hasattr(obj, "shape") and obj.ndim == 4:
        return [obj[i] for i in range(obj.shape[0])]
    return [obj]


def process_alpha_channel(
    rgb: Any,
    alpha: Any,
    policy: str = "opaque",
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Any:
    """Process RGBA array according to the specified alpha policy.

    Policies:
      - 'opaque' / 'discard_if_opaque': only strip alpha if all pixels are fully opaque.
        Raises ValueError if any transparent/translucent pixels exist.
      - 'matte_white' / 'white': composite over solid white background.
      - 'matte_black' / 'black': composite over solid black background.
      - 'composite': composite over custom bg_color (r, g, b).
      - 'discard_unchecked' / 'ignore': unconditionally discard alpha.
    """
    import numpy as np

    is_uint8 = np.issubdtype(alpha.dtype, np.integer)
    if is_uint8:
        is_opaque = bool(np.all(alpha == 255))
        max_alpha = 255.0
    else:
        is_opaque = bool(np.all(alpha >= 0.999))
        max_alpha = 1.0

    if is_opaque:
        # Fully opaque: safely strip alpha without data loss
        return rgb

    p = policy.lower().replace("-", "_")
    if p in ("opaque", "discard_if_opaque"):
        raise ValueError(
            "Image contains non-opaque alpha (transparency). Image quality metrics operate on "
            "opaque RGB. To evaluate transparent images, specify an explicit alpha_policy: "
            "'matte_white' (composite over white), 'matte_black' (composite over black), "
            "or 'discard_unchecked' (ignore alpha channel)."
        )
    elif p in ("matte_white", "white"):
        bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    elif p in ("matte_black", "black"):
        bg = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elif p in ("composite", "bg"):
        bg = np.array(bg_color, dtype=np.float32)
    elif p in ("discard_unchecked", "ignore"):
        return rgb
    else:
        raise ValueError(
            f"Unknown alpha_policy '{policy}'. Supported policies: "
            "'opaque', 'matte_white', 'matte_black', 'composite', 'discard_unchecked'."
        )

    # Perform alpha compositing: out = rgb * a + bg * (1 - a)
    a = (alpha.astype(np.float32) / max_alpha)[..., None]
    rgb_f = rgb.astype(np.float32)
    out = rgb_f * a + bg * (1.0 - a)
    if is_uint8:
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out


def normalize_image(
    img: Any,
    alpha_policy: str = "noise",
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[str, Any]:
    """Normalize input image to ('file', path_str), ('rgba', bytes, w, h), or ('rgb', bytes, w, h).

    Supports:
      - str / os.PathLike / Path -> file path
      - PIL.Image.Image -> RGB or RGBA bytes
      - numpy.ndarray (H, W, 3), (3, H, W), (H, W), or RGBA (H, W, 4) -> RGB or RGBA bytes
      - torch.Tensor -> detached, moved to CPU, converted to RGB/RGBA bytes
    """
    if is_path_like(img):
        return ("file", str(img))

    # Check for PyTorch tensor without requiring hard torch dependency
    if hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):
        img = img.detach().cpu().numpy()

    # Check for PIL Image without requiring hard PIL import at top level
    if hasattr(img, "mode") and hasattr(img, "size") and hasattr(img, "tobytes"):
        import numpy as np

        w, h = img.size
        if img.mode in ("RGBA", "LA", "PA"):
            rgba = img.convert("RGBA")
            return ("rgba", rgba.tobytes(), w, h)
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return ("rgb", img.tobytes(), w, h)

    # Check for NumPy array or array-like
    if hasattr(img, "shape") and hasattr(img, "dtype"):
        import numpy as np

        arr = np.asarray(img)

        # Handle 4D batches with batch size 1 e.g. (1, 3, H, W) or (1, H, W, 3)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)

        # Handle RGBA in CHW format (4, H, W) -> transpose to (H, W, 4)
        if arr.ndim == 3 and arr.shape[0] == 4 and arr.shape[2] != 4:
            arr = np.transpose(arr, (1, 2, 0))

        # Handle RGB in CHW format (3, H, W) -> transpose to (H, W, 3)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[0] == 1:
            # (1, H, W) grayscale
            arr = np.squeeze(arr, axis=0)

        # Handle Grayscale (H, W) -> duplicate to RGB (H, W, 3)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(
                f"Unsupported array shape: {arr.shape}. Expected (H, W, 3), (3, H, W), (H, W), or (H, W, 4)."
            )

        # Handle floating point [0.0, 1.0] or [0.0, 255.0]
        if np.issubdtype(arr.dtype, np.floating):
            max_val = np.nanmax(arr) if arr.size > 0 else 1.0
            if max_val <= 1.05:
                arr = np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        h, w, c = arr.shape
        arr = np.ascontiguousarray(arr)
        if c == 4:
            return ("rgba", arr.tobytes(), w, h)
        return ("rgb", arr.tobytes(), w, h)

    raise TypeError(
        f"Unsupported image type: {type(img)!r}. "
        "Expected file path (str/Path), PIL Image, NumPy array, or PyTorch tensor."
    )
