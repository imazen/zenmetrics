"""Explicit strict results destinations, shared by execution and harvesting."""
from pathlib import Path

ROOTED = {"v2_lodo_mlp.py": "cells", "v2_confirm_fit.py": "confirm/cells"}


def explicit_root(argv, fit_root):
    if "--dest" not in argv:
        return None
    if argv[0] not in ROOTED or any(flag not in argv for flag in ("--strict-admission", "--train-only", "--root")):
        raise ValueError("explicit fit destination requires a strict training-only rooted cell")
    dest = Path(argv[argv.index("--dest") + 1])
    data = Path(argv[argv.index("--root") + 1])
    relative = dest.relative_to(fit_root)
    suffix = tuple(ROOTED[argv[0]].split("/"))
    if (len(relative.parts) != len(suffix) + 3 or relative.parts[1:-2] != suffix
            or any(p in ("", ".", "..") or p.startswith(".") for p in relative.parts)
            or relative.parts[0] == data.name or data.parent != fit_root
            or not dest.resolve().is_relative_to(fit_root.resolve())
            or dest.resolve().is_relative_to(data.resolve())):
        raise ValueError("unsafe strict destination; results must be outside admitted data")
    return str(relative.parent.parent)
