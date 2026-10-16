"""Explicit strict results destinations, shared by execution and harvesting."""
from pathlib import Path

ROOTED = {"steer_exact_smoke.py": "cells", "v2_lodo_mlp.py": "cells", "v2_confirm_fit.py": "confirm/cells", "subjective_cell.py": "cells"}


def explicit_root(argv, fit_root):
    if "--dest" not in argv:
        return None
    required = ("--root",) if argv[0] == "steer_exact_smoke.py" else ("--strict-admission", "--train-only", "--root")
    if argv[0] not in ROOTED or any(flag not in argv for flag in required):
        raise ValueError("explicit fit destination requires a strict training-only rooted cell or bounded exact probe")
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
