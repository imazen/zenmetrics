#!/usr/bin/env python3
"""zen-metrics external metric server.

Reads (reference, distorted) RGB8 frame pairs from stdin, runs a
PyTorch reference-implementation metric, writes JSON score lines to
stdout. Designed to be spawned exactly once per zen-metrics-cli
process by the `external-metric` plugin (see
`crates/zen-metrics-cli/src/metrics/external.rs` for the wire
protocol).

Wire protocol per scoring request:
    request:
        line:  {"w": <int>, "h": <int>}\\n
        bytes: w*h*3 reference RGB8 (row-major, tightly packed)
        bytes: w*h*3 distorted RGB8 (same layout)
    response:
        line:  {"score": <float>}\\n   (success)
        or     {"error": "<message>"}\\n  (per-request failure;
               server continues)

The server emits a single `{"ready":true,"name":"<metric>"}` line on
stdout before accepting requests. stderr is inherited from the parent
so traceback / debug output appears in worker logs untouched.

Usage:
    metric_server.py <metric-name>

The metric implementation is loaded from a sibling module named
`<metric-name>_impl.py`, which must expose:
    class Scorer:
        name: str                                  # for the ready banner
        def __init__(self) -> None: ...            # one-time setup
        def score(self, ref: bytes, dist: bytes,
                  w: int, h: int) -> float: ...    # per-pair score
"""
import importlib
import json
import os
import sys
import traceback


def _read_exact(buf, n):
    """Read exactly n bytes from a binary file-like, or raise EOFError."""
    out = bytearray()
    while len(out) < n:
        chunk = buf.read(n - len(out))
        if not chunk:
            raise EOFError(
                f"upstream closed stdin after {len(out)}/{n} bytes"
            )
        out.extend(chunk)
    return bytes(out)


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: metric_server.py <metric-name>\n")
        return 2
    name = sys.argv[1]

    # Allow caller to add the impl module's directory to sys.path via
    # ZEN_METRICS_PYTHON_DIR. The Docker images set this to /opt/python
    # so the impl modules can live alongside this server file.
    impl_dir = os.environ.get("ZEN_METRICS_PYTHON_DIR")
    if impl_dir and impl_dir not in sys.path:
        sys.path.insert(0, impl_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    module_name = f"{name}_impl"
    try:
        impl_module = importlib.import_module(module_name)
        scorer = impl_module.Scorer()
    except Exception as e:  # noqa: BLE001 — surface anything during init
        sys.stderr.write(
            f"failed to initialise metric '{name}': {type(e).__name__}: {e}\n"
        )
        sys.stderr.write(traceback.format_exc())
        return 1

    sys.stdout.write(
        json.dumps({"ready": True, "name": getattr(scorer, "name", name)}) + "\n"
    )
    sys.stdout.flush()

    stdin_bin = sys.stdin.buffer
    while True:
        header_line = stdin_bin.readline()
        if not header_line:
            # Upstream closed; clean exit.
            return 0
        try:
            header = json.loads(header_line.decode("ascii").strip())
            w = int(header["w"])
            h = int(header["h"])
            if w <= 0 or h <= 0 or w * h > 256 * 1024 * 1024:
                raise ValueError(f"invalid dims w={w} h={h}")
            nbytes = w * h * 3
            ref = _read_exact(stdin_bin, nbytes)
            dist = _read_exact(stdin_bin, nbytes)
            value = float(scorer.score(ref, dist, w, h))
            sys.stdout.write(json.dumps({"score": value}) + "\n")
        except EOFError as e:
            # Upstream closed mid-request — propagate as error and exit.
            sys.stdout.write(
                json.dumps({"error": f"EOF: {e}"}) + "\n"
            )
            sys.stdout.flush()
            return 0
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(
                json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n"
            )
            sys.stderr.write(traceback.format_exc())
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
