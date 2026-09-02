#!/usr/bin/env python3
"""G-CROP gate regression test: a bogus extractor MUST NOT clear the gate.

Why this file exists. `avifdoe_build_budget_corpus.py` used to document
`--features-cmd` as the way to satisfy G-CROP and flipped the manifest off
`PENDING` when it was passed -- while never executing it (the script imported no
`subprocess`). So the one gate the AVIF-DOE plan calls "the one place the wave
can silently change what an image is" was satisfiable by a no-op. These cases
pin the replacement shut, in both directions: the extractor is really RUN and
its failures are fatal (cases 1-4), and features that are not what they claim
are caught by the native control (cases 5-7).

Run standalone (`python3 scripts/jobsys/test_avifdoe_budget_gate.py`) or under
pytest. Hermetic: no corpus, no parquet, no network, ~instant.
"""
import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avifdoe_build_budget_corpus as B  # noqa: E402

FAILS, RAN = [], []


def case(name):
    def deco(fn):
        RAN.append(name)
        try:
            fn()
        except AssertionError as e:
            FAILS.append(f"{name}: {e}")
        except Exception as e:  # a crash is a failure too
            FAILS.append(f"{name}: unexpected {type(e).__name__}: {e}")
        return fn
    return deco


def rejects(fn, why):
    """The gate must refuse by EXITING, not by returning a passing value."""
    try:
        fn()
    except SystemExit as e:
        assert e.code not in (0, None), f"{why}: exited {e.code!r}, want non-zero"
        return
    raise AssertionError(f"{why}: the gate accepted it")


def _rows(n=2):
    return [dict(corpus_key=f"k{i}.png", crop_sha256=f"sha{i}",
                 content_class="cc", source_path=f"P{i}",
                 transform="native" if i == 0 else "crop-native",
                 parent_cluster=i) for i in range(n)]


def _script(tmp, body):
    p = os.path.join(tmp, "fake_extractor.sh")
    with open(p, "w") as f:
        f.write("#!/bin/sh\n" + body)
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
    return p


# --------------------------------------------------------------- run_extractor
@case("1. a missing extractor binary is fatal")
def _():
    with tempfile.TemporaryDirectory() as t:
        rejects(lambda: B.run_extractor(os.path.join(t, "nope"), t, _rows(),
                                        os.path.join(t, "w")),
                "nonexistent --extractor")


@case("2. a non-zero exit is fatal")
def _():
    with tempfile.TemporaryDirectory() as t:
        rejects(lambda: B.run_extractor(_script(t, "exit 3\n"), t, _rows(),
                                        os.path.join(t, "w")),
                "extractor rc=3")


@case("3. THE ORIGINAL HOLE — rc=0 that extracts nothing is fatal")
def _():
    with tempfile.TemporaryDirectory() as t:
        # Exactly the old `--features-cmd` behaviour: accepted, did nothing.
        rejects(lambda: B.run_extractor(_script(t, "exit 0\n"), t, _rows(),
                                        os.path.join(t, "w")),
                "no-op extractor")


@case("4. a short row count is fatal (a silently dropped decode is not a pass)")
def _():
    with tempfile.TemporaryDirectory() as t:
        # rc=0, writes a well-formed TSV -- but only one row for two references.
        sc = _script(t, 'out=""\n'
                        'while [ $# -gt 0 ]; do\n'
                        '  [ "$1" = "--output" ] && out="$2"\n'
                        '  shift\n'
                        'done\n'
                        'printf "image_path\\tsource\\tfeat_a\\n" > "$out"\n'
                        'printf "x\\tk0.png\\t0\\n" >> "$out"\n')
        rejects(lambda: B.run_extractor(sc, t, _rows(), os.path.join(t, "w")),
                "1 row for 2 references")


# -------------------------------------------------------------- check_clusters
def _model(tmp):
    import numpy as np
    p = os.path.join(tmp, "m.npz")
    np.savez(p, content_names=np.array(["a", "b", "c"], dtype=object),
             fill=np.zeros(3), kept_cols=np.array([True, True, True]),
             mu=np.zeros(3), sd=np.ones(3),
             centroids=np.array([[0.0, 0, 0], [10.0, 10, 10]]))
    return p


def _with_stub_parents(fn):
    import numpy as np
    real = B.stored_parent_vectors
    B.stored_parent_vectors = lambda *_a, **_k: {
        "P0": np.array([0.0, 0, 0]), "P1": np.array([10.0, 10, 10])}
    try:
        return fn()
    finally:
        B.stored_parent_vectors = real


def _ext(a0, a1):
    f = lambda k, v: dict(source=k, feat_a=str(v[0]), feat_b=str(v[1]),
                          feat_c=str(v[2]))
    return [f("k0.png", a0), f("k1.png", a1)]


@case("5. honest features pass, and the crop's preservation is reported")
def _():
    with tempfile.TemporaryDirectory() as t:
        v = _with_stub_parents(lambda: B.check_clusters(
            _model(t), _ext((0, 0, 0), (9, 9, 9)), _rows(), "unused.parquet", 0.5))
        assert v["k0.png"]["preserved"] is True, "native must preserve"
        assert v["k1.png"]["assigned_cluster"] == 1, v["k1.png"]
        assert v["k1.png"]["preserved"] is True, v["k1.png"]


@case("6. features that land the NATIVE control in the wrong cluster are fatal")
def _():
    with tempfile.TemporaryDirectory() as t:
        rejects(lambda: _with_stub_parents(lambda: B.check_clusters(
            _model(t), _ext((10, 10, 10), (9, 9, 9)), _rows(),
            "unused.parquet", 0.5)),
            "native assigned to the wrong cluster")


@case("7. the drift tolerance bites even when the cluster still matches")
def _():
    with tempfile.TemporaryDirectory() as t:
        # (0.6,0,0) is still nearest centroid 0 -- so only the z-displacement
        # arm can catch it. Both arms are load-bearing.
        rejects(lambda: _with_stub_parents(lambda: B.check_clusters(
            _model(t), _ext((0.6, 0, 0), (9, 9, 9)), _rows(),
            "unused.parquet", 0.5)),
            "native z-displacement over tolerance")


for n in RAN:
    print(("FAIL " if any(f.startswith(n) for f in FAILS) else "ok   ") + n)
for f in FAILS:
    print("  " + f, file=sys.stderr)
print(f"\nRESULT: {'PASS' if not FAILS else 'FAIL'} ({len(RAN) - len(FAILS)}/{len(RAN)})")
sys.exit(1 if FAILS else 0)
