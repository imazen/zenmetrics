#!/usr/bin/env python3
# HDR phase-2 corpus (zensim campaign appendix S): emit the encode-declare cells
# JSONL for the registered grid — 1,140 PQ-PNG sources (76 origins x 15 scales)
# x 3 codec arms x 30 q points, all cells hdr:true.
#
#   usage: hdrgrid_cells.py <source_dir> <out_cells.jsonl> [--sha-cache <tsv>]
#                           [--arms <codec>:<knobs-json>[,...]] [--q-grid <csv>]
#
# --arms / --q-grid override the appendix-S defaults below WITHOUT changing them.
# They exist so a later registered grid can DRIVE this emitter instead of forking
# it (the AVIF high-bit-depth arm's Track T2 is the first caller:
# benchmarks/avif_hdr_arm_plan_2026-09-02.md section 4.3). Omit both and the
# output is byte-identical to before they existed -- gated by
# its `--self-test` mode below.
#
#   T2-a: --arms 'zenav1-svt:{"preset":N}' per preset, --q-grid <the 29-point ladder>
#   T2-b: --arms 'zenavif:{"speed":N}'     per speed,  --q-grid <the 9-point ladder>
#
# Grid, verbatim from appendix S (S.4):
#   - q: step 5 across 0..70 (0,5,...,70) + step 2 across 70..100 (72,...,100)
#     = 30 points, dense at BOTH ends per the sweep discipline.
#   - sizes: ALL 15 ladder steps (every file in the source dir).
#   - preset: the QUALITY tier — svt preset 6 / jxl effort 7 / uhdr crate
#     defaults (gainmap q85, scale 4; recorded as defaults, not knobs).
#   - arms: zenjxl / zenav1-svt / jpeg-gainmap. The AVIF arm is ABSENT per the
#     standing B5 record (additive-later; rows are independent so adding it
#     later touches nothing).
#
# `image_path` is the source BASENAME (workers resolve it against the run's
# corpus prefix); `source_sha` is the sha256 of the source bytes (the encode
# job's content-addressed input). The sha pass reads ~7.8 GB once; --sha-cache
# persists it so re-declares are instant.
import hashlib
import json
import os
import sys

def q_grid():
    """The appendix-S default: step 5 across 0..70 + step 2 across 70..100."""
    qs = list(range(0, 71, 5)) + list(range(72, 101, 2))
    assert len(qs) == 30, len(qs)
    return qs

def parse_q_grid(spec):
    """`--q-grid 5,15,25` -> [5, 15, 25]. Rejects an empty or non-integer list
    loudly rather than emitting a silently-shorter grid."""
    qs = [int(x) for x in spec.split(",") if x.strip() != ""]
    if not qs:
        raise SystemExit("--q-grid parsed to an empty list: %r" % spec)
    return qs

def parse_arms(spec):
    """`--arms 'zenav1-svt:{"preset":6},zenavif:{"speed":4}'` -> the ARMS shape.

    Split on the FIRST colon only, so the knob JSON keeps its own colons. The
    knobs are re-serialised through json.dumps with the same separators the
    defaults use, so an arm spelled here and an arm spelled in ARMS produce
    byte-identical `knob_tuple_json` -- which matters because that string is
    part of the content-addressed cell identity."""
    arms = []
    for chunk in spec.split("},"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.endswith("}"):
            chunk += "}"
        codec, sep, knobs = chunk.partition(":")
        codec = codec.strip()
        if not codec or not sep:
            raise SystemExit("--arms entry is not <codec>:<knobs-json>: %r" % chunk)
        try:
            parsed = json.loads(knobs)
        except ValueError as e:
            # A malformed arm must exit CLEANLY, naming the arm. Letting the
            # JSONDecodeError escape gives a traceback that does not say which
            # arm was wrong -- and the empty-knobs case reaches here looking
            # like `"}"` after the brace repair above, so this is the only
            # place that can catch it. (Found by --self-test.)
            raise SystemExit("--arms %r has unparseable knobs %r: %s" % (codec, knobs, e))
        arms.append((codec, json.dumps(parsed, separators=(",", ":"))))
    if not arms:
        raise SystemExit("--arms parsed to an empty list: %r" % spec)
    return arms

ARMS = [
    ("zenjxl", json.dumps({"effort": 7}, separators=(",", ":"))),
    ("zenav1-svt", json.dumps({"preset": 6}, separators=(",", ":"))),
    ("jpeg-gainmap", "{}"),
]

def self_test():
    """`--self-test`: the overrides must not have moved the defaults.

    The claim in the header is that omitting --arms/--q-grid reproduces the
    pre-override output byte-for-byte. What can actually drift is the two
    default constants and the knob SPELLING, since `knob_tuple_json` is part of
    the content-addressed cell identity -- an arm spelled through --arms that
    serialised differently from the same arm in ARMS would silently mint a
    second identity for one encode."""
    qs = q_grid()
    assert qs == list(range(0, 71, 5)) + list(range(72, 101, 2)), qs
    assert len(qs) == 30 and qs[0] == 0 and qs[-1] == 100, qs
    assert [c for c, _ in ARMS] == ["zenjxl", "zenav1-svt", "jpeg-gainmap"], ARMS

    # --arms must reproduce each default arm's knob string EXACTLY.
    spec = ",".join("%s:%s" % (c, k) for c, k in ARMS)
    assert parse_arms(spec) == ARMS, (parse_arms(spec), ARMS)

    assert parse_q_grid("5,15,25") == [5, 15, 25]
    assert parse_q_grid(" 5 , 15 ") == [5, 15]
    for bad in ("", "   ", ",,"):
        try:
            parse_q_grid(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError("--q-grid %r should have exited" % bad)
    for bad in ("", "zenavif", "zenavif:"):
        try:
            parse_arms(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError("--arms %r should have exited" % bad)
    print("hdrgrid_cells self-test OK: defaults unchanged, spellings identical")

def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    src_dir, out_path = sys.argv[1], sys.argv[2]
    sha_cache = None
    if "--sha-cache" in sys.argv:
        sha_cache = sys.argv[sys.argv.index("--sha-cache") + 1]
    arms = ARMS
    if "--arms" in sys.argv:
        arms = parse_arms(sys.argv[sys.argv.index("--arms") + 1])
    qs_override = None
    if "--q-grid" in sys.argv:
        qs_override = parse_q_grid(sys.argv[sys.argv.index("--q-grid") + 1])
    names = sorted(n for n in os.listdir(src_dir) if n.endswith(".png"))
    if not names:
        sys.exit("no .png sources in %s" % src_dir)
    cache = {}
    if sha_cache and os.path.exists(sha_cache):
        for line in open(sha_cache):
            n, s = line.rstrip("\n").split("\t")
            cache[n] = s
    shas = {}
    for i, n in enumerate(names):
        if n in cache:
            shas[n] = cache[n]
            continue
        h = hashlib.sha256()
        with open(os.path.join(src_dir, n), "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        shas[n] = h.hexdigest()
        if (i + 1) % 100 == 0:
            print("sha %d/%d" % (i + 1, len(names)), file=sys.stderr)
    if sha_cache:
        with open(sha_cache, "w") as f:
            for n in names:
                f.write("%s\t%s\n" % (n, shas[n]))
    qs = qs_override if qs_override is not None else q_grid()
    cells = 0
    with open(out_path, "w") as out:
        for n in names:
            for codec, knobs in arms:
                for q in qs:
                    out.write(json.dumps({
                        "image_path": n,
                        "codec": codec,
                        "q": q,
                        "knob_tuple_json": knobs,
                        "source_sha": shas[n],
                        "hdr": True,
                    }, separators=(",", ":")) + "\n")
                    cells += 1
    expect = len(names) * len(arms) * len(qs)
    assert cells == expect, (cells, expect)
    print("wrote %d cells (%d sources x %d arms x %d q) -> %s"
          % (cells, len(names), len(arms), len(qs), out_path))

if __name__ == "__main__":
    main()
