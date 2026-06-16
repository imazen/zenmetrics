#!/usr/bin/env python3
"""Reference job executor for the zen job system (the ZEN_EXEC contract).

The worker (`zenfleet-worker`) runs ONE executor process per claimed job and speaks a dead-simple
contract (see crates/zenfleet-worker/src/lib.rs::exec_command):

    stdin   <- the DesiredJob as JSON (one object, then EOF)
    stdout  -> the job's OUTPUT BYTES (these get content-addressed to blobs/<sha256> in R2)
    exit 0  =  success;  non-zero exit = a deterministic failure (FAILED ledger row, error_class)

That's the whole interface. ANY program honoring it plugs in as ZEN_EXEC — bash, python, a Rust
binary, whatever. The synthetic demo uses `/bin/cat` (echoes the job JSON straight back, which proves
the pipeline without doing real work). THIS file shows the shape of a *real* executor: branch on the
job kind, do the work, write the result bytes.

The DesiredJob JSON looks like:
    {"kind": {"kind": "metric", "metric": "cvvdp"},
     "inputs": ["<sha256-of-an-input-blob>", ...],
     "cell": {"image_path": "corpus/img-00001.png", "codec": "zenjpeg", "q": 80, "knob_tuple_json": "{}"}}
or for an encode:
    {"kind": {"kind": "encode", "codec": "zenjpeg", "q": 80, "knobs": "{}"},
     "inputs": ["<sha256-of-the-source-image>"],
     "cell": {...}}

To run REAL work you fill in the TODO branches below (fetch inputs from R2 by sha, shell out to the
codec/scorer, emit the bytes) and BAKE this — plus whatever it shells out to — into a worker image,
then point the worker at it with ZEN_EXEC=/usr/local/bin/example_executor.py.
"""
import json
import sys


def main() -> int:
    job = json.load(sys.stdin)             # the DesiredJob
    kind = job["kind"]["kind"]
    out = sys.stdout.buffer                 # write RAW BYTES here (this becomes the blob)

    if kind == "metric":
        # REAL metric job: you have (reference, distorted). Resolve them, score, emit the score bytes.
        #   ref_sha, dist_sha = job["inputs"][0], job["inputs"][1]   # or your own input convention
        #   fetch each blob from R2 (s5cmd cp s3://$ZEN_BUCKET/$ZEN_RUN/blobs/<sha> ...), then e.g.:
        #   score = subprocess.run(["zenmetrics","score","--metric",job["kind"]["metric"],ref,dist],
        #                          capture_output=True, check=True).stdout
        #   out.write(score); return 0
        # PLACEHOLDER (proves the contract end-to-end without a real metric):
        out.write(json.dumps({"metric": job["kind"]["metric"], "cell": job["cell"],
                              "score": None, "note": "placeholder — wire a real scorer here"}).encode())
        return 0

    if kind == "encode":
        # REAL encode job: fetch the source (inputs[0]), run the codec at cell.q / knobs, emit the
        # encoded bytes on stdout (they get content-addressed — re-scoring never re-encodes, goal G).
        #   src = fetch_blob(job["inputs"][0])
        #   encoded = subprocess.run(["<your-encoder>","--q",str(job["cell"]["q"]),src,"-"],
        #                            capture_output=True, check=True).stdout
        #   out.write(encoded); return 0
        out.write(b"")  # PLACEHOLDER
        return 0

    # Unknown kind → non-zero so it lands as a FAILED row (not silently "done").
    sys.stderr.write(f"executor: unhandled job kind {kind!r}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
