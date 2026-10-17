# ort-webgpu parity receipts — external file pointer

The full per-pair parity receipt (568 rows: commit, engine, model, stem,
size, got, want, abs_err, jnd_err, PASS/OOM/LOADFAIL) is generated data
over the 30 KB commit cap and lives on local disk:

    /mnt/v/output/zenmetrics/ort_webgpu_2026-10-17/parity.tsv

- **Host:** `i265` (the path is on i265's local disk, not a network mount)
- **sha256:** `209d7a163109b5a7821cb5a62bc4e76c0a03e2b2eaa71f5eb04eb0783444984a`
- **Data commit:** `107edec1006e827ff630108617a5abe6f540c173`
  (run-time snap export `1aa75d0e661c`, identical tree)
- **Command:** `ZENMETRICS_NN_ENGINE=ort-webgpu ZENMETRICS_NN_MODELS=<model>
  ZENMETRICS_NN_MODEL_DIR=<dir> ZENMETRICS_NN_LADDER=~/tmp/nn-phase0/ladder
  just nn-parity <dir> <ladder> ort-webgpu model-tests,ort-webgpu <model>` —
  see `parity.tsv`'s own header inside the file for the per-model dirs and
  the gate definition.
- **Regenerate:** rerun the parity campaign on `ort-webgpu` per
  `README.md` "Parity" section; scores are deterministic
  (`with_deterministic_compute`) and the test asserts bit-identical
  repeats.

The committed rollup is `parity_summary.tsv` in this directory
(per model × size: row count, worst abs err, worst JND err, status).
