# zenmetrics justfile — common dev/CI commands.
# `just` (1.x) required. Run `just` with no args to list recipes.

# Feature combo CI uses for the GPU-less zenmetrics-api CPU job: wgpu so
# the -gpu crates build without the CUDA SDK, but `cuda` OFF so the
# matrix's CPU-vs-CUDA parity layer compiles out (see ci.yml
# cpu-metrics-tests). Kept here verbatim so `just test-cpu` reproduces CI.
CPU_FEATURES := "all-metrics,cpu-metrics,wgpu,pixels,encoded"

# gmsd MDSI author-score gate (116 pairs). The caller decides whether it runs:
# GMSD_MDSI_GATE=require runs it and needs GMSD_MDSI_TARGETS=<path to the
# target table>; GMSD_MDSI_GATE=skip skips it, and the recipes below echo that
# (libtest reports a skipped gate as `ok`, so the recipe says so).
# The default here is `skip`; CI or a release run must set `require`.
GMSD_MDSI_GATE := env_var_or_default("GMSD_MDSI_GATE", "skip")
GMSD_MDSI_TARGETS := env_var_or_default("GMSD_MDSI_TARGETS", "")

# List recipes.
default:
    @just --list

# NEVER `cargo fmt --all` — rustfmt can follow `mod`/path-dep edges into the
# patched sibling repos (../zensim, ../../butteraugli, …) and rewrite files
# we don't own. `cargo metadata --no-deps` lists exactly this workspace's
# members (verified to exclude every sibling path-dep); we expand it to a
# `-p NAME` list so only in-repo crates are touched.

# Format in-repo workspace packages only (sibling-safe) + regenerate the
# public-API surface snapshots (docs/public-api/). The snapshot runner lives
# in the workspace-excluded apidoc/ package, so it is never built or run by
# plain `cargo test` or any CI job.
fmt:
    cargo fmt $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Formatting check over the same in-repo package set (sibling-safe).
fmt-check:
    cargo fmt --check $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')

# zenmetrics-api optimized-CPU backend tests — EXACT mirror of CI's
# `cpu-metrics-tests` job (Backend::Cpu dispatch + backend×metric×size
# matrix, GPU-less: cuda off → CPU-vs-CUDA parity layer gated out).

# Run the GPU-less CPU-backend test suite (mirrors CI).
test-cpu:
    cargo test -p zenmetrics-api --no-default-features --features {{CPU_FEATURES}} \
        --test it -- backend_matrix cpu_dispatch

# The full matrix including the CPU-vs-CUDA parity layer needs an NVIDIA
# GPU; default features turn `cuda` on so the `#[cfg(feature = "cuda")]`
# parity tests compile in and run.

# Run the full backend matrix locally, including CPU-vs-CUDA parity (needs GPU).
test-matrix-gpu:
    cargo test -p zenmetrics-api --features cpu-metrics --test it backend_matrix

# Quick default-feature check of the umbrella crate.
check:
    cargo check -p zenmetrics-api

# ghcr package-name guard: fail if any active-infra file references a
# ghcr.io/imazen/<name> that isn't a canonical package in ghcr-packages.json.
# One package per artifact; variants are TAGS. See docs/GHCR_PACKAGES.md.
ghcr-check:
    python3 scripts/ci/check_ghcr_packages.py

# Strict: also fail on grandfathered splinters still referenced in infra.
# Flip CI to this once ghcr-packages.json's `deprecated` map is empty.
ghcr-check-strict:
    python3 scripts/ci/check_ghcr_packages.py --strict

# Audit the LIVE ghcr.io/imazen packages against the manifest (needs `gh`
# authed with read:packages). Prints orphans + a commented migrate/delete recipe.
ghcr-audit:
    python3 scripts/ci/audit_ghcr_org.py

# Fleet-tooling guard: fail if a new launch_*/onstart_*/fleet*/*_watch script
# appears outside the canonical set in fleet-tools.json. ONE tool per concern —
# add a subcommand to `scripts/jobsys/fleet`, not a new script.
fleet-check:
    python3 scripts/ci/check_fleet_tools.py

# Strict: also fail on grandfathered forks (the post-Phase-E gate).
fleet-check-strict:
    python3 scripts/ci/check_fleet_tools.py --strict

# Same patterns scripts/safe_push.sh gates the outgoing diff with —
# scripts/lib/hygiene_patterns.txt is their one owner. CI: ci.yml.
# hygiene: address/identifier check over every tracked text file.
hygiene-check:
    python3 scripts/ci/check_hygiene.py --self-test
    python3 scripts/ci/check_hygiene.py

# TMPDIR discipline (ban RAM-backed tmp everywhere): unset/tmpfs TMPDIR must be
# rejected loud at worker boot. Pure-logic shell test, no cloud/GPU/secrets.
test-tmpdir-discipline:
    bash crates/zenfleet-worker/tests/tmpdir_discipline_test.sh

# --- quality kit (zenutils) -------------------------------------------------
# Advisory sweep — report only, never gates CI. Kit resolution: $ZENUTILS,
# then ../zenutils (workspace layout), then .quality-kit/ clone
# (`just quality-bootstrap`). See ../zenutils/quality/README.md.
ZENUTILS := env_var_or_default("ZENUTILS", justfile_directory() / "../zenutils")
QUALITY := ZENUTILS / "quality"

# Advisory quality sweep: fmt, clippy census, API exposure, stale docs,
# unused deps, cargo deny, typos, complexity hotspots, shellcheck.
quality *flags:
    @q="{{QUALITY}}"; [ -x "$q/quality.sh" ] || q="{{justfile_directory()}}/.quality-kit/quality"; \
    [ -x "$q/quality.sh" ] || { echo "kit not found — run: just quality-bootstrap"; exit 2; }; \
    "$q/quality.sh" --root "{{justfile_directory()}}" {{flags}}

# Quick sweep — skips compile-heavy clippy/deny stages.
quality-quick:
    @just quality --quick

# Fetch the quality kit if ../zenutils isn't checked out.
quality-bootstrap:
    @if [ -d "{{justfile_directory()}}/../zenutils/quality" ]; then \
      echo "kit already at ../zenutils"; \
    else \
      git clone --quiet https://github.com/imazen/zenutils \
        "{{justfile_directory()}}/.quality-kit" && \
      echo "cloned kit into .quality-kit (gitignored)"; \
    fi

# Public-API exposure/YAGNI report only.
api-surface *flags:
    @q="{{QUALITY}}"; [ -d "$q" ] || q="{{justfile_directory()}}/.quality-kit/quality"; \
    python3 "$q/api-report.py" "{{justfile_directory()}}" {{flags}}

# Stale-doc scan only (dead links, dead script refs, dead just recipes).
docs-check *flags:
    @q="{{QUALITY}}"; [ -d "$q" ] || q="{{justfile_directory()}}/.quality-kit/quality"; \
    python3 "$q/check-stale-docs.py" "{{justfile_directory()}}" {{flags}}

# Statement coverage for the default suite (llvm-cov; lcov.info written).
# Advisory — informs refactoring risk. Heavy: first run compiles the tree.
coverage:
    cargo llvm-cov --workspace --lcov --output-path lcov.info
    @cargo llvm-cov report --summary-only 2>/dev/null | tail -15 || true

# gmsd crate tests (all tiers the host has, banded parallelism). The MDSI
# author-score gate follows GMSD_MDSI_GATE / GMSD_MDSI_TARGETS above.
test-gmsd:
    @if [ "{{GMSD_MDSI_GATE}}" = "skip" ]; then echo 'MDSI author-score gate: SKIPPED (GMSD_MDSI_GATE=skip)'; fi
    GMSD_MDSI_GATE={{GMSD_MDSI_GATE}} GMSD_MDSI_TARGETS={{GMSD_MDSI_TARGETS}} \
        cargo test -p gmsd --release --features parallel

# The MDSI author-score gate on its own: fails unless GMSD_MDSI_TARGETS names the table.
test-gmsd-mdsi-gate:
    GMSD_MDSI_GATE=require GMSD_MDSI_TARGETS={{GMSD_MDSI_TARGETS}} \
        cargo test -p gmsd --release --features parallel -- author_score_gate

# The unpublished Margarine research instruments; does not format sibling crates.
margarine-check:
    python3 -m unittest discover -s crates/margarine -p 'test_*.py'
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path crates/margarine/Cargo.toml -j 2
    nice -n 19 cargo clippy --manifest-path crates/margarine/Cargo.toml --all-targets -j 2 -- -D warnings

# Caller may wrap this in run-heavy on Linux; use a fresh output directory.
margarine-bootstrap scores output draws="2000" seed="20260926":
    nice -n 19 crates/margarine/target/release/margarine-eval --bootstrap-all "{{scores}}" "{{output}}" "{{draws}}" "{{seed}}"

# Requires an interpreter with crates/margarine/requirements-training.txt.
margarine-fit-check python:
    cd crates/margarine && OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 19 "{{python}}" -m unittest fit_probe_checks

# Fresh-process memory plus interleaved timing; caller sets RAYON_NUM_THREADS.
margarine-resources crops output commit:
    nice -n 19 python3 crates/margarine/resource_sweep.py "{{crops}}" crates/margarine/target/release/margarine-box3 "{{output}}" --build-commit "{{commit}}"

# Broad-blur approximation; shared arithmetic and strided seam contracts.
margarine-multirate-check:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path crates/margarine/Cargo.toml --features multirate -j 2
    nice -n 19 cargo clippy --manifest-path crates/margarine/Cargo.toml --features multirate --all-targets -j 2 -- -D warnings

margarine-compact-check:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path crates/margarine/Cargo.toml --features compact -j 2
    nice -n 19 cargo clippy --manifest-path crates/margarine/Cargo.toml --features compact --all-targets -j 2 -- -D warnings

# Wrap these recipes with run-heavy on Linux; one heavy command at a time.
margarine-candidate-check features:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path crates/margarine/Cargo.toml --features "{{features}}" -j 2
    nice -n 19 cargo test --release --manifest-path crates/margarine/Cargo.toml --features "{{features}}" -j 2
    nice -n 19 cargo clippy --manifest-path crates/margarine/Cargo.toml --features "{{features}}" --all-targets -j 2 -- -D warnings
    nice -n 19 cargo build --release --manifest-path crates/margarine/Cargo.toml --features "{{features}}" --bin margarine-box3 -j 2

margarine-direct-eval pairs binaries output candidate teacher commit ingress="aic-rgb8":
    nice -n 19 python3 crates/margarine/score_manifest.py "{{pairs}}" "{{binaries}}" "{{output}}" --candidate "{{candidate}}" --teacher "{{teacher}}" --build-commit "{{commit}}" --ingress "{{ingress}}"

margarine-direct-resources crops binary output candidate commit rows="128" columns="512":
    nice -n 19 python3 crates/margarine/resource_sweep.py "{{crops}}" "{{binary}}" "{{output}}" --direct "{{candidate}}" --strip-rows "{{rows}}" --tile-columns "{{columns}}" --build-commit "{{commit}}"

# A single-size diagnostic cannot fit fixed overhead or qualify the size curve.
margarine-direct-timing binary reference distorted output rows="128":
    nice -n 19 "{{binary}}" --bench-direct "{{rows}}" "{{reference}}" "{{distorted}}" "{{output}}"

margarine-direct-profile binary reference distorted output commit rows="128":
    mkdir "{{output}}"
    printf '%s\n' "{{commit}}" > "{{output}}/build_commit.txt"
    shasum -a 256 "{{binary}}" "{{reference}}" "{{distorted}}" > "{{output}}/inputs.sha256"
    nice -n 19 valgrind --tool=callgrind --callgrind-out-file="{{output}}/callgrind.out" "{{binary}}" --memory-native "{{rows}}" "{{reference}}" "{{distorted}}" > "{{output}}/run.log" 2>&1
    callgrind_annotate --inclusive=no --threshold=99 "{{output}}/callgrind.out" > "{{output}}/flat.txt"

# Linux CPU samples; instrumented timings are not qualification measurements.
margarine-direct-perf binary reference distorted output commit rows="128":
    mkdir "{{output}}"
    printf '%s\n' "{{commit}}" > "{{output}}/build_commit.txt"
    shasum -a 256 "{{binary}}" "{{reference}}" "{{distorted}}" > "{{output}}/inputs.sha256"
    sudo -n env RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 TMPDIR="$HOME/tmp" ZENBENCH_NO_SAVE=1 nice -n 19 perf record -F 997 --call-graph dwarf,8192 -o "{{output}}/perf.data" -- "{{binary}}" --bench-direct "{{rows}}" "{{reference}}" "{{distorted}}" "{{output}}/instrumented.json" > "{{output}}/run.log" 2>&1
    sudo -n perf report --stdio --no-children --percent-limit 0.5 -i "{{output}}/perf.data" > "{{output}}/flat.txt" 2>&1

# Point-label diagnostics; participant significance remains a separate gate.
margarine-disagreements ledger output candidate teacher_maps candidate_maps commit:
    nice -n 19 python3 crates/margarine/disagreements.py "{{ledger}}" "{{output}}" --candidate "{{candidate}}" --teacher-maps "{{teacher_maps}}" --candidate-maps "{{candidate_maps}}" --build-commit "{{commit}}"

margarine-kadid-opinions raw dmos output commit:
    nice -n 19 python3 crates/margarine/prepare_kadid_opinions.py "{{raw}}" "{{dmos}}" "{{output}}" --build-commit "{{commit}}"

margarine-panels scored evaluator output candidate commit:
    nice -n 19 python3 crates/margarine/evaluate_manifest.py "{{scored}}" "{{evaluator}}" "{{output}}" --candidate "{{candidate}}" --build-commit "{{commit}}"

margarine-ordinary-panels scored evaluator output candidate commit:
    nice -n 19 python3 crates/margarine/evaluate_manifest.py "{{scored}}" "{{evaluator}}" "{{output}}" --candidate "{{candidate}}" --build-commit "{{commit}}" --ordinary-only

margarine-quality-bands evaluator scores output bands="5":
    nice -n 19 "{{evaluator}}" --quality-bands "{{scores}}" "{{output}}" "{{bands}}"

margarine-aic-intervals scored labels output candidate commit:
    nice -n 19 python3 crates/margarine/interval_disagreements.py "{{scored}}" "{{labels}}" "{{output}}" --candidate "{{candidate}}" --build-commit "{{commit}}"

margarine-participant-pairs scored opinions evaluator output candidate commit draws="2000" seed="20260926":
    nice -n 19 python3 crates/margarine/participant_pairs.py "{{scored}}" "{{opinions}}" "{{evaluator}}" "{{output}}" --candidate "{{candidate}}" --build-commit "{{commit}}" --draws "{{draws}}" --seed "{{seed}}"

margarine-live1-participants scored opinions evaluator output candidate commit draws="2000" seed="20260926":
    nice -n 19 python3 crates/margarine/participant_pairs.py "{{scored}}" "{{opinions}}" "{{evaluator}}" "{{output}}" --live1 --candidate "{{candidate}}" --build-commit "{{commit}}" --draws "{{draws}}" --seed "{{seed}}"

# Published Release 1 cohorts retain their separate rating normalizations.
margarine-live1-inputs root output destination commit:
    nice -n 19 python3 crates/margarine/prepare_live1.py "{{root}}" "{{output}}" --destination-root "{{destination}}" --build-commit "{{commit}}"

margarine-audited-eval pairs audit binaries output candidate commit teacher="":
    if [ -n "{{teacher}}" ]; then set -- --teacher "{{teacher}}"; else set --; fi; nice -n 19 python3 crates/margarine/score_manifest.py "{{pairs}}" "{{binaries}}" "{{output}}" --input-audit "{{audit}}" --candidate "{{candidate}}" --build-commit "{{commit}}" --ingress common-srgb "$@"

# The named command uses the frozen primary max candidate and streamed geometry.
margarine-build:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo build --release --manifest-path crates/margarine/Cargo.toml --no-default-features --features rayon,avx512,simd-malta,row-malta --bin margarine -j 2

margarine-cli-replay scored binary output commit:
    nice -n 19 python3 crates/margarine/cli_replay.py "{{scored}}" "{{binary}}" "{{output}}" --candidate simd-row-malta --build-commit "{{commit}}"

# Offline view of the committed point panels and measured resource curves.
margarine-report output commit:
    python3 crates/margarine/report.py "{{output}}" --build-commit "{{commit}}"

# Compare pre/post relocation builds on the caller's fixed pair manifest.
margarine-relocation-check pairs before after output:
    nice -n 19 python3 crates/margarine/relocation_check.py "{{pairs}}" "{{before}}" "{{after}}" "{{output}}"

margarine-vendor-check:
    python3 crates/margarine/verify_sources.py
