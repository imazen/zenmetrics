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

# E29 local preparation; caller supplies frozen source, binaries and build metadata.
fit-pack-e29 source bin_dir build_meta out:
    TMPDIR=$HOME/tmp/e29 ~/work/zen/scripts/run-heavy --mem 8G --jobs 1 -- python3 scripts/jobsys/pack_fit_program.py --source {{source}} --executor scripts/jobsys/fit_cell_exec.py --bin-dir {{bin_dir}} --build-meta {{build_meta}} --profile v2e29 --out {{out}}

# Local manifest generation only: this does not enqueue a fleet job.
fit-declare-e29 ctl spec out:
    {{ctl}} declare-fits --spec {{spec}} --out {{out}}

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
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine --check
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
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine --check
    nice -n 19 cargo test --manifest-path crates/margarine/Cargo.toml --features multirate -j 2
    nice -n 19 cargo clippy --manifest-path crates/margarine/Cargo.toml --features multirate --all-targets -j 2 -- -D warnings

margarine-compact-check:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine --check
    nice -n 19 cargo test --manifest-path crates/margarine/Cargo.toml --features compact -j 2
    nice -n 19 cargo clippy --manifest-path crates/margarine/Cargo.toml --features compact --all-targets -j 2 -- -D warnings

# Wrap these recipes with run-heavy on Linux; one heavy command at a time.
margarine-candidate-check features:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine --check
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
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine --check
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

# Claim ownership regressions and full worker validation; callers use run-heavy.
worker-claim-regressions:
    cargo test --locked -p zenfleet-worker --lib workerfix_ -- --nocapture

worker-tests:
    cargo test --locked -p zenfleet-worker

worker-clippy:
    cargo clippy --locked -p zenfleet-worker --all-targets -- -D warnings

# CI-pinned lock check and snapshot export regression (use run-heavy).
lock-check *args:
    bash scripts/ci/lock.sh --check {{args}}

# Every remote branch and open PR with a verdict (merged/stale/conflict/...); read-only.
branch-status *args:
    python3 scripts/ci/branch_status.py {{args}}

lock-snapshot-tests:
    python3 scripts/ci/test_lock_snapshot.py

# Run cargo in the CI-pinned sibling snapshot, for dev trees whose sibling
# checkouts have drifted from ci/sibling-pins.tsv (a plain `cargo` there can
# fail to resolve). Exports `rev` (`@` = the working copy, snapshotted first;
# any other revision is exported as committed), checks the lock,
# then runs `cargo --locked <args>` in the snapshot with its own target dir.
# Example: just snap-cargo @ test -p zenstats
# `fmt` is refused without `--check`: it would format the snapshot copy and
# never reach the tree (use `just fmt`).
snap-cargo rev +args:
    @case " {{args}} " in " fmt "*) case " {{args}} " in *" --check "*) ;; *) echo "snap-cargo: fmt edits the snapshot, not this tree; run 'just fmt' (fmt --check is fine here)" >&2; exit 2 ;; esac ;; esac
    if [ "{{rev}}" = "@" ]; then bash scripts/ci/lock.sh --check; else bash scripts/ci/lock.sh --check --rev {{rev}}; fi
    cd "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/work/zenmetrics" && CARGO_TERM_COLOR=never CARGO_TARGET_DIR="${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target" cargo --locked {{args}}

# zenmetrics-cli feature-combination matrix: 12 rows — every supported
# `--no-default-features --features <F>` shape, plus the bare
# `--no-default-features` build, plus the default build. Prints a PASS/FAIL
# row per combination and exits nonzero if any fails. CI runs the same list
# as plain `cargo --locked check` (ci.yml, `cli-feature-matrix` job) — keep
# the two lists in sync.
cli-feature-matrix:
    #!/usr/bin/env bash
    set -uo pipefail
    total=0
    fails=0
    run_combo() {
        local label="$1" out status=0
        shift
        out=$(just snap-cargo @ check "$@" 2>&1) || status=$?
        total=$((total + 1))
        if [ "$status" -eq 0 ]; then
            printf 'PASS  %s\n' "$label"
        else
            printf 'FAIL  %s\n' "$label"
            printf '%s\n' "$out"
            fails=$((fails + 1))
        fi
    }
    for f in cpu-metrics sweep sweep,png sweep,hdr jobexec jobexec,hdr orchestrator \
             gpu,gpu-cuda sweep,png,gpu,gpu-cuda gpu,gpu-wgpu; do
        run_combo "$f" -p zenmetrics-cli --no-default-features --features "$f"
    done
    run_combo bare -p zenmetrics-cli --no-default-features
    run_combo default -p zenmetrics-cli
    printf '%s\n' "---- $((total - fails))/$total combinations passed ----"
    [ "$fails" -eq 0 ]

# Matched CPU-metric wall/counter runs; pass a preserved baseline or candidate example.
ranked-wall binary threads output w="3355" h="2516" reps="5":
    bash scripts/bench/ranked-metric-wall.sh "{{binary}}" "{{threads}}" "{{output}}" "{{w}}" "{{h}}" "{{reps}}"

# Preserve native perf samples under an explicit disk-backed output prefix.
ranked-profile binary output w="3355" h="2516" reps="30":
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- sudo -n --preserve-env=IWSSIM_WALL_STRIP_HEIGHT perf record -F 997 --call-graph dwarf,8192 -o "{{output}}.data" -- taskset -c 2 env RAYON_NUM_THREADS=1 "{{binary}}" "{{w}}" "{{h}}" "{{reps}}"
    sudo -n perf report -i "{{output}}.data" --stdio --no-children --percent-limit 1 > "{{output}}.report"

# Absolute-luminance end-to-end HDR probe, with explicit generation and thread count.
ranked-hdr-probe binary version side reps threads output cpus="2":
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS="{{threads}}" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- perf stat -e instructions,cycles -- taskset -c "{{cpus}}" "{{binary}}" "{{side}}" "{{reps}}" "{{version}}" 2>&1 | tee "{{output}}"

# Exact coefficient audit for the proposed libvmaf separable low-pass.
lpf-rank-check:
    python3 scripts/bench/check-lpf-rank.py

# Existing new-metrics-wall driver, bounded zenbench rounds and retained TSV.
ranked-zenbench binary threads prefix metrics size="1024":
    bash scripts/bench/ranked-metric-zenbench.sh "{{binary}}" "{{threads}}" "{{prefix}}" "{{metrics}}" "{{size}}"

# Sample the same bounded production latency fixture; profiling is a separate run.
ranked-zenbench-profile binary prefix metrics size="1024":
    TMPDIR="$HOME/tmp" RAYON_NUM_THREADS=1 NMW_CONTENT=synth NMW_MODES=lat NMW_MIN_ROUNDS=30 NMW_GROUP_WALL_S=120 NMW_CELL_MAX_S=30 ZENBENCH_PIN=2 ZENBENCH_DATA_DIR="$HOME/tmp/zm-ranked-profile-evidence" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- sudo -n --preserve-env=TMPDIR,RAYON_NUM_THREADS,NMW_CONTENT,NMW_MODES,NMW_MIN_ROUNDS,NMW_GROUP_WALL_S,NMW_CELL_MAX_S,ZENBENCH_PIN,ZENBENCH_DATA_DIR perf record -F 997 --call-graph dwarf,8192 -o "{{prefix}}.data" -- taskset -c 2 "{{binary}}" "{{size}}" "{{prefix}}.tsv" "{{metrics}}" > "{{prefix}}.log" 2>&1
    sudo -n perf report -i "{{prefix}}.data" --stdio --no-children --percent-limit 1 > "{{prefix}}.report"

# Prepare a fresh-process exploratory A/B with A/A and an exact score-bit checker.
ranked-pair-plan baseline candidate baseline_rev candidate_rev output threads="1" cpus="2":
    python3 -B scripts/bench/ranked-metric-pair.py plan "{{baseline}}" "{{candidate}}" "{{baseline_rev}}" "{{candidate_rev}}" "{{output}}" --threads "{{threads}}" --cpus "{{cpus}}"

ranked-hdr-pair-plan baseline candidate baseline_rev candidate_rev output version="v3" side="250" threads="1" cpus="2":
    python3 -B scripts/bench/ranked-metric-pair.py plan "{{baseline}}" "{{candidate}}" "{{baseline_rev}}" "{{candidate_rev}}" "{{output}}" --threads "{{threads}}" --cpus "{{cpus}}" --hdr-version "{{version}}" --side "{{side}}"

# Pair full production fleet scoring with every scalar product checked exactly.
ranked-fleet-pair-plan baseline candidate baseline_rev candidate_rev case control runner output:
    python3 -B scripts/bench/ranked-metric-pair.py fleet-plan "{{baseline}}" "{{candidate}}" "{{baseline_rev}}" "{{candidate_rev}}" "{{case}}" "{{control}}" "{{runner}}" "{{output}}"

ranked-pair runner plan output cpus="2":
    TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- "{{runner}}" exec --cpu-policy auto --cpu-set "{{cpus}}" "{{runner}}" -- pair "{{plan}}" --output "{{output}}" --wait 30

# Explicit system scope when the user manager lacks delegated cpuset control.
ranked-pair-scope runner plan output cpus="2":
    sudo -n systemd-run --scope --quiet --property=AllowedCPUs={{cpus}} --property=MemoryMax=16G --property=MemorySwapMax=0 runuser -u "$(id -un)" -- env TMPDIR="$HOME/tmp" ZENBENCH_CPU_SET="{{cpus}}" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- "{{runner}}" pair "{{plan}}" --output "{{output}}" --wait 30

# Local V40 package verification; no enqueue or publication.
v40-fit-tools:
    PYTHONPATH=scripts/jobsys python3 -m unittest scripts.jobsys.test_v40_contract scripts.jobsys.test_fit_tools

v40-program source binaries metadata contract output:
    python3 scripts/jobsys/pack_fit_program.py --source {{source}} --executor scripts/jobsys/fit_cell_exec.py --bin-dir {{binaries}} --build-meta {{metadata}} --profile v40 --data benchmarks/v40_fit_contract_2026-10-07.json={{contract}} --out {{output}}

e33-program source binaries metadata contract output:
    python3 scripts/jobsys/pack_fit_program.py --source {{source}} --executor scripts/jobsys/fit_cell_exec.py --bin-dir {{binaries}} --build-meta {{metadata}} --profile e33 --data benchmarks/e33_fit_contract_2026-10-10.json={{contract}} --out {{output}}

e33-image bundle image:
    docker build --build-arg FIT_PROGRAM_SHA=$(sha256sum {{bundle}}/program.tar.gz | cut -d' ' -f1) -t {{image}} {{bundle}}/image-context

v40-image bundle image:
    docker build --build-arg FIT_PROGRAM_SHA=$(sha256sum {{bundle}}/program.tar.gz | cut -d' ' -f1) -t {{image}} {{bundle}}/image-context

v40-harvest-refusals bundle attempt:
    python3 scripts/jobsys/v40_harvest_refusals.py --bundle {{bundle}} --attempt {{attempt}}

v40-postfit-checks:
    bash -n scripts/jobsys/v40_postfit.sh
    shellcheck scripts/jobsys/v40_postfit.sh
    bash -c 'bash scripts/jobsys/v40_postfit.sh /nonexistent uh4 > "$HOME/tmp/v40/postfit-refusal.log" 2>&1; test "$?" -eq 2'

# Every CPU metric the CLI scores from sRGB8, at 1 and 8 threads: zenbench at
# 1024², plus an 8.4 MP frame by default (8t zenbench, 1t one call per metric
# with max RSS). Each leg runs under run-heavy; nothing else heavy alongside.
cpu-metrics-1t8t out *sizes:
    nice -n 19 cargo build --release -p cpu-profile --bin new-metrics-wall
    scripts/cpu_metrics_1t8t.sh "{{out}}" {{sizes}}
    python3 scripts/cpu_metrics_1t8t_analyze.py "{{out}}"

# sol-math lane validation uses the CI-pinned siblings, never the dev-tree lock.
solmath-lock mode="--check":
    TMPDIR="$HOME/tmp" LOCK_SNAP_DIR="$HOME/tmp/solmath-validation" scripts/ci/lock.sh {{mode}}

solmath-test *features:
    cd "$HOME/tmp/solmath-validation/work/zenmetrics" && TMPDIR="$HOME/tmp" CARGO_TARGET_DIR="{{justfile_directory()}}/target/solmath" ~/work/zen/scripts/run-heavy --mem 12G -- cargo test --locked --manifest-path "$HOME/tmp/solmath-validation/work/zenmetrics/Cargo.toml" -p zenmetrics-math -p cvvdp {{features}}

solmath-csf-bench *args:
    cd "$HOME/tmp/solmath-validation/work/zenmetrics" && TMPDIR="$HOME/tmp" CARGO_TARGET_DIR="{{justfile_directory()}}/target/solmath" RAYON_NUM_THREADS=8 ~/work/zen/scripts/run-heavy --mem 12G -- taskset -c 2-9 cargo bench --locked --manifest-path "$HOME/tmp/solmath-validation/work/zenmetrics/Cargo.toml" -p cvvdp --bench csf_wall {{args}}

solmath-check *args:
    cd "$HOME/tmp/solmath-validation/work/zenmetrics" && TMPDIR="$HOME/tmp" CARGO_TARGET_DIR="{{justfile_directory()}}/target/solmath" ~/work/zen/scripts/run-heavy --mem 12G -- cargo {{args}}

benchmark-contract-check:
    TMPDIR="${TMPDIR:-$HOME/tmp}" python3 -B -m unittest discover -s scripts/jobsys -p test_benchmark_exec.py
    just snap-cargo @ test -p zenfleet-core -p zenfleet-worker -p zenfleet-ctl --lib --bins

benchmark-study-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/jobsys -p test_benchmark_study.py

benchmark-pair-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/jobsys -p test_benchmark_pair.py

benchmark-pair +args:
    python3 -B scripts/jobsys/build_benchmark_pair.py {{args}}

benchmark-analysis-check:
    TMPDIR="$HOME/tmp" python3 -B -m unittest discover -s scripts/jobsys -p test_benchmark_analysis.py

benchmark-fit-oracle:
    python3 -B scripts/jobsys/check_benchmark_fit_oracle.py

benchmark-study-analyze +args:
    python3 -B scripts/jobsys/analyze_benchmark_study.py {{args}}

benchmark-study +args:
    python3 -B scripts/jobsys/build_benchmark_study.py {{args}}

benchmark-study-package-check runner output:
    python3 -B scripts/jobsys/check_benchmark_study_package.py {{runner}} {{output}}

benchmark-executor-check runner output:
    python3 -B scripts/jobsys/check_benchmark_exec.py {{runner}} {{output}}

# Separate artifacts: the memory binary must never supply latency claims.
iqa-fleet-build output:
    mkdir {{output}}
    just snap-cargo @ build -p cpu-profile --bin new-metrics-wall
    cp "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target/debug/new-metrics-wall" {{output}}/latency
    just snap-cargo @ build -p cpu-profile --bin new-metrics-wall --features heap-phases
    cp "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target/debug/new-metrics-wall" {{output}}/memory

# Release artifacts for study qualification; keep timing and heap instrumentation separate.
iqa-study-build output:
    mkdir {{output}}
    just snap-cargo @ build --release -p cpu-profile --bin new-metrics-wall
    cp "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target/release/new-metrics-wall" {{output}}/latency
    just snap-cargo @ build --release -p cpu-profile --bin new-metrics-wall --features heap-phases
    cp "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target/release/new-metrics-wall" {{output}}/memory

iqa-gpu-study-build backend output:
    python3 -c 'import sys; assert sys.argv[1] in ("wgpu", "cuda")' "{{backend}}"
    mkdir {{output}}
    just snap-cargo @ build --release -p cpu-profile --bin new-metrics-wall --features study-gpu-{{backend}}
    cp "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target/release/new-metrics-wall" {{output}}/latency
    just snap-cargo @ build --release -p cpu-profile --bin new-metrics-wall --features study-gpu-{{backend}},heap-phases
    cp "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target/release/new-metrics-wall" {{output}}/memory

# Run on a distro SDK source host; preserve the hash-manifested runtime directory.
cuda-runtime-sdk output:
    python3 -B scripts/cuda/export-runtime-sdk.py "{{output}}"

iqa-gpu-check binaries backend output adapter *args:
    python3 -B scripts/jobsys/check_iqa_gpu.py {{binaries}} {{backend}} {{output}} "{{adapter}}" {{args}}

iqa-pair-lines-check binaries case output:
    python3 -B scripts/jobsys/check_iqa_pair_lines.py {{binaries}} {{case}} {{output}}

iqa-hdr-check binaries output *args:
    python3 -B scripts/jobsys/check_iqa_hdr.py {{binaries}} {{output}} {{args}}

iqa-study-check binaries output *args:
    python3 -B scripts/jobsys/check_iqa_study.py {{binaries}} {{output}} {{args}}

# Use prequalified small cases to exercise all native routes through zenfleet.
iqa-deployment-bench binaries runner sdr hdr output machine domain:
    python3 -B scripts/jobsys/benchmark_iqa_deployment.py {{binaries}} {{runner}} {{sdr}} {{hdr}} {{output}} {{machine}} {{domain}}

# Argument-based orchestration also works through nested SSH without shell quoting.
iqa-control-tree output:
    tar -czf "{{output}}" scripts/jobsys/check_iqa_study.py scripts/jobsys/check_iqa_hdr.py scripts/jobsys/check_iqa_gpu.py scripts/jobsys/check_iqa_deployment.py scripts/jobsys/build_benchmark_study.py scripts/jobsys/benchmark_iqa_deployment.py scripts/jobsys/benchmark_exec.py benchmarks/heaptrack/drivers/cpu_profile/src crates/zenmetrics-cli/src/metrics/mod.rs

iqa-deployment-campaign binaries runner checker_tree output machine domain *args:
    python3 -B scripts/jobsys/check_iqa_deployment.py "{{binaries}}" "{{runner}}" "{{checker_tree}}" "{{output}}" "{{machine}}" "{{domain}}" {{args}}

# Stage the hash-bound scoped CUDA runtime with every qualified GPU case.
iqa-gpu-deployment-bench binaries runner controls output machine domain sdk *args:
    python3 -B scripts/jobsys/benchmark_iqa_deployment.py {{binaries}} {{runner}} - - {{output}} {{machine}} {{domain}} --gpu-only --gpu-controls {{controls}} --gpu-sdk {{sdk}} {{args}}

iqa-fleet-check binaries output:
    python3 -B scripts/jobsys/check_iqa_fleet.py {{binaries}} {{output}}

benchmark-image-check image scratch:
    mkdir -p {{scratch}}
    docker run --rm --memory=6g --memory-swap=6g --cpus=2 --mount type=bind,source={{scratch}},target=/scratch --mount type=bind,source={{justfile_directory()}}/scripts/jobsys,target=/controls,readonly --entrypoint python3 {{image}} /controls/check_benchmark_exec.py /usr/local/bin/zenbench /scratch/proof /usr/local/bin/benchmark-exec /usr/local/bin/zenfleet-worker

iqa-heap-oracle binary case output:
    python3 -B scripts/jobsys/check_iqa_heap_oracle.py {{binary}} {{case}} {{output}}

iqa-bundle-check binaries runner case output:
    python3 -B scripts/jobsys/check_iqa_bundle.py {{binaries}} {{runner}} {{case}} {{output}}

benchmark-native-check runner worker reader output:
    python3 -B scripts/jobsys/check_benchmark_exec.py {{runner}} {{output}} scripts/jobsys/benchmark_exec.py {{worker}} {{reader}}

# Install locked dashboard dependencies without dependency lifecycle scripts.
dashboard-web-check npm_bin="npm":
    cd crates/zenfleet-dash/web && "{{npm_bin}}" ci --ignore-scripts --no-audit --no-fund --allow-remote=all
    cd crates/zenfleet-dash/web && "{{npm_bin}}" run build

vmaf-portable-check:
    cargo check --locked -p vmaf --lib --features simd,avx512 --target aarch64-unknown-linux-gnu
    cargo check --locked -p vmaf --lib --features simd,avx512 --target x86_64-unknown-linux-gnu

checkpoint-i686-vmaf-check:
    ZEN_CHECKPOINT_WORKSPACES="$(cd ../.. && pwd)" CROSS_CONFIG="$(pwd)/ci/checkpoint-cross.toml" cross check --locked -p vmaf --lib --features simd,avx512 --target i686-unknown-linux-gnu

checkpoint-native profile target source runner output:
    python3 scripts/checkpoint/package.py --profile {{profile}} --target {{target}} --source {{source}} --runner-source {{runner}} --out {{output}}

checkpoint-browser-build output:
    CHECKPOINT_OUT={{output}} bash scripts/checkpoint/build-browser.sh

checkpoint-browser-check package report oracle:
    python3 scripts/checkpoint/check-browser.py {{package}} {{report}} --oracle {{oracle}}

checkpoint-release-prepare run source output:
    python3 scripts/checkpoint/prepare-release.py --run {{run}} --source {{source}} --out {{output}}

checkpoint-gpu-contract-check expectation features:
    ZENMETRICS_TEST_GPU={{expectation}} cargo test --locked -p zenmetrics-cli --features {{features}} --test cli butteraugli_cpu_and_gpu_agree -- --exact

checkpoint-gpu-refusal-check features:
    cargo test --locked -p zenmetrics-cli --features {{features}} --test cli require_gpu

checkpoint-release-publish run source tag output:
    python3 scripts/checkpoint/publish-release.py --run {{run}} --source {{source}} --tag {{tag}} --out {{output}}

checkpoint-release-controls:
    python3 scripts/checkpoint/test_release.py
    python3 scripts/checkpoint/test_archive.py

checkpoint-i686-cli-tests:
    ZEN_CHECKPOINT_WORKSPACES="$(cd ../.. && pwd)" CROSS_CONFIG="$(pwd)/ci/checkpoint-cross.toml" ZENMETRICS_TEST_GPU=absent cross test --locked -p zenmetrics-cli --all-targets --features sweep,bmp,gif,tiff,hdr-gainmap,avif-svt,hdr-svt,jpegai,orchestrator,orchestrator-cpu-all,feature-jobs --target i686-unknown-linux-gnu

# Assemble a verified browser release for Cloudflare Pages (output must be new).
site-prepare archive source tag out:
    python3 scripts/checkpoint/prepare-pages.py --archive '{{archive}}' --source '{{source}}' --tag '{{tag}}' --out '{{out}}'

# Check isolated Rayon execution locally; requires Playwright and Chromium.
site-check site:
    python3 scripts/checkpoint/check-pages.py '{{site}}'

# Check the deployed viewer while retaining the local unisolated refusal control.
site-check-live site url:
    python3 scripts/checkpoint/check-pages.py '{{site}}' --url '{{url}}'

# Qualify the all-metric browser viewer against independent native spatial outputs.
viewer-check site oracle report:
    python3 scripts/checkpoint/check-pages.py '{{site}}' --oracle '{{oracle}}' --report '{{report}}'

# Assemble the matching current-source viewer (not an old checkpoint frontend).
viewer-site build source out:
    python3 scripts/checkpoint/prepare-pages.py --build '{{build}}' --source '{{source}}' --out '{{out}}'

# --- zenmetrics-nn (learned metrics) ----------------------------------------
# Export one LPIPS network (alex | vgg | squeeze) from the upstream `lpips`
# package and write CPU-fp32 goldens on a ladder dir
# (crates/zenmetrics-nn/models/lpips/export.py).
nn-export-lpips python ladder out net sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/lpips/export.py '{{ladder}}' '{{out}}' '{{net}}' '{{sizes}}'

# Export DISTS from the upstream `DISTS_pytorch` package, with goldens.
nn-export-dists python ladder out sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/dists/export.py '{{ladder}}' '{{out}}' '{{sizes}}'

# Export one ST-LPIPS network (alex | vgg) from `stlpips_pytorch`, with goldens.
nn-export-stlpips python ladder out net sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/stlpips/export.py '{{ladder}}' '{{out}}' '{{net}}' '{{sizes}}'

# Export DeepDC from `DeepDC_PyTorch`, with goldens.
nn-export-deepdc python ladder out sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/deepdc/export.py '{{ladder}}' '{{out}}' '{{sizes}}'

# Export CKDN from a clone of researchmm/CKDN plus its trained checkpoint.
nn-export-ckdn python ladder out upstream ckpt sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/ckdn/export.py '{{ladder}}' '{{out}}' '{{upstream}}' '{{ckpt}}' '{{sizes}}'

# Export A-FINE from a clone of ChrisDud0257/AFINE plus its afine.pth and
# OpenAI's ViT-B-32.pt (URLs in upstream's README/afine.py).
nn-export-afine python ladder out upstream afine_pth clip_pth sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/afine/export.py '{{ladder}}' '{{out}}' '{{upstream}}' '{{afine_pth}}' '{{clip_pth}}' '{{sizes}}'

# Export the DreamSim ensemble from the pip `dreamsim` package plus its
# downloaded ensemble checkpoint dir (upstream downloads it on first use).
nn-export-dreamsim python ladder out ckpt:
    {{python}} -I crates/zenmetrics-nn/models/dreamsim/export.py '{{ladder}}' '{{out}}' '{{ckpt}}'

# Export ARNIQA from a clone of miccunifi/ARNIQA plus its release weights
# dir (ARNIQA.pth + regressor_kadid10k.pth), with goldens.
nn-export-arniqa python ladder out upstream weights sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/arniqa/export.py '{{ladder}}' '{{out}}' '{{upstream}}' '{{weights}}' '{{sizes}}'

# Export MANIQA from a clone of IIGROUP/MANIQA plus its release weights
# dir (ckpt_koniq10k.pt), with goldens.
nn-export-maniqa python ladder out upstream weights sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/maniqa/export.py '{{ladder}}' '{{out}}' '{{upstream}}' '{{weights}}' '{{sizes}}'

# Export LIQE from a clone of zwx8981/LIQE plus its checkpoint
# (LIQE.pt), with goldens.
nn-export-liqe python ladder out upstream ckpt sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/liqe/export.py '{{ladder}}' '{{out}}' '{{upstream}}' '{{ckpt}}' '{{sizes}}'

# Export HyperIQA from the official KonIQ checkpoint (koniq_pretrained.pkl);
# upstream checkout + pkl paths, goldens from oracle.py.
nn-export-hyperiqa python upstream ckpt out goldens ladder:
    {{python}} -u -I crates/zenmetrics-nn/models/hyperiqa/export.py '{{upstream}}' '{{ckpt}}' '{{out}}' '{{goldens}}' '{{ladder}}'

# HyperIQA goldens via upstream models.py on the deterministic 5x5 grid
# (upstream's eval crops at random — see models/hyperiqa/card.toml).
nn-goldens-hyperiqa python upstream ckpt out ladder:
    {{python}} -I crates/zenmetrics-nn/models/hyperiqa/oracle.py '{{upstream}}' '{{ckpt}}' '{{out}}' {{ladder}}/*.png

nn-export-paq2piq python upstream ckpt out goldens ladder nonsquare:
    {{python}} -I crates/zenmetrics-nn/models/paq2piq/export.py '{{out}}' --upstream '{{upstream}}' --checkpoint '{{ckpt}}' --goldens '{{goldens}}' --ladder '{{ladder}}' --ladder '{{nonsquare}}'

nn-goldens-paq2piq python upstream ckpt out ladder:
    {{python}} -I crates/zenmetrics-nn/models/paq2piq/oracle.py --upstream '{{upstream}}' --checkpoint '{{ckpt}}' '{{ladder}}' > '{{out}}'

# Export WaDIQaM (dmaniry/deepIQA `models/{fr,nr}_{live,tid}_weighted.model`,
# h5py-only — no chainer needed). `weights` is the directory holding the
# official checkpoints. The recipe does not forward --goldens/--ladder (the
# oracle-vs-ONNX check); run export.py directly for that after
# nn-goldens-wadiqam produces the TSVs.
nn-export-wadiqam python out weights:
    {{python}} -I crates/zenmetrics-nn/models/wadiqam/export.py '{{out}}' --weights '{{weights}}'

# WaDIQaM goldens via upstream's chainer model classes on CPU
# (needs the chainer venv — see models/wadiqam/card.toml; chainer's
# cuda shim is patched inside oracle.py).
nn-goldens-wadiqam python upstream model top mode col out ladder nonsquare:
    {{python}} -I crates/zenmetrics-nn/models/wadiqam/oracle.py '{{upstream}}' '{{model}}' '{{top}}' '{{mode}}' '{{col}}' '{{out}}' '{{ladder}}' '{{nonsquare}}'

# Export MUSIQ from the official KonIQ checkpoint NPZ (koniq_ckpt.npz).
# Goldens come from oracle.py under the legacy JAX env, not this recipe.
nn-export-musiq python npz out goldens ladder:
    {{python}} -u -I crates/zenmetrics-nn/models/musiq/export.py '{{npz}}' '{{out}}' '{{goldens}}' '{{ladder}}'

# MUSIQ goldens via the upstream JAX oracle (jax 0.3.25 + flax 0.3.3 venv —
# see models/musiq/card.toml; tensorflow CPU, eager mode).
nn-goldens-musiq python ladder upstream npz out sizes="64,128,256,512,1024":
    {{python}} -I crates/zenmetrics-nn/models/musiq/oracle.py '{{ladder}}' '{{upstream}}' '{{npz}}' '{{out}}' '{{sizes}}'

# Parity of zenmetrics-nn against the exported goldens (CPU, rten). Runs in
# the pinned lock snapshot, so the paths are made absolute first.
# engine: rten | ort-cuda (ort-cuda needs ORT_DYLIB_PATH and the CUDA/cuDNN libs
# on the library path).
nn-parity model_dir ladder engine="rten" features="model-tests" models="lpips_alex,lpips_vgg,lpips_squeeze,stlpips_alex,stlpips_vgg,dists,deepdc,ckdn,afine,dreamsim,arniqa,maniqa,liqe,musiq,hyperiqa,paq2piq,wadiqam_fr_live,wadiqam_fr_tid,wadiqam_nr_live,wadiqam_nr_tid,pieapp":
    ZENMETRICS_NN_MODEL_DIR="$(realpath '{{model_dir}}')" ZENMETRICS_NN_LADDER="$(realpath '{{ladder}}')" ZENMETRICS_NN_ENGINE='{{engine}}' ZENMETRICS_NN_MODELS='{{models}}' \
        {{just_executable()}} snap-cargo @ test --release -p zenmetrics-nn --features '{{features}}' --test parity -- --nocapture --test-threads=1

# Crate-scoped statistical API validation in the pinned dependency snapshot.
stats-helpers-check:
    just snap-cargo @ test -p zenmetrics-stats
    just snap-cargo @ clippy -p zenmetrics-stats --all-targets -- -D warnings

stats-helpers-serial-check:
    just snap-cargo @ test -p zenmetrics-stats --no-default-features

# --- no-reference classical metrics (piqe, niqe, brisque) ---------------------
# Regenerate a crate's goldens TSV with the oracle venv
# (~/tmp/nr-classic-scratch/venv, built per the nr-classic lane brief).
nr-golden crate python="$HOME/tmp/nr-classic-scratch/venv/bin/python":
    {{python}} -I crates/{{crate}}/tests/golden_{{crate}}.py > crates/{{crate}}/tests/goldens_{{crate}}.tsv

# Score all 1,008 KonJND pairs for one NR metric (PairedNoReference: `<m>` on
# the distorted image, `<m>_ref` on the reference) into a TSV for the JND
# estimate. `bin` is a zenmetrics-cli binary built with `--features nr`.
nr-konjnd bin metric out pairs="$HOME/tmp/konjnd-pairs/pairs_abs.tsv":
    {{bin}} batch --metric {{metric}} --pairs '{{pairs}}' --output '{{out}}' --use-legacy-scheduler

# Crate tests (behaviour + oracle parity) in the pinned lock snapshot.
nr-test crate:
    {{just_executable()}} snap-cargo @ test -p {{crate}}

# AIC LDR-FLIP raw-map gate. Run under run-heavy with the shared heavy lock.
# python is the isolated venv containing flip-evaluator==1.7, numpy and Pillow.
aic-flip-verify python out:
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_flip_2026-10-15/flip_probe.rs -o "$HOME/tmp/aic-variants-scratch/flip-probe"
    {{python}} benchmarks/aic_flip_2026-10-15/verify_flip.py aic {{out}}/flip-aic.json
    {{python}} benchmarks/aic_flip_2026-10-15/verify_flip.py ladder {{out}}/flip-ladder.json

# Round 3 library, complete workspace fmt, CI lint flags, and CLI inventory.
aic-flip-ci:
    bash benchmarks/aic_flip_2026-10-15/check_ci.sh

# Independent complex steerable-pyramid convention search; pyrtools==1.0.11.
aic-cw-search python out:
    {{python}} benchmarks/aic_flip_2026-10-15/search_cw.py {{out}}/cw-search.json
    {{python}} benchmarks/aic_flip_2026-10-15/cw_variants.py {{out}}/cw-variants.json S01_AVIF_01.png S15_PNG_15.png S53_JAIB_07.png S66_AVIF_15.png
    {{python}} benchmarks/aic_flip_2026-10-15/cw_grid_panel.py {{out}}/cw-grid-panel.json

# Selected CW-SSIM raw RGB / independent Python gate (true DFT-bin grid).
aic-cw-verify python out:
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_flip_2026-10-15/cw_probe.rs -o "$HOME/tmp/aic-variants-scratch/cw-probe"
    {{python}} benchmarks/aic_flip_2026-10-15/verify_cw_explicit.py {{out}}/cw-final.json fftfreq

# BRISQUE Round 6 review; caller holds shared heavy lock/run-heavy scope.
brisque-review-build:
    just snap-cargo @ build -p brisque --example review --release

brisque-review-prepare python:
    {{python}} benchmarks/brisque_review_2026-10-15/prepare.py

brisque-review-run binary label:
    {{binary}} ratios "$HOME/tmp/aic-variants-scratch/round6/ratios-input.bin" "$HOME/tmp/aic-variants-scratch/round6/{{label}}-ratios.bin"
    {{binary}} panel "$HOME/tmp/aic-variants-scratch/round6/panel.paths" > "$HOME/tmp/aic-variants-scratch/round6/{{label}}-panel.tsv"

brisque-review-analyze python label *flags:
    {{python}} benchmarks/brisque_review_2026-10-15/analyze.py {{label}} {{flags}}

# Final BRISQUE review gates, matching CI's all-member fmt and Lint flags.
brisque-review-ci:
    #!/usr/bin/env bash
    set -euo pipefail
    just snap-cargo @ test -p brisque --all-targets
    just snap-cargo @ build -p brisque --no-default-features
    just snap-cargo @ test -p brisque --no-default-features --lib
    bash scripts/ci/lock.sh --check
    mapfile -t brisque_fmt_packages < <(cd "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/work/zenmetrics" && cargo metadata --locked --no-deps --format-version 1 | python3 -c 'import json,sys; [print(p["name"]) for p in json.load(sys.stdin)["packages"]]')
    brisque_fmt_args=()
    for package in "${brisque_fmt_packages[@]}"; do brisque_fmt_args+=(-p "$package"); done
    just snap-cargo @ fmt "${brisque_fmt_args[@]}" --check
    just snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items

brisque-review-params python:
    {{python}} crates/brisque/tests/gen_params.py

# Run after an ordinary review build/export; changes only the isolated snapshot.
brisque-review-computed-build:
    just brisque-review-build
    python3 benchmarks/brisque_review_2026-10-15/build_computed.py

# Export qualification gates without loading model weights.
nn-export-checks:
    python3 -I crates/zenmetrics-nn/models/test_export_checks.py

# Review gates in the CI-pinned sibling snapshot (caller uses run-heavy).
nn-review-tests:
    just snap-cargo @ test -p zenmetrics-nn
    just snap-cargo @ test --release -p zenmetrics-nn

nn-review-format:
    bash scripts/ci/lock.sh --check
    cd "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/work/zenmetrics" && cargo fmt --check $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')

nn-review-lint:
    just snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items

# Round-10 review: caller serializes these through run-heavy/flock.
nn-opt2-review-build rev out:
    bash benchmarks/nn_opt2_2026-10-15/review_build.sh '{{rev}}' '{{out}}'

nn-opt2-review-tests:
    just snap-cargo @ test -p zenmetrics-nn
    just snap-cargo @ test --release -p zenmetrics-nn

nn-opt2-review-run stage binaries models ladder out:
    python3 benchmarks/nn_opt2_2026-10-15/review_run.py '{{stage}}' '{{binaries}}' '{{models}}' '{{ladder}}' '{{out}}'

nn-opt2-review-ci:
    bash benchmarks/nn_opt2_2026-10-15/review_ci.sh

# PaQ-2-PiQ review: run under the shared heavy-job lock and run-heavy.
nn-review-paq2piq-gates:
    bash benchmarks/paq2piq_review_2026-10-16.sh

nn-paq2piq-small-inputs python ladder nonsquare:
    {{python}} -I crates/zenmetrics-nn/models/paq2piq/small_inputs.py '{{ladder}}' '{{nonsquare}}'

# Default scheduler, explicitly excluding any inherited legacy override.
nn-paq2piq-cli-smoke binary models ladder:
    python3 -I crates/zenmetrics-nn/models/paq2piq/cli_smoke.py '{{binary}}' '{{models}}' '{{ladder}}'

# Official PieAPP release is executed only as a documented black-box CLI.
nn-pieapp-oracle python upstream reference distorted log sampling="dense":
    python3 -I crates/zenmetrics-nn/models/pieapp/oracle_cli.py --python '{{python}}' --upstream '{{upstream}}' --reference '{{reference}}' --distorted '{{distorted}}' --log '{{log}}' --sampling '{{sampling}}'

# Paper/tensor-metadata reconstruction candidates; no official source imports.
nn-pieapp-probe python checkpoint reference distorted:
    {{python}} -I crates/zenmetrics-nn/models/pieapp/probe.py --checkpoint '{{checkpoint}}' --reference '{{reference}}' --distorted '{{distorted}}'

# Score a TSV using explicitly accepted NC PieAPP weights; verify repeatability.
nn-pieapp-score models pairs out engine="rten":
    just snap-cargo @ run --release -p zenmetrics-nn --features ort --example pieapp_score -- '{{models}}' '{{pairs}}' '{{out}}' '{{engine}}'

# Complete PieAPP landing checks; invoke under run-heavy and the shared lock.
nn-pieapp-gates models ladder:
    bash -c "$(cat benchmarks/pieapp_gates_2026-10-16.sh)" -- '{{models}}' '{{ladder}}'

nn-pieapp-cli-smoke binary models ladder:
    python3 -I crates/zenmetrics-nn/models/pieapp/cli_smoke.py '{{binary}}' '{{models}}' '{{ladder}}'

nn-pieapp-core-check:
    just snap-cargo @ test -p zenmetrics-nn
    just snap-cargo @ clippy -p zenmetrics-nn --all-targets --features ort,model-tests -- -D warnings

# Fixed convention qualified through the public oracle.
nn-pieapp-export python checkpoint out:
    {{python}} -I crates/zenmetrics-nn/models/pieapp/export.py --checkpoint '{{checkpoint}}' --out '{{out}}'

# The release's public patch-error/weight entry point, used only as an opaque oracle.
nn-pieapp-blackbox-probe python upstream checkpoint reference distorted:
    {{python}} crates/zenmetrics-nn/models/pieapp/blackbox_probe.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --reference '{{reference}}' --distorted '{{distorted}}'

# Opaque official entry-point oracle; caller holds the heavy-job lock.
nn-pieapp-entry-oracle python pairs out device:
    {{python}} crates/zenmetrics-nn/models/pieapp/oracle.py --upstream "$HOME/tmp/pieapp-oracle/upstream" --checkpoint "$HOME/tmp/pieapp-oracle/upstream/weights/PieAPPv0.1.pth" --pairs {{pairs}} --out {{out}} --device {{device}}

nn-pieapp-calibration-summary pairs scores out:
    python3 crates/zenmetrics-nn/models/pieapp/summarize.py '{{pairs}}' '{{scores}}' '{{out}}'

nn-pieapp-goldens pairs oracle out:
    python3 crates/zenmetrics-nn/models/pieapp/goldens.py '{{pairs}}' '{{oracle}}' '{{out}}'

# WD reference work: caller wraps in run-heavy and the shared heavy-job lock.
nn-wd-reference stage python reference out ladder=(env("HOME") / "tmp/nn-phase0/ladder") nonsquare=(env("HOME") / "tmp/nn-phase0/ladder-nonsquare"):
    {{python}} -I crates/zenmetrics-nn/models/wd/reference.py '{{stage}}' --reference '{{reference}}' --out '{{out}}' --ladder '{{ladder}}' --ladder '{{nonsquare}}'

nn-wd-score models pairs out engine="rten":
    {{just_executable()}} snap-cargo @ run --release -p zenmetrics-nn --features ort --example wd_score -- '{{models}}' '{{pairs}}' '{{out}}' '{{engine}}'

nn-wd-parity models ladder engine="rten":
    ZENMETRICS_NN_MODEL_DIR='{{models}}' ZENMETRICS_NN_LADDER='{{ladder}}' ZENMETRICS_NN_ENGINE='{{engine}}' {{just_executable()}} snap-cargo @ test --release -p zenmetrics-nn --features model-tests,ort --test wd_parity -- --nocapture --test-threads=1

nn-wd-kon-jnd scores out:
    python3 -I crates/zenmetrics-nn/models/wd/konjnd.py '{{scores}}' '{{out}}'

nn-wd-cli-smoke binary models ladder:
    python3 -I crates/zenmetrics-nn/models/wd/cli_smoke.py '{{binary}}' '{{models}}' '{{ladder}}'

nn-wd-gates models ladder:
    bash benchmarks/wd_port_gates_2026-10-16.sh '{{models}}' '{{ladder}}'

# AIC probe has its own hard one-hour clock after acquiring the heavy lock.
nn-wd-aic-probe models python reference site images config previous pairs out:
    bash benchmarks/wd_aic_probe_2026-10-16.sh '{{models}}' '{{python}}' '{{reference}}' '{{site}}' '{{images}}' '{{config}}' '{{previous}}' '{{pairs}}' '{{out}}'

# Compact an existing WD convention probe; no inference.
nn-wd-aic-report probe native site config out:
    python3 -I crates/zenmetrics-nn/models/wd/aic_report.py --probe '{{probe}}' --native '{{native}}' --site '{{site}}' --config '{{config}}' --out '{{out}}'

# Round 4 oracle panels; caller holds shared heavy lock and run-heavy scope.
aic-more-verify python:
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/butter_original_probe.rs -o "$HOME/tmp/aic-variants-scratch/round4/butter-original-probe"
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ssimulacra1_probe.rs -o "$HOME/tmp/aic-variants-scratch/round4/ssim1-probe"
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ws_hv_probe.rs -o "$HOME/tmp/aic-variants-scratch/round4/ws-probe"
    {{python}} benchmarks/aic_more_2026-10-15/verify_butter.py
    {{python}} benchmarks/aic_more_2026-10-15/verify_ssim1.py
    {{python}} benchmarks/aic_more_2026-10-15/verify_ws.py
    {{python}} benchmarks/aic_more_2026-10-15/cambi_oracle.py
    {{python}} benchmarks/aic_more_2026-10-15/verify_cambi.py

aic-more-ci:
    bash benchmarks/aic_more_2026-10-15/check_ci.sh

# Round 5 review controls; run under shared heavy lock and run-heavy.
aic-review-bogus:
    just snap-cargo @ test -p zenmetrics-cli --features sweep --bin zenmetrics fail_on_bogus_tests

aic-review-pins python:
    {{python}} benchmarks/aic_review_2026-10-15/pin_oracles.py

aic-review-libs:
    just snap-cargo @ test -p zenmetrics-cli --lib
    just snap-cargo @ test -p vmaf --no-default-features --lib
    just snap-cargo @ test -p vmaf --features simd,avx512 --lib

aic-review-ssim-order python:
    rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ssimulacra1_probe.rs -o "$HOME/tmp/aic-variants-scratch/round5/ssim1-row-block-probe"
    AIC_RESULTS_DIR="$HOME/tmp/aic-variants-scratch/round5" AIC_SSIM1_PROBE="$HOME/tmp/aic-variants-scratch/round5/ssim1-row-block-probe" {{python}} benchmarks/aic_more_2026-10-15/verify_ssim1.py

aic-review-rust-pins:
    rustc --edition 2024 -O benchmarks/aic_review_2026-10-15/pins_rust.rs -o "$HOME/tmp/aic-variants-scratch/round5/pins-rust"
    "$HOME/tmp/aic-variants-scratch/round5/pins-rust"

aic-review-cambi python:
    cp "$LOCK_SNAP_DIR/target/release/examples/aic_more_probe" "$HOME/tmp/aic-variants-scratch/round5/aic-more-probe"
    AIC_RESULTS_DIR="$HOME/tmp/aic-variants-scratch/round5" AIC_CAMBI_PROBE="$HOME/tmp/aic-variants-scratch/round5/aic-more-probe" {{python}} benchmarks/aic_more_2026-10-15/verify_cambi.py

aic-review-oracles python:
    bash benchmarks/aic_review_2026-10-15/check_oracles.sh {{python}}

aic-review-ci:
    bash benchmarks/aic_review_2026-10-15/check_ci.sh

aic-review-pin-sensitivity:
    python3 benchmarks/aic_review_2026-10-15/check_pin_sensitivity.py

aic-review-cambi-reference python:
    AIC_RESULTS_DIR="$HOME/tmp/aic-variants-scratch/round5/clean-cambi-reference" {{python}} benchmarks/aic_more_2026-10-15/cambi_oracle.py

# Label-free indexed planner/executor checks, including an independent PNG decoder.
palette-check:
    PYTHONPATH=scripts/jobsys python3 -m unittest scripts/jobsys/test_palette_plan.py
    just snap-cargo @ test -p zenmetrics-cli --no-default-features --features png,sweep,cpu-metrics --bin zenmetrics palette_ladder
    just snap-cargo @ test -p zenmetrics-cli --no-default-features --features png,sweep,cpu-metrics,palette-joint --bin zenmetrics palette_ladder
    just snap-cargo @ clippy -p zenmetrics-cli --no-default-features --features png,sweep,cpu-metrics,palette-joint --bin zenmetrics -- -D warnings

budget-preflight-check:
    PYTHONPATH=scripts/jobsys python3 -m unittest scripts/jobsys/test_budget_preflight.py

registered-fit-check:
    TMPDIR="$HOME/tmp" PYTHONPATH=scripts/jobsys python3 -m unittest scripts/jobsys/test_registered_program.py scripts/jobsys/test_registered_fit.py scripts/jobsys/test_fit_tools.py scripts/jobsys/test_strict_fit_harvest.py scripts/jobsys/test_e33_contract.py scripts/jobsys/test_e33_wall_cap.py

# Invented local-smoke transport and existing fit/harvest negative controls.
steer-smoke-transport-check:
    python3 -m unittest discover -s scripts/jobsys -p 'test_steer_smoke_transport.py'
    python3 -m unittest discover -s scripts/jobsys -p 'test_fit_tools.py'

# Operator wrappers retain the canonical packer/launcher and caller-supplied caps.
fit-pack-bounded-smoke source bin_dir build_meta pins contract out:
    python3 scripts/jobsys/pack_fit_program.py --source "{{source}}" --executor scripts/jobsys/fit_cell_exec.py --bin-dir "{{bin_dir}}" --build-meta "{{build_meta}}" --profile bounded-smoke --source-pins "{{pins}}" --source-pins-sha256 "$(sha256sum '{{pins}}' | cut -d' ' -f1)" --data "benchmarks/e33_fit_contract_2026-10-10.json={{contract}}" --out "{{out}}"

lan-fit-smoke-launch host run role image:
    bash scripts/jobsys/lan_score_launch.sh "{{host}}" "{{run}}" "{{role}}" cpu "{{image}}"

# IQA competition harness: every heavy stage runs under the caller's resource lock.
iqa-compete-build mode="cached":
    bash benchmarks/iqa_compete_2026-10-16/build_cpu.sh '{{mode}}'

iqa-compete-prepare binary reference distorted output:
    '{{binary}}' prepare '{{reference}}' '{{distorted}}' '{{output}}'

iqa-compete-run binary pairs output sizes="cid22,2k" threads="1,20":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --sizes '{{sizes}}' --threads '{{threads}}'

iqa-compete-charts raw output python=(env_var_or_default("IQA_COMPETE_PYTHON", env_var("HOME") + "/tmp/iqa-compete/python-venv/bin/python")):
    '{{python}}' benchmarks/iqa_compete_2026-10-16/charts.py '{{raw}}' --output '{{output}}'

iqa-compete-check-charts raw directory rows svgs label="":
    python3 benchmarks/iqa_compete_2026-10-16/check_charts.py '{{raw}}' '{{directory}}' --expected-rows '{{rows}}' --expected-svgs '{{svgs}}' --bar-label '{{label}}'

iqa-compete-smoke binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --metrics psnr,ssim2,gmsd,fsim --threads 1,20 --sizes cid22,2k

# PieAPP uses its own slow-cell lane; both lanes contribute to the full inventory.
iqa-compete-cpu-i265 binary pairs output models model_map sizes="cid22,2k" threads="1,20" batch="1100":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --sizes '{{sizes}}' --threads '{{threads}}' --model-root '{{models}}' --models-json '{{model_map}}' --exclude-metrics pieapp --batch-seconds '{{batch}}'

iqa-compete-libvmaf-build source output:
    bash benchmarks/iqa_compete_2026-10-16/build_libvmaf.sh '{{source}}' '{{output}}'

iqa-compete-libjxl-fetch output:
    python3 benchmarks/iqa_compete_2026-10-16/fetch_libjxl.py '{{output}}'

iqa-compete-libjxl-build source output:
    bash benchmarks/iqa_compete_2026-10-16/build_libjxl.sh '{{source}}' '{{output}}'

iqa-compete-model-parity reference candidate pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/check_model_adapter.py '{{reference}}' '{{candidate}}' '{{pairs}}' '{{output}}'

iqa-compete-planes binary pairs output:
    '{{binary}}' prepare-planes '{{pairs}}' '{{output}}'

iqa-compete-native engine binary rust pairs planes output sizes="cid22,2k":
    python3 benchmarks/iqa_compete_2026-10-16/native_run.py '{{engine}}' '{{binary}}' '{{rust}}' '{{pairs}}' '{{planes}}' '{{output}}' --sizes '{{sizes}}'

iqa-compete-native-probe vmaf jxl rust pairs planes output:
    python3 benchmarks/iqa_compete_2026-10-16/native_probe.py '{{vmaf}}' '{{jxl}}' '{{rust}}' '{{pairs}}' '{{planes}}' '{{output}}'

# Round 13b public-oracle controls; individual steps use the heavy-job lock.
nn-pieapp-review-head python upstream checkpoint pairs out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_head_probe.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --pairs '{{pairs}}' --out '{{out}}'

nn-pieapp-review-cli python upstream pairs entry out logs:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_cli_panel.py --python '{{python}}' --upstream '{{upstream}}' --pairs '{{pairs}}' --entry_scores '{{entry}}' --out '{{out}}' --logs '{{logs}}'

# Run outside the lock: the orchestrator acquires/releases it for each step.
nn-pieapp-review models ladder output:
    bash -c "$(cat benchmarks/pieapp_review_2026-10-16.sh)" -- '{{models}}' '{{ladder}}' '{{output}}'

nn-pieapp-review-summary directory log out:
    python3 -I crates/zenmetrics-nn/models/pieapp/review_summary.py '{{directory}}' '{{log}}' '{{out}}'

nn-pieapp-review-shards models out:
    python3 -I crates/zenmetrics-nn/models/pieapp/review_shards.py '{{models}}' '{{out}}'

nn-pieapp-review-step phase models ladder shards output:
    bash -c "$(cat benchmarks/pieapp_review_step_2026-10-16.sh)" -- '{{phase}}' '{{models}}' '{{ladder}}' '{{shards}}' '{{output}}'

nn-pieapp-review-weight-sign python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_sign.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}'

nn-pieapp-review-weight-joint python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_sign.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}' --joint --steps 500

# Public-tensor diagnostic outside RGB code-value range; never native image parity.
nn-pieapp-review-weight-diagnostic python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_sign.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}' --joint --steps 1000 --lower -4096 --upper 4096 --rate 32

nn-pieapp-review-weight-control python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_control.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}'

nn-pieapp-review-weight-identify python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_identify.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}'

nn-pieapp-review-reexport oldmodels models output:
    bash -c "$(cat benchmarks/pieapp_reexport_2026-10-16.sh)" -- '{{oldmodels}}' '{{models}}' '{{output}}'

nn-pieapp-review-konjnd-shards pairs output:
    python3 -I crates/zenmetrics-nn/models/pieapp/review_konjnd_shards.py '{{pairs}}' '{{output}}'

nn-pieapp-review-konjnd-compare baseline directory output:
    python3 -I crates/zenmetrics-nn/models/pieapp/review_konjnd_compare.py '{{baseline}}' '{{directory}}' '{{output}}'

# Syntax-check independent probes without importing or executing an oracle.
nn-pieapp-review-python-check:
    python3 -I -m compileall -q crates/zenmetrics-nn/models/pieapp

iqa-compete-collect output +ledgers:
    python3 benchmarks/iqa_compete_2026-10-16/collect.py {{ledgers}} --output '{{output}}'

iqa-compete-cuda-i265 binary pairs output models model_map reference qualified mode="nn" sizes="cid22,2k":
    bash benchmarks/iqa_compete_2026-10-16/run_cuda.sh '{{binary}}' '{{pairs}}' '{{output}}' '{{models}}' '{{model_map}}' '{{sizes}}' '{{mode}}' '{{reference}}' '{{qualified}}'

iqa-compete-clippy:
    bash benchmarks/iqa_compete_2026-10-16/build_cpu.sh cached clippy

iqa-compete-model-manifest default overrides output:
    python3 benchmarks/iqa_compete_2026-10-16/model_manifest.py '{{default}}' '{{overrides}}' '{{output}}'

iqa-compete-python-setup output:
    bash benchmarks/iqa_compete_2026-10-16/python_setup.sh '{{output}}'

iqa-compete-python package python binary pairs output reference qualified sizes="cid22,2k" threads="1,20":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --threads '{{threads}}' --sizes '{{sizes}}' --python-package '{{package}}' --python-executable '{{python}}' --reference-jsonl '{{reference}}' --reference-jsonl '{{qualified}}'

iqa-compete-chunk-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_chunks.py

iqa-compete-python-preflight python:
    '{{python}}' benchmarks/iqa_compete_2026-10-16/python_preflight.py

# Repair only the owned competitor environment; existing oracle sites are read-only.
iqa-compete-python-deps python:
    uv pip install --python '{{python}}' --no-deps ffmpeg-python==0.2.0 matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 fonttools==4.67.0 kiwisolver==1.5.1 pyparsing==3.3.3 python-dateutil==2.9.0.post0 six==1.17.0 PyEXR==0.5.0 OpenEXR==3.5.2

# Force multiple short holds while preserving the production ten-round criterion.
iqa-compete-chunk-e2e binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --metrics psnr --sizes cid22 --threads 1 --chunk-budget-ms 3000 --sample-target-ns 500000000
    python3 benchmarks/iqa_compete_2026-10-16/check_chunk_control.py '{{output}}'

# Linux-only reservation recovery uses actual process liveness and flock guards.
iqa-compete-linux-admission-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_admission_recovery.py

# One-second interruption control; production scopes still default to 1100 seconds.
iqa-compete-linux-timeout-control binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --metrics psnr --sizes cid22 --threads 1 --scope-seconds 1 --sample-target-ns 2000000000
    python3 benchmarks/iqa_compete_2026-10-16/check_timeout_control.py '{{output}}'

# Capability metadata is independent of a model's startup or scope outcome.
iqa-compete-capability-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_nomap_metadata.py

# Retry host admission refusals without accepting or discarding partial samples.
iqa-compete-admission-classification-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_admission_classification.py


# Shorter group budget reserves room for slow model warmup and batch overrun.
iqa-compete-pieapp-cpu binary pairs output models model_map sizes="cid22,2k" threads="1,20":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --sizes '{{sizes}}' --threads '{{threads}}' --model-root '{{models}}' --models-json '{{model_map}}' --metrics pieapp --chunk-budget-ms 600000

# Explicit metric subset for isolated admission repairs; preserves old ledgers.
iqa-compete-selected binary pairs output metrics sizes="cid22,2k" threads="1,20" batch="1100":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --metrics '{{metrics}}' --sizes '{{sizes}}' --threads '{{threads}}' --batch-seconds '{{batch}}'

iqa-compete-reference-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_references.py

iqa-compete-requalify raw inventory output +references:
    python3 benchmarks/iqa_compete_2026-10-16/requalify.py '{{raw}}' '{{inventory}}' --output '{{output}}' {{references}}

iqa-compete-parity-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_parity.py

iqa-compete-ort-thread-control python output:
    '{{python}}' benchmarks/iqa_compete_2026-10-16/check_ort_threads.py '{{output}}'

iqa-compete-ort-patch-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_ort_patch.py

# Scope the formatter to workspace packages; path-dependency repositories are separate.
iqa-compete-fmt-check:
    bash scripts/ci/lock.sh --check
    cd "$LOCK_SNAP_DIR/work/zenmetrics" && just fmt-check

iqa-compete-coverage raw inventory pairs output *options:
    python3 benchmarks/iqa_compete_2026-10-16/coverage.py '{{raw}}' '{{inventory}}' '{{pairs}}' '{{output}}' {{options}}

iqa-compete-coverage-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_coverage.py

iqa-compete-dispatch-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_dispatch_failure.py

iqa-compete-ort-thread-control-no-telemetry python output:
    '{{python}}' benchmarks/iqa_compete_2026-10-16/check_ort_threads.py '{{output}}' --disable-telemetry

iqa-compete-platform-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_collect_platforms.py

# Reuse qualified smoke/maps and retain every diagnostic attempt in its source ledger.
iqa-compete-collect-i265 output:
    python3 benchmarks/iqa_compete_2026-10-16/collect.py --output '{{output}}' "$HOME/tmp/iqa-compete/native-smoke/i265.jsonl" "$HOME/tmp/iqa-compete/smoke-placement/i265.jsonl" "$HOME/tmp/iqa-compete/cvvdp-smoke/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-first/i265.jsonl" "$HOME/tmp/iqa-compete/extra-map-smoke/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-qualified-reruns/i265.jsonl" "$HOME/tmp/iqa-compete/pieapp-first/i265.jsonl" "$HOME/tmp/iqa-compete/pieapp-cpu-bounded/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-admission-repairs/i265.jsonl" "$HOME/tmp/iqa-compete/gpu-extended/i265.jsonl" "$HOME/tmp/iqa-compete/nn-cuda-first/i265.jsonl" "$HOME/tmp/iqa-compete/nn-cuda-budgeted/i265.jsonl"

iqa-compete-batch-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_batch.py

# Individual cells keep their run-heavy/time limits; this controller owns bounded holds.
iqa-compete-batch-smoke binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --metrics psnr,gmsd --sizes cid22 --threads 1 --batch-seconds 1100

iqa-compete-controller-tests: iqa-compete-chunk-tests iqa-compete-linux-admission-tests iqa-compete-capability-tests iqa-compete-admission-classification-tests iqa-compete-reference-tests iqa-compete-parity-tests iqa-compete-ort-patch-tests iqa-compete-coverage-tests iqa-compete-dispatch-tests iqa-compete-platform-tests iqa-compete-batch-tests

# Learned CPU continuation shares prior observations; classical repairs have a separate lane.
iqa-compete-learned-cpu binary pairs output models model_map sizes="cid22,2k" threads="1,20":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --sizes '{{sizes}}' --threads '{{threads}}' --model-root '{{models}}' --models-json '{{model_map}}' --metrics learned --exclude-metrics pieapp --batch-seconds 1100

# Audit the owner-requested 113-name inventory, independent of any old build ledger.
iqa-compete-coverage-i265 raw pairs output sizes="cid22,2k":
    python3 benchmarks/iqa_compete_2026-10-16/coverage.py '{{raw}}' benchmarks/iqa_compete_2026-10-16/frozen-inventory.json '{{pairs}}' '{{output}}' --sizes '{{sizes}}' --gpu-runtime cuda --nn-cuda

# Freeze data outside git before extending a long-running multi-host matrix.
iqa-compete-freeze-models default overrides manifest output:
    python3 benchmarks/iqa_compete_2026-10-16/freeze_models.py '{{default}}' '{{overrides}}' '{{manifest}}' '{{output}}'

iqa-compete-freeze-model-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_freeze_models.py

# No metric execution: fresh exact-pair comparisons, immutable receipts and audit.
iqa-compete-refresh-i265 raw pairs output sizes="cid22,2k":
    python3 benchmarks/iqa_compete_2026-10-16/refresh.py '{{raw}}' '{{pairs}}' '{{output}}' --sizes '{{sizes}}' --gpu-runtime cuda --nn-cuda
