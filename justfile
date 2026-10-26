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

# Canonical by-construction runner transport, archive/receipt negative controls.
byc-fit-check:
    TMPDIR="$HOME/tmp" PYTHONPATH=scripts/jobsys python3 -m unittest scripts.jobsys.test_byc_fit scripts.jobsys.test_fit_tools scripts.jobsys.test_registered_program

byc-fit-data root score_root label out:
    python3 scripts/jobsys/pack_fit_data_v2.py --byc-root "{{root}}" --byc-score-root "{{score_root}}" --byc-label "{{label}}" --out "{{out}}"

# Frozen multi-GB root compression uses the shared local admission lock.
byc-fit-data-admitted root score_root label out:
    TMPDIR="$HOME/tmp" flock "$HOME/tmp/heavy.lock" "$HOME/work/zen/scripts/run-heavy" --mem 8G --jobs 1 -- just byc-fit-data "{{root}}" "{{score_root}}" "{{label}}" "{{out}}"

byc-fit-spec program data label folds seeds cells out *flags:
    python3 scripts/jobsys/build_fit_spec.py --grid byc --program-sha "{{program}}" --data-sha "{{data}}" --byc-label "{{label}}" --byc-folds "{{folds}}" --byc-seeds "{{seeds}}" --existing-root "{{cells}}" --out "{{out}}" {{flags}}

# Input staging is separate from declaration so replay controls can finish first.
byc-fit-stage data program run:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/lib/s3env.sh
    bash scripts/jobsys/stage_fit_inputs.sh "{{run}}" "{{data}}" "{{program}}"

byc-fit-enqueue ctl spec run data program out:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/lib/s3env.sh
    mkdir -p "{{out}}"
    "{{ctl}}" declare-fits --spec "{{spec}}" --out "{{out}}/manifest.json"
    "{{ctl}}" ids --manifest "{{out}}/manifest.json" > "{{out}}/ids.tsv"
    just byc-fit-stage "{{data}}" "{{program}}" "{{run}}"
    printf '%s\n' '{"paused":false,"drain":false}' > "{{out}}/control.json"
    aws --endpoint-url "$EP" s3 cp "{{out}}/manifest.json" "s3://zentrain/jobs/{{run}}/manifest.json" --only-show-errors
    aws --endpoint-url "$EP" s3 cp "{{out}}/control.json" "s3://zentrain/jobs/{{run}}/control.json" --only-show-errors

byc-fit-status ctl run:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/lib/s3env.sh
    "{{ctl}}" report --run "{{run}}" --endpoint "$EP"

# Publish the native first-wins DONE snapshot before starting additional claimants.
byc-fit-compact ctl run out:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/lib/s3env.sh
    "{{ctl}}" compact --run "{{run}}" --endpoint "$EP" --out-dir "{{out}}" --upload

byc-fit-coverage grid live_workers out +options:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/lib/s3env.sh
    export TMPDIR="$HOME/tmp"
    python3 scripts/jobsys/fit_grid_coverage.py --grid-manifest "{{grid}}" --live-workers "{{live_workers}}" --json-out "{{out}}" --endpoint "$EP" {{options}}

# Log/receipt progress only; uses the zensim owner and opens no target tables.
byc-fit-local-progress source study arm folds:
    BYC_CELLS="{{study}}/cells" python3 "{{source}}/scripts/rev4_featpot/byc_cells.py" progress --arm "{{arm}}" --folds "{{folds}}" | python3 -c 'import json,sys; d=json.load(sys.stdin); e=[r["latest_logged_epoch"] for r in d["cells"] if r["state"]=="training" and r["latest_logged_epoch"] is not None]; print(json.dumps(dict(arm=d["arm"],complete=d["complete"],requested=d["requested"],active_epoch_min=min(e) if e else None,active_epoch_max=max(e) if e else None)))'

# Relay the cell owner's progress reader from an already running worker.
byc-fit-worker-progress host role arm folds seeds="0,1,2":
    ssh "{{host}}" docker exec -e BYC_CELLS=/var/tmp/rev4-featpot/byc/cells "zen-score-{{role}}" python3 /opt/fleet-fits/program/scripts/rev4_featpot/byc_cells.py progress --arm "{{arm}}" --folds "{{folds}}" --seeds "{{seeds}}"

# Caller sets live-sized CPU/RAM caps and disk-backed host scratch explicitly.
byc-fit-lan-worker host run role image:
    bash scripts/jobsys/lan_score_launch.sh "{{host}}" "{{run}}" "{{role}}" cpu "{{image}}"

byc-fit-harvest ctl run out cells *flags:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/lib/s3env.sh
    "{{ctl}}" compact --run "{{run}}" --endpoint "$EP" --out-dir "{{out}}"
    stage=$(realpath -m "{{cells}}/../fleet-harvest/{{run}}")
    python3 scripts/jobsys/harvest_fit_cells.py --manifest "{{out}}/manifest.json" --ids "{{out}}/ids.tsv" --ledger "{{out}}/snap_{{run}}.parquet" --blobs-prefix "s3://zentrain/jobs/{{run}}/blobs" --endpoint "$EP" --scratch "$stage" --byc-install-root "{{cells}}" {{flags}}

# Native execution keeps canonical fit paths private; caller supplies run-heavy admission.
byc-fit-native namespace program scratch +command:
    bash scripts/jobsys/native_fit_namespace.sh "{{namespace}}" "{{program}}" "{{scratch}}" {{command}}

# A bounded native batch shares one hard cap and the owner's global heavy lock.
byc-fit-native-batch runtime:
    TMPDIR="$HOME/tmp" flock "$HOME/tmp/heavy.lock" "$HOME/work/zen/scripts/run-heavy" --mem 42G --jobs 12 -- bash scripts/jobsys/native_fit_batch.sh "{{runtime}}"

# Read aggregate cgroup accounting; single-process RSS cannot bound a fit tree.
byc-fit-native-memory runtime:
    python3 -c 'from pathlib import Path; import json; r=Path("{{runtime}}"); p=Path("/sys/fs/cgroup")/r.joinpath("scope.cgroup").read_text().strip().split("::",1)[1].lstrip("/"); print(json.dumps(dict(scope=str(p), **{k:(p/k).read_text().strip() for k in ("memory.current","memory.peak","memory.max","memory.swap.max","memory.events")})))'

# Add a bounded native launcher to the existing scope and share its original lock.
byc-fit-native-join runtime lock_pid +command:
    sudo -n python3 scripts/jobsys/join_native_fit_scope.py --runtime "{{runtime}}" --lock-pid "{{lock_pid}}" -- {{command}}

byc-fit-receipt cells manifest ids ledger out:
    python3 scripts/jobsys/byc_harvest_receipt.py --cells "{{cells}}" --manifest "{{manifest}}" --ids "{{ids}}" --ledger "{{ledger}}" --out "{{out}}"

# Caller supplies immutable source/binary pins; shared local lock serializes builds and packing.
byc-fit-prepare source bin_dir metadata pins columns worker out image control_data="" extra_arms="" shared_lock="1":
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR="$HOME/tmp"
    export ZEN_FIT_BUILDER=byc-cells-20261025
    export ZEN_BUILD_CACHE_REF=ghcr.io/imazen/zenfleet-worker:buildcache-fit-byc
    mkdir -p "$TMPDIR"
    admission=(flock "$HOME/tmp/heavy.lock")
    if [ "{{shared_lock}}" = 0 ]; then
      test -n "{{control_data}}" && test -z "{{extra_arms}}"
      admission=()
    fi
    "${admission[@]}" "$HOME/work/zen/scripts/run-heavy" --mem 8G --jobs 1 -- bash scripts/jobsys/prepare_byc_fit.sh "{{source}}" "{{bin_dir}}" "{{metadata}}" "{{pins}}" "{{columns}}" "{{worker}}" "{{out}}" "{{image}}" "{{control_data}}" "{{extra_arms}}"

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

# Metadata-only operator tool; scientific worker images remain unchanged.
stopped-claim-release-check:
    just snap-cargo @ test -p zenfleet-worker --example release_owned_claim
    just snap-cargo @ test -p zenfleet-worker --lib release
    just snap-cargo @ clippy -p zenfleet-worker --example release_owned_claim -- -D warnings

stopped-claim-release-build:
    just snap-cargo @ build -p zenfleet-worker --example release_owned_claim

# Run under the caller's heavy admission, including the shared native fit scope.
claim-clock-check-build:
    just snap-cargo @ test --release -p zenfleet-worker --lib
    just snap-cargo @ clippy --release -p zenfleet-worker --bin zenfleet-worker -- -D warnings
    just snap-cargo @ build --release -p zenfleet-worker --bin zenfleet-worker

# CLI/NN feature-combination matrix: 14 rows — every supported
# `--no-default-features --features <F>` shape, plus the bare
# `--no-default-features` build, the default build, and standalone NN WebGPU. Prints a PASS/FAIL
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
             gpu,gpu-cuda sweep,png,gpu,gpu-cuda gpu,gpu-wgpu nn-webgpu; do
        run_combo "$f" -p zenmetrics-cli --no-default-features --features "$f"
    done
    run_combo "zenmetrics-nn/ort-webgpu" -p zenmetrics-nn --features ort-webgpu
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

# Regenerate or verify the shared metric cards and README tables.
model-cards-refresh python:
    '{{python}}' -B scripts/dashboard/model_cards.py

model-cards-check python:
    '{{python}}' -B -m unittest discover -s scripts/dashboard -p test_model_cards.py
    '{{python}}' -B scripts/dashboard/model_cards.py --check

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

# Catch browser/CLI catalog drift before compiling WASM or starting Chromium.
browser-coverage-check:
    python3 scripts/checkpoint/test_browser_coverage.py

# Private model exports and native baseline are supplied explicitly by the caller.
browser-local-weights-check site models native report:
    python3 scripts/checkpoint/check-local-weights.py "{{site}}" "{{models}}" "{{native}}" "{{report}}"

# Benchmark selection controls preserve explicit opt-in methods.
bench-iqa-selection-check:
    python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_selection.py

# All cheap competition-controller controls.
bench-iqa-python-check:
    mkdir -p "$HOME/tmp"
    TMPDIR="$HOME/tmp" python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p 'test_*.py'

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

# Export AHIQ from a clone of IIGROUP/AHIQ (AHIQ_SRC, needs model/) plus its
# epoch-32 .pth from upstream's Google Drive link. patch_size 8 = upstream.
nn-export-ahiq python ladder out upstream ckpt patch_size="8":
    AHIQ_SRC='{{upstream}}' {{python}} -I crates/zenmetrics-nn/models/ahiq/export.py '{{ladder}}' '{{out}}' '{{ckpt}}' '{{patch_size}}'

nn-export-dmm python ladder out upstream:
    {{python}} -I crates/zenmetrics-nn/models/dmm/export.py '{{ladder}}' '{{out}}' '{{upstream}}'

# Parity of zenmetrics-nn against the exported goldens (CPU, rten). Runs in
# the pinned lock snapshot, so the paths are made absolute first.
# engine: rten | ort-cuda (ort-cuda needs ORT_DYLIB_PATH and the CUDA/cuDNN libs
# on the library path).
nn-parity model_dir ladder engine="rten" features="model-tests" models="lpips_alex,lpips_vgg,lpips_squeeze,stlpips_alex,stlpips_vgg,dists,deepdc,ckdn,afine,dreamsim,ahiq,arniqa,maniqa,liqe,musiq,hyperiqa,paq2piq,pieapp,wadiqam_fr_live,wadiqam_fr_tid,wadiqam_nr_live,wadiqam_nr_tid":
    ZENMETRICS_NN_MODEL_DIR="$(realpath '{{model_dir}}')" ZENMETRICS_NN_LADDER="$(realpath '{{ladder}}')" ZENMETRICS_NN_ENGINE='{{engine}}' ZENMETRICS_NN_MODELS='{{models}}' \
        {{just_executable()}} snap-cargo @ test --release -p zenmetrics-nn --features '{{features}}' --test parity -- --nocapture --test-threads=1

# DMM parity has its own lane: the metric's `wt` term consumes the
# orthonormal basis its SVD solver draws inside degenerate singular
# clusters. Historical CPU/CUDA observations differ; see
# crates/zenmetrics-nn/models/dmm/card.toml. This recipe gates the
# port against netlib-sgesdd-f32 goldens scored on the SAME engine's taps
# (models/dmm/dmm_goldens_netlib_<engine>.tsv via examples/dmm_tap_dump +
# models/dmm/gen_netlib_goldens.py); the MKL-draw dmm_goldens.tsv deltas
# are printed as a recorded measurement, not asserted.
nn-parity-dmm model_dir ladder engine="rten" features="model-tests" revision="@":
    ZENMETRICS_NN_MODEL_DIR="$(realpath '{{model_dir}}')" ZENMETRICS_NN_LADDER="$(realpath '{{ladder}}')" ZENMETRICS_NN_ENGINE='{{engine}}' ZENMETRICS_NN_MODELS='dmm' \
        {{just_executable()}} snap-cargo '{{revision}}' test --release -p zenmetrics-nn --features '{{features}}' --test parity -- --nocapture --test-threads=1

# Produces candidate floors in model_dir; reviewing/committing a replacement
# of models/dmm/dmm_goldens_netlib_*.tsv requires owner approval.
nn-dmm-netlib-goldens python model_dir ladder tapdir engine="rten" features="model-tests" approval="":
    test '{{approval}}' = 'owner-approved-floors'
    ZENMETRICS_NN_ENGINE='{{engine}}' {{just_executable()}} snap-cargo @ run --release -p zenmetrics-nn --features '{{features}}' --example dmm_tap_dump -- "$(realpath '{{model_dir}}')" "$(realpath '{{ladder}}')" "$(realpath -m '{{tapdir}}')/$(echo '{{engine}}' | tr - _)"
    '{{python}}' crates/zenmetrics-nn/models/dmm/gen_netlib_goldens.py '{{model_dir}}' '{{tapdir}}' "$(echo '{{engine}}' | tr - _)" --owner-approved-floors

# Diagnostic subset only; nn-parity-dmm always validates the full panel.
nn-dmm-probe model_dir ladder output rows engine="rten" features="model-tests" mode="scores-only":
    ZENMETRICS_NN_ENGINE='{{engine}}' {{just_executable()}} snap-cargo @ run --release -p zenmetrics-nn --features '{{features}}' --example dmm_tap_dump -- "$(realpath '{{model_dir}}')" "$(realpath '{{ladder}}')" "$(realpath -m '{{output}}')" '{{rows}}' '{{mode}}'

# Compare current small-row trunk inputs to the original oracle's tap dump.
# Every output directory must be fresh; mismatch/missing files fail explicitly.
nn-dmm-tap-identity model_dir ladder baseline output:
    #!/usr/bin/env bash
    set -euo pipefail
    {{just_executable()}} nn-dmm-probe '{{model_dir}}' '{{ladder}}' '{{output}}/rten' '1415:128' rten model-tests,ort taps
    {{just_executable()}} nn-dmm-probe '{{model_dir}}' '{{ladder}}' '{{output}}/ort_cuda' '1415:128' ort-cuda model-tests,ort taps
    for engine in rten ort_cuda; do
        for side in ref dist; do
            for tap in 0 1; do
                cmp '{{baseline}}'/$engine/1415_128_${side}_tap${tap}.npy '{{output}}'/$engine/1415_128_${side}_tap${tap}.npy
            done
        done
    done
    echo 'DMM tap identity: all eight NPY files are byte-identical'

# One immutable source, both native engines, all 48 frozen golden rows.
# Outer caller supplies the shared lock and a sub-20-minute run-heavy limit.
nn-dmm-bisect-panel revision output model_dir ladder:
    #!/usr/bin/env bash
    set -uo pipefail
    failed=0
    mkdir -p "$(dirname '{{output}}')"
    for label in rten ort-cuda ort-cuda-repeat; do
        engine=$label
        if [ "$label" = ort-cuda-repeat ]; then engine=ort-cuda; fi
        features=model-tests
        if [ "$engine" = ort-cuda ]; then features=model-tests,ort; fi
        if timeout --kill-after=10 900 {{just_executable()}} nn-parity-dmm '{{model_dir}}' '{{ladder}}' "$engine" "$features" '{{revision}}' > '{{output}}'-$label.log 2>&1; then
            echo "PANEL_PASS $label {{revision}}"
        else
            echo "PANEL_FAIL $label {{revision}}"
            failed=1
        fi
    done
    exit "$failed"

nn-dmm-bisect-all model_dir ladder output:
    bash benchmarks/nn_port_2026-10-15/dmm_bisect_run.sh '{{model_dir}}' '{{ladder}}' '{{output}}'

nn-dmm-bisect-summary directory output *flags:
    python3 benchmarks/nn_port_2026-10-15/dmm_bisect_summary.py '{{directory}}' '{{output}}' {{flags}}

# Read-only progress; completed score counts do not imply a passing gate.
nn-dmm-bisect-progress directory:
    #!/usr/bin/env bash
    set -euo pipefail
    while IFS=$'\t' read -r change revision; do
        [ "$change" = change ] && continue
        log='{{directory}}'/$change.runner.log
        echo "$change $revision"
        if [ -s "$log" ]; then
            rg '^(PANEL_|run-heavy: done)' "$log" || true
            for engine in rten ort-cuda ort-cuda-repeat; do
                panel='{{directory}}'/$change-$engine.log
                if [ -f "$panel" ]; then
                    count=$(rg -c 'dmm .*: got ' "$panel" || true)
                    echo "  $engine: ${count:-0}/24 score observations"
                fi
            done
        else
            echo '  not started'
        fi
    done < benchmarks/nn_port_2026-10-15/dmm_bisect_sources_2026-10-17.tsv

# Package explicitly selected completed measurements; preserve earlier archives.
nn-dmm-bisect-pack directory output cases:
    #!/usr/bin/env bash
    set -euo pipefail
    set -o noclobber
    files=()
    IFS=, read -ra selected <<< '{{cases}}'
    for change in "${selected[@]}"; do
        [[ "$change" =~ ^[a-z][a-z0-9_]*$ ]]
        rg -q "^$change"$'\t' benchmarks/nn_port_2026-10-15/dmm_bisect_sources_2026-10-17.tsv
        rg -q '^run-heavy: done ' '{{directory}}'/$change.runner.log
        files+=("$change.runner.log")
        for engine in rten ort-cuda ort-cuda-repeat; do
            rg -q '^Pinned rows eligible for promotion:' '{{directory}}'/$change-$engine.summary
            for suffix in log tsv summary; do
                test -s '{{directory}}'/$change-$engine.$suffix
                files+=("$change-$engine.$suffix")
            done
        done
    done
    tar -cz -C '{{directory}}' "${files[@]}" -C '{{justfile_directory()}}' \
        benchmarks/nn_port_2026-10-15/dmm_bisect_sources_2026-10-17.tsv \
        benchmarks/nn_port_2026-10-15/dmm_bisect_results_2026-10-17.md \
        benchmarks/nn_port_2026-10-15/dmm_bisect_isolation_2026-10-17.txt \
        benchmarks/nn_port_2026-10-15/dmm_oracle_info_2026-10-17.txt > '{{output}}'
    sha256sum '{{output}}'

nn-dmm-bisect-copy archive destination:
    rsync -a --ignore-existing -- '{{archive}}' '{{destination}}'

# Compare independent existing receipts without running inference again.
nn-dmm-baseline-gate-identity gate directory:
    #!/usr/bin/env python3
    import csv
    from pathlib import Path
    import re
    segments = re.split(r'(?m)^\+ G ', Path('{{gate}}').read_text())
    pattern = re.compile(r'dmm (\S+) (\S+): got (\S+) netlib ')
    for label, engine in [('rten', 'rten'), ('cuda', 'ort-cuda')]:
        selected = [part for part in segments if part.startswith(f'parity-dmm-{label} ')]
        assert len(selected) == 1, (engine, 'ambiguous or missing gate segment')
        observations = pattern.findall(selected[0])
        assert len(observations) == 24, (engine, 'incomplete gate panel')
        actual = {(image, size): float(value) for image, size, value in observations}
        with (Path('{{directory}}') / f'baseline-{engine}.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        assert len(rows) == 24, (engine, 'incomplete baseline panel')
        expected = {(row['image'], row['size']): float(row['dmm']) for row in rows}
        assert len(actual) == len(expected) == 24
        assert actual.keys() == expected.keys()
        differences = [key for key in actual if actual[key].hex() != expected[key].hex()]
        assert not differences, (engine, 'independent baseline scores differ', differences)
        print(f'{engine}: independent full gate vs bisection baseline, 24/24 score bits identical')

nn-dmm-bisect-isolation:
    python3 benchmarks/nn_port_2026-10-15/dmm_bisect_isolation.py

nn-dmm-bisect-sources output:
    python3 benchmarks/nn_port_2026-10-15/dmm_bisect_sources.py '{{output}}'

nn-dmm-oracle-info python:
    '{{python}}' -I benchmarks/nn_port_2026-10-15/dmm_oracle_info.py

nn-dmm-fix-summary log model_dir engine output:
    python3 benchmarks/nn_port_2026-10-15/dmm_fix_summary.py '{{log}}' '{{model_dir}}/dmm_goldens.tsv' "crates/zenmetrics-nn/models/dmm/dmm_goldens_netlib_$(echo '{{engine}}' | tr - _).tsv" '{{output}}'

# Run under the shared heavy lock and run-heavy with an outer <20-minute
# deadline. Group short checks to avoid repeated admission waits.
nn-port-compile-gate:
    #!/usr/bin/env bash
    set -uo pipefail
    failed=0
    gate() {
        local name="$1"
        shift
        if "$@"; then echo "GATE_PASS $name"; else echo "GATE_FAIL $name"; failed=1; fi
    }
    gate lock bash scripts/ci/lock.sh --check
    # Dev sibling versions differ from CI pins. fmt-check's metadata command
    # can rewrite the dev lock; preserve its exact input before the next gate.
    format_check() {
        local saved result
        saved=$(mktemp "$TMPDIR/nn-port-fmt-lock.XXXXXXXXXX") || return
        cp Cargo.lock "$saved" || return
        {{just_executable()}} fmt-check
        result=$?
        if ! cmp -s "$saved" Cargo.lock; then
            cp "$saved" Cargo.lock || return
        fi
        return "$result"
    }
    gate fmt format_check
    gate ci-clippy {{just_executable()}} snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items
    gate cli-clippy {{just_executable()}} snap-cargo @ clippy --keep-going -p zenmetrics-cli --features sweep,orchestrator,nn,nr -- -D warnings
    gate cli-ort-clippy {{just_executable()}} snap-cargo @ clippy --keep-going -p zenmetrics-cli --features sweep,orchestrator,nn,nr,nn-ort -- -D warnings
    gate cli-webgpu-clippy {{just_executable()}} snap-cargo @ clippy --keep-going -p zenmetrics-cli --features sweep,orchestrator,nn,nr,nn-webgpu -- -D warnings
    gate nn-clippy {{just_executable()}} snap-cargo @ clippy -p zenmetrics-nn --all-targets --features model-tests,ort -- -D warnings
    gate nn-webgpu-clippy {{just_executable()}} snap-cargo @ clippy -p zenmetrics-nn --all-targets --features model-tests,ort-webgpu -- -D warnings
    gate browser {{just_executable()}} snap-cargo @ check -p zenmetrics-browser --all-targets
    gate nn-test timeout --kill-after=10 900 {{just_executable()}} snap-cargo @ test -p zenmetrics-nn
    gate browser-list {{just_executable()}} snap-cargo @ test -p zenmetrics-browser --lib listed_metrics_are_compiled_in
    gate benchmark-study python3 scripts/jobsys/test_benchmark_study.py
    gate python {{just_executable()}} nn-port-python-check
    gate dmm-source-isolation {{just_executable()}} nn-dmm-bisect-isolation
    gate matrix {{just_executable()}} cli-feature-matrix
    exit "$failed"

# Syntax-check portable Python probes without importing their model dependencies.
nn-port-python-check:
    PYTHONPYCACHEPREFIX="$HOME/tmp/nn-port-scratch/pycache" python3 -m py_compile benchmarks/nn_port_2026-10-15/dmm_bisect_isolation.py benchmarks/nn_port_2026-10-15/dmm_bisect_sources.py benchmarks/nn_port_2026-10-15/dmm_bisect_summary.py benchmarks/ahiq_aic_probe/ahiq_aic_probe.py benchmarks/nn_aic2026_crosscheck_2026-10-15/deepdc_variants.py benchmarks/nn_aic2026_crosscheck_2026-10-15/deepdc_variants2.py benchmarks/nn_runtime_shootout_2026-10-14/scripts/bench.py benchmarks/nn_runtime_shootout_2026-10-14/scripts/bench_gpu.py benchmarks/dmm_aic_probe/dmm_aic_probe.py benchmarks/nn_port_2026-10-15/dmm_fix_summary.py benchmarks/nn_port_2026-10-15/dmm_oracle_info.py crates/zenmetrics-nn/models/dmm/gen_netlib_goldens.py

# Advisory quality scan against pinned siblings, using the installed ignored kit.
# Run under the same shared lock and bounded run-heavy wrapper as compile gates.
nn-port-quality revision="@":
    #!/usr/bin/env bash
    set -euo pipefail
    {{just_executable()}} snap-cargo '{{revision}}' metadata --no-deps --format-version 1 > /dev/null
    export ZENUTILS="{{justfile_directory()}}/.quality-kit"
    export CARGO_TARGET_DIR="${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/target"
    cd "${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap}/work/zenmetrics"
    {{just_executable()}} quality --jobs "${CARGO_BUILD_JOBS:-16}"

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

# Historical combined qualification; use nn-pieapp-review for bounded steps.
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
nn-pieapp-entry-oracle python pairs out device upstream checkpoint:
    {{python}} crates/zenmetrics-nn/models/pieapp/oracle.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --pairs '{{pairs}}' --out '{{out}}' --device '{{device}}'

nn-pieapp-calibration-summary pairs scores out:
    python3 crates/zenmetrics-nn/models/pieapp/summarize.py '{{pairs}}' '{{scores}}' '{{out}}'

nn-pieapp-goldens pairs oracle out:
    python3 crates/zenmetrics-nn/models/pieapp/goldens.py '{{pairs}}' '{{oracle}}' '{{out}}'

# WD reference work: caller wraps in run-heavy and the shared heavy-job lock.
nn-wd-reference stage python reference out ladder=(env("HOME") / "tmp/nn-phase0/ladder") nonsquare=(env("HOME") / "tmp/nn-phase0/ladder-nonsquare"):
    {{python}} -I crates/zenmetrics-nn/models/wd/reference.py '{{stage}}' --reference '{{reference}}' --out '{{out}}' --ladder '{{ladder}}' --ladder '{{nonsquare}}'

nn-wd-score models pairs out engine="rten":
    {{just_executable()}} snap-cargo @ run --release -p zenmetrics-nn --features ort --example wd_score -- '{{models}}' '{{pairs}}' '{{out}}' '{{engine}}'

# A-FINE sub-score dump for the AIC-4 convention search
# (benchmarks/afine_aic_2026-10-16/). ort-cuda needs ORT_DYLIB_PATH +
# cuDNN 9.19 libs on LD_LIBRARY_PATH (see benchmarks receipt).
nn-afine-aic-dump models pairs out engine="rten" variants="":
    {{just_executable()}} snap-cargo @ run --release -p zenmetrics-nn --features model-tests,ort --example afine_aic_dump -- '{{models}}' '{{pairs}}' '{{out}}' '{{engine}}' '{{variants}}'

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
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/butter_original_probe.rs -o "${AIC_BUTTER_PROBE:-$HOME/tmp/aic-variants-scratch/round4/butter-original-probe}"
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ssimulacra1_probe.rs -o "${AIC_SSIM1_PROBE:-$HOME/tmp/aic-variants-scratch/round4/ssim1-probe}"
    TMPDIR="$HOME/tmp/aic-variants-scratch" rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ws_hv_probe.rs -o "${AIC_WS_PROBE:-$HOME/tmp/aic-variants-scratch/round4/ws-probe}"
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

# Round 14 preserved CPU baseline and profiles; caller owns heavy-job lock.
aic-perf-big-baseline output metrics="cambi butter":
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/baseline.sh)" -- '{{output}}' '{{metrics}}'

aic-perf-big-baseline-step phase output metric width height threads rev="@" label="base":
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/baseline_step.sh)" -- '{{phase}}' '{{output}}' '{{metric}}' '{{width}}' '{{height}}' '{{threads}}' '{{rev}}' '{{label}}'

aic-perf-big-parity base candidate output scratch:
    python3 benchmarks/aic_perf_big_2026-10-16/parity.py '{{base}}' '{{candidate}}' '{{output}}' '{{scratch}}'

# Call under a bounded shared heavy-job scope.
aic-perf-big-measure binary width height threads reps="5":
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/measure.sh)" -- '{{binary}}' '{{width}}' '{{height}}' '{{threads}}' '{{reps}}'

aic-perf-big-cambi-scout output:
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/cambi_scout.sh)" -- '{{output}}'

aic-perf-big-butter-scout output:
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/butter_scout.sh)" -- '{{output}}'

aic-perf-big-assembly binary output:
    python3 -I benchmarks/aic_perf_big_2026-10-16/assembly.py '{{binary}}' '{{output}}'

aic-perf-big-oracles output python="python3":
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/oracles.sh)" -- '{{output}}' '{{python}}'

aic-perf-big-interleave cambi_base butter_base candidate output metrics="cambi butter":
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/interleave.sh)" -- '{{cambi_base}}' '{{butter_base}}' '{{candidate}}' '{{output}}' '{{metrics}}'

aic-perf-big-interleave-step base candidate output width height threads:
    bash -c "$(cat benchmarks/aic_perf_big_2026-10-16/interleave_step.sh)" -- '{{base}}' '{{candidate}}' '{{output}}' '{{width}}' '{{height}}' '{{threads}}'

aic-perf-big-residuals parity pairs published config output:
    python3 -I benchmarks/aic_perf_big_2026-10-16/residuals.py '{{parity}}' '{{pairs}}' '{{published}}' '{{config}}' '{{output}}'

aic-perf-big-summary scratch output stage="final":
    python3 -I benchmarks/aic_perf_big_2026-10-16/summarize.py '{{scratch}}' '{{output}}' --stage '{{stage}}'

# Scorer-budget regression and refreshed standalone executable; caller owns heavy lock.
aic-perf-big-thread-budget output:
    just snap-cargo @ test -p vmaf --features parallel,simd,avx512 --lib
    just snap-cargo @ build --release -p vmaf --features parallel,simd,avx512 --example cambi_wall
    mkdir '{{output}}'
    cp "$LOCK_SNAP_DIR/target/release/examples/cambi_wall" '{{output}}/cambi-candidate'
    sha256sum '{{output}}/cambi-candidate'
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

# Extra documented probe arguments select joint/diagnostic controls.
# Out-of-range tensors are diagnostics, never native image parity.
nn-pieapp-review-weight-sign python upstream checkpoint out *args:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_sign.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}' {{args}}

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
# Label-free indexed planner/executor checks, including an independent PNG decoder.
palette-check:
    just palette-pool-check
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

fit-pack-steer source bin_dir build_meta pins contract out:
    python3 scripts/jobsys/pack_fit_program.py --source "{{source}}" --executor scripts/jobsys/fit_cell_exec.py --bin-dir "{{bin_dir}}" --build-meta "{{build_meta}}" --profile steer --source-pins "{{pins}}" --source-pins-sha256 "$(sha256sum '{{pins}}' | cut -d' ' -f1)" --data "benchmarks/e33_fit_contract_2026-10-10.json={{contract}}" --out "{{out}}"

lan-fit-smoke-launch host run role image:
    bash scripts/jobsys/lan_score_launch.sh "{{host}}" "{{run}}" "{{role}}" cpu "{{image}}"

# Generated-input effective-dispatch receipts for the reviewed quantizer API.
palette-dispatch-receipts out:
    ZEN_PALETTE_RECEIPT_DIR='{{out}}' just snap-cargo @ test -p zenmetrics-cli --no-default-features --features png,sweep,cpu-metrics --bin zenmetrics palette_ladder::tests::effective_dispatch_receipts_match_native_and_png_outputs -- --nocapture
    ZEN_PALETTE_RECEIPT_DIR='{{out}}' just snap-cargo @ test -p zenmetrics-cli --no-default-features --features png,sweep,cpu-metrics,palette-joint --bin zenmetrics palette_ladder::tests::effective_dispatch_receipts_match_native_and_png_outputs -- --nocapture

# Invented metadata only: sealed palette pool validation and declaration contracts.
palette-pool-check:
    PYTHONPATH=scripts/jobsys python3 -m unittest scripts/jobsys/test_palette_plan.py scripts/jobsys/test_palette_pool_authority.py scripts/jobsys/test_budget_preflight.py

# A source-closed E36 executor, built in the CI sibling-pin snapshot.
e36-native-build snapshot target:
    cd {{quote(snapshot)}} && cargo build --release --locked -p zenmetrics-cli --no-default-features --features png,jpeg,webp,avif,avif-svt,jxl,cpu-metrics,cpu-cvvdp,sweep,jobexec --target-dir {{quote(target)}} --message-format=json

e36-native-local-test snapshot target:
    cd {{quote(snapshot)}} && cargo test --release --locked -p zenmetrics-cli --no-default-features --features png,jpeg,webp,avif,avif-svt,jxl,cpu-metrics,cpu-cvvdp,sweep,jobexec --target-dir {{quote(target)}} --bin zenmetrics local_variant_requires_content_hash_and_remains_borrowed

# Static GitHub Pages benchmark dashboard; stdlib data adapters, no Cargo build.
dashboard-build output="target/dashboard":
    python3 scripts/dashboard/build.py --output {{quote(output)}}

dashboard-data-check:
    TMPDIR="$HOME/tmp" python3 -m unittest discover -s scripts/dashboard -p 'test_*.py'

dashboard-smoke output="target/dashboard":
    python3 scripts/dashboard/check.py {{quote(output)}}

dashboard-clean-check:
    TMPDIR="$HOME/tmp" python3 scripts/dashboard/clean_check.py

dashboard-plot-check:
    node scripts/dashboard/test-plots.mjs

dashboard-generated-check:
    python3 scripts/dashboard/catalog.py --check
    python3 scripts/dashboard/extract_markdown.py --check

dashboard-correlations source output="benchmarks/dashboard/quality":
    python3 scripts/dashboard/correlate.py --source {{quote(source)}} --output-dir {{quote(output)}} --draws 1000 --seed 20261017

# Private derived-data executor admission; no fitting or new public Rust API.
e36-tools-check source target:
    cd {{quote(source)}} && cargo test --locked --release -p zenfleet-ctl --bin zenfleet-ctl derived_declare --target-dir {{quote(target)}}
    cd {{quote(source)}} && cargo test --locked --release -p zenfleet-worker --lib e36_ladder_deadline_tests --target-dir {{quote(target)}}
    cd {{quote(source)}} && cargo clippy --locked --release -p zenfleet-ctl -p zenfleet-worker --bins --target-dir {{quote(target)}} -- -D warnings
    cd {{quote(source)}} && cargo build --locked --release -p zenfleet-ctl --bin zenfleet-ctl -p zenfleet-worker --bin zenfleet-worker --target-dir {{quote(target)}}

# Verify optional entrypoint hash/path admission without network or workers.
lan-entrypoint-test:
    python3 -m unittest discover -s scripts/jobsys -p test_lan_entrypoint_binding.py

dashboard-speed-coverage site="target/dashboard" output="target/dashboard-speed-coverage.md":
    python3 scripts/dashboard/coverage.py {{quote(site)}} --output {{quote(output)}}

# Metadata-only E36 mask/TV bindings and legacy checkpoint refusal checks.
e36-fit-contract-check:
    python3 -m unittest discover -s scripts/jobsys -p 'test_e36_fit_contract.py'
    just registered-fit-check
    python3 -m unittest discover -s scripts/jobsys -p 'test_e33_wall_cap.py'

iqa-compete-build mode="cached":
    bash benchmarks/iqa_compete_2026-10-16/build_cpu.sh '{{mode}}'

iqa-compete-prepare binary reference distorted output:
    '{{binary}}' prepare '{{reference}}' '{{distorted}}' '{{output}}'

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-run-supplemental binary pairs output sizes="cid22,2k" threads="1,20":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --sizes '{{sizes}}' --threads '{{threads}}'

iqa-compete-charts raw output python=(env_var_or_default("IQA_COMPETE_PYTHON", env_var("HOME") + "/tmp/iqa-compete/python-venv/bin/python")):
    '{{python}}' benchmarks/iqa_compete_2026-10-16/charts.py '{{raw}}' --output '{{output}}'

iqa-compete-check-charts raw directory rows svgs label="":
    python3 benchmarks/iqa_compete_2026-10-16/check_charts.py '{{raw}}' '{{directory}}' --expected-rows '{{rows}}' --expected-svgs '{{svgs}}' --bar-label '{{label}}'

iqa-compete-smoke binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --metrics psnr,ssim2,gmsd,fsim --threads 1,20 --sizes cid22,2k

# PieAPP uses its own slow-cell lane; both lanes contribute to the full inventory.
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-cpu-i265-supplemental binary pairs output models model_map sizes="cid22,2k" threads="1,20" batch="1100":
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

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-native-supplemental engine binary rust pairs planes output sizes="cid22,2k":
    python3 benchmarks/iqa_compete_2026-10-16/native_run.py '{{engine}}' '{{binary}}' '{{rust}}' '{{pairs}}' '{{planes}}' '{{output}}' --sizes '{{sizes}}'

iqa-compete-native-probe vmaf jxl rust pairs planes output:
    python3 benchmarks/iqa_compete_2026-10-16/native_probe.py '{{vmaf}}' '{{jxl}}' '{{rust}}' '{{pairs}}' '{{planes}}' '{{output}}'

# Round 13b public-oracle controls; individual steps use the heavy-job lock.
nn-pieapp-review-weight-joint python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_sign.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}' --joint --steps 500

# Public-tensor diagnostic outside RGB code-value range; never native image parity.
nn-pieapp-review-weight-diagnostic python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_sign.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}' --joint --steps 1000 --lower -4096 --upper 4096 --rate 32

nn-pieapp-review-weight-control python upstream checkpoint out:
    {{python}} crates/zenmetrics-nn/models/pieapp/review_weight_control.py --upstream '{{upstream}}' --checkpoint '{{checkpoint}}' --out '{{out}}'

iqa-compete-collect output +ledgers:
    python3 benchmarks/iqa_compete_2026-10-16/collect.py {{ledgers}} --output '{{output}}'

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-cuda-i265-supplemental binary pairs output models model_map reference qualified mode="nn" sizes="cid22,2k":
    bash benchmarks/iqa_compete_2026-10-16/run_cuda.sh '{{binary}}' '{{pairs}}' '{{output}}' '{{models}}' '{{model_map}}' '{{sizes}}' '{{mode}}' '{{reference}}' '{{qualified}}'

iqa-compete-clippy:
    bash benchmarks/iqa_compete_2026-10-16/build_cpu.sh cached clippy

iqa-compete-model-manifest default overrides output:
    python3 benchmarks/iqa_compete_2026-10-16/model_manifest.py '{{default}}' '{{overrides}}' '{{output}}'

iqa-compete-python-setup output:
    bash benchmarks/iqa_compete_2026-10-16/python_setup.sh '{{output}}'

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-python-supplemental package python binary pairs output reference qualified sizes="cid22,2k" threads="1,20":
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
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-pieapp-cpu-supplemental binary pairs output models model_map sizes="cid22,2k" threads="1,20":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --sizes '{{sizes}}' --threads '{{threads}}' --model-root '{{models}}' --models-json '{{model_map}}' --metrics pieapp --chunk-budget-ms 600000

# Explicit metric subset for isolated admission repairs; preserves old ledgers.
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-selected-supplemental binary pairs output metrics sizes="cid22,2k" threads="1,20" batch="1100":
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

iqa-compete-supervision-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_supervision.py

iqa-compete-host-progress raw inventory archive output threads *options:
    python3 benchmarks/iqa_compete_2026-10-16/host_progress.py '{{raw}}' '{{inventory}}' '{{archive}}' '{{output}}' --threads '{{threads}}' {{options}}

iqa-compete-cell-receipts raw output:
    python3 benchmarks/iqa_compete_2026-10-16/charts.py '{{raw}}' --output '{{output}}' --summaries-only

iqa-compete-copy-receipts host files output:
    mkdir -p '{{output}}'
    rsync -ar --files-from='{{files}}' -- '{{host}}:tmp/iqa-compete/task4/' '{{output}}/'

# Reuse qualified smoke/maps and retain every diagnostic attempt in its source ledger.
iqa-compete-collect-i265 output:
    python3 benchmarks/iqa_compete_2026-10-16/collect.py --output '{{output}}' "$HOME/tmp/iqa-compete/native-smoke/i265.jsonl" "$HOME/tmp/iqa-compete/smoke-placement/i265.jsonl" "$HOME/tmp/iqa-compete/cvvdp-smoke/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-first/i265.jsonl" "$HOME/tmp/iqa-compete/extra-map-smoke/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-qualified-reruns/i265.jsonl" "$HOME/tmp/iqa-compete/pieapp-first/i265.jsonl" "$HOME/tmp/iqa-compete/pieapp-cpu-bounded/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-admission-repairs/i265.jsonl" "$HOME/tmp/iqa-compete/gpu-extended/i265.jsonl" "$HOME/tmp/iqa-compete/nn-cuda-first/i265.jsonl" "$HOME/tmp/iqa-compete/nn-cuda-budgeted/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-accounted-v15/i265.jsonl" "$HOME/tmp/iqa-compete/cpu-map-repairs/i265.jsonl"

iqa-compete-batch-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_batch.py

# Individual cells keep their run-heavy/time limits; this controller owns bounded holds.
iqa-compete-batch-smoke binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --metrics psnr,gmsd --sizes cid22 --threads 1 --batch-seconds 1100

iqa-compete-controller-tests: iqa-compete-chunk-tests iqa-compete-linux-admission-tests iqa-compete-capability-tests iqa-compete-admission-classification-tests iqa-compete-reference-tests iqa-compete-parity-tests iqa-compete-ort-patch-tests iqa-compete-coverage-tests iqa-compete-dispatch-tests iqa-compete-platform-tests iqa-compete-batch-tests

# Learned CPU continuation shares prior observations; classical repairs have a separate lane.
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-learned-cpu-supplemental binary pairs output models model_map sizes="cid22,2k" threads="1,20":
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

iqa-compete-competitor-coverage raw pairs output sizes="cid22,2k" implementations="libvmaf,libjxl,scikit-image,piq,torchmetrics,pytorch-msssim,lpips,DISTS-pytorch,stlpips-pytorch,DeepDC-PyTorch,flip-evaluator,cvvdp":
    python3 benchmarks/iqa_compete_2026-10-16/competitor_coverage.py '{{raw}}' '{{pairs}}' '{{output}}' --sizes '{{sizes}}' --implementations '{{implementations}}'

iqa-compete-competitor-coverage-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_competitor_coverage.py

iqa-compete-luma-map-patch-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_luma_maps_patch.py

iqa-compete-check-map-repairs raw baseline output:
    python3 benchmarks/iqa_compete_2026-10-16/check_map_repairs.py '{{raw}}' '{{baseline}}' '{{output}}'

# Process accounting controls for the owner-authorized shared-host protocol.
iqa-compete-contention-tests:
    python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_contention.py

# Read-only retention of observation files referenced by the other host lane.
iqa-compete-retain-host-scopes raw output:
    python3 benchmarks/iqa_compete_2026-10-16/merge_host_scopes.py '{{raw}}' '{{output}}'

# Explicit immutable cohort export before appending newer continuation results.
iqa-compete-collect-binary raw output binary_sha:
    python3 benchmarks/iqa_compete_2026-10-16/collect.py '{{raw}}' --output '{{output}}' --only-binary-sha '{{binary_sha}}'

# Owner coverage pass: one pair per size, n=5/3/1, one bounded scope per cell.
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-coverage-rust-supplemental binary pairs output models model_map sizes="aic-small,aic-medium,aic-large,4k,8k":
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --sizes '{{sizes}}' --threads 1,all --model-root '{{models}}' --models-json '{{model_map}}' --coverage-first

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-coverage-native-supplemental engine binary rust pairs planes output reference:
    python3 benchmarks/iqa_compete_2026-10-16/native_run.py '{{engine}}' '{{binary}}' '{{rust}}' '{{pairs}}' '{{planes}}' '{{output}}' --sizes cid22,aic-small,aic-medium,aic-large,2k,4k,8k --coverage-first --reference-jsonl '{{reference}}'

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-coverage-python-supplemental package python binary pairs output reference:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --threads 1,all --sizes cid22,aic-small,aic-medium,aic-large,2k,4k,8k --python-package '{{package}}' --python-executable '{{python}}' --reference-jsonl '{{reference}}' --coverage-first

# Light controller orchestration; each child acquires the heavy lock per cell.
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-coverage-pass-supplemental binary pairs output models model_map ort cuda:
    python3 benchmarks/iqa_compete_2026-10-16/coverage_pass.py '{{binary}}' '{{pairs}}' '{{output}}' '{{models}}' '{{model_map}}' --ort-library '{{ort}}' --cuda-libraries '{{cuda}}'

iqa-compete-adopt-scope config log contention output:
    python3 benchmarks/iqa_compete_2026-10-16/adopt_scope.py '{{config}}' '{{log}}' '{{contention}}' '{{output}}'

# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-coverage-competitors-supplemental rust jxl pairs planes python flip_python reference vmaf_ledger output wait_pid *options:
    python3 benchmarks/iqa_compete_2026-10-16/coverage_competitors.py '{{rust}}' '{{jxl}}' '{{pairs}}' '{{planes}}' '{{python}}' '{{flip_python}}' '{{reference}}' '{{vmaf_ledger}}' '{{output}}' --wait-pid '{{wait_pid}}' {{options}}

iqa-compete-publish-binary snapshot log binary metadata:
    python3 benchmarks/iqa_compete_2026-10-16/publish_binary.py '{{snapshot}}' '{{log}}' '{{binary}}' '{{metadata}}'

iqa-compete-coverage-progress:
    python3 benchmarks/iqa_compete_2026-10-16/coverage_progress.py benchmarks/iqa_compete_2026-10-16/frozen-inventory.json --baseline '{{env_var("HOME")}}/tmp/iqa-compete/refresh-accounted-first2-v1/raw.jsonl' --baseline '{{env_var("HOME")}}/tmp/iqa-compete/pieapp-cpu-accounted/i265.jsonl' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-rust-v16' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-native-vmaf' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-competitors-v16' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-butter8k-isolated-v17'

iqa-compete-stop-idle-controller pid script root proof *options:
    python3 benchmarks/iqa_compete_2026-10-16/stop_idle_controller.py '{{pid}}' '{{script}}' '{{root}}' '{{proof}}' {{options}}

iqa-compete-resume-coverage manifest output label:
    python3 benchmarks/iqa_compete_2026-10-16/resume_coverage.py '{{manifest}}' '{{output}}' --label '{{label}}'

iqa-compete-coverage-snapshot output:
    python3 benchmarks/iqa_compete_2026-10-16/coverage_progress.py benchmarks/iqa_compete_2026-10-16/frozen-inventory.json --baseline '{{env_var("HOME")}}/tmp/iqa-compete/refresh-accounted-first2-v1/raw.jsonl' --baseline '{{env_var("HOME")}}/tmp/iqa-compete/pieapp-cpu-accounted/i265.jsonl' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-rust-v16' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-native-vmaf' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-competitors-v16' '{{env_var("HOME")}}/tmp/iqa-compete/coverage-butter8k-isolated-v17' --freeze-dir '{{output}}' --save '{{output}}/progress.json'

# Only missing 8K Butteraugli cells; avoid co-resident score/map scorers.
# Historical protocol; supplemental evidence only. Use bench-iqa for canonical CPU comparisons.
iqa-compete-butter8k-isolated-supplemental binary pairs output:
    python3 benchmarks/iqa_compete_2026-10-16/run.py '{{binary}}' '{{pairs}}' '{{output}}' --host i265 --metrics butteraugli-gpu --sizes 8k --threads 1,all --gpu-runtime cuda --coverage-first --isolate-modes --batch-seconds 1100 --batch-reserve-seconds 1000

iqa-compete-landing-gates output snapshot *options:
    python3 benchmarks/iqa_compete_2026-10-16/landing_gates.py '{{output}}' '{{snapshot}}' {{options}}

# Match deploy assembly without requiring committed derived copies to be current.
dashboard-regenerate:
    python3 scripts/dashboard/regenerate.py

# Canonical scalar CPU comparison: frozen RGB, cold3 + warm10 at 100ms, threads1/8.
# Requires explicit --binary/--pairs/--build-receipt/--source-commit/--bundle/--output/--lock/--run-heavy.
# Build beforehand under the host heavy lock; see docs/IQA_CANONICAL_COMMANDS.md.
bench-iqa sizes="cid22,aic-small,aic-medium,aic-large,2k,4k" metrics="all" *options:
    python3 benchmarks/iqa_compete_2026-10-16/canonical_bench.py '{{sizes}}' '{{metrics}}' {{options}}

# Current changed/added scope, including PSNR, FLIP, WS-HV and base HDR revisions.
bench-iqa-changed *options:
    just bench-iqa cid22,aic-small,aic-medium,aic-large,2k,4k changed {{options}}

# Compare Rust with explicit, hash-bound original/reference implementations.
bench-iqa-canonical plan output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/compare_canonical_budgeted.py '{{plan}}' '{{output}}' {{options}}

# Matched classical evaluator-first calls; the plan declares constructor/I/O scope.
bench-iqa-canonical-classical-first plan output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/compare_classical_first.py '{{plan}}' '{{output}}' {{options}}

# Same clocks and content checks; plans must also bind the successor controller.
bench-iqa-canonical-classical-first-verified plan output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/compare_classical_first_verified.py '{{plan}}' '{{output}}' {{options}}

# Caller supplies shared heavy admission; these controls execute no scorers.
bench-iqa-canonical-classical-first-verified-check:
    python3 -B benchmarks/iqa_compete_2026-10-16/test_classical_first_verified.py -v
    python3 -O -B benchmarks/iqa_compete_2026-10-16/test_classical_first_verified.py -v

# Caller admits setup/check/cell/finish; queue admits each cell itself.
bench-iqa-canonical-hdr-numerical +options:
    python3 -I -B benchmarks/iqa_compete_2026-10-16/hdr_original_grid_numerical.py {{options}}

# Caller supplies shared heavy admission and the frozen real-row fixtures.
bench-iqa-canonical-hdr-numerical-check retained_rows:
    HDR_NUMERICAL_RETAINED_ROWS='{{retained_rows}}' python3 -I -B benchmarks/iqa_compete_2026-10-16/test_hdr_original_grid_numerical.py -v
    HDR_NUMERICAL_RETAINED_ROWS='{{retained_rows}}' python3 -I -O -B benchmarks/iqa_compete_2026-10-16/test_hdr_original_grid_numerical.py -v

# Fixed-stage confidence interval for a resident-model first-call flag.
bench-iqa-canonical-first-confirm plan case output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/confirm_scored_first_evaluation.py '{{plan}}' '{{case}}' '{{output}}' {{options}}

# Caller supplies heavy admission; captures the complete native probe process.
bench-iqa-canonical-first-profile plan case output:
    python3 -B benchmarks/iqa_compete_2026-10-16/profile_first_evaluation.py '{{plan}}' '{{case}}' '{{output}}'

# Read finalized classical cold-call summaries; retain a new snapshot each time.
bench-iqa-canonical-classical-first-flags output regressions *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/capture_classical_first_flags.py '{{output}}' --regressions '{{regressions}}' {{options}}

# Caller supplies heavy admission; inert aggregate/case metadata controls.
bench-iqa-canonical-classical-first-flags-check:
    python3 -B benchmarks/iqa_compete_2026-10-16/test_classical_first_flags.py -v
    python3 -O -B benchmarks/iqa_compete_2026-10-16/test_classical_first_flags.py -v

# Caller supplies heavy admission for the fixed-stage bootstrap analysis.
bench-iqa-canonical-classical-first-interval results output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/classical_first_intervals.py '{{results}}' '{{output}}' {{options}}

# Caller supplies heavy admission; controls execute no metric subprocesses.
bench-iqa-canonical-classical-first-interval-check:
    python3 -I -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_classical_first_intervals.py -v
    python3 -I -O -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_classical_first_intervals.py -v

# Caller supplies shared heavy admission; complete cold-worker CPU stack capture.
bench-iqa-canonical-classical-first-profile plan case output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/profile_classical_first.py '{{plan}}' '{{case}}' '{{output}}' {{options}}

# Explicit CUDA scalar first calls; each block acquires shared admission.
bench-iqa-canonical-gpu-first plan output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/compare_gpu_first_verified.py '{{plan}}' '{{output}}' {{options}}

# Caller supplies admission; these inert controls launch no scorer or CUDA process.
bench-iqa-canonical-gpu-first-check:
    python3 -I -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_gpu_first.py -v
    python3 -I -O -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_gpu_first.py -v

bench-iqa-canonical-first-confirm-test:
    python3 -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_first_confirmation.py -v
    python3 -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_scored_first_confirmation.py -v

# Finish missing blocks only, after the original coordinator has stopped.
bench-iqa-canonical-resume output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/resume_canonical_comparison.py '{{output}}' {{options}}

bench-iqa-canonical-plan +options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_canonical_comparison.py {{options}}

# Extend an existing reference contract onto a hashed size/content grid.
bench-iqa-canonical-grid-plan source pairs output +options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_canonical_grid.py '{{source}}' '{{pairs}}' '{{output}}' {{options}}

# Preserve an independently reviewed timing scope in a new numerical grid.
bench-iqa-canonical-grid-scope-plan source pairs scoped output +options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_canonical_grid_scope.py '{{source}}' '{{pairs}}' '{{scoped}}' '{{output}}' {{options}}

bench-iqa-canonical-grid-scope-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_canonical_grid_scope

# Caller supplies heavy admission; original weighted HaarPSI maps and current frozen rlibs.
bench-iqa-canonical-haarpsi-maps python +options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/check_haarpsi_maps.py {{options}}

# Includes a bit-exact original-loader control before resident-parameter timing.
bench-iqa-canonical-niqe-resident python +options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/canonical_niqe_resident_cell.py {{options}}

bench-iqa-canonical-niqe-plan +options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_niqe_resident.py {{options}}

# Caller supplies heavy admission; no timed-code changes or new map tolerance.
bench-iqa-canonical-butteraugli-maps python +options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/check_butteraugli_maps.py {{options}}

# Freeze and run the manifest scope in ascending retained duration.
bench-iqa-canonical-changed manifest output *options:
    just bench-iqa-canonical-ordered-plan '{{manifest}}' '{{output}}.plan.json'
    just bench-iqa-canonical '{{output}}.plan.json' '{{output}}' {{options}}

# Compare the exact manifest scope; select thread budgets explicitly (default 1,8).
bench-iqa-canonical-compare manifest output threads="1,8" *options:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 1 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/prepare_scoped_canonical.py '{{manifest}}' '{{output}}.plan.json' --threads '{{threads}}'
    just bench-iqa-canonical '{{output}}.plan.json' '{{output}}' {{options}}

# Keep numerical gates and withhold timing ratios for explicitly caveated scopes.
bench-iqa-canonical-scoped-plan manifest output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_scoped_canonical.py '{{manifest}}' '{{output}}' {{options}}

bench-iqa-canonical-scoped-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_scoped_canonical

# One current native worker plus explicit original-reference plans, changed routes only.
bench-iqa-canonical-changed-plan manifest output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_ordered_canonical.py '{{manifest}}' '{{output}}' {{options}}

bench-iqa-canonical-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_compare_canonical test_canonical_samples test_canonical_intervals test_resume_canonical test_reference_publication test_canonical_rounds test_canonical_grid

bench-iqa-canonical-interval results output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/canonical_intervals.py '{{results}}' '{{output}}' {{options}}

bench-iqa-canonical-publish output +cohorts:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_canonical_comparison.py '{{output}}' {{cohorts}}

# Caller supplies shared heavy admission; verifies retained raw profiles before export.
bench-iqa-canonical-profiles-publish index output:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_canonical_profiles.py '{{index}}' '{{output}}'

bench-iqa-canonical-dssim-build source output:
    bash benchmarks/iqa_compete_2026-10-16/build_dssim_reference.sh '{{source}}' '{{output}}'

bench-iqa-canonical-ssim1-build source output:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_ssimulacra1_reference.py '{{source}}' '{{output}}'

bench-iqa-canonical-gmsd-build archive output:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_gmsd_reference.py '{{archive}}' '{{output}}'

bench-iqa-canonical-gpu-build source target:
    CARGO_TARGET_DIR='{{target}}' CARGO_PROFILE_RELEASE_LTO=thin CUDARC_CUDA_VERSION=12000 RUSTFLAGS='--cfg iqa_compete_ingress --cfg iqa_compete_cached' cargo build --locked --release --manifest-path '{{source}}/Cargo.toml' -p cpu-profile --features compete,metrics-parallel,study-gpu-cuda --bin iqa-compete

bench-iqa-canonical-cpu-build source target:
    CARGO_TARGET_DIR='{{target}}' CARGO_PROFILE_RELEASE_LTO=thin RUSTFLAGS='--cfg iqa_compete_ingress --cfg iqa_compete_cached' cargo build --locked --release --manifest-path '{{source}}/Cargo.toml' -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr --bin iqa-compete

bench-iqa-canonical-profile binary config output:
    perf record -F 999 --call-graph dwarf -o '{{output}}.data' -- '{{binary}}' '{{config}}'
    perf report --stdio --no-children -i '{{output}}.data' > '{{output}}.txt'

bench-iqa-canonical-original-butter-build source output:
    c++ -O3 -std=c++17 -ffp-contract=off -I '{{source}}' benchmarks/iqa_compete_2026-10-16/butteraugli_original_rgb.cpp '{{source}}/butteraugli/butteraugli.cc' -lpng -ljpeg -pthread -o '{{output}}'

# Explicit original-model exports; invoke under the host heavy lock/resource wrapper.
bench-iqa-canonical-export-original source python ladder output:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir '{{output}}'
    for net in alex vgg squeeze; do
        '{{python}}' -I '{{source}}/crates/zenmetrics-nn/models/lpips/export.py' '{{ladder}}' '{{output}}' "$net" 512
    done
    '{{python}}' -I '{{source}}/crates/zenmetrics-nn/models/dists/export.py' '{{ladder}}' '{{output}}' 512
    for net in alex vgg; do
        '{{python}}' -I '{{source}}/crates/zenmetrics-nn/models/stlpips/export.py' '{{ladder}}' '{{output}}' "$net" 512
    done
    '{{python}}' -I '{{source}}/crates/zenmetrics-nn/models/deepdc/export.py' '{{ladder}}' '{{output}}' 512

bench-iqa-canonical-model-verify root output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/verify_copied_models.py '{{root}}' '{{output}}' {{options}}

bench-iqa-canonical-copy-assets manifest output:
    python3 -B benchmarks/iqa_compete_2026-10-16/copy_canonical_assets.py '{{manifest}}' '{{output}}'

bench-iqa-canonical-original-nn-plan +options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_original_nn_comparison.py {{options}}

bench-iqa-canonical-author-setup python config metric pairs *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/canonical_author_network_setup.py '{{python}}' '{{config}}' '{{metric}}' '{{pairs}}' {{options}}

# Isolated legacy author runtime; invoke under shared resource admission.
bench-iqa-canonical-wadiqam-env output:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR='{{env_var("HOME")}}/tmp'
    test ! -e '{{output}}'
    uv venv --python /usr/bin/python3 '{{output}}'
    uv pip install --python '{{output}}/bin/python' setuptools==80.10.2 wheel numpy==1.23.5
    uv pip install --python '{{output}}/bin/python' --no-build-isolation chainer==7.8.1 h5py==3.16.0 imageio==2.38.0

bench-iqa-canonical-dreamsim-env python parent output:
    #!/usr/bin/env bash
    set -euo pipefail
    export TMPDIR='{{env_var("HOME")}}/tmp'
    test ! -e '{{output}}'
    uv venv --python '{{python}}' '{{output}}'
    printf '%s\n' '{{parent}}' > '{{output}}/lib/python3.12/site-packages/reference-parent.pth'
    uv pip install --python '{{output}}/bin/python' --no-deps dreamsim==0.2.1 peft open_clip_torch

# Link an explicitly supplied original libvmaf archive; no replacement kernel.
bench-iqa-canonical-vmaf-build source library output:
    c++ -O3 -Wall -Wextra -ffp-contract=off -std=c++17 -I '{{source}}/libvmaf/include' benchmarks/iqa_compete_2026-10-16/libvmaf_rgb_compare.cpp '{{library}}' -pthread -lm -o '{{output}}'

bench-iqa-canonical-daala-build source library output:
    c++ -O3 -Wall -Wextra -ffp-contract=off -std=c++17 -I '{{source}}/libvmaf/include' benchmarks/iqa_compete_2026-10-16/libvmaf_daala_rgb.cpp '{{library}}' -pthread -lm -o '{{output}}'

bench-iqa-canonical-nr-goldens source target root:
    ZENMETRICS_NR_GOLDEN_ROOT='{{root}}' CARGO_TARGET_DIR='{{target}}' cargo test --locked --release --manifest-path '{{source}}/Cargo.toml' -p brisque -p niqe -p piqe --all-features --test parity -- --nocapture

# Historical portable 5/3/1 panel: supplemental evidence, never canonical replacements.
bench-iqa-supplemental sizes="cid22,aic-small,aic-medium,aic-large,2k,4k" metrics="all" *options:
    cargo build --locked --release -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr --bin iqa-compete --bin bench-iqa
    '{{env_var_or_default("CARGO_TARGET_DIR","target")}}/release/bench-iqa{{if os() == "windows" { ".exe" } else { "" }}}' '{{sizes}}' '{{metrics}}' {{options}}

# Standalone checks for exact changed scope and retained timing order.
bench-iqa-changed-check:
    #!/usr/bin/env bash
    set -euo pipefail
    task_output=$(mktemp -d "$HOME/tmp/changed-benchmark-selection-XXXXXXXX")
    rustc --edition 2024 --test benchmarks/heaptrack/drivers/cpu_profile/src/benchmark_selection_tests.rs -o "$task_output/tests"
    "$task_output/tests"
    python3 -B -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_current_benchmark_scope.py -v

# Historical Python/GPU/NN panels have their own protocol; supplemental only.
bench-iqa-advanced-supplemental sizes="cid22,aic-small,aic-medium,aic-large,2k,4k" metrics="all" *options:
    python3 benchmarks/iqa_compete_2026-10-16/bench.py '{{sizes}}' '{{metrics}}' {{options}}

bench-iqa-check:
    bash benchmarks/iqa_compete_2026-10-16/check_portable.sh

# Optional adapter contract checks; no external checkout or C++ source required.
bench-iqa-identity-check:
    just snap-cargo @ test -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr --bin bench-iqa executable_identity

bench-iqa-optional-check:
    just snap-cargo @ test -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr --bin bench-iqa optional_dvifm

# IQA method-complexity tallies: evaluator tests and schema validation (docs/IQA_COMPLEXITY.md).
complexity-check:
    python3 scripts/complexity/test_nn_tallies.py
    python3 scripts/complexity/test_nn_schedules.py
    cd scripts/complexity && python3 -m unittest test_evaluate
    python3 scripts/complexity/evaluate.py --check benchmarks/iqa_complexity_2026-10-17/tallies/*.json
    python3 scripts/complexity/coverage.py --check
    python3 scripts/complexity/paper.py --check

# Regenerate the complexity README coverage table and the draft paper from committed inputs.
complexity-regenerate:
    python3 scripts/complexity/coverage.py
    python3 scripts/complexity/paper.py

# Metric-cost brief (one self-contained HTML + PDF) from the current data; see scripts/complexity/brief.sh.
brief:
    scripts/complexity/brief.sh

# Same, then upload to the Cloudflare Pages project in BRIEF_PROJECT.
brief-deploy:
    scripts/complexity/brief.sh --deploy

# Rebuild every 5 minutes from master and redeploy when the page changed (run it in a visible tab).
brief-watch workspace="../zenmetrics--brief-watch":
    scripts/complexity/brief_watch.sh {{workspace}}

# Integrity of the committed AIC2026 reproduction; add --reference combined.csv to re-derive the compare tables.
aic-repro-check *args:
    python3 scripts/aic2026/repro_check.py {{args}}

# Browser check of a built dashboard: every tab at two sizes, every control button, drawer, print mode.
# Unit tests for how the results-sync status becomes dashboard checks.
brief-sync-check:
    cd scripts/complexity && python3 -m unittest test_brief_sync

brief-check site="~/tmp/metric-brief/site" shots="~/tmp/metric-brief/check":
    uv run -q --with playwright scripts/complexity/brief_check.py {{site}} --shots {{shots}} ${BRIEF_CHROME:+--chrome "$BRIEF_CHROME"}

# Run one integration-test module per process on the native wgpu backend.
mac-validation-module package module:
    cargo test --locked -p {{package}} --no-default-features --features wgpu --test it {{module}} -- --nocapture --test-threads=1

# Owned-child timeout and missing-cell scheduler controls, with no model weights.
complexity-driver-check:
    python3 -m unittest discover -s scripts/complexity -p test_driver.py

# Host-specific bounded NN 4K measurements; configure CX4K_* inputs explicitly.
complexity-drive-nn-4k phase="all":
    bash scripts/complexity/drive_nn_4k.sh "{{phase}}"

# Frozen ONNX graph audits; model files are explicit and never silently skipped.
complexity-pair-graph-audit models receipt:
    python3 scripts/complexity/test_nn_tallies.py --pair-graphs "{{models}}" --receipt "{{receipt}}"

complexity-nn-schedule-audit models receipt:
    python3 scripts/complexity/test_nn_schedules.py --graphs "{{models}}" --receipt "{{receipt}}"

# Crash containment and immutable portable-bundle controls (real child processes).
iqa-durable-check:
    mkdir -p "$HOME/tmp"
    TMPDIR="$HOME/tmp" python3 benchmarks/iqa_compete_2026-10-16/test_durable.py
    TMPDIR="$HOME/tmp" python3 benchmarks/iqa_compete_2026-10-16/test_bundle.py
    TMPDIR="$HOME/tmp" python3 benchmarks/iqa_compete_2026-10-16/test_admission_recovery.py

iqa-durable-prepare binary pairs output revision *options:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_durable.py --binary '{{binary}}' --pairs '{{pairs}}' --output '{{output}}' --source-commit '{{revision}}' --sizes cid22,aic-small,aic-medium,aic-large,2k,4k --modes score --threads 1,8 --cold-repeats 3 {{options}}

# Start a copied, frozen ARM64 bundle; no compiler or model download needed.
bench-iqa-mac bundle output action="install" *options:
    python3 '{{bundle}}/mac_service.py' '{{action}}' '{{bundle}}' '{{output}}' --run-heavy '{{bundle}}/tools/run-heavy' --lock "$HOME/tmp/mac-heavy.lock" {{options}}

bench-iqa-mac-progress bundle output *options:
    python3 benchmarks/iqa_compete_2026-10-16/mac_service.py progress '{{bundle}}' '{{output}}' {{options}}

bench-iqa-mac-check-in plan logs *options:
    python3 benchmarks/iqa_compete_2026-10-16/mac_service.py check-in '{{plan}}' '{{logs}}' {{options}}

iqa-mac-coverage-prepare core remaining nn_cpu nn_metal report output:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_coverage.py '{{core}}' '{{remaining}}' '{{nn_cpu}}' '{{nn_metal}}' '{{report}}' '{{output}}'

iqa-durable-nn-prepare template binary build inventory models output engine *options:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_nn_durable.py '{{template}}' '{{binary}}' '{{build}}' '{{inventory}}' '{{models}}' '{{output}}' --engine '{{engine}}' {{options}}

iqa-durable-nn-check python="python3":
    TMPDIR="$HOME/tmp" '{{python}}' benchmarks/iqa_compete_2026-10-16/test_nn_bundle.py

iqa-nn-timing-check bundle output *options:
    python3 benchmarks/iqa_compete_2026-10-16/check_nn_timing.py '{{bundle}}' '{{output}}' {{options}}

iqa-cpu-timing-check bundle output *options:
    python3 benchmarks/iqa_compete_2026-10-16/check_cpu_timing.py '{{bundle}}' '{{output}}' {{options}}

iqa-dvifmish-prepare template binary build output results +options:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_dvifmish.py prepare '{{template}}' '{{binary}}' '{{build}}' '{{output}}' '{{results}}' {{options}}

iqa-dvifmish-install plan:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_dvifmish.py install '{{plan}}'

iqa-dvifmish-timing-check bundle output +options:
    python3 benchmarks/iqa_compete_2026-10-16/check_dvifmish_timing.py '{{bundle}}' '{{output}}' {{options}}

iqa-durable-build-receipt binary output command_file +options:
    python3 benchmarks/iqa_compete_2026-10-16/record_durable_build.py '{{binary}}' '{{output}}' --command-file '{{command_file}}' {{options}}

iqa-durable-run manifest output wrapper lock *options:
    python3 benchmarks/iqa_compete_2026-10-16/durable.py run '{{manifest}}' '{{output}}' --run-heavy '{{wrapper}}' --lock '{{lock}}' --memory 12G --jobs 8 {{options}}

# Frozen family sequence in a restarting Linux user service.
bench-iqa-linux plan action="status":
    python3 benchmarks/iqa_compete_2026-10-16/linux_service.py '{{action}}' '{{plan}}'

iqa-linux-coverage-prepare cpu nn_cpu nn_cuda output binary build:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_linux_sequence.py '{{cpu}}' '{{nn_cpu}}' '{{nn_cuda}}' '{{output}}' --lock "$HOME/tmp/i265-heavy.lock" --binary '{{binary}}' --build '{{build}}'

iqa-timing-worker-check binary:
    TMPDIR="$HOME/tmp" IQA_TIMING_BINARY='{{binary}}' python3 benchmarks/heaptrack/drivers/cpu_profile/src/bin/test_iqa_compete_timing.py

iqa-timing-boundary-check:
    mkdir -p "$HOME/tmp/iqa-timing-tests"
    rustc --edition=2024 --test benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete_timing/tests.rs -o "$HOME/tmp/iqa-timing-tests/boundary"
    "$HOME/tmp/iqa-timing-tests/boundary"

iqa-mac-service-check bundle output wrapper lock:
    python3 benchmarks/iqa_compete_2026-10-16/check_mac_service.py '{{bundle}}' '{{output}}' --run-heavy '{{wrapper}}' --lock '{{lock}}'

iqa-mac-coverage-check:
    TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/test_prepare_coverage.py

# Independent resource accounting; caller uses the existing shared heavy lock.
iqa-resource-accounting-build:
    cargo build --locked --release --manifest-path benchmarks/iqa_compete_2026-10-16/resource-accounting/Cargo.toml

iqa-resource-accounting-run binary output:
    bash benchmarks/iqa_compete_2026-10-16/resource-accounting/run.sh '{{binary}}' '{{output}}'

iqa-resource-accounting-external diagnostic worker config output:
    bash benchmarks/iqa_compete_2026-10-16/resource-accounting/external.sh '{{diagnostic}}' '{{worker}}' '{{config}}' '{{output}}'

# Read-only strict Mac WebGPU session diagnostics; shared heavy admission is internal.
iqa-webgpu-node-audit python bundle output:
    '{{python}}' benchmarks/iqa_compete_2026-10-16/audit_webgpu_nodes.py --bundle '{{bundle}}' --output '{{output}}'

iqa-webgpu-node-classify directory output *options:
    python3 benchmarks/iqa_compete_2026-10-16/classify_webgpu_nodes.py '{{directory}}' '{{output}}' {{options}}

# Immutable scalar queue replacement; boundary/rollout acquire the shared lock.
iqa-scalar-priority action config *options:
    python3 benchmarks/iqa_compete_2026-10-16/scalar_priority.py '{{action}}' '{{config}}' {{options}}

iqa-scalar-priority-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_scalar_priority.py

# Project recorded durable attempts into a fixed chart snapshot (new output only).
iqa-chart-export root output host:
    python3 benchmarks/iqa_compete_2026-10-16/export_chart_snapshot.py {{root}} {{output}} --host {{host}}

# Install an endpoint guard without changing a frozen measurement plan.
iqa-window-guard plan:
    python3 benchmarks/iqa_compete_2026-10-16/scalar_priority.py window-guard {{plan}}

# AIC convention checks and private-oracle invocation are documented in the receipt.
aic-convention-check:
    just snap-cargo @ test -p hdrvdp
    just snap-cargo @ test -p zenmetrics-cli --lib
    python3 scripts/checkpoint/test_browser_coverage.py

# Real-process refusal control; build cpu-profile with zenmetrics-cli/nr first.
complexity-refusal-check bin:
    python3 scripts/complexity/check_refusals.py --bin "{{bin}}"

# Caller serializes these with the shared heavy lock and resource wrapper.
aic-perf bin out metrics shapes threads="1" cpus="2":
    python3 scripts/complexity/perf_deltas.py --bin "{{bin}}" --out "{{out}}" --metrics "{{metrics}}" --shapes "{{shapes}}" --threads {{threads}} --cpu "{{cpus}}" --event-profile portable --repeats 3 --timeout-s 900

# Each audit attempt takes the shared lock and resource wrapper itself.
aic-refusal-inventory bin out logs shapes="64x64,1024x1024,3840x2160" threads="1,8":
    python3 scripts/complexity/refusal_inventory.py --lock "$HOME/tmp/dev-heavy.lock" --bin "{{bin}}" --out "{{out}}" --logs "{{logs}}" --shapes "{{shapes}}" --threads "{{threads}}"

# Preliminary draft weighted counts, including explicitly partial private aggregates.
complexity-draft-compare:
    python3 scripts/complexity/compare_draft.py

# Caller supplies resource admission; retain OS peak RSS for one prepared-pair process.
complexity-process-rss bin metric shape threads cpus report:
    env RAYON_NUM_THREADS={{threads}} OMP_NUM_THREADS={{threads}} RTEN_NUM_THREADS={{threads}} /usr/bin/time -v -o "{{report}}" taskset -c "{{cpus}}" "{{bin}}" "{{shape}}" "{{metric}}" 1

# Requires all six inventory panels; refuses incomplete or duplicate cells.
aic-refusal-publish logs out *inputs:
    python3 scripts/complexity/publish_refusals.py --logs "{{logs}}" --out "{{out}}" {{inputs}}

# Verify only landing navigation/layout; metric numerical checks stay separate.
dashboard-landing-check python site chromium report:
    "{{python}}" scripts/dashboard/check_landing.py "{{site}}" --chromium "{{chromium}}" --report "{{report}}"

# Independent HDR-VDP upstream research (full logs/receipts kept in caller's scratch).
hdrvdp-canonical-oracle tree pairs output version peak bits ppd encoding matrix *options:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/run_oracle.py '{{tree}}' '{{pairs}}' '{{output}}' '{{version}}' '{{peak}}' '{{bits}}' '{{ppd}}' '{{encoding}}' '{{matrix}}' {{options}}

hdrvdp-canonical-grid grid:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/grid.py '{{grid}}'

hdrvdp-canonical-rust probe pairs output *options:
    test ! -e '{{output}}'
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" RAYON_NUM_THREADS=8 '{{probe}}' '{{pairs}}' '{{output}}' {{options}} > '{{output}}.log' 2>&1

hdrvdp-canonical-harness-check scratch published config:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/check_harness.py '{{scratch}}' '{{published}}' '{{config}}'

hdrvdp-canonical-stratify data scores config aggregate private:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/stratify.py '{{data}}' '{{scores}}' '{{config}}' '{{aggregate}}' '{{private}}'

hdrvdp-canonical-collect scratch published config output:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/collect.py '{{scratch}}' '{{published}}' '{{config}}' '{{output}}'

hdrvdp-canonical-compare published config column scores field *reference:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/compare.py '{{published}}' '{{config}}' '{{column}}' '{{scores}}' '{{field}}' {{reference}}

hdrvdp-canonical-prepare-v2 tree newtree headers:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_v2_octave.py '{{tree}}' '{{newtree}}' '{{headers}}'

hdrvdp-canonical-prepare-v306 tree newtree:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_v306.py '{{tree}}' '{{newtree}}'

hdrvdp-canonical-build-rust manifest:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" cargo build --release --manifest-path '{{manifest}}' -p vdpprobe

hdrvdp-canonical-prepare-rust newdir:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_rust_probe.py '{{newdir}}'

hdrvdp-canonical-controls scratch output:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/summarize_controls.py '{{scratch}}' '{{output}}'

hdrvdp-canonical-prepare-symmetric tree newtree:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_symmetric_control.py '{{tree}}' '{{newtree}}'

hdrvdp-canonical-check-symmetric:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" octave --quiet benchmarks/hdrvdp_canonical_2026-10-19/scripts/check_symmetric_control.m

hdrvdp-canonical-pixels pairs output *native:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" OPENBLAS_NUM_THREADS=1 python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/pixel_control.py '{{pairs}}' '{{output}}' {{native}}

hdrvdp-canonical-build-decode source newdir:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_native_decode.py '{{source}}' '{{newdir}}'
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" cargo build --locked --release --manifest-path '{{newdir}}/Cargo.toml' -p vdpprobe --bin native-decode

hdrvdp-canonical-decode binary images output:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" '{{binary}}' '{{images}}' '{{output}}'

hdrvdp-canonical-build-stages newdir:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_native_stages.py '{{newdir}}'

hdrvdp-canonical-stages v221 v222 v307 probe output:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" OPENBLAS_NUM_THREADS=1 python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/spectral_control.py '{{v221}}' '{{v222}}' '{{v307}}' '{{probe}}' '{{output}}'

hdrvdp-canonical-prepare-cubic tree newtree:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_cubic_control.py '{{tree}}' '{{newtree}}'

hdrvdp-canonical-check-cubic:
    flock "$HOME/tmp/dev-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G -- env TMPDIR="$HOME/tmp/hdrvdpcx" octave --quiet benchmarks/hdrvdp_canonical_2026-10-19/scripts/check_cubic_control.m

hdrvdp-canonical-prepare-matrix tree native newtree:
    python3 benchmarks/hdrvdp_canonical_2026-10-19/scripts/prepare_matrix_control.py '{{tree}}' '{{native}}' '{{newtree}}'

# Finite Mac GPU timing policy and strict numerical refusal controls.
iqa-finite-gpu-policy-tests:
    python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_finite_gpu_policy.py

# Std-only deferred observation controls; call under the shared heavy lock.
iqa-score-observations-tests:
    rustc --edition=2024 -D warnings --test benchmarks/heaptrack/drivers/cpu_profile/src/iqa_compete_score_observations.rs -o "$HOME/tmp/iqa-score-observations-tests"
    "$HOME/tmp/iqa-score-observations-tests"

# Freeze replacement executable/runners; existing assets and source bundle stay immutable.
iqa-worker-successor source binary build output:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_worker_successor.py '{{source}}' '{{binary}}' '{{build}}' '{{output}}'

# Fixed-factor scoring regressions; run inside the usual host resource wrapper.
fixed-factor-check rev="@":
    just snap-cargo {{rev}} test -p fsim --release --features avx512 --all-targets
    just snap-cargo {{rev}} test -p fsim --features avx512 --doc
    just snap-cargo {{rev}} test -p zenmetrics-cli --no-default-features --lib ssim_pyiqa
    just snap-cargo {{rev}} clippy -p fsim --features avx512 --all-targets -- -D warnings

# Verify native metric consumer feature propagation against the pinned graph.
metric-dispatch-check rev="@":
    just snap-cargo {{rev}} check -p zenmetrics-cli --no-default-features --features cpu-fsim,cpu-vmaf,cpu-cvvdp,cpu-iwssim,orchestrator,orchestrator-cpu-cvvdp,orchestrator-cpu-iwssim

# Lean VMAF adapter regression tests include the independent PSNR plane check.
vmaf-adapter-check rev="@":
    just snap-cargo {{rev}} test -p zenmetrics-cli --no-default-features --features cpu-vmaf --lib metrics::vmaf

# Explicit named weight-consent propagation; no default consent.
iqa-weight-consent-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_weight_consents.py

# Requires explicit owner authorization for a separate consent-only attempt budget.
iqa-consent-retry config:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_consent_retry.py '{{config}}'

clean-panel-check:
    TMPDIR="$HOME/tmp" python3 -m unittest discover -s benchmarks/clean_panel_2026-10-19 -p 'test_*.py'

# Negative control only: run acceptance assertions against an exported old candidate.

clean-panel-before-check historical_directory:
    TMPDIR="$HOME/tmp" python3 benchmarks/clean_panel_2026-10-19/baseline_regressions.py {{historical_directory}}

# Synthetic extraction only: no metric graph, scorer or timing run.

clean-panel-nr-check:
    #!/usr/bin/env bash
    set -euo pipefail
    directory=$(mktemp -d "$HOME/tmp/clean-panel-nr-control.XXXXXX")
    rustc --edition=2024 --test benchmarks/clean_panel_2026-10-19/nr_observer.rs -o "$directory/control"
    "$directory/control"

# Isolated nonmetric monitor diagnostics; invoke under run-heavy and shared lock.

clean-panel-monitor-build directory:
    cargo test --manifest-path {{directory}}/source/Cargo.toml --locked --offline --no-default-features --lib --no-run --message-format=json


clean-panel-monitor-control binary source_manifest output:
    python3 benchmarks/clean_panel_2026-10-19/monitor_control.py {{binary}} {{source_manifest}} {{output}}

# Focused repair controls; retained generic diagnostic failure remains unchanged.

clean-panel-monitor-repair-check current output retained_author retained_reviewer:
    python3 benchmarks/clean_panel_2026-10-19/monitor_repair_controls.py {{current}} {{output}} {{retained_author}} {{retained_reviewer}}

# Bind portable scalar timing rows to their worker memory/resource receipts.
iqa-portable-collect output +roots:
    python3 benchmarks/iqa_compete_2026-10-16/collect_portable_results.py '{{output}}' {{roots}}

iqa-portable-collect-check:
    python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_portable_collection.py

# Caller holds the same host benchmark lock; one unchanged-config scope per flagged row.
iqa-portable-retry-flagged parent output wrapper:
    python3 benchmarks/iqa_compete_2026-10-16/retry_portable_flagged.py '{{parent}}' '{{output}}' '{{wrapper}}'

# Include sealed completed attempts while a phase report still lists them pending.
iqa-chart-export-live root output host *options:
    python3 benchmarks/iqa_compete_2026-10-16/export_chart_snapshot.py '{{root}}' '{{output}}' --host '{{host}}' --sealed-attempts {{options}}

iqa-chart-export-tests:
    python3 benchmarks/iqa_compete_2026-10-16/test_chart_export.py

iqa-canonical-cpu-check:
    python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_canonical_cpu.py

iqa-canonical-cpu-prepare *args:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_canonical_cpu.py {{args}}


# Fixed canonical IQA recipe arguments, immutable output and shared-lock controls.
iqa-canonical-command-check:
    TMPDIR="$HOME/tmp" python3 benchmarks/iqa_compete_2026-10-16/test_canonical_commands.py

# Immutable retry subsets for flagged warm rows, preserving original remaining caps.
iqa-canonical-gate-retry bundle results output lock:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_gate_retry.py '{{bundle}}' '{{results}}' '{{output}}' --lock '{{lock}}'

iqa-canonical-gate-retry-check:
    TMPDIR="$HOME/tmp" python3 benchmarks/iqa_compete_2026-10-16/test_gate_retry.py

# Freeze only unfinished approved changed routes, retaining original attempt budgets.
bench-iqa-changed-successor bundle results output metrics lock:
    python3 benchmarks/iqa_compete_2026-10-16/prepare_changed_successor.py '{{bundle}}' '{{results}}' '{{output}}' '{{metrics}}' --lock '{{lock}}'

bench-iqa-changed-successor-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -m unittest benchmarks/iqa_compete_2026-10-16/test_changed_successor.py

# Read finalized saved results without stopping workers. Raw captures stay outside git.
bench-iqa-capture-saved spec output host:
    python3 benchmarks/iqa_compete_2026-10-16/capture_saved_results.py '{{spec}}' '{{output}}' --host '{{host}}'

# Publish timing-only allowlisted saved-result projections; refuses existing output.
bench-iqa-publish-saved captures private_aggregates output:
    python3 benchmarks/iqa_compete_2026-10-16/publish_saved_results.py '{{captures}}' '{{private_aggregates}}' '{{output}}'

bench-iqa-publish-saved-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -m unittest benchmarks/iqa_compete_2026-10-16/test_saved_publication.py

# Read-only host capture followed by guarded commit/push of allowlisted timing data.
iqa-results-sync config:
    python3 benchmarks/iqa_compete_2026-10-16/sync_saved_results.py '{{config}}'

# After a repaired safe_push, verify ancestry and identical committed data before
# updating only private sync bookkeeping. This never pushes or edits repository files.
iqa-results-sync-recover config revision:
    python3 benchmarks/iqa_compete_2026-10-16/sync_saved_results.py '{{config}}' --recover-published '{{revision}}'

iqa-results-sync-check:
    TMPDIR="$HOME/tmp" python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p test_result_sync.py

# Explicit resident original mDCTpsnr, preserving its compiled fast-math contract.
bench-iqa-canonical-mdctpsnr-build source output:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_mdctpsnr_reference.py '{{source}}' '{{output}}'

bench-iqa-canonical-author-plan manifest output:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_author_nn_comparison.py '{{manifest}}' '{{output}}'

bench-iqa-canonical-queue manifest *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/compare_canonical_queue.py '{{manifest}}' {{options}}

bench-iqa-canonical-status manifest *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/status_canonical_queue.py '{{manifest}}' {{options}}

bench-iqa-canonical-round-audit output raw-manifest +cohorts:
    python3 -B benchmarks/iqa_compete_2026-10-16/audit_canonical_rounds.py '{{output}}' '{{raw-manifest}}' {{cohorts}}

bench-iqa-canonical-wadiqam-inference python manifest receipt:
    SETUPTOOLS_USE_DISTUTILS=stdlib '{{python}}' -B benchmarks/iqa_compete_2026-10-16/prepare_wadiqam_inference.py '{{manifest}}' '{{receipt}}'

bench-iqa-canonical-warm-profile binary config output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/profile_canonical_warm.py '{{binary}}' '{{config}}' '{{output}}' {{options}}

bench-iqa-canonical-flags manifest output regression-log:
    python3 -B benchmarks/iqa_compete_2026-10-16/audit_canonical_flags.py '{{manifest}}' '{{output}}' '{{regression-log}}'

bench-iqa-canonical-flag-profiles index:
    python3 -B benchmarks/iqa_compete_2026-10-16/profile_canonical_queue.py '{{index}}'


# Caller supplies heavy admission; exports the pinned revision before compiling.
bench-iqa-canonical-cpu-revision-build revision output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/build_primary_cpu.py '{{revision}}' '{{output}}' {{options}}

# Caller supplies heavy admission; verify every tracked in-tree compiled input.
bench-iqa-canonical-worker-verify build output:
    python3 -B benchmarks/iqa_compete_2026-10-16/verify_classical_worker.py '{{build}}' '{{output}}'

# Caller supplies heavy admission; common raw crop, native warm timing only.
bench-iqa-native-order plan output:
    python3 -B benchmarks/iqa_compete_2026-10-16/measure_native_order.py '{{plan}}' '{{output}}'

# Caller supplies heavy admission; checks NR reference-image auxiliary columns.
bench-iqa-canonical-nr-auxiliary python manifest output:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/check_nr_auxiliary.py '{{manifest}}' '{{output}}'

bench-iqa-canonical-nr-auxiliary-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 TMPDIR="$HOME/tmp" python3 -B -m unittest test_nr_auxiliary

# The manifest can bind a current scheduling table; legacy manifests retain the 27-route table.
bench-iqa-canonical-ordered-plan manifest output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/prepare_ordered_canonical.py '{{manifest}}' '{{output}}' {{options}}

bench-iqa-canonical-ordered-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 TMPDIR="$HOME/tmp" python3 -B -m unittest test_ordered_canonical

# Caller supplies heavy admission; immutable models load before the first-evaluation timer.
bench-iqa-canonical-nn-first-build plan target output:
    python3 -B benchmarks/iqa_compete_2026-10-16/build_resident_nn_first.py '{{plan}}' '{{target}}' '{{output}}'

# Observe an existing queue; each scored-case completion is audited without starting inference.
bench-iqa-canonical-watch manifest output regression-log progress-log pid:
    python3 -B benchmarks/iqa_compete_2026-10-16/watch_canonical_flags.py '{{manifest}}' '{{output}}' '{{regression-log}}' '{{progress-log}}' '{{pid}}'

# Full cropped-set canonical score evaluation; supply the immutable local manifest.
aic-canonical-run manifest *options:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_run.py {{manifest}} {{options}}

aic-canonical-compare python manifest summary:
    {{python}} benchmarks/aic2026_full_repro_2026-10-20/canonical_compare.py {{manifest}} --summary {{summary}}

aic-canonical-check python:
    {{python}} -m unittest discover -s benchmarks/aic2026_full_repro_2026-10-20 -p 'test_canonical_*.py'

# Partial files survive 9p ENOMEM; repeat this recipe after memory compaction.
aic-canonical-inputs source destination:
    ~/work/claudehints/scripts/run-heavy --mem 12G --jobs 12 -- rsync -a --partial --bwlimit=12000 {{source}} {{destination}}

# Bind unique reference/distorted bytes; caller supplies shared resource admission.
aic-canonical-input-hashes pairs inventory receipt *options:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_input_hashes.py {{pairs}} {{inventory}} --receipt {{receipt}} {{options}}

# Capture installed reference files and explicit assets without loading models.
aic-canonical-reference-inventory python output receipt *options:
    {{quote(python)}} benchmarks/aic2026_full_repro_2026-10-20/canonical_reference_inventory.py {{quote(output)}} {{quote(receipt)}} {{options}}

aic-canonical-fsim-vsi python manifest summary reference sha256:
    {{python}} benchmarks/aic2026_full_repro_2026-10-20/compare_fsim_vsi.py {{manifest}} {{summary}} --published-reference {{reference}} --reference-sha256 {{sha256}}

aic-canonical-campaign manifest *options:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_campaign.py {{manifest}} {{options}}

# Read retained stage logs; this does not acquire resources or qualify scores.
aic-canonical-progress scratch:
    tail -n 6 {{scratch}}/*.campaign.log

# Partial observations only; output includes reference rows and stays outside git.
aic-canonical-partial python manifest metric output *options:
    {{quote(python)}} benchmarks/aic2026_full_repro_2026-10-20/canonical_partial.py {{quote(manifest)}} {{quote(metric)}} {{quote(output)}} {{options}}

aic-canonical-provenance manifest runtime_versions *options:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_provenance.py {{manifest}} {{runtime_versions}} {{options}}

# Copy the requested exports with backups and verify all three destination hashes.
aic-canonical-mirror manifest receipt label:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_mirror.py {{quote(manifest)}} {{quote(receipt)}} --label {{quote(label)}}

# Separate diagnostic of author MAD components and documented comparison semantics.
aic-canonical-mad-probe manifest output sources +identities:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_mad_probe.py {{quote(manifest)}} {{quote(output)}} --source-bindings {{quote(sources)}} {{identities}}

# Diagnostic only: three studio-709 rounding conventions, with no acceptance gate.
aic-canonical-psnr-rounding python pairs native output *options:
    {{python}} benchmarks/aic2026_full_repro_2026-10-20/diagnose_psnr_rounding.py {{pairs}} {{native}} {{output}} {{options}}

# Full-precision local AIC CSV regeneration; caller wraps heavy runs in run-heavy.
aic-rescore +args:
    python3 scripts/complexity/aic_rescore.py {{args}}

# Compare July and supplied CSV revisions without changing source values.
aic-csv-reconcile +args:
    python3 scripts/complexity/aic_csv_reconcile.py {{args}}

# OS-backed resume, precision, identity and CSV reconciliation controls.
aic-rescore-check:
    python3 -m unittest discover -s scripts/complexity -p 'test_aic*.py' -v

aic-precision-check binary reference distorted output:
    python3 scripts/complexity/check_aic_precision.py --binary '{{binary}}' --reference '{{reference}}' --distorted '{{distorted}}' --output '{{output}}'



# Caller supplies heavy admission; the author callable runs on explicit PyTorch CUDA.
bench-iqa-canonical-cuda-reference python +options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/canonical_cuda_reference_cell.py {{options}}

# Controller acquires both heavy locks and a 12 GiB scope per alternating block.
bench-iqa-canonical-nn-first plan output:
    python3 -B benchmarks/iqa_compete_2026-10-16/compare_first_evaluation.py '{{plan}}' '{{output}}'

# Caller supplies heavy admission; records the whole first-evaluation probe lifetime.
bench-iqa-canonical-nn-first-profile plan case output:
    python3 -B benchmarks/iqa_compete_2026-10-16/profile_first_evaluation.py '{{plan}}' '{{case}}' '{{output}}'

# Callers supply heavy admission and explicit frozen native/reference builds.
bench-iqa-canonical-dssim-maps python +options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/check_dssim_maps.py {{options}}

bench-iqa-canonical-flip-maps python build inputs output *options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/check_flip_maps.py '{{build}}' '{{inputs}}' '{{output}}' {{options}}

# Publish resident-model first calls separately from warm comparison tables.
bench-iqa-canonical-nn-first-publish output receipt +roots:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_nn_first_evaluation.py '{{output}}' '{{receipt}}' {{roots}}

# Original Google Butteraugli, distinct from the libjxl comparator; caller holds heavy admission.
bench-iqa-canonical-butter-original-maps python +options:
    '{{python}}' -B benchmarks/iqa_compete_2026-10-16/check_butteraugli_original_maps.py {{options}}

# Prepare a new schedule from completed canonical bundles without re-scoring them.
aic-canonical-successor predecessor scratch output freeze lock1 lock2:
    python3 benchmarks/aic2026_full_repro_2026-10-20/canonical_successor.py {{quote(predecessor)}} {{quote(scratch)}} {{quote(output)}} {{quote(freeze)}} --locks {{quote(lock1)}} {{quote(lock2)}} --rows-per-worker 32

# Retained original/resident HDR controls; no new numerical or timing qualification.
bench-iqa-canonical-hdr-controls-publish golden preflight output:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_hdr_reference_controls.py '{{golden}}' '{{preflight}}' '{{output}}'

# Exact retained warm cases for the site; caller supplies shared heavy admission.
bench-iqa-canonical-summary source inventory output:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_spotcheck_summary.py '{{source}}' '{{inventory}}' '{{output}}'

bench-iqa-canonical-summary-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_spotcheck_summary

# Small four-case score-control batch; source/math and per-child checks stay frozen.
bench-iqa-canonical-hdr-warm-controls cohort worker output +options:
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 8 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/run_hdr_reference_controls.py '{{cohort}}' '{{worker}}' '{{output}}' {{options}}

# Four original-cell protocol calls after exact ABBA controls; no timing acceptance.
bench-iqa-canonical-hdr-cell-controls manifest output:
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 8 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/run_hdr_cell_controls.py '{{manifest}}' '{{output}}'

# Execute an independently reviewed frozen VMAF driver under local heavy admission.
bench-iqa-canonical-vmaf-fullcall driver +options:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 8 -- env TMPDIR="$HOME/tmp" python3 -B '{{driver}}' --admitted {{options}}

# Verify retained VMAF evidence and freeze a new reference plan without scoring.
bench-iqa-canonical-vmaf-plan *options:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 1 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/prepare_vmaf_fullcall_plan.py {{options}}

# Pure receipt fixtures, including optimized-Python gate controls.
bench-iqa-canonical-vmaf-plan-check:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 1 -- env TMPDIR="$HOME/tmp" PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_vmaf_fullcall_plan
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 1 -- env TMPDIR="$HOME/tmp" PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -O -B -m unittest test_vmaf_fullcall_plan

# Caller supplies heavy admission; retain cold-process captures separately from warm evidence.
bench-iqa-canonical-classical-first-profiles-publish index output:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_classical_first_profiles.py '{{index}}' '{{output}}'

# Caller supplies heavy admission; retain size-capped recordings without executing workers.
bench-iqa-canonical-classical-first-profile-adopt-capped plan case capture output:
    python3 -B benchmarks/iqa_compete_2026-10-16/adopt_capped_classical_profile.py '{{plan}}' '{{case}}' '{{capture}}' '{{output}}'

# Self-contained admission for the original-archive first-call reference build.
bench-iqa-canonical-dssim-first-build parent output parent_sha256:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 1 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/build_dssim_first_reference.py '{{parent}}' '{{output}}' --parent-sha256 '{{parent_sha256}}'

# Synthetic clock/lifetime check; no real metric or timing comparison.
bench-iqa-canonical-dssim-first-boundary-check output compiler:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 1 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/check_dssim_first_boundary.py '{{output}}' --compiler '{{compiler}}'

# Independently bound original/candidate references; no native comparison.
bench-iqa-canonical-first-boundary-control plan output:
    mkdir -p "$HOME/tmp"
    flock "$HOME/tmp/lilith-heavy.lock" flock "$HOME/tmp/lilith-wsl-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 28 -- env TMPDIR="$HOME/tmp" python3 -B benchmarks/iqa_compete_2026-10-16/first_boundary_control.py '{{plan}}' '{{output}}'

# Retains the frozen plan and every completed original three-block outcome.
bench-iqa-canonical-block-resume plan output *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/resume_canonical_campaign.py '{{plan}}' '{{output}}' {{options}}

bench-iqa-canonical-queue-resume manifest *options:
    python3 -B benchmarks/iqa_compete_2026-10-16/resume_canonical_queue.py '{{manifest}}' {{options}}

bench-iqa-canonical-recovery-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_resume_canonical_campaign

# Original three-block CPU constructor/first-evaluation campaigns only.
bench-iqa-canonical-classical-first-resume plan output helper helper-sha256:
    python3 -B benchmarks/iqa_compete_2026-10-16/resume_classical_first_campaign.py '{{plan}}' '{{output}}' --original-helper '{{helper}}' --original-helper-sha256 '{{helper-sha256}}'

bench-iqa-canonical-classical-first-recovery-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_resume_classical_first_campaign

# The original publisher stays frozen; recovered block paths remain explicit.
bench-iqa-canonical-nn-first-publish-recovered output receipt +cohorts:
    python3 -B benchmarks/iqa_compete_2026-10-16/publish_nn_first_evaluation_recovered.py '{{output}}' '{{receipt}}' {{cohorts}}

bench-iqa-canonical-nn-first-recovery-check:
    PYTHONPATH=benchmarks/iqa_compete_2026-10-16 python3 -B -m unittest test_nn_first_recovery_publication test_nn_first_publication

# Historical Butteraugli maps on identical calibration pixels, with frozen Git pins.
margarine-reference-build:
    cargo build --release --locked --manifest-path crates/margarine/reference-audit/Cargo.toml

margarine-reference-maps pairs output:
    crates/margarine/reference-audit/target/release/margarine-reference-audit "{{pairs}}" "{{output}}"

margarine-reference-summary pairs maps output:
    python3 crates/margarine/reference_maps.py "{{pairs}}" "{{maps}}" "{{output}}"

margarine-linear-check:
    cargo fmt --manifest-path crates/margarine/Cargo.toml -p margarine --check
    cargo test --locked --manifest-path crates/margarine/Cargo.toml --no-default-features --features rayon,avx512,linear-steering
    cargo test --locked --release --manifest-path crates/margarine/Cargo.toml --no-default-features --features rayon,avx512,linear-steering
    cargo clippy --locked --manifest-path crates/margarine/Cargo.toml --no-default-features --features rayon,avx512,linear-steering --all-targets -- -D warnings
    cargo build --locked --release --manifest-path crates/margarine/Cargo.toml --no-default-features --features rayon,avx512,linear-steering --bin margarine-steer

margarine-loop-replay-build:
    cargo build --release --locked --manifest-path crates/margarine/loop-replay/Cargo.toml

margarine-loop-replay pairs output:
    crates/margarine/loop-replay/target/release/margarine-loop-replay "{{pairs}}" "{{output}}"

margarine-linear-calibrate pairs replay binary output:
    python3 crates/margarine/linear_calibrate.py "{{pairs}}" "{{replay}}" "{{binary}}" "{{output}}"

margarine-replay-oracle pairs replay output:
    python3 crates/margarine/oracle_replay.py "{{pairs}}" "{{replay}}" "{{output}}"
