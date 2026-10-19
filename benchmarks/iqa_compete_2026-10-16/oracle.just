# Artifact integrity only; these recipes do not import or execute a metric.
# Run from any directory with just --justfile <absolute-path>/oracle.just.
# Selfcheck writes scratch artifacts: supply a fresh disk-backed evidence directory.
fixed-factor-selfcheck evidence +sources:
    python3 -B check_fixed_factor_oracle.py selfcheck --evidence '{{evidence}}' {{sources}}

fixed-factor-check manifest:
    python3 -B check_fixed_factor_oracle.py check --manifest '{{manifest}}'

mad-helper-selfcheck evidence +sources:
    python3 -B check_mad_helper_patch.py selfcheck --evidence '{{evidence}}' {{sources}}

mad-helper-check manifest:
    python3 -B check_mad_helper_patch.py check --manifest '{{manifest}}'

native-scope-check source binary:
    python3 -B check_libjxl_scope.py '{{source}}' --binary '{{binary}}'

inventory-source-check:
    python3 -B ../port_spotcheck_2026-10-18/make_inventory_source.py ../port_spotcheck_2026-10-18 --verify

# Caller supplies both host locks, run-heavy and a complete output log.
[positional-arguments]
ssim2-primary-build ssim2_archive lcms_archive output:
    bash build_ssimulacra2_primary.sh "$1" "$2" "$3"

[positional-arguments]
ssim2-primary-rgb-build ssim2_archive lcms_archive output:
    bash build_ssimulacra2_primary.sh "$1" "$2" "$3" rgb-input

[positional-arguments]
ssim2-primary-rgb-owned-build ssim2_archive lcms_archive output hwy_archive:
    bash build_ssimulacra2_primary.sh "$1" "$2" "$3" rgb-input "$4"

[positional-arguments]
ssim2-primary-rgb-check binary archive source_root evidence:
    python3 -B check_ssimulacra2_primary.py --binary "$1" --archive "$2" --source-root "$3" --evidence "$4"

# Calls both canonical binaries on synthetic pixels. Requires both host heavy locks.
[positional-arguments]
ssim2-primary-rgb-ingress-check binary archive source_root evidence official:
    python3 -B check_ssimulacra2_primary.py --binary "$1" --archive "$2" --source-root "$3" --evidence "$4" --official "$5"

[positional-arguments]
ssim2-primary-owned-ingress-check binary archive source_root evidence official hwy_archive hwy_source hwy_prefix:
    python3 -B check_ssimulacra2_primary.py --binary "$1" --archive "$2" --source-root "$3" --evidence "$4" --official "$5" --hwy-archive "$6" --hwy-source-root "$7" --hwy-prefix "$8"

# Canonical metric calls; caller supplies the host locks and run-heavy.
[positional-arguments]
octave-primary manifest reference distorted width height threads samples evidence:
    python3 -B octave_primary.py --manifest "$1" --reference "$2" --distorted "$3" --width "$4" --height "$5" --threads "$6" --samples "$7" --evidence "$8"

# Fresh CI-pinned source export. Caller supplies both locks, run-heavy and log.
# Do not repeat export over an existing snapshot; compile it separately instead.
[positional-arguments]
primary-rust-freeze snapshot revision:
    test ! -e "$1" && test ! -L "$1"
    LOCK_SNAP_DIR="$1" bash ../../scripts/ci/lock.sh --check --rev "$2"

[positional-arguments]
primary-rust-compile snapshot:
    env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS CARGO_TARGET_DIR="$1/target" CARGO_PROFILE_RELEASE_LTO=thin cargo build --manifest-path "$1/work/zenmetrics/Cargo.toml" --locked --release -p cpu-profile --features compete,metrics-parallel --bin iqa-compete

[positional-arguments]
octave-rust-parity binary build_receipt canonical manifest reference distorted threads evidence:
    python3 -B probe_octave_parity.py --binary "$1" --build-receipt "$2" --canonical "$3" --manifest "$4" --reference "$5" --distorted "$6" --threads "$7" --evidence "$8"

[positional-arguments]
spotcheck-raw-prepare plan output:
    python3 -B prepare_spotcheck_raw.py "$1" "$2"

spotcheck-raw-check:
    python3 -B -m unittest -v test_spotcheck_raw

[positional-arguments]
octave-primary-memory manifest reference distorted width height threads samples evidence:
    python3 -B octave_primary.py --manifest "$1" --reference "$2" --distorted "$3" --width "$4" --height "$5" --threads "$6" --samples "$7" --evidence "$8" --measure-memory

[positional-arguments]
octave-rust-parity-memory binary build_receipt canonical manifest reference distorted threads evidence:
    python3 -B probe_octave_parity.py --binary "$1" --build-receipt "$2" --canonical "$3" --manifest "$4" --reference "$5" --distorted "$6" --threads "$7" --evidence "$8" --measure-memory

# Controller wraps each child with both heavy-job locks and run-heavy itself.
# Do not wrap this recipe in either lock.
[positional-arguments]
octave-panel inputs references rust_build output threads:
    nice -n 19 ionice -c 3 python3 -B run_octave_panel.py --inputs "$1" --references "$2" --rust-build "$3" --output "$4" --threads "$5"

[positional-arguments]
octave-panel-resume inputs references rust_build output threads:
    nice -n 19 ionice -c 3 python3 -B run_octave_panel.py --inputs "$1" --references "$2" --rust-build "$3" --output "$4" --threads "$5" --resume

# Nonmetric live-child control; caller supplies both locks and run-heavy.
[positional-arguments]
oracle-memory-timeout-check evidence:
    python3 -B check_oracle_memory_timeout.py "$1"

# The comparison controller supplies both locks and a resource wrapper per group.
[positional-arguments]
ssim-candidate-compare baseline candidate inputs output:
    nice -n 19 ionice -c 3 python3 -B compare_ssim_candidate.py --baseline "$1" --candidate "$2" --inputs "$3" --output "$4"

[positional-arguments]
fixed-factor-create name factor source output:
    python3 -B fixed_factor_oracle.py patch --name "$1" --factor "$2" --source-path "$3" --out "$4"

# Caller supplies both locks, run-heavy, GNU-time receipt, and complete log.
[positional-arguments]
pyiqa-ssim-probe python manifest inputs output threads:
    "$1" -B probe_pyiqa_ssim.py --manifest "$2" --inputs "$3" --output "$4" --threads "$5"

# The controller supplies both locks, deadlines and run-heavy for every case.
[positional-arguments]
pyiqa-ssim-panel python inputs references output:
    nice -n 19 ionice -c 3 python3 -B run_pyiqa_ssim_panel.py --python "$1" --inputs "$2" --references "$3" --output "$4"

[positional-arguments]
ssim-candidate-publish input output:
    python3 -B publish_ssim_candidate.py "$1" "$2"
