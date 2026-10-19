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
