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
