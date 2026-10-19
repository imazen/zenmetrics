# Read-only checkpoint status and the separate CMS call-stack observation.
checkpoint-status:
    gh run view 39275363269 -R imazen/zenmetrics --json status,conclusion,jobs --jq '{status,conclusion,jobs:[.jobs[]|{name,status,conclusion}]}'
    gh run view 39275131060 -R imazen/zenmetrics --json status,conclusion,jobs --jq '{status,conclusion,jobs:[.jobs[]|select(.conclusion != "success" and .conclusion != "skipped")|{name,status,conclusion}]}'
    gh run view 39275131008 -R imazen/zenmetrics --json status,conclusion,jobs --jq '{status,conclusion,jobs:[.jobs[]|{name,status,conclusion}]}'

cms-callers canonical panel output:
    python3 -B trace_ssim2_cms_callers.py --canonical '{{canonical}}' --panel '{{panel}}' --output '{{output}}'

haarpsi-maps-publish panel output:
    python3 -B publish_haarpsi_maps.py '{{panel}}' '{{output}}'

portable-math stage:
    python3 -B run_checkpoint_math.py '{{stage}}'

portable-math-remote-only stage:
    nice -n19 ionice -c3 python3 -B run_checkpoint_math.py '{{stage}}' --remote-only

# Run under run-heavy and both shared heavy-job locks.
checkpoint-integrated-check snapshot:
    python3 -B verify_integrated_snapshot.py '{{snapshot}}'
