# SSIMULACRA2 CMS caller observation, 2026-10-19

Missing: the same runtime trace across the remaining sizes, content and
thread counts. This 64×64 photo observation does not establish complete
runtime closure or qualify a performance comparison.

The original zero-`cmsDoTransform` hypothesis failed: the retained primary
binary made 2,088 such calls during its two score invocations. Its failed
diagnostic log is preserved at
`/home/lilith/tmp/portcheck-ssim2-cms-runtime-20261019T1440/`; its assertion
was not weakened to accept the observation.

A separate GDB [caller trace](ssim2-cms-callers.json) records all 2,088 calls
and their stacks. Every recorded stack contains `JxlCmsInit`. Of those calls,
2,032 go directly through `ProfileEquivalentToICC`, eight directly through
`ColorEncoding::SetFieldsFromICC`, and 48 through black-point/conversion
setup inside transform creation. No recorded call comes from the row
transformation callback. This distinguishes initialization work from image
row conversion for this input; it does not show that LCMS is unused.

Both returned scores are exactly `90.86388011956463`, matching the retained
scalar study. The binary hash remains
`9d0fc033b44580055909b52e2cabad87d2c5f6fb28c4fe970391738fe3f67ac1`,
from the pinned
[Cloudinary v2.1 source](https://github.com/cloudinary/ssimulacra2/tree/fb5d0dbe866e5113d320666a09ddbc8d145b5fde).
The trace records command arguments, script/source/output hashes and the
earlier study's manifest hashes. Debugger stop/resume overhead is excluded
from every metric timing claim.

The command is `ssim2-primary-cms-callers` in
[`oracle.just`](../iqa_compete_2026-10-16/oracle.just), under both host locks
and `run-heavy --mem 12G --jobs 1`. Raw trace:
`/home/lilith/tmp/portcheck-ssim2-cms-callers-20261019T1518/`.
Receipt: `rc=0 6s | peak-RSS 0.06GiB | min-avail 13383MiB | peak-load 2.07`.
