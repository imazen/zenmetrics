# Original HDR sRGB byte and runtime controls

`first_boundary_hdrvdp_admitted.py prepare PARENT_PLAN NEW_FOLDER` copies a
source-bound all24 control into a distinct folder. Preparation does not run
Octave. The emitter copies the original `reference_codes`, `minstd` and
`distort` function bodies literally; only the raw-output wrapper is new.

The admitted `run NEW_FOLDER NEW_RESULTS` command first inventories installed
Octave core/image M/OCT files, configuration, ELF dependencies and their recursive
`ldd` closure. It retains every resolution log and compares content hashes and
resolved dependency edges after each process. ASLR addresses remain in logs
but are excluded from closure equality. Unbound Octave search/init overrides
and loader injection overrides refuse execution.

The original-function emitter must reproduce every byte of all fourteen raw
files before either metric arm runs. Then system image 2.12 and the unchanged
two-file image 2.20 resize overlay each run all24 original sRGB goldens against
the existing `5e-3` bound. A numerical failure or nonzero system-arm exit does
not suppress the resize arm. Source or runtime closure changes refuse further
execution. Complete stdout/stderr, commands, process memory receipts, closure
snapshots and outcomes remain separate for every stage.

The caller supplies heavy admission around the entire driver. These are
diagnostic controls, with zero qualified timing cells. Preparation and source
inspection do not establish raw-byte equality or numerical acceptance; the
coordinator executes and reviews those controls separately.
