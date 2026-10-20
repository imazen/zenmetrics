# DSSIM full-call reference successor

## Root execution, 2026-10-20

The frozen builder linked successfully, and all 24/24 pair/budget comparisons
passed: 48 fresh children retained 144 finite scalar observations, with exact
f64 bits across old/new COLD and SAMPLE values in each case. The C, Python and
header bytes match the retained source bundle. The run completed in 47 s with
peak-RSS 1.42 GiB, minimum available 13,980 MiB and peak load 5.61.

The [control receipt](../canonical_recheck_2026-10-19/dssim-fullcall-controls.json)
binds the new ELF, source, compiler, static archive and linked runtime. It does
not recover the older executable's missing C-source witness or assign a new
numerical bound. Separate paired timings on the ordinary current40 worker are
in progress. The preparation notes below remain the record before execution.

Source preparation only: this adapter has not been compiled or scored by this
lane. Root admits the frozen build/control command separately. The old oracle,
plans, scores and timing caveats remain unchanged.

The [new C adapter](dssim_fullcall_reference.c) times `dssim_new`, both
`dssim_create_image_rgb` calls, `dssim_compare`, both `dssim_free_image` calls and
`dssim_free` in every observation. It uses the existing author C API and archive
without modifying metric arithmetic. Raw RGB reads, argument validation,
locale/thread-environment setup and stdout transport are outside those clocks.
No scorer call precedes the one `COLD`; the requested positive number of
`SAMPLE` calls follows. Image/context failures exit nonzero with diagnostics.

The new adapter reads the thread argument and sets Rayon/OMP/BLAS environment
caps before its first library call. The older C harness does **not** read
argv[5] or set environment variables; its outer runner supplies those caps.
Neither arrangement establishes observed library utilization. Controls use
explicit budgets1/8, each in a fresh process.

## Frozen source and link receipt

[The builder](build_dssim_fullcall_reference.py) links the retained author
`dssim-core`3.5.1 static archive; it never invokes Cargo. The retained lineage
names author revision `0e44c9b7fe91a5265c8e463436c28512186fe9cd` from
[kornelski/dssim](https://github.com/kornelski/dssim). Archive SHA256:
`4de9551888ecdb2af6516543d108557d91950a0974f9cee104b1554d76d72ea1`.
This successor does not reconstruct the old archive's complete build history.

The private source bundle has an externally supplied `freeze.json` digest. It
binds the exact new C/builder/control/docs, header, original pair manifest,
source/lineage metadata, compiler executable path/hash, fixed flags, retained
archive/old executable and inspection tools. Compiler version text is captured
under root admission before linking; its exact argv/output hashes enter the
new build receipt. Backend/linker paths and hashes are captured before/after.

The builder refuses an existing output directory, copies the exact C/header
and archive into its new link directory, records compile argv and complete
stdout/stderr, verifies ELF output and records its SHA256. Source/assets are
hashed before/after; recursive `ldd` closure identities and unedited transcripts
are retained around linking. The new ELF's linked closure enters the receipt.
No compiler, `ldd` or scorer is invoked by source-bundle preparation itself.

Compiler flags are `-O3 -std=c11 -Wall -Wextra -Werror -fno-fast-math
-ffp-contract=off`, followed by the frozen include/source/archive paths and
`-ldl -pthread -lm`. Compiler search-path override variables are removed.
These flags affect the C wrapper/link only; the author archive is reused byte
for byte. The binary is private evidence, never a committed artifact.

## Exact scalar controls

[The control driver](check_dssim_fullcall_reference.py) uses only standard
library orchestration and frozen utility definitions from the builder. It
requires a passed, externally digest-bound build receipt and rechecks the new
ELF, source/header/archive identities and linked runtime. It retains commands,
thread environments, process outcomes, raw stdout/stderr, parsed observations
and exact f64 bits. Per-child timeout cleanup targets only the group it starts.

The exact original `primary-raw-crops/1` manifest SHA256 is
`83d61ef3ce1877a20e07a2c8f8ac0df0a761a0a77903a5784b011a74b17f0712`.
It supplies three contents at64x64,255x257,1024x1024 and3840x2160. The driver
preserves original source/crop metadata and checks RGB lengths and hashes
before/after children. The manifest's historical B/C decode-lineage limitation
remains; this fixture binds raw bytes, not a new decoder equivalence claim.

The prepared gate is24pair/budget comparisons,48fresh old/new processes. With
the frozen two samples each, retain144scalars: one `COLD` plus two `SAMPLE`
observations per child. Every scalar must be finite; all old/new observations
within each comparison must have identical binary64 bits. Missing/extra/wrong
protocol records, process errors, size refusals, timeouts, altered inputs,
source/binary drift or dependency-closure drift fail. Remaining independent
cases still run after a score/protocol refusal, with failures retained.
Cross-budget identity is not an additional acceptance gate.

Runtime/source snapshots are retained before/after the controls. A terminal
result requires24/24passed comparisons and unchanged identities. Raw elapsed
observations are retained as control evidence; the driver computes no speed
ratios, calibrated timing estimates or tolerance adjustments. Existing native
numerical bounds and primary-canonical dispositions are untouched.

Finite old/new equivalence would establish this fixture's scalar agreement.
It cannot recover the old binary's missing C-wrapper source witness or by
itself establish native parity, resource admission, first-call scope equality
or a benchmark speed qualification. The new source-bound reference can support
a separately reviewed minimal paired timing successor after controls pass.

## Admission interface

The private frozen bundle and exact paths/digests/command are recorded in
`~/tmp/handoff/dssim-fullcall-oracle-20261020/report.md`. Root runs the complete
`build_dssim_fullcall_reference.py all` command under both shared locks and
`run-heavy --mem 12G --jobs 8`; compiler and old/new children are sequential.
`all` binds the just-written build-receipt digest into the control argv and
retains both driver logs. Existing build/control outputs are never overwritten.
No other agent notification, production edits, tests, CI or scoring is
authorized by source preparation. No human-approved exception is claimed.
