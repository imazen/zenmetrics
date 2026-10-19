# mad-helper-oracle — hash-guarded ownership overlay for the MAD `ical_std` helper

Benchmark glue for the AIC2026 canonical-oracle work. It patches a
scratch copy of the **original upstream** `ical_std.c` so the compiled
helper can run under Octave without the upstream cleanup-path fault.
It is **not** a metric, scorer, runner or shipped dependency, and the
patched copy is a *compatibility reference*, never the canonical
artifact.

## Why the patch exists — measured evidence

- Original `ical_std.c` (`4a3370ec…`) compiled under Octave's `mex.h`
  (`typedef void mxArray`) with only const-qualifier warnings at
  `:23`/`:24` — the `mxDestroyArray(TMP)` call on a `double*`
  (line 159) produced **no compiler warning**.
- On the x86 Octave 6.4 build tested, the compiled original
  `ical_std.mex` (`24d5beb5…`) **SIGSEGVs on the first valid call**
  (16×16 finite doubles), inside `mxArray::~mxArray`/`mxDestroyArray`
  with `rdi = 0x404e849c986c42b4` — a computed output double
  interpreted as an object pointer.
- Disassembly of the exact MEX (`0x1609 mov %rbp,%rdi` → `0x1616 jmp
  mxDestroyArray@plt`, DWARF line 159) proves the faulting call **is**
  the `:159` statement — a tail call; the absent `mexFunction` frame is
  tail-call collapse, not deferred corruption.
- `ical_stat.c`'s original MEX (`f6d6e305…`) ran clean on the same
  controls and is **unchanged** — this overlay does not touch it.

## What the patch is (and is not)

Exactly three regions, ownership only: declare `mxArray *tmpArr`, split
`mxCreateDoubleMatrix`/`mxGetPr` so `tmpArr` holds the handle, destroy
`tmpArr` instead of `TMP`. All loops, `/255.0` and `/63.0` divisors,
indexing, types and the `[std, modified_std, mean]` output order are
byte-identical outside those regions **by construction** — and the
patcher additionally *enforces* it: the produced stream must sha256-
equal the copy that already passed the contained controls
(`79fb753e…`); anything else fails.

Validated scope — and its limits:

- The compat build ran all **six** helper cases (16×16, 17×19, 64×64 ×
  textured + constant) cleanly and produced outputs **byte/SHA256
  identical** to the imazen `.m` shim on those cases — independently
  verified, 18/18 pairs (`~/tmp/portcheck-mad-artifact-review-
  20261019/verification.json`; `bytes_equal`+sha match per file, not
  merely maxabs=0). Six hand-made matrices are *bounded coverage*, not
  input-space proof, and the shim is itself a port of the same C —
  the agreement is corroboration, not an independent oracle.
- The original helper produced **no outputs in these controls** (it
  faulted before returning on its first call), so no
  original-vs-compat output comparison exists; the claim is "runs the
  same loops on tested cases without crashing", not "proven equal to
  upstream on all inputs".
- This qualifies **one helper MEX** — `ical_std`. It says nothing about
  the MAD metric score, `hi_index`/`lo_index`, timing, or the 19-cell
  panel: **0/19 unchanged.**

## Provenance and terms

- Source: `Netflix/vmaf` `matlab/STMAD_2011_MatlabCode/ical_std.c` @
  commit `229f2478ab53ae0fd6b07328802294ba15fd0cdd`, URL
  `https://raw.githubusercontent.com/Netflix/vmaf/229f2478ab53ae0fd6b07328802294ba15fd0cdd/matlab/STMAD_2011_MatlabCode/ical_std.c`,
  sha256 `4a3370ec…c2f2`. The file carries **no license header** — terms
  unclear; private oracle/test material only. Every produced copy
  retains the upstream content (and thus its attribution); **no copied
  upstream source is committed to this repository or shipped** — only
  the patcher, checker and this doc are.
- Original compiled MEX `24d5beb5…` remains the canonical (failing)
  artifact; compat build `ical_std_compat.mex` `4e804359…` (gcc 11.4.0,
  overlay `mkoctfile 6.4.0`, `-DMEX_DEBUG -O2 -flto`, overlay include
  dirs — full argv in the toolchain report) is the reference build of
  the patched bytes.
- `ical_stat` original MEX `f6d6e305…` — separate, unchanged, already
  clean.

## Module-name requirement (downstream staging rule)

Octave resolves MEX by **filename == function name**. Future `hi_index`
calls `ical_std`, so the compatibility module must be staged as
`ical_std.mex` **inside a separately identified compatibility
directory** (e.g. `staging-compat/`) — never next to, over, or in place
of the original artifacts — and the harness must record `which` output
(the resolved path) per run so the scored artifact's identity is
auditable. Original `ical_std.c`/`ical_std.mex` stay read-only
canonical artifacts.

## Usage

```
python3 patch_mad_helper.py list --source-root <mirror-root>
python3 patch_mad_helper.py patch --name mad-ical_std-c \
    --source-root <mirror-root> \
    --out <new-scratch-dir>/ical_std_compat.c \
    --manifest <new-scratch-dir>/ical_std_compat.manifest.json
python3 check_mad_helper_patch.py check --manifest <manifest>
python3 check_mad_helper_patch.py selfcheck \
    --source-root <mirror-root> --evidence <new-evidence-dir>
```

The mirror resolves `mad/STMAD-ical_std.c` under `--source-root`.
`--source-path` takes two different forms by command: `patch` accepts a
plain `--source-path FILE` (for its `--name`), while `list`/`selfcheck`
accept `--source-path mad-ical_std-c=<path>` (repeatable, name-keyed).
All outputs are `O_EXCL`-created; rerunning against the same output
path is safe (reports `exists-identical` for identical content, refuses
anything different). The tool performs no metric imports, compile,
execution or network access — compiling the produced copy, running
helper controls and any oracle integration are separate, explicitly
authorized steps.
