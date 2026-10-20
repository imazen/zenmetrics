# Full-call VMAF retained-evidence verifier

`prepare_vmaf_fullcall_plan.py` prepares a new photo512 reference plan from the
retained [full-call controls](vmaf-fullcall-reference.md). It never runs a compiler,
ldd, scorer, model or subprocess. Existing plans and observations stay unchanged.
Only the six model routes at one/eight threads are selected. Their original input,
convention, callable arguments and numerical bounds are copied exactly; v1/4K
bounds remain null. Complete model lifetime is matched scope, independently of
native parity, numerical acceptance, utilization or timing acceptance.

The caller supplies independently checked SHA256 values for the original freeze,
build result, controls result and normal native `verified-build.json`. The latter
must link its ordinary `build-receipt.json`, source map, outer build log and binary;
these paths follow the retained build layout (`work/zenmetrics/`, `cargo.log`,
and the adjacent `<build-directory>.log`). An arbitrary scorer receipt is refused.
These digests are trust anchors, not values to derive from untrusted replacement
receipts. The original frozen producer and old/new clock-source hashes are pinned
in the verifier; it hashes their bytes and does not import the producer.

```sh
python3 -B prepare_vmaf_fullcall_plan.py \
  --source ORIGINAL_PLAN --build BUILD/build-receipt.json \
  --controls CONTROLS/result.json --native NATIVE_BINARY \
  --native-receipt NATIVE_BUILD/verified-build.json \
  --freeze-sha REVIEWED_FREEZE_SHA --build-sha REVIEWED_BUILD_SHA \
  --controls-sha REVIEWED_CONTROLS_SHA \
  --native-receipt-sha REVIEWED_NATIVE_RECEIPT_SHA --output NEW_PLAN
```

Verification reparses raw VERSION/SCOPE/COLD+two SAMPLE records, requires finite
scalars and exact recorded f64 bits, and checks all 72 observations. Exact arm
names, binaries, case identities, argv, environments and all recorded process
outcomes must agree. Nonempty stderr is preserved under the producer's policy;
there is no new score or stderr threshold. Duplicate/nonfinite receipt JSON is refused,
and validation uses explicit exceptions rather than optimization-sensitive
assertions. Hashed source and asset files remain opaque bytes, including native
test fixtures whose `.json` extension does not imply a valid receipt.

The output binds source/header/archive identities, original/native receipts,
compiler query/link records, both sets of asset/runtime snapshots, recursive
retained ldd transcripts/receipts, raw child output and per-arm input snapshots.
The closure walk is replayed from retained text. Before/after comparisons follow
the producer: existing build runtime identities persist; the new ELF may extend
the final build closure; both control closures equal that final closure. No
missing-evidence skips are supported. Bound bytes are rechecked before writing.

Validation finishes before output publication. Complete serialized bytes are
written and synced to a sibling temporary file, then linked atomically to the
new output name; an existing file or competing publication refuses replacement.
Temporary files are removed on failure. The output parent must already exist.
This does not prevent later mutation by another writer: the consuming runner
must still enforce every plan binding.

`test_vmaf_fullcall_plan.py` constructs synthetic retained files and fake inert
ELF/archive bytes; no executable or library is invoked. Its controls cover valid
scope-only preservation, forged arms/argv/environment/outcomes/protocol/bits,
nonfinite/duplicate JSON, missing/altered historical evidence, input/archive drift,
null bounds and atomic output refusal/failure/races. Run only under the caller's
resource admission:

```sh
python3 -B -m unittest discover -s SOURCE_DIRECTORY -p test_vmaf_fullcall_plan.py -v
# The same controls can also be admitted with Python -O; no verifier assertion disappears.
```

Retained hashes do not establish the original static archive's historical compiler
or rebuild reproducibility. The verifier does not independently reconstruct native
source-to-binary provenance, observe library utilization, or qualify performance.
Trust in the separately reviewed normal native build receipt remains explicit.
Exact old/new reference scores preserve reference continuity; they do not invent
a native parity bound or qualify prior asymmetric timings.
