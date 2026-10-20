# Saved CPU results captured 2026-10-20

Partial saved-result snapshot, taken while measurements continued. Capture
reads atomic report files and verifies finalized sealed attempt/config/log
hashes. An active, unfinished attempt emits no observation. This is a capture
window per host, not a simultaneous distributed transaction.

| Host | Capture window UTC | Saved attempt rows | Preferred qualified cells |
| --- | --- | ---: | ---: |
| mac | 00:57:16.479033–00:57:41.577354 | 2254 | 2092 |
| i265 | 00:57:16.477510–00:57:41.287565 | 3505 | 3332 |
| r7900x | 00:57:17.163762 | 3643 | 3487 |

Use each host's `*-observations.csv`, selecting `preferred=True` for qualified
chart rows. Keep binary, input and protocol identities as distinct series.
Failed/flagged attempts remain present but never become zero timing values.
Preferred selection uses exact metric/binary/input/geometry/thread/phase/cold
repeat/engine identity, then a qualified saved attempt; it does not pool
implementation revisions or combine cold repeats into warm timings.

All six DVIFMISH routes have observations on every host. Mac/i265 each have
48 preferred cells per route; r7900x has 31–32 per route at this capture.
Private weightless `cw-ssim-pyiqa` contributes 48 preferred cells per host,
carried forward from its completed, hash-verified allowlisted aggregate.
Its original capture time and build/source identities remain in each private
protocol; it was not remeasured for this snapshot. No other private NN panel
is represented as completed.

The only observed Zensim name here is `zensim`. A separate W-C or another bake
is not identified by the frozen worker and is not claimed. No NN/GPU timing
campaign was launched to produce this snapshot.

[sources.json](sources.json) binds per-campaign capture UTC, frozen manifests,
binaries, builds, source revisions, input hashes and sealed-result hashes.
[coverage.json](coverage.json) records pending/refused/exhausted and flagged
saved attempts per campaign. Parent and successor campaign status counts
**overlap**: never sum them as completed coverage. Active Mac/i265 queues now
contain only unfinished changed27 cells; r7900x keeps the full non-NN queue.
[protocol.json](protocol.json) specifies three fresh-process constructor-inclusive
cold repeats (RGB loading outside clocks), ten clean 100ms-target full-pair
warm rounds, scalar 1/8 workers, and unchanged resource/native gates.
Portable 5/3 panels, maps and 8K rows are excluded.

This export includes worker-scope peak RSS evidence, not metric heap usage or
GPU VRAM. Timing qualification does not establish native oracle parity or
AIC published-column agreement. No raw inputs, per-image scores, model URLs,
private source, graphs or weights are included. Raw captures remain private.

Reproduce the projection with `just bench-iqa-publish-saved CAPTURES
PRIVATE_AGGREGATES NEW_OUTPUT`. Captures are made by
`just bench-iqa-capture-saved SPEC NEW_CAPTURE HOST`. The spec explicitly names
saved campaign directories; immutable completed captures refuse replacement.
