# Codec-decoded video input — scope (2026-09-27)

`cvvdp::VideoScorer` is deliberately **codec-agnostic**: it consumes
decoded frame buffers (`push_frame`/`push_frame_u16`/`push_frame_f32`)
plus an explicit `fps`, so the metric never sees a container. pycvvdp
takes the opposite tack (`video_source_file` + ffmpeg decoders baked
into the library). This document scopes what it would take to feed
`VideoScorer` from compressed video in the Zen workspace — it is a
design brief, not a shipped feature.

## Decoder candidates in-workspace

| Crate | Codec | Shape | Notes |
|---|---|---|---|
| `rav1d-safe` | AV1 | `Decoder::decode(&[u8]) -> Option<Frame>`; `Frame::{planes(), color_info(), timestamp(), duration(), pixel_layout(), bit_depth()}` | safe-Rust rav1d fork; full CICP-style metadata (`ColorInfo{primaries, transfer_characteristics, matrix_coefficients, color_range}`), content-light/mastering-display side data. OBU-level — needs a container demuxer above it. |
| `zencodec` | — | shared image-codec traits | single-image surface; would need a `VideoDecoder` trait extension or stay below it |
| `zenav1-svt`, `zenav1-svt-c`, `zenrav1e`, `zenav1-aom` | AV1 | encoders | not decoders — no input role here |
| (absent) | H.264/HEVC/VP9/AVC | — | no Zen decoder exists; outside pure-Rust policy these would come from an external crate or be out of scope |

No demuxer exists in the workspace. AV1 in IVF/OBU Annex-B is
parseable in ~200 lines; MP4/MKV/WebM would need a real demuxer
(`zenavif`-adjacent MP4 parsing exists for stills but not a timed
track reader).

## API boundary — the proposal

Keep `VideoScorer` untouched. Add a separate crate or a
`zenmetrics-cli`-internal adapter:

```text
encoded file → demuxer → decoder → DecodedFrame { planes, layout,
    bit_depth, color: Cicp/ColorInfo, ts: Rational, duration }
→ ColorConverter (YUV→RGB under declared matrix/range/primaries,
  transfer → sRGB-8/16 or linear f32)
→ VideoScorer::push_frame_*
```

The converter is the correctness-critical seam, not the decoder:
VideoScorer's `push_frame` contract is **sRGB-8** (`_u16`/linear f32
variants exist). Feeding YUV BT.709 limited-range planes as if they
were sRGB is exactly the class of silent-relabeling bug
`AGENTS.md`'s pixel rules forbid — so the adapter must carry color
metadata end-to-end and *convert*, not reinterpret.

## Metadata the adapter must preserve

From `rav1d_safe::Frame::color_info()` et al:

- `matrix_coefficients` (YUV→RGB matrix — BT.601/709/2020, identity RGB)
- `color_range` (limited vs full — changes the scale offsets)
- `primaries` + `transfer_characteristics` (sRGB / PQ / HLG →
  display-referred conversion; cvvdp has its own display photometry,
  so the sane target is linearized-to-display OR sRGB per the scorer's
  contract — decide once, document)
- `bit_depth` + `pixel_layout` (I400/420/422/444 — chroma upsample
  position needs the layout + chroma sample location)
- `timestamp`/`duration` (rational, container units)
- orientation / crop if the container applies them (display matrix,
  clean aperture)

## Stream pairing & sync policy

Scoring needs `(ref_frame, dist_frame)` pairs. Compressed streams give
timestamps, not indices:

- **Same fps, aligned start** — trivial index pairing (the common
  benchmark case: reference + encode of the same source).
- **Different frame counts / dropped or duplicated frames** — pair by
  timestamp; skip-or-dupe policy must be *explicit* (default: error on
  mismatch > half a frame interval).
- **Different fps** — do NOT silently resample video frames; resample
  `q_per_ch` post-hoc via `temp_resample`, or error. Temporal pooling
  is fps-dependent; pairing must never invent frames upstream of the
  temporal filter.
- **VFR** — cvvdp assumes constant fps (temporal filter taps are
  built from a single rate). Policy: require CFR after decode or
  reject. `VideoScorer`'s `fps` arg should come from container
  timebase, not user flag, when wired.
- **PTS reorder (B-frames)** — decoder emits display order; timestamps
  resolve ordering.

## Decode-error policy

- Corrupt/truncated packet: error the run (a scored pair of unequal
  real content is worse than no score) — never substitute a
  repeated/black frame silently.
- Missing metadata: error unless the caller passes an explicit
  `--assume-color bt709-limited`-style override; the override is
  recorded in output provenance.
- Bit depth > supported path: 10/12-bit planes → `push_frame_u16` or
  f32 conversion; never truncate to u8 silently.

## Memory / streaming

The adapter is the one place a whole-clip blowup can sneak in. Rules:
frames are pulled through `Decoder::decode`/`get_frame` lazily
(iterator style), converted one at a time, and dropped after
`push_frame` returns — peak = 2 decoded frames + the scorer's bounded
ring (`fl` frames, or u8 in `low_memory`). No frame caching in the
adapter; PTS reorder is the decoder's problem.

## CLI UX sketch

```bash
zenmetrics score-video \
    --reference ref.ivf --distorted enc.ivf \
    [--display-model standard_4k] [--fps <override>] ...
```

- Container probing picks demuxer (IVF first; MP4 later).
- `--fps` becomes optional/derived when timestamps exist.
- Color provenance printed into JSON output (`color_info` of each
  stream + the conversion applied).

## Phases

1. **Phase A — IVF+AV1 only** (`rav1d-safe` + ~200-line IVF demuxer),
   8-bit 4:2:0 BT.709 limited → sRGB-8 conversion wired through
   `push_frame`. Provenance fields in CLI JSON. This covers the
   "score an SVT encode vs its source" benchmark case.
2. **Phase B** — 10/12-bit (`push_frame_u16`), I444/422 layouts,
   HDR metadata surfaced (`content_light`/`mastering_display` → warn
   when `display-model` mismatches the signal).
3. **Phase C** — MP4/WebM demuxing; explicit timestamp pairing with
   a documented skip/dupe policy; other codecs if/when Zen decoders
   exist.
4. **Out of scope** — re-encoding, scaling/crop beyond metadata
   application, foveation input from container metadata, audio.

## Why `VideoScorer` stays frame-buffer based

pycvvdp's `video_source_file*` couples decode, colorspace assumption
(ffmpeg output *is* assumed sRGB), and scoring. That is precisely the
relabeling hazard the workspace pixel rules exist for. The adapter
keeps provenance explicit and lets tests inject synthetic frames
without a decoder in the loop — the same reason `push_frame` is the
conformance harness's entry point today.
