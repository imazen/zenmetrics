# hdr10_sky_192x108 — real HDR10 video fixture

96×54, 24 frames @ 30 fps, RGB48 (16-bit) PNG, full-range
BT.2020-primaries samples with ST 2084 (PQ) encoding. `ref/` is the
source; `dist_q8/` is `ref` pushed through a per-sample 8-bit
roundtrip (`round(v/257)*257`) — a realistic banding distortion that
destroys exactly the sub-8-bit precision this fixture exists to
exercise.

(The directory name predates the current crop: it started life as a
192×108 window, shrunk to 96×54 to keep the in-repo payload small —
~400 KB total — per the workspace large-files rule. The crop origin
was re-picked so the nit range matches the original: v10 ≈ 387–872
≈ 390–1030 nits.)

## Source

`hdr-pq-sky.mp4` from <https://github.com/JonaNorman/HDRSample>
(`sample/src/main/assets/video/`): HEVC `yuv420p10le`, BT.2020
non-constant-luminance matrix, SMPTE ST 2084 PQ transfer, 30 fps.
That repository has no explicit license file; the files are published
as Android-sample test assets. Only a small derivative crop is
committed here, for test-fixture use.

## Extraction (reproduces the committed frames)

```sh
ffmpeg -ss 2.0 -i hdr-pq-sky.mp4 -frames:v 24 \
    -vf "crop=96:54:760:80" -pix_fmt rgb48le -f rawvideo ref.raw
# dist_q8: numpy roundtrip (a/257).round().astype(u16)*257
ffmpeg -f rawvideo -pix_fmt rgb48le -s 96x54 -i ref.raw \
    -pix_fmt rgb48be ref/f%02d.png
ffmpeg -f rawvideo -pix_fmt rgb48le -s 96x54 -i dist.raw \
    -pix_fmt rgb48be dist_q8/f%02d.png
```

ffmpeg's `rgb48le` conversion applies the source's BT.2020 NCL matrix
and yields full-range, PQ-encoded RGB samples (v/65535 = the PQ code
value). Verified genuine 10-bit content: ~99.9% of samples have
nonzero low-8 bits; 10-bit code range in this crop ≈ 387–872
(≈390–1030 nits).
