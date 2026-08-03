# TRCC + Levita Vision on ryzen5800xt — investigation, 2026-08-03

Getting the Thermalright **Levita Vision** LCD (on ryzen5800xt's CPU cooler) to play
video under Linux via [TRCC Linux](https://github.com/Lexonight1/thermalright-trcc-linux),
the community port of Thermalright's Windows LCD Control Center.

**Where it ended:** solid colours display on demand; **rendered images and video do
not** — the panel falls back to the Thermalright logo. TRCC also **OOM-killed itself
twice at ~62 GB RSS**. Both are almost certainly the same root cause and both need
source edits, which is the next session's job. Everything below is measured on the
box, not inferred.

---

## 1. The hardware and the software

| | |
|---|---|
| Box | `ryzen5800xt` (LAN IP in homefleet NODES.md), Ubuntu 26.04, Python 3.14, 64 GB RAM |
| Device | `lsusb` → `Bus 001 Device 005: ID 87ad:70db ChiZhu Tech USBDISPLAY` |
| TRCC | **9.9.5**, installed from the project's `.deb` (`trcc-linux-latest_all.deb`) |
| Installed package | `/usr/lib/python3/dist-packages/trcc` ← **edit here to test hot-fixes** |
| Source clone | `~/trcc` on the node, rev **`f6e9de1`** (upstream `main`) |
| udev | `sudo trcc setup` ran; the GPU-sensor extras step fails (`No module named pip`) — harmless, unrelated |
| Service (ours) | `~/.config/systemd/user/trcc-play.service` + `~/trcc-play.sh`, linger enabled |

Upstream already flags this panel as unproven: `doc/TESTERS_WANTED.md` lists
**"Levita Vision | HID LCD or LED | Need tester"**, and `doc/CHANGELOG.md` (~line 765)
documents issue #149 — a Levita right-side camera notch — noting the Levita "shares
the chipset" with GrandVision and that scoped-lookup cross-contamination "is how
Levita was" broken before. That is exactly the bug class we hit.

## 2. What works vs. what doesn't (all verified on the panel by eye)

| Operation | Result | Evidence |
|---|---|---|
| `device connect` | **works** | reports `resolution: 1600×720`, `model_id: 64` |
| `display set-brightness` | **works** | `Brightness set to 90%` |
| `display color <key> #rrggbb` | **WORKS — visible** | `Sent 32131 bytes`; user confirmed red, then a 6-colour cycle ending cyan |
| `display load-image` (real 1600×720 PNG) | **sends, NOT visible** | `Theme 'image:frame-1600x720' loaded and sent (250444 bytes)` → panel stayed on logo |
| `display load-video` → theme | **sends, NOT visible** | `playing Theme.zt (722 frame(s))` → static |
| `display play-video` + `display play` ticker | **pumps, then OOM** | 881 CPU ticks/6 s, 20 GB written to the endpoint, then killed |
| `trcc daemon` + `TRCC_DAEMON=1` | **broken** | daemon opens the device at discovery, then its own `ConnectDevice` fails: *"USB device 87AD:70DB interface is in use by another process"* |
| `trcc qtgui` headless | **unusable** | forces xcb (see §5); under `xvfb-run` it starts but sits at 0 % CPU and never touches the device |

**The shape of it: the direct-fill path works, the render path does not.** Solid
colours bypass the renderer; images, themes and video all go through it.

## 3. Bug 1 — 62 GB OOM (the worst one)

TRCC was OOM-killed **twice**, both times at ~62 GB resident on a 64 GB box:

```
[Mon Aug  3 06:49:21 2026] Out of memory: Killed process 11960 (trcc)
    total-vm:87024540kB, anon-rss:62395440kB
[Mon Aug  3 07:21:49 2026] Out of memory: Killed process 16694 (trcc)
    total-vm:86616044kB, anon-rss:62358856kB
```

PID 16694 was the `display play` ticker — **the OOM is why the panel reverted to the
Thermalright logo mid-playback**, and why later pushes had no visible effect: the
process driving the screen was dead.

Scale check on the theory that it decodes everything into RAM: 5763 frames ×
1600×720 × 4 B ≈ 26 GB, which is the right order of magnitude but still short of 62 GB
— so there is likely a *second* multiplier (double-buffering, or geometry arithmetic
using mismatched width/height). Worth an `/usr/bin/time -v` or heaptrack run on a
10-frame clip to get the per-frame cost honestly, per the workspace no-extrapolation
rule.

## 4. Bug 2 — the device descriptor is wrong (probable root cause of the render path)

`~/trcc/src/trcc/core/registry.py:106`:

```python
    (0x87AD, 0x70DB): ProductInfo(
        vid=0x87AD, pid=0x70DB,
        vendor="ChiZhu Tech",
        product="GrandVision 360 AIO",
        wire=Wire.BULK, kind=Kind.LCD,
        device_type=4, fbl=72,
        native_resolution=(480, 480),   # <-- WRONG for Levita Vision
        orientations=(0, 90, 180, 270),
        model="GRAND_VISION",
        button_image="A1GRAND VISION",
    ),
```

`trcc device list` therefore announces *"ChiZhu Tech GrandVision 360 AIO (wire=bulk,
resolution=480×480)"* — but the **wire handshake** (`trcc device connect`) reports
**1600×720, model_id 64**. One USB id, two different panels.

`display load-video`'s own help says it transcodes "to the device's native
resolution", i.e. to the **480×480 from this table**, not the 1600×720 the panel
actually is. A wrong-geometry frame is exactly what would make the panel discard
input and show its idle logo — and plausibly what makes the buffer arithmetic
explode in §3.

**The fix is not simply editing 480×480 → 1600×720 in place**: the same `(0x87AD,
0x70DB)` entry legitimately serves GrandVision hardware. It needs a discriminator —
`model_id` (64 here) from the handshake, or the handshake resolution overriding the
static table once connected. Check how `Kind`/`ProductInfo` is consumed by the
renderer before choosing.

## 5. The process/USB-ownership model (this cost hours — read before debugging)

- **Every plain `trcc …` CLI call is its own process.** It opens the USB interface and
  drops it on exit, so nothing persists. Playback needs a long-lived process.
- **`trcc daemon` is a trap here.** It opens the device during discovery, then its own
  `ConnectDevice` reports the interface "in use by another process" — it collides with
  itself. Same failure via `TRCC_DAEMON=1` CLI calls.
- **`trcc shell` is the workable holder** — one `App` across commands. Ours is fed by a
  FIFO held open on fd 3 (`/run/user/1000/trcc.cmd`) so writers can come and go
  without EOF-ing the shell:
  ```bash
  echo "display color 87ad:70db #ff0000" > /run/user/1000/trcc.cmd
  ```
- **`play-video` only ARMS playback; `display play` is the pump.** Buried in
  `~/trcc/src/trcc/ui/cli/display.py:297` — *"Frames advance on each `display play`
  tick"* — with the ticker itself at `:436`: *"Run the render-and-send ticker until
  Ctrl-C. Dispatches TickDisplay every tick … advances an active video playback and
  renders."* Arming without ticking is silent: it logs "playing N frames" and nothing
  moves. **This is the single most misleading thing in the CLI.**
- **The Qt GUI cannot run headless as shipped**: `adapters/render/qt.py:81` sets
  `QT_QPA_PLATFORM=offscreen`, but `ui/qapp.py:54` **pops** it, forcing a real
  platform → aborts without a display. `xvfb-run trcc qtgui` starts (needs
  `xvfb libxcb-cursor0`) but never drives the device.

## 6. Reproducing from scratch

```bash
ssh zen@ryzen5800xt   # IP in homefleet NODES.md
sudo pkill -f "python3 /usr/local/bin/[t]rcc"     # bracket avoids self-match, see §8
trcc device connect 87ad:70db                      # -> 1600×720, model_id 64
trcc display color 87ad:70db "#ff0000"             # panel turns RED (works)
trcc display load-image 87ad:70db ~/media/frame-1600x720.png   # sends, panel unchanged
# arm + pump (will OOM within a minute or two):
trcc display play-video 87ad:70db ~/media/bofuri-30s.mp4 --fps 15
trcc display play 87ad:70db --interval 0.066
```

Media staged on the node (`~/media/`): `bofuri-30s.mp4` (30 s, 1600×720, 15 fps,
9.5 MB — the sane one), `bofuri-1600x720.mp4` (4 min, 52 MB), `bofuri-loop-480.mp4`
(the first, mis-sized 480×480 cut), `frame-1600x720.png` (single frame, 673 KB).
Themes TRCC baked: `~/.trcc-user/single-video/bofuri-30s/Theme.zt` **59 MB / 722
frames**; `…/bofuri-1600x720/Theme.zt` **400 MB / 5763 frames**.

Source clip: `/mnt/user/media/library/anime/BOFURI …/Season 01/… S01E01 …mkv` on the
tower. Re-cut with the tower's docker ffmpeg (Unraid is Docker-only):

```bash
docker run --rm -v "<dir>":/in:ro -v /mnt/user/coefficient/tmp:/out \
  --entrypoint ffmpeg lscr.io/linuxserver/ffmpeg:latest \
  -ss 90 -t 30 -i "/in/<file>.mkv" \
  -vf "scale=1600:-2,crop=1600:720,fps=15" -an -c:v libx264 -preset fast -crf 23 \
  /out/bofuri-30s.mp4
```

## 7. Plan for the source-editing session

1. **Instrument the render path first.** Find where a frame's target geometry is
   chosen (start at `core/registry.py` consumers and `adapters/render/`), and log the
   width/height actually used for `load-image`. If it is 480×480 on this panel, §4 is
   confirmed as the render-path root cause.
2. **Fix the descriptor without breaking GrandVision** — discriminate on the
   handshake's `model_id` (64) / reported resolution rather than editing the shared
   `(0x87AD, 0x70DB)` row in place.
3. **Then re-test the ladder**: colour → still image → 30 s video + ticker. Colour
   already passes, so it is a clean regression baseline.
4. **Chase the OOM separately** — it may vanish once geometry is right, but a display
   utility must not allocate 62 GB regardless. Measure with heaptrack/`time -v` on a
   10-frame clip; do not extrapolate.
5. **Upstream it.** The project explicitly wants a Levita tester and we have unusually
   good evidence (working/broken matrix, OOM dmesg lines, the descriptor mismatch,
   byte counts per push). File the issue even if we carry a local patch.
6. Hot-fix testing goes in `/usr/lib/python3/dist-packages/trcc` (installed) or run
   from the clone; keep the two in sync when reporting.

## 8. Gotchas that cost time (do not rediscover)

- **`pkill -f trcc` kills your own ssh command** — its command line contains "trcc".
  Bit twice. Use `pkill -f "python3 /usr/local/bin/[t]rcc"`.
- **A 400 MB `Theme.zt` blocks `trcc shell` for minutes**; injected commands queue
  silently behind it and look dead. Keep startup theme-free.
- **Nested heredocs over ssh mangle quoting** — write scripts locally, `scp` them.
- **`trcc devices` does not exist** (it is `trcc device list`); `device select` is
  1-based (`select 0` → "ordinal 0 out of range").
- **Log lines lie about success.** "playing Theme.zt (N frames)", "loaded and sent
  (N bytes)" all appear while the panel shows a logo. Only eyes on the panel count —
  and `display color` is the cheapest ground-truth probe.

## 9. Unrelated but adjacent

The RAM work this rode on top of is **finished and validated**: 4×16 GB mixed Corsair
`CMK32GX4M2D3600C18` + Neo Forza `NMGD416E82-4400G` all trained at 3600 MT/s,
48.6 GB/s copy, 1 h y-cruncher over 50 GB = 60/60 algorithm passes, zero errors. See
[`NODES.md`](../NODES.md) and zensysbench
`benchmarks/ryzen5800xt_{baseline_pre-upgrade,post-upgrade_ddr4-3600}_*`.
