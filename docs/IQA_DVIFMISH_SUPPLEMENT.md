# dvifmish durable IQA supplements

dvifmish is a benchmark-only optional dependency of the unpublished
`cpu-profile` driver. Enable it with `compete-dvifmish`, or set
`IQA_DVIFMISH=1` when using `just iqa-compete-build durable`. The dependency
is pinned to `74ec5eeae118ead266c7b66b56e56cb5f3bef372`. Its manifest retains
`publish = false`; this work does not publish the crate, binaries or weights.
No metric-library or public CLI API is added.

The six routes are the four `port-min-v0.2`, `port-2d-v0.2`,
`port-lap-v0.2`, `port-lap-v0.3` presets, prefixed `dvifmish-`, and
`dvifmish-talk-faithful-luma-float` / `dvifmish-talk-faithful-luma-int`.
The latter use the default talk-inspired preset. Other fitted presets and
the streaming API are outside this supplement's coverage.

Port scalars are JND values. Talk-inspired scalars are raw distortion E.
Every native map is the library's per-plane/per-level block contribution
field in raw distortion units. It is not a full-resolution JND map. The
inventory and each timing row retain these units, the preset name and its
SHA-256. Timed map calls request and consume the actual native field; they
do not paint, interpolate or substitute scalar-only work.

Preset parsing and validation happen inside the existing constructor-inclusive
cold clock. First-call integer table initialization is also cold work.
Each cold sample has a fresh process. Warm calls reuse parsed constants and
run the same ten zenbench rounds with 100-ms targets as the base study.
RGB loading remains outside the compute clock. Both library APIs receive
the explicit one/eight-worker budget; small levels may use fewer workers.
Linux uses the existing checked P-core boundary and shared heavy lock.

Adapter controls compare these six routes against direct library calls with
exact scalar/map/thread bits, including odd dimensions. They do not loosen
the crate's existing numerical controls or establish new perceptual parity.
The pinned lock resolves through the existing isolated-snapshot check.
