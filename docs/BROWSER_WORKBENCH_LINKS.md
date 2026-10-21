# Browser workbench links

The browser workbench at https://metrics.imazen.org/ runs metrics in the
page. This file fixes the URL contract other pages (the model dashboard,
docs, issues) use to open it in a given state, and the links it publishes
back. The contract is versioned by this file; additions are allowed,
renames and removals are not.

## Opening the workbench

Parameters go in the query string. A fragment carrying the same
`key=value` pairs (`#metric=psnr&map=squared_error`) is read the same way,
so a link can survive hosts that strip queries. The query wins when both
name a key.

```
https://metrics.imazen.org/?metric=<cli-name>[&map=<map-name>][&codec=<codec>][&q=<quality>][&distance=<jxl distance>][&sweep=jxl][&model=<zensim model>][&display=<cvvdp display>][&ep=<nn provider>]
```

| Key | Values | Meaning |
| --- | --- | --- |
| `metric` | a CLI metric name (`zenmetrics list-metrics`), e.g. `psnr`, `ssim2`, `lpips-alex` | The metric to score. Names are the CLI's `MetricKind` names, never display labels. |
| `map` | a native map name the metric publishes, e.g. `diffmap`, `ssim_0`, `gms` | The spatial map to show. Default: the metric's fused `diffmap` when it has one, else its first map. |
| `codec` | `jpeg`, `webp`, `avif`, `jxl` | Encode the reference with this encoder to make the distorted image. Default: the demo (`jpeg`, quality 40). |
| `q` | 1–100 | Encoder quality for `jpeg`, `webp`, `avif` (and `jxl` when no `distance` is given). |
| `distance` | 0.1–25 | JPEG XL Butteraugli distance (only with `codec=jxl`). |
| `sweep` | `jxl` | Open the JPEG XL impact panel and run its distance sweep for `metric`. |
| `model` | `b`, `c`, `d`, `a`, `preview-v0.1`, `preview-v0.2` | Zensim model (only for `metric=zensim`). |
| `display` | `standard_4k`, `standard_fhd` | CVVDP display (only for `metric=cvvdp`). |
| `ep` | `webgpu`, `wasm` | Execution provider for learned (NN) metrics. Default `webgpu`. |

Rules:
- The workbench never substitutes. A `metric` it cannot run opens with that
  metric selected, its coverage status and the measured reason shown, and no
  score. An unknown metric, map, codec or out-of-range number is reported as
  an error naming the parameter; nothing else is scored in its place.
- An NN metric requested with `ep=webgpu` on a browser without a WebGPU
  adapter fails with the adapter error. It does not fall back to CPU; the
  page offers `ep=wasm` as a separate, labelled choice. Every NN result names
  the execution provider that produced it.
- Unknown keys are ignored, so later additions do not break old links.
- The page rewrites its own URL (`history.replaceState`) as the user changes
  metric, map, codec, quality or distance, so the address bar is always a
  link to the current state.

Dashboard deep link, for a model card's "Run in browser" button:

```
https://metrics.imazen.org/?metric=<cli-name>
https://metrics.imazen.org/?metric=<cli-name>&codec=jxl&sweep=jxl
```

## Coverage file

`https://metrics.imazen.org/coverage.json` (source: `web/metrics/coverage.json`)
lists every CLI metric name with its browser status. It is served with
`Access-Control-Allow-Origin: *`, so the dashboard can fetch it cross-origin.

```json
{
  "schema": 1,
  "workbench": "https://metrics.imazen.org/",
  "contract": "docs/BROWSER_WORKBENCH_LINKS.md",
  "workbench_url": "https://metrics.imazen.org/?metric={name}",
  "dashboard_url": "https://metric-brief.pages.dev/#models&m={name}",
  "metrics": [
    {"name": "psnr", "status": "score+map", "module": "full", "maps": ["squared_error"], "reason": null},
    {"name": "ssim2-gpu", "status": "unavailable", "module": null, "maps": [], "reason": "GPU implementation …"}
  ]
}
```

Every CLI name appears once, in the CLI's order. `workbench_url` and
`dashboard_url` are templates: replace `{name}` with the entry's `name`.
`maps` lists the map names the metric's evaluation published in the check.

`status` is one of:

| Status | Meaning |
| --- | --- |
| `score+map` | CPU WebAssembly score and at least one native spatial map. |
| `score-only` | CPU WebAssembly score; the metric publishes no spatial map in the browser. |
| `nn-webgpu` | Learned metric run by onnxruntime-web on a WebGPU adapter. |
| `nn-cpu-wasm` | Learned metric run in the browser on CPU only. |
| `unavailable` | Not runnable in the browser; `reason` gives the measured cause. |

`module` names the asset group that serves the metric: `core` (loaded with
the page), `full` (the CPU metric module, imported on demand), `nn` (the
learned-metric runtime and that model's weights, fetched on demand), or
`null` when unavailable. Every status
other than `unavailable` is executed in headless Chromium by
`scripts/checkpoint/check-pages.py` before deployment, and the check fails
if the file and the measured behaviour disagree. Learned metrics run on a
WebGPU adapter there (SwiftShader in CI, a subset covering every model kind
by default; the full set's GPU receipt is in
`benchmarks/browser_nn_2026-10-17/`). An `nn-cpu-wasm` entry also carries a
`reason`: its WebGPU result missed the parity gate.

## Links from the workbench

For the selected metric the workbench links:
- its model card on the metric brief: `https://metric-brief.pages.dev/#models&m=<cli-name>`
  (opens the card in the details panel);
- its speed on the brief: `#speed&q=<cli-name>` under `https://metric-brief.pages.dev/`.
  The former `#agreement` link is retired with the GitHub Pages dashboard (2026-10-21):
  the agreement data is outdated and no longer published;
- this coverage entry (`coverage.json`).
