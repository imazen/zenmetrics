# Fixed-factor oracle patching (benchmark glue)

`fixed_factor_oracle.py` writes **scratch copies** of verified oracle
sources with the automatic downscale-factor assignment replaced by a
literal `k` in {1,2,3,4}; `check_fixed_factor_oracle.py` independently
verifies each produced copy differs from its source at exactly one byte
region and runs the negative-control suite. Pure stdlib; originals are
never modified; no metric/oracle code is executed; no network.

Roles: **primary** = the oracle that binds the ported metric's parity
contract — the authors' MATLAB implementation for FSIM/VSI, and pyiqa
for the `*-pyiqa` SSIM-family metric names (whose oracle is pyiqa
itself). **diagnostic** = a separately labeled variant comparison —
pyiqa's FSIM/VSI patch does not displace MATLAB parity.

| name | role | source | factor line |
|---|---|---|---|
| fsim-matlab | primary | author release `FeatureSIM.m` ([FSIM files](https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m), edu/research-only) | `F = max(1,round(minDimension / 256));` @72 |
| vsi-matlab | primary | author release `VSI.m` ([VSI files](https://cslinzhang.github.io/home/VSI/Files/VSI.m), edu/research-only) | `F = max(1,round(minDimension / 256));` @55 |
| fsim-pyiqa | diagnostic | pyiqa 0.1.16 [`pyiqa/archs/fsim_arch.py`](https://github.com/chaofengc/IQA-PyTorch/blob/18dd7a19694e94aac21019170e3f5e63d6b4e19e/pyiqa/archs/fsim_arch.py) | `kernel_size = max(1, round(min(x.shape[-2:]) / 256))` @77 |
| vsi-pyiqa | diagnostic | pyiqa 0.1.16 [`pyiqa/archs/vsi_arch.py`](https://github.com/chaofengc/IQA-PyTorch/blob/18dd7a19694e94aac21019170e3f5e63d6b4e19e/pyiqa/archs/vsi_arch.py) | `kernel_size = max(1, round(min(vs_x.size()[-2:]) / 256))` @118 |
| ssim-pyiqa | primary | pyiqa 0.1.16 [`pyiqa/archs/ssim_arch.py`](https://github.com/chaofengc/IQA-PyTorch/blob/18dd7a19694e94aac21019170e3f5e63d6b4e19e/pyiqa/archs/ssim_arch.py) | `f = max(1, round(min(X.size()[-2:]) / 256))` @47 |

Upstream: MATLAB files are the authors' releases (edu/research-only
terms in each file header); pyiqa is
[github.com/chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
[tag v0.1.16](https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.16)
(commit `18dd7a19694e94aac21019170e3f5e63d6b4e19e`, matching the
installed wheel's `scm_version.json` node — verified byte-identical to
the installed package files, see `fixed-factor-upstream-binding.md` in
the handoff dir). Every source is bound by full sha256 in the registry;
any drift is a loud failure, not a re-patch.

## Source resolution (no built-in host paths)

Each registered source carries a `relpath` (`fsim/FeatureSIM.m`,
`vsi/VSI.m`, `pyiqa/archs/<file>.py`). Resolve explicitly:

- `--source-path <name>=<path>` (repeatable) — exact file, or
- `--source-root <dir>` (repeatable) — the first `<dir>/<relpath>` that
  exists wins; the registered sha256 still binds either way.

## Runtime contracts (recorded, applied at the call site)

- **pyiqa SSIM/MS-SSIM**: the patched `f` takes effect only when the
  caller passes `downsample=True` — a constructor/function argument, not
  a source mutation.
- **pyiqa `ms_ssim`**: with `downsample=True` the patched factor applies
  at **every pyramid level** (per-level consequence). This patcher fixes
  `f` once in the shared `ssim()`; there is no separate pre-pool mutation.

## Usage

```sh
python3 fixed_factor_oracle.py list --source-root /path/to/mirror-root
python3 fixed_factor_oracle.py patch --name vsi-pyiqa --factor 4 \
    --source-path /path/to/site-packages/pyiqa/archs/vsi_arch.py \
    --out /path/to/vsi_arch.k4.py                        # creates copy + .manifest.json
python3 check_fixed_factor_oracle.py check \
    --manifest /path/to/vsi_arch.k4.py.manifest.json      # independent diff check
python3 check_fixed_factor_oracle.py selfcheck \
    --evidence /path/to/fresh-evidence-dir \
    --source-root /path/to/mirror-root \
    --source-root /path/to/site-packages                  # 5 sources x 4 factors + negatives
```

Write discipline: source, output and manifest must resolve to distinct
paths before anything is read or written (alias → loud failure). Every
artifact is created with `O_EXCL` — an existing *different* file is
preserved and fails loudly; an existing *identical* output is reported
`exists-identical`, not rewritten. Re-running selfcheck requires a fresh
evidence dir. Other loud failures: hash drift, duplicate or missing
target line, illegal factor.

The checker binds the manifest to the registry: source sha256 must equal
`spec.sha256` (a co-rewritten source+manifest is still rejected), and
schema, role, origin, runtime_contract, target bytes and output status
must match the registry exactly.

Semantics note (from `fixed-factor-current-source.md`): Zenmetrics'
`decimate` windows for k2 are interior-aligned with `avg_pool2d(2)`;
k3/k4 are shifted −1. The patched-oracle experiments target k4 for the
interior-alignment diagnostic and odd-dimension k2 for ceil/floor tail
behavior; all outcomes are measured, none presumed.
