# avif-autotune-2026-09-04 — pointer

The canonical AVIF autotune training view + the two v1 bakes. Nothing here is
in git: the view alone is 14 MB. Record:
[`avif_autotune_v1_2026-09-04.md`](avif_autotune_v1_2026-09-04.md). Consumer
contract (tracked copy): [`avif_autotune_contract_2026-09-04.md`](avif_autotune_contract_2026-09-04.md).

## Where

| where | path | state |
|---|---|---|
| local | `/mnt/v/zen/avif-autotune-2026-09-04/` | 25 files, 14133761 B |
| LAN | `s3://zentrain/analysis/avif-autotune-2026-09-04/` | see §mirrors below |
| Tower | `/mnt/user/coefficient/output/avif-autotune-2026-09-04/` | see §mirrors below |

## Build commits

```json
{
  "zenanalyze": "18971ef9ec3d69121cee2623a360447368e9f2eb",
  "zenmetrics": "3351b53d7ffeea16d97f20d8240298670a28e595",
  "zensim": "ccef06991a971b484725a4677fba44fe038ede52"
}
```

## Files (sha256)

| file | bytes | sha256 |
|---|--:|---|
| `AUTOTUNE_CONTRACT.md` | 14463 | `44c4ffd2ea2b4696c93d4f90fd06e8980b665d2d16f856216e99a11f5570814c` |
| `_MANIFEST.json` | 12726 | `6c457787c47579b19e2d04a8ed85ba1627fead0d87fd6f73f15821c9a7a6f472` |
| `cells_hdr.parquet` | 254691 | `22171301c7a4686818c6f1fa996fc547cdd116227782780f5172489d14d70810` |
| `features_avif_autotune.tsv` | 83521 | `fb5378250ec89b03f206694e6dcc6a890da8bc92c79437fe65976b86df33c68f` |
| `models/zenavif_autotune_v1_core.bin` | 161756 | `c2bfb016cbf2c02cb948e6e57bf608599716eeb7276f83f814e18ce50de7f9f4` |
| `models/zenavif_autotune_v1_core.json` | 2437108 | `fab9cd06d36d41eacc7ddd7bbe819c1f8a6e063a5f72a919ce409276f481d077` |
| `models/zenavif_autotune_v1_core.log` | 3215 | `46e4fe676461ab14e122b2aa2dbabf8587d19198f82f734b62cb95d9ff9ef82a` |
| `models/zenavif_autotune_v1_core.manifest.json` | 278817 | `958d4fb026aebb3a073dc6cb9d60d390feb5a1d72c81ef94a4f5d7094c2ca059` |
| `models/zenavif_autotune_v1_full.bin` | 270572 | `ef91a4b971189a2a9640abcd116af0a0c645e51ab5030476e6020d4b804fd478` |
| `models/zenavif_autotune_v1_full.json` | 4087966 | `ef68b8d5abc09b6219bd80f11a6a96afa3aaa1752979fdb136a1311d8e41ad2b` |
| `models/zenavif_autotune_v1_full.log` | 8674 | `c4944b099093e206e192fd01b3bd9519b6291aa7bb3245b2af305273e1eb9e24` |
| `models/zenavif_autotune_v1_full.manifest.json` | 708372 | `2d712aac77fd951cf850b1b087c2c2d8168294819b92c23816c86d5afa8fb6dd` |
| `pareto_avif_autotune.parquet` | 3662245 | `aa7f055a8ff63bf6a7d8de244872502bd3eb352cba87054a4beca25cd7cb57e4` |
| `pareto_avif_autotune_core.parquet` | 1990175 | `49bd24701bcfabe00710ab6cd31a9faca66769faa73c552c158041d912c3fed4` |
| `sidecar_alias_arms.tsv` | 1035 | `91139fafbc4ee6912d0c521c79ee9758fd4602e78783078b04bb3b208c91e788` |
| `sidecar_inert_arm_census.tsv` | 7123 | `43d8d01a249a759ea6d956178e5c024099d0f49219d63fd3ee436513abf66bdb` |
| `sidecar_speed_alias.tsv` | 104 | `13ad8a3e3759551de439bf3e6522461e67cb88c1ad788bc24861c82794fe0857` |
| `validation/zenavif_autotune_v1_core_cellmap.json` | 12457 | `74f0ee7788527b95345e75410b77edde2e3b0b44b6329755cfa1d5785b03c49d` |
| `validation/zenavif_autotune_v1_core_encode_ms_lut.json` | 7187 | `5bc51ca5b6adc905197de701a1a1f6a6d72a5283b96ae113a909139e9575318d` |
| `validation/zenavif_autotune_v1_core_quality_lut.json` | 16751 | `6f6f6ade8cbe5f1e16adc2b8bcfca141e69422688d1ac22ad326aa35922698be` |
| `validation/zenavif_autotune_v1_core_validation.json` | 2709 | `56650f959ed922adaeac5d38c2472897156a003872f347cbd4f9488abeccd447` |
| `validation/zenavif_autotune_v1_full_cellmap.json` | 37787 | `9ec3c4757aadb50d1e81df156ecee7ef8bd14b20eea7568ac0c99002c3547b43` |
| `validation/zenavif_autotune_v1_full_encode_ms_lut.json` | 20935 | `f9becca9d2ad0250a7cfe6b159fba51018fece8c3334802ff04e8d41c4130020` |
| `validation/zenavif_autotune_v1_full_quality_lut.json` | 49208 | `918ad253f481ad57d686b44b62c30e11cdadb03d4d8fde1000613116568a2ef2` |
| `validation/zenavif_autotune_v1_full_validation.json` | 4164 | `2b085f12bad1b3cab5b76c2e12d2280de4cb431424e92063fcde0108a653dd42` |

Regenerate: see §6 of the contract. Every number in the record is reproducible
from these files plus the two `scripts/jobsys/avif_autotune_*.py` tools.
