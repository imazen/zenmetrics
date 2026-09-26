# Frozen TRAIN colour report — 2026-09-24

QUARANTINED; reporting only. No selection or promotion from these results.
All four arms share the same 1,539 keyed RGB8 pairs. Labels were first read
at 12:44:40Z after source, population, signs, predictions and both numeric gates
were frozen. No held-out human sets were read. CPU fast-ssim2 was rescored on
worker 2 from main f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4; stored GPU
columns were not substituted. Teacher overlap is an in-sample caveat for that arm.

Signed SROCC from `zen_stats.panel_batch`; reference-clustered percentile CIs
from the existing `band_reliability.marginal_ci`, 2,000 replicates, seed 20260924.
KonFiG has only three reference clusters: its reference-clustered percentile interval degenerates (upper bound 1.0, lower bound ≈ the point estimate), so it is reported as "not estimable (3 clusters)" and is not a 95% CI (Opus review correction 9).

| Set/type | n / refs | Arm | Signed SROCC | 95% CI |
|---|---:|---|---:|---|
| kadid_train/ALL_COLOUR | 1000 / 40 | gmsd | 0.751324 | [0.707220, 0.795131] |
| kadid_train/ALL_COLOUR | 1000 / 40 | mdsi | 0.850117 | [0.825475, 0.875178] |
| kadid_train/ALL_COLOUR | 1000 / 40 | ms_gmsdc | 0.789221 | [0.763427, 0.816631] |
| kadid_train/ALL_COLOUR | 1000 / 40 | fast_ssim2 | 0.857081 | [0.838298, 0.875901] |
| konfig_train/ALL_COLOUR | 39 / 3 | gmsd | 0.647992 | not estimable (3 clusters) |
| konfig_train/ALL_COLOUR | 39 / 3 | mdsi | 0.721683 | not estimable (3 clusters) |
| konfig_train/ALL_COLOUR | 39 / 3 | ms_gmsdc | 0.714375 | not estimable (3 clusters) |
| konfig_train/ALL_COLOUR | 39 / 3 | fast_ssim2 | 0.758833 | not estimable (3 clusters) |
| tid_train/ALL_COLOUR | 500 / 25 | gmsd | 0.250263 | [0.170307, 0.332687] |
| tid_train/ALL_COLOUR | 500 / 25 | mdsi | 0.747839 | [0.692909, 0.798350] |
| tid_train/ALL_COLOUR | 500 / 25 | ms_gmsdc | 0.869164 | [0.839925, 0.896309] |
| tid_train/ALL_COLOUR | 500 / 25 | fast_ssim2 | 0.775872 | [0.728587, 0.820939] |
| kadid_train/04 | 200 / 40 | gmsd | 0.851815 | [0.797449, 0.895965] |
| kadid_train/04 | 200 / 40 | mdsi | 0.868815 | [0.821506, 0.913074] |
| kadid_train/04 | 200 / 40 | ms_gmsdc | 0.825405 | [0.763055, 0.876542] |
| kadid_train/04 | 200 / 40 | fast_ssim2 | 0.883807 | [0.844843, 0.916534] |
| kadid_train/05 | 200 / 40 | gmsd | 0.745262 | [0.647745, 0.832900] |
| kadid_train/05 | 200 / 40 | mdsi | 0.730107 | [0.638891, 0.809598] |
| kadid_train/05 | 200 / 40 | ms_gmsdc | 0.773246 | [0.678082, 0.859469] |
| kadid_train/05 | 200 / 40 | fast_ssim2 | 0.846622 | [0.783710, 0.901770] |
| kadid_train/06 | 200 / 40 | gmsd | 0.866657 | [0.826253, 0.902346] |
| kadid_train/06 | 200 / 40 | mdsi | 0.870084 | [0.832378, 0.903743] |
| kadid_train/06 | 200 / 40 | ms_gmsdc | 0.886034 | [0.854897, 0.913738] |
| kadid_train/06 | 200 / 40 | fast_ssim2 | 0.883076 | [0.845970, 0.916827] |
| kadid_train/07 | 200 / 40 | gmsd | 0.418305 | [0.315277, 0.528917] |
| kadid_train/07 | 200 / 40 | mdsi | 0.552063 | [0.455216, 0.636389] |
| kadid_train/07 | 200 / 40 | ms_gmsdc | 0.468161 | [0.347779, 0.587572] |
| kadid_train/07 | 200 / 40 | fast_ssim2 | 0.532393 | [0.442545, 0.621044] |
| kadid_train/08 | 200 / 40 | gmsd | 0.870053 | [0.834710, 0.908200] |
| kadid_train/08 | 200 / 40 | mdsi | 0.901214 | [0.871793, 0.929780] |
| kadid_train/08 | 200 / 40 | ms_gmsdc | 0.867121 | [0.826347, 0.904709] |
| kadid_train/08 | 200 / 40 | fast_ssim2 | 0.887991 | [0.854047, 0.921479] |
| konfig_train/colordiffusion | 39 / 3 | gmsd | 0.647992 | not estimable (3 clusters) |
| konfig_train/colordiffusion | 39 / 3 | mdsi | 0.721683 | not estimable (3 clusters) |
| konfig_train/colordiffusion | 39 / 3 | ms_gmsdc | 0.714375 | not estimable (3 clusters) |
| konfig_train/colordiffusion | 39 / 3 | fast_ssim2 | 0.758833 | not estimable (3 clusters) |
| tid_train/02 | 125 / 25 | gmsd | 0.868213 | [0.828403, 0.903488] |
| tid_train/02 | 125 / 25 | mdsi | 0.879362 | [0.840669, 0.913265] |
| tid_train/02 | 125 / 25 | ms_gmsdc | 0.873690 | [0.835077, 0.907422] |
| tid_train/02 | 125 / 25 | fast_ssim2 | 0.847988 | [0.802618, 0.890912] |
| tid_train/18 | 125 / 25 | gmsd | -0.293797 | [-0.498103, -0.049672] |
| tid_train/18 | 125 / 25 | mdsi | 0.800072 | [0.692019, 0.893585] |
| tid_train/18 | 125 / 25 | ms_gmsdc | 0.629508 | [0.507403, 0.774919] |
| tid_train/18 | 125 / 25 | fast_ssim2 | 0.772926 | [0.659021, 0.876925] |
| tid_train/22 | 125 / 25 | gmsd | 0.910224 | [0.879747, 0.941628] |
| tid_train/22 | 125 / 25 | mdsi | 0.913385 | [0.882207, 0.945595] |
| tid_train/22 | 125 / 25 | ms_gmsdc | 0.920995 | [0.897447, 0.948149] |
| tid_train/22 | 125 / 25 | fast_ssim2 | 0.909674 | [0.885291, 0.936328] |
| tid_train/23 | 125 / 25 | gmsd | 0.852813 | [0.748172, 0.928634] |
| tid_train/23 | 125 / 25 | mdsi | 0.882374 | [0.806555, 0.937380] |
| tid_train/23 | 125 / 25 | ms_gmsdc | 0.865335 | [0.765997, 0.934324] |
| tid_train/23 | 125 / 25 | fast_ssim2 | 0.824985 | [0.698817, 0.913452] |

Raw report: `/var/tmp/gmsd-chroma/colour_v2/report.json`, SHA256
`1d1eef5c1e67950230b210b55432434f846a188bcf2e609e1ed3adbb3cc810d8`.
Frozen input SHA256 `c5db39717f6f5c317634c7f7d35ef6d83bf7fc6af6bdb20e1ea1e2672717bb68`.
Peer table SHA256 `e5dc78f8e984d74120517582a00b055171af5976a227ab05035c66bf6eb37722`.

Exact re-execution uses `remote_compute.sh colour-report` with the recorded
worker environment, immutable inputs and owner hashes. Preserve/rename the old
output before any replay. Verbatim commands/timestamps/exit codes/log hashes are
in `/var/tmp/gmsd-chroma/commands.jsonl` (`colour_freeze_v1`,
`remote_colour_report_v1`); the remote heavy log is retained alongside the local logs.

## Re-run with the clean-room MDSI (2026-09-25)

The MDSI arm was re-scored with the clean-room implementation (implementation commit `19b8c03c87934b1df2e078aa689534d17777713c`, predictions `colour_v3/predictions.tsv` sha256 `8e71a70901d020c592956687d7c5d19feb0873aed60b8c46943dd2ccae5ec646`, report `colour_v3/report.json` sha256 `8663ff467048c8ca05185f8b56c51a28b73ff6b583c700b5db5ac44925bd8593`) through the same frozen population, panel and statistics owner. All 52 rows (point SROCC and reference-clustered interval) are **identical to the table above** to the stored precision, as expected from scores that agree to about 1e-10. Same TRAIN roles; no held-out label read. The KonFiG intervals remain not estimable (3 clusters).
