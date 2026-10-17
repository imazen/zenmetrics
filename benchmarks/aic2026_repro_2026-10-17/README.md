# AIC2026 published columns never compared before: WaDIQaM, WD, PieAPP (2026-10-17)

Our implementations scored on the 61 AIC2026 crop pairs (840×944, the public
cropped set; `metrics_subset.csv` holds the published values) and compared
with the published columns. Scores are in the per-metric TSVs (image names
relative to the cropped set). Binary: `zenmetrics` built with
`--features nn,nr` from the CI-pinned snapshot of master (see the run log
hash in the lead's notes), rten engine, 8 threads, models from the frozen
exported sets, weights consent as in `benchmarks/iqa_compete_2026-10-16/run.py`.

commit: 783c64c6
command: zenmetrics batch --metric <m> --pairs pairs.tsv --output <m>.tsv --jobs 1
compare: scripts/complexity/aic2026_column_compare.py metrics_subset.csv <scores_dir> wadiqam-fr-live WaDIQaM

## WaDIQaM: not reproduced by any official checkpoint

| ours | score used | Pearson | Spearman | our range |
|---|---|---:|---:|---|
| wadiqam-fr-live | score | −0.732 | −0.741 | 18.7…46.4 |
| wadiqam-fr-tid | score | −0.520 | −0.486 | 9.6…35.1 |
| wadiqam-nr-live | score | −0.479 | −0.474 | 0.0…63.6 |
| wadiqam-nr-live | dist − ref | −0.451 | −0.411 | −54.1…63.4 |
| wadiqam-nr-tid | score | +0.107 | +0.177 | 13.4…59.3 |
| wadiqam-nr-tid | dist − ref | −0.224 | −0.219 | −20.8…15.2 |

The published column ranges −0.039…0.018 (median −0.013). Our outputs are
DMOS-like (higher is worse), so a negative correlation is the expected sign;
its magnitude (at best 0.74) and the scale show the column comes from a
different model or normalisation than the official `dmaniry/deepIQA`
`top="weighted"` checkpoints.

## WD: ranks agree, scale does not

Default `wd` (five scale outputs) on the same 61 pairs:

| column | Pearson | Spearman | published median | ours median | pub/ours (median ratio) |
|---|---:|---:|---:|---:|---:|
| WD_s1 | 0.937 | 0.944 | 1.885 | 0.570 | 3.39 |
| WD_s2 | 0.906 | 0.923 | | | |
| WD_s3 | 0.856 | 0.894 | 1.134 | 0.232 | 4.71 |
| WD_s4 | 0.820 | 0.859 | | | |
| WD_s5 | 0.811 | 0.858 | 0.423 | 0.100 | 3.99 |

The ratio differs per scale and per pair, so no single constant maps ours to
the published columns; agreement also falls with scale. Consistent with the
earlier convention search (`benchmarks/wd_port_2026-10-16/`): the published
pipeline differs in its per-scale mapping or feature normalisation.

## DVIFM: the released builds do not reproduce the columns

`dvifmish` port mode (Rust ports of the author's four released C++ variants,
agreeing with his builds to 1e-7–5e-6 JND) on the same 61 pairs, JND output:

| variant | vs `proposal-DVIFM` | vs `proposal-DVIFM-0.2` | vs `-0.2-use_chroma` |
|---|---:|---:|---:|
| port-min-v0.2 | 0.611 | 0.879 | 0.865 |
| port-2d-v0.2 | 0.655 | 0.824 | 0.755 |
| port-lap-v0.2 | 0.733 | 0.931 | 0.809 |
| port-lap-v0.3 | 0.751 | 0.941 | 0.830 |

(Spearman; median |Δ| 0.66–1.11 JND.) The releases were retuned on the AIC2026
training data, after the columns were computed, so the columns came from
earlier builds. Command: `dvifmish batch --pairs pairs.tsv --out-dir scores
--preset port-min-v0.2 --preset port-2d-v0.2 --preset port-lap-v0.2 --preset port-lap-v0.3`.

## IDFIQA: reproduced by the authors' AIC-4 script

The authors' AIC-4 evaluation script (`amirkhaki/variance_guided_iqa`,
commit 40df3a5, `aic4-eval.py`, single-pair mode, CPU, torch 2.14.1:
EfficientNet-B4 at 380×380, top 70 % of channels by reference variance, Gram
matrices, 4×4-window structure term, `JND = 15·max(0, 1 − s)`) reproduces
`proposal-IDFIQA`: median |Δ| 4.7e-6, max 3.3e-4 JND, Pearson and Spearman
1.0000. The repository has no licence, so neither the code nor its scores
per pair are included here. The later `amirkhaki/idfiqa` experiment
repository's VGG16 baseline is a different model (Spearman −0.88).
