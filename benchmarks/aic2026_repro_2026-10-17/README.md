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
