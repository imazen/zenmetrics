# nr-classic JND estimates — KonJND-1k, 2026-10-15

Method per metric: `zenmetrics batch --metric <m> --pairs
~/tmp/konjnd-pairs/pairs_abs.tsv --output benchmarks/nr_classic_2026-10-15/
<m>_konjnd.tsv --use-legacy-scheduler` (zenmetrics-cli built with
`--features nr`), which scores all 1,008 KonJND pairs through
`PairedNoReference` — `<m>` on the distorted image, `<m>_ref` on the
reference. JND = median over pairs of `|<m> − <m>_ref|`.

## piqe

|Δ| distribution over the 1,008 pairs (score units, PIQE range 0–100):

| set  | n    | p25       | median     | p75        | min      | max      | mean      |
|------|------|-----------|------------|------------|----------|----------|-----------|
| all  | 1008 | 5.901124  | **12.486448** | 21.169273 | 0.000707 | 70.5910  | 15.03790  |
| JPEG | 504  | 4.478168  | 9.690457   | 17.063066  | 0.000707 | 66.9036  |           |
| BPG  | 504  | 8.609569  | 15.420883  | 24.921583  | 0.031338 | 70.5910  |           |

`piqe::JND = 12.486448`. Parity gate: |Δ_oracle| ≤ 0.012486448.

Output table: `piqe_konjnd.tsv` (141 KB, not committed — regenerable with
the command above; copy at `~/tmp/nr-classic-scratch/piqe_konjnd.tsv` on
i265).

## niqe

|Δ| distribution over the 1,008 pairs (score units, NIQE typical range
~0–10):

| set  | n    | p25       | median    | p75       | min      | max      | mean     |
|------|------|-----------|-----------|-----------|----------|----------|----------|
| all  | 1008 | 0.286027  | **0.602329** | 1.025230 | 0.003430 | 5.264905 | 0.750610 |
| JPEG | 504  | 0.222284  | 0.506864  | 0.847026  | 0.003430 | 4.585708 | 0.645474 |
| BPG  | 504  | 0.370550  | 0.710947  | 1.180932  | 0.004041 | 5.264905 | 0.855745 |

`niqe::JND = 0.602329`. Parity gate: |Δ_oracle| ≤ 0.000602329.

Output table: `niqe_konjnd.tsv` (~120 KB, not committed — regenerable
with the command above; copy at `~/tmp/nr-classic-scratch/niqe_konjnd.tsv`
on i265).

## brisque

pending
