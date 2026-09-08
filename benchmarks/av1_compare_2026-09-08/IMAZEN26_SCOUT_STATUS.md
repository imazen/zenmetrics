# Interrupted training scout: retained results

The full 1,078-input scout is **stopped and incomplete**. All three Nomad allocations report `ClientStatus=complete`, `DesiredStatus=stop`. Serial mode buffered its ledger until the end of the population pass; the configured pass timeout could discard ledger rows for already uploaded successful artifacts. No per-pass ledger object was saved. Resume requires canonical recovery/checkpointing; inspect unpushed fleet work on dev first (currently blocked by SSH authentication).

Two population artifacts remain in the canonical object store. Both archive SHA-256 values match their content-addressed blob keys. The artifact checker matched all declared arms and three rounds, executor/source identities, encoded payload hashes, timing cohorts, and independent SVT reconstruction witnesses: **468 timing rows, 156 cells, 84 reconstruction checks**. This is 2 of 1,078 population jobs, not sufficient evidence for routing or representative selection. The separate successful smoke adds 234 rows for origin 1002.

Example retained QP 27 anchors follow. Time is the median of three rounds. Each origin has its own hardware cohort; do not pool their timings. Equal QP does not imply equal quality. These points do not establish a matched-quality or matched-time winner.

| Origin | Backend/configuration | Bytes | bpp | SSIMULACRA2 | Median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 8402 | zenav1-svt -1 | 7,687 | 0.5089 | 70.9403 | 6236.159 |
| 8402 | zenav1-svt -1 intra-edge | 7,727 | 0.5116 | 71.1160 | 6462.601 |
| 8402 | zenav1-svt -1 restoration-unit search | 7,687 | 0.5089 | 70.9403 | 6259.918 |
| 8402 | libaom 0 | 8,945 | 0.5922 | 74.2238 | 1094.933 |
| 8402 | zenav1-aom 0 | 8,945 | 0.5922 | 74.2238 | 2725.375 |
| 9334 | zenav1-svt -1 | 8,712 | 0.4004 | 70.5195 | 2134.443 |
| 9334 | zenav1-svt -1 intra-edge | 8,605 | 0.3955 | 71.6862 | 2259.895 |
| 9334 | zenav1-svt -1 restoration-unit search | 8,730 | 0.4012 | 70.5911 | 2240.559 |
| 9334 | libaom 0 | 9,757 | 0.4484 | 72.7595 | 1454.911 |
| 9334 | zenav1-aom 0 | 9,757 | 0.4484 | 72.7595 | 3857.454 |

Artifact URIs, source/build/cohort identities and verification counts are recorded in [the preparation receipt](imazen26_training_scout_preparation.json). Local raw status, object inventory, downloaded artifacts and verifier output are under `~/tmp/av1-training-scout-2026-09-08/retained-population` and its parent. No ledger outcomes were fabricated and no completed encode was rerun during collection.
