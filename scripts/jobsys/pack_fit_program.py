#!/usr/bin/env python3
"""Pack the preregistered P0 program and current-main Rust binaries for the image."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


SCRIPTS = {
    "scripts/rev4_featpot/mlp_probe.py": "4e7b18d0c5cd28f633af249111f2f06cb988db376b3429b3783a25f8e0e0617a",
    "scripts/rev4_featpot/data.py": "2906e383aa3891a94154cafdaec24a770da72baa64e5a0cbf689845455a7916c",
    "scripts/rev4_featpot/admit_bank.py": "e80ec845dd4d2c186d43e1cf76a23168ee240ab33ac137ffb31571bfa07cd4cf",
    "scripts/rev4_featpot/linear_probe.py": "1deef7b7d49b375b2ee51d32ca586b82a14b3c14ba5472451e4c0a7de86820b7",
    "scripts/rev4_featpot/mlp_importance.py": "6d28cdd9ecb1784be7194947e16287531354372436411a4511af0255bfafb559",
    "scripts/rev4_featpot/stability_lasso.py": "8dd081008595780ca49a40fd3e560d286bc9d783cef4c8b6a0e4ab2ab5fc62f2",
    "scripts/lib/zen_stats.py": "6e2bed69441195674e3a397f11a8eeb4b40e51dd271ebbd124e6702b68eb3d35",
}
# P2 (GMSD-peer MLP) and D2 (source-held-out MLP) program era: the potential lane's committed tip
# b168bba26ad0 (quarantine/codex/featbank-potential), files pinned by sha.
P2D2_SCRIPTS = {
    "scripts/rev4_featpot/admit_bank.py": "e80ec845dd4d2c186d43e1cf76a23168ee240ab33ac137ffb31571bfa07cd4cf",
    "scripts/rev4_featpot/data.py": "2906e383aa3891a94154cafdaec24a770da72baa64e5a0cbf689845455a7916c",
    "scripts/rev4_featpot/linear_probe.py": "24d7ac8b1e90170358f749d7fe12d09c7f911b4d4ae8775a179e6ed135e03a1d",
    "scripts/rev4_featpot/lodo_bvls.py": "43d2f445afe7658c14177870718629fcd5a3390082ae27ee22d5ca07e09f9081",
    "scripts/rev4_featpot/mlp_probe.py": "df6e6aebbced4debcb3e297707ec17b4bdb78786ecee70d0f57180271031d787",
    "scripts/rev4_featpot/p2_data.py": "d30149c9f322095d98e124388d441c7258176ba20ee8b9e5c0c70ec7d58111a8",
    "scripts/rev4_featpot/p2_lodo_mlp.py": "8bbca5b0f937979222df8ff21ff10d6fa4a897c2dce4cb39bf44c7c4b7c14745",
    "scripts/rev4_featpot/p2_mlp_importance.py": "aad42677c9fe93f79ab735c2b218755af0f7957aa3f2902b6a947fc993f4d1b5",
    "scripts/rev4_featpot/p2_mlp.py": "7fba1eb80e5345db0b05dab6bd4f7b842e95239871613d2e56ea3ff14be5b0b3",
    "scripts/rev4_featpot/stability_lasso.py": "8dd081008595780ca49a40fd3e560d286bc9d783cef4c8b6a0e4ab2ab5fc62f2",
    "scripts/lib/zen_stats.py": "6e2bed69441195674e3a397f11a8eeb4b40e51dd271ebbd124e6702b68eb3d35",
}
# Rev4 potential Instrument v2 (R915 sampling): zensim commit 1ee08bae (v2 scripts), e7248c65 (zen_stats)
# (benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R1 + erratum R1.1 two-family layout + revision R3:
# distance-oriented oracles, `<spec>@h<w>` human-leg weight), files pinned by sha. Its bin-dir is the R1.1 mixed set:
# v8 trainer + panel, bake_dial_refit from zensim 86fc02bb (admitted by
# benchmarks/rev4_featpot_v2_predictor_parity_2026-10-01.json). Previous pins (zensim cce9ab19): v2_common 1e22400d…,
# v2_lodo_mlp 6d7d2d60….
V2_SCRIPTS = {
    # zensim ecc69a1f: amendment R4 (EPOCH_RULE = "last", train_and_select), CANONTAB v2c --root support, R2.3 screen specs.
    # zensim 3fc72546 (design log E23): seed indices 10-19 as a second disjoint stream; 0-9 unchanged.
    # zensim e930eb5a (Rev5 tables): refuse_nonfinite_kept, table_revision, dense_bake (was bbb5f682 at 3fc72546).
    "scripts/rev4_featpot/v2_common.py": "efc6ad5623cdb882b2956fcf9fdf8c369b2acd3c4930aaea47ef55528918c8e6",
    # zensim 538d3549 (TRAINEROPT): dev panels every 17th epoch under EPOCH_RULE == "last", sparse read_curve; the
    # trajectory and final weights do not depend on --log-every (gate: zensim benchmarks/traineropt_WORKLOG.md).
    # zensim 3fc72546: --seed-index accepts 0..19 (v2_common.N_SEEDS); otherwise as c156fb4c.
    # zensim e930eb5a: NaN guard on kept columns + dense-bake predict on Rev5 tables (was 1f581300 at 3fc72546).
    "scripts/rev4_featpot/v2_lodo_mlp.py": "cb9ca9282ce739a12dd0bd40fee71e89806f2edfca10964c2eaa5039af657eb4",
    # zensim 6ffcb814 + c482fddf2a9c (design log E13): curated SafeSyn teacher legs, recipe token ts<rule>; uuid-named scratch copy.
    # zensim 9dfc50b8: the E15 coverage-pool pin follows the pool manifest's formula_revision (Rev4 6b00349c, Rev5 6bf584ac;
    # was 6916ee53, which refused the Rev5 pool and failed every E24 cell).
    "scripts/rev4_featpot/v2_teacher.py": "0b511e7daad3c26129b0937bce9283287fa8bfe59d3b4415e8e67dfcad6fddaf",
    # Revision R2 confirmatory full-data fits (features only).
    # zensim 01a662b3: set:/sel: specs + the cv/cf coverage leg (set-compare confirmatory read, 2026-10-04).
    # zensim e930eb5a: the same NaN guard (was 24a417b3 at 01a662b3).
    "scripts/rev4_featpot/v2_confirm_fit.py": "2d1a92c0fb6fc4a64eea9ba8e0c7da39817ed24932ddade956737f9c994634af",
    # zensim e7248c65 (EFFAUDIT D8: render_indexed_jobs + rendered_jobs=; panel_batch unchanged).
    "scripts/lib/zen_stats.py": "68532bad3b3482bc734508183872d9253893cbacd9d3299d523894fc38158a6a",
    # Design log E5/E5b epoch-selection cells.
    "scripts/rev4_featpot/e5_epochs.py": "059e5eeb4d2efc167b9e99ab95e4f0b689f1f8935eb7c62cc78ad640ebce4f70",
}
# Program v25b (2026-10-04): v25 + zensim 9dfc50b8 v2_teacher (revision-aware E15 pool pin); image fit-v2r5-v25b-w<worker>.
# v25's E24 jobset failed all 100 cells on the Rev4-only pool pin and was paused before any cell succeeded.
# Program v25 (2026-10-04): V2_SCRIPTS at zensim e930eb5a (Rev5-table guards); profile v2r5 adds V2R5_DATA for the Rev5 root
# (image zenfleet-worker:fit-v2r5-v25-w<worker>).
# Program v24 (image zenfleet-worker:fit-v2-v24-w<worker>, 2026-10-04): v23 + zensim 01a662b3 v2_confirm_fit (set:/sel: specs and
# the coverage leg) for the set-compare confirmatory fits.
# Program v23 (image zenfleet-worker:fit-v2-v23-w<worker>, 2026-10-04): v21 binaries and data + zensim 3fc72546 v2_common /
# v2_lodo_mlp (seed indices 10-19; indices 0-9 byte-identical in effect) + the current fit_cell_exec.py (ZEN_ERROR_CLASS, per-cell
# flock). Numbered v23 because the tag fit-v2-v22 already names a program-v21 image (superseded, see CLAUDE.md).
# Program v21 (image zenfleet-worker:fit-v2-v21, 2026-10-03): v20 + zensim a7a3168d scripts (design log E15: cv<w>:cf<mask>
# coverage leg over the pinned pool) + the E15 coverage pool, keys and manifest (V2_DATA).
# Program v20 (image zenfleet-worker:fit-v2-v20, 2026-10-03): v19 with the compliant E14 table (20 KADIS types; 6/9/10/15 are
# third-party generated and excluded, zensim 6c5d38fa) and the matching v2_teacher pin. v19 was retired before any E14 cell finished.
# Program v19 (image zenfleet-worker:fit-v2-v19, 2026-10-03): v18 + zensim d1587e29 scripts (design log E14: ko<w> KADIS
# ordinal ladder leg; e13 scoring shared) + the E14 ordinal table and manifest (V2_DATA). Specs without ko/ts run as v17.
# Program v18 (image zenfleet-worker:fit-v2-v18, 2026-10-03): v17 binaries + zensim 6ffcb814 scripts (design log E13: ts<rule>
# SafeSyn teacher curation, v2_teacher.py) + the E13 strata data file (V2_DATA). Specs without a ts token run exactly as v17.
# Program v17 (image zenfleet-worker:fit-v2-v17, 2026-10-02): v16 + zensim_mlp_train 605d20e0 from zensim e176cc1e (TRAINEROPT3:
# pair forward in one w1 walk + the next pair's forward fused into the Adam row walk; 25/25 cells byte-identical to f9d076c6).
# Program v16 (image zenfleet-worker:fit-v2-v16, 2026-10-02): v15 + zensim_mlp_train f9d076c6 from zensim f7270995 (TRAINEROPT2:
# group-lasso prox fused into the w1 Adam rows; 12/12 cells byte-identical to a5f40576) + zensim bfd22c24 sel:<id> subset specs.
# Program v15 (image zenfleet-worker:fit-v2-v15, 2026-10-02): v14 + zensim 324078f6 (set:/core+ specs resolve candidate arms
# from the pinned keep lists; v14 imported restore_data, which the program does not pack, so every set: cell failed).
# Program v14 (image zenfleet-worker:fit-v2-v14, 2026-10-01): v13 + zensim 34c41507 set:<groups> specs (E9″, no exempt core).
# Program v13 (image zenfleet-worker:fit-v2-v13, 2026-10-01): v12 + zensim a14a30d4 multi-group core specs (core+X+Y, E9′).
# Program v12 (image zenfleet-worker:fit-v2-v12, 2026-10-01): v11 + zensim 30c9ca7d block specs (r0-<block>, core,
# core+<x>; design log E9).
# Program v11 (image zenfleet-worker:fit-v2-v11, 2026-10-01): v10 binaries + zensim fb24f308 scripts (recipe tokens
# @h<w>:H<n>:gl<lambda>, design log E8; specs without tokens run exactly as before).
# Program v10 (image zenfleet-worker:fit-v2-v10, 2026-10-01): v9 + zensim_mlp_train from zensim 538d3549 (TRAINEROPT: fused
# Adam visits only kept rows; byte-identical final weights; sha a5f40576) + the v2_lodo_mlp pin below.
# Program v9 (93dc93d0, image zenfleet-worker:fit-v2-v9, 2026-10-01): these V2_SCRIPTS + v8 trainer/panel (zensim cafed5ca)
# + bake_dial_refit from a local merge of zensim main b926258e and pr/signedfeat a659715e, so bakes reading the
# SIGNEDFEAT columns f1825-f1852 load (zensim benchmarks/rev4_featpot_effaudit/v9_predictor_gate_2026-10-01.json).
# Data files a v2 program carries (--data NAME=PATH), pinned by sha. E13 strata: per-row codec/quality of the v2-canon SafeSyn
# fit table (zensim benchmarks/e13_safesyn_strata_2026-10-03.pointer.md; rebuilt by e13_teacher.py strata).
V2_DATA = {
    "data/e13/safesyn_fit_strata.npz": "6baf5d1b2cb012963cdfa19d94ea2717bca17066a49cf9660979669d5662929c",
    # E14 KADIS ordinal ladder table + its trainer manifest (zensim benchmarks/e14_kadis_ordinal_2026-10-03.pointer.md).
    "data/e14/kadis_ordinal.parquet": "ffc245a0e39bd85d7527a08fd96bdd93b9fb266ad0334d4401eca06c95812012",
    "data/e14/kadis_ordinal.parquet.manifest.json": "5b47e46181dfba7ab0995ec684cf2b9fcc632754f76bbf9471939398adcf5e31",
    # E15 coverage pool + its row keys + trainer manifest (zensim benchmarks/e15_coverage_pool_2026-10-03.pointer.md).
    "data/e15/coverage_pool.parquet": "6b00349c8aca6613aeb1591f8411e738e3c70798274c7df9017dfbc8844848b3",
    "data/e15/coverage_pool.keys.parquet": "bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf",
    "data/e15/coverage_pool.parquet.manifest.json": "b11ef05944a656acb64a0cabf7f88abd748d8ff0cada1a7f240f20a738c3c068",
}
# Rev5 tables (zensim benchmarks/rev5_spec_2026-10-04.md §6): the v2r5 profile carries the Rev5 E15 coverage pool (same 42,021
# rungs / 9,594 ladders, basic+peaks+v2 at Rev5, every other slot NaN) and the same SafeSyn strata (row order unchanged). The
# E14 KADIS ordinal table is a Rev4 extraction and is NOT shipped (a ko cell on a Rev5 root then fails loudly instead of mixing).
V2R5_DATA = {
    "data/e13/safesyn_fit_strata.npz": V2_DATA["data/e13/safesyn_fit_strata.npz"],
    "data/e15/coverage_pool.parquet": "6bf584ac70579bdf9a0242b7ccfd688ff0182cb5c75e35085ba71967207f8f5e",
    "data/e15/coverage_pool.keys.parquet": "bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf",
    "data/e15/coverage_pool.parquet.manifest.json": "a5992e8a3f25771964b281b1a17db713952d5e7c0a6c65e53ad31990e213e6a5",
}

# E26: native HDR fit-only leg. Its admitted TRAIN table is carried in the
# content-addressed data pack, never the program or a VAL/confirmation leg.
V2R5HDR_SCRIPTS = {**V2_SCRIPTS,
    "scripts/rev4_featpot/v2_common.py": "03ce8d76333f6d49682a6c5e8184a1f8c4946bf3b5cb7760075a36704f74d814",
    "scripts/rev4_featpot/v2_lodo_mlp.py": "64d53b79489f5e69dce42005f84b28d7b852e034de4aa1724df9065cb5cbcece",
    "scripts/rev4_featpot/v2_teacher.py": "fbab8e780b587f53ca6fe983c383a04ae4ae292ca011b471932eee74b9a30327",
    "scripts/rev4_featpot/v2_confirm_fit.py": "80bef53b92b39b3ac971790f66ef8aa21e0c9d570e1f8ce752c707f8eab0dea8",
}

# E27 on the E26 + SHIPPATH merge: pooled-rank / within-reference absolute
# HDR legs. Keep the frozen E26 profile; include safe-path/coverage runtime owner.
V2R5HDR27_SCRIPTS = {**V2R5HDR_SCRIPTS,
    "scripts/rev4_featpot/v2_common.py": "cd3af4fff63c9672f834533b382a9e35d3a4ce93c3baf8287bb88bbc0e670604",
    "scripts/rev4_featpot/v2_lodo_mlp.py": "ad8052869fb01c0c4aabbcddb028b0617ea28a5983395c6ec27a08799c8af0e4",
    "scripts/rev4_featpot/v2_teacher.py": "8185465d4e59c9e2e4dd05d99d46e49b17757d942d0dfd99430c010c9268529c",
    "scripts/rev4_featpot/v2_confirm_fit.py": "887b5a199901478476d8bb31b25ac0c7634ece217f331f9b3c8a60683986e507",
    "scripts/rev4_featpot/v2c_wide.py": "6f9364fe8eed5270860215d8d9b0ab50cb5b26cc189e31187cbe766598b9a291",
}


# E28: registered within-dataset pooled rank/Pearson objectives and grouped NM.
# Preserve prior program profiles; these hashes bind the new research program.
V2R5S2RECIPE28_SCRIPTS = {**V2R5HDR27_SCRIPTS,
    "scripts/rev4_featpot/v2_common.py": "921bfa79f832ceb2fc28a25673b656f8e33368d6e798f65fe1efd5a5f6e7770d",
    "scripts/rev4_featpot/v2_lodo_mlp.py": "0853cd7d4213f57a6edb7c2816b2f1a556490174f3d328170bada15e7c807379",
    "scripts/lib/zen_stats.py": "3a0ba0b03baca9e4a1f57b62ac09bbfed478fcc6aa456828e3dc33a06135e972",
    "scripts/rev4_featpot/e28_recipe.py": "852faf2b1ee77e751823b16063d239b5418d84d95d483ce8d399da22f10f8001",
    "scripts/rev4_featpot/e28_simplex.py": "7d5cd8d799f92e351889cefc5a9a94495c2f8506dcb3f5bc2cced27ac5992c92",
    "benchmarks/e28_teacher_pin_2026-10-07.json": "ddbc9c22cd08f545f76c50dd60947062b285d937df862f61153ff1685d9bc599",
    "benchmarks/e28_nm_grouping_2026-10-07.json": "266df7f42568640035427cf850d77150c50ba34893c4a349b972237c1310aa08",
    "benchmarks/e28_admission_inventory_2026-10-07.json": "71067920cbe93f05ef12d13317e4332af8f2ec73ca37b403d8bae670e30afe47",
}

# D1 strict four-source production/E30 program; admitted coverage travels in data.
D1_SCRIPTS = {'scripts/rev4_featpot/v2_common.py': '6be699bad4fbaad816c39d0240ae0f8d1e32340409b46da19fda0e603d07453c',
 'scripts/rev4_featpot/v2_lodo_mlp.py': 'a2e731a7fb8e2b639e99094309051773bb809477513692213d78e0d1bebb48a5',
 'scripts/rev4_featpot/v2_confirm_fit.py': '51010567202e73a51d224829a131054ed3af710eb846dcd50fff72f876f7753a',
 'scripts/rev4_featpot/v2_teacher.py': '8185465d4e59c9e2e4dd05d99d46e49b17757d942d0dfd99430c010c9268529c',
 'scripts/rev4_featpot/v2c_wide.py': '6f9364fe8eed5270860215d8d9b0ab50cb5b26cc189e31187cbe766598b9a291',
 'scripts/rev4_featpot/v2_human_role.py': '350370cd7a41c38bae090219221df5e70d890b54668cdb146dfad79a3bdb05a1',
 'scripts/rev4_featpot/v2_production_pack.py': '6ed1edcb9070ed3804e2583623982af850011b69f89a346c64fa87e2434ca982',
 'scripts/rev4_featpot/rev5_bank.py': 'e42eec57fed223c19dd3b72d2a3c0476c20067ad614b9d76de16d5b95fdda0a7',
 'scripts/lib/zen_stats.py': '3a0ba0b03baca9e4a1f57b62ac09bbfed478fcc6aa456828e3dc33a06135e972',
 'scripts/lib/assessment_identity.py': '2bc420285d81661932cd117b2e080d61419846dc02b0173c8113e45c48840fee',
 'benchmarks/shippath_human_role_D1_2026-10-07.json': '1baaa0ae980757d69e351240cb9c0c4d3c5369bde2eb52addafcd49845dbe94a',
 'scripts/rev4_featpot/e30_four_source.py': '13b86ad249e9be81d44707d28e69f2d810ecadb4ab20952d314e3694223e520d',
 'scripts/rev4_featpot/e21_cheap_recipe.py': '7f60d0e23d1005eed14e42dfac91ae8bd5d5ba136023703d5ccc9a3225afa18f',
 'scripts/rev4_featpot/e13_teacher.py': 'dbd6b7533a42753825dbaa31b838ff346a01a75b332b94168b5c7c59d8ba9116',
 'scripts/rev4_featpot/e24_rev5.py': '5fa9e700c087252f8c859ac3cc7e8fc69fdb5b1760124b89a1e09444bc289311',
 'scripts/rev4_featpot/external_sets.py': '078f98b3ad02703acd60c165f44c2902bfd0fbd3ff61946dad5a61322df9c85a',
 'benchmarks/costset2_2026-10-03.candidate_ids.json': '0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17',
 'scripts/rev4_featpot/v2_smoke_contract.py': 'c74d967c69d0d8562fd1264f8da95213289eb277366706092e2fc0c0591a4d3f',
 'benchmarks/shippath_qualified_fit_contract_2026-10-07.json': '49de01953a55f3a02eb831efb55459f213bbe8f4c66c8ff01753837502dac30d'}

# E29 strict four-source SDR plus explicit native HDR research subset.
E29_SCRIPTS = {'scripts/rev4_featpot/v2_common.py': '0e38bcfd652633fc37fe49888f67b26bce8777c0cb1b64a57944f6bc8da0502b', 'scripts/rev4_featpot/v2_lodo_mlp.py': '22b7335f0d20af1d56f1106e5f908cd57cd98a06faef47ca14c0034ca6b7bdd0', 'scripts/rev4_featpot/v2_confirm_fit.py': '51010567202e73a51d224829a131054ed3af710eb846dcd50fff72f876f7753a', 'scripts/rev4_featpot/v2_teacher.py': '8185465d4e59c9e2e4dd05d99d46e49b17757d942d0dfd99430c010c9268529c', 'scripts/rev4_featpot/v2c_wide.py': '6f9364fe8eed5270860215d8d9b0ab50cb5b26cc189e31187cbe766598b9a291', 'scripts/rev4_featpot/v2_human_role.py': 'f0e8b896c82d93765f7f236b37265f6aef961b66ea8ddad9259988ce80d1367e', 'scripts/rev4_featpot/v2_production_pack.py': '6ed1edcb9070ed3804e2583623982af850011b69f89a346c64fa87e2434ca982', 'scripts/rev4_featpot/rev5_bank.py': '9e6b87f605e54e7e452e2d121ff34bb6479692443f8f424f02b02eaa6aacc557', 'scripts/lib/zen_stats.py': '3a0ba0b03baca9e4a1f57b62ac09bbfed478fcc6aa456828e3dc33a06135e972', 'scripts/lib/assessment_identity.py': '2bc420285d81661932cd117b2e080d61419846dc02b0173c8113e45c48840fee', 'benchmarks/shippath_human_role_D1_2026-10-07.json': '1baaa0ae980757d69e351240cb9c0c4d3c5369bde2eb52addafcd49845dbe94a', 'scripts/rev4_featpot/e30_four_source.py': '13b86ad249e9be81d44707d28e69f2d810ecadb4ab20952d314e3694223e520d', 'scripts/rev4_featpot/e21_cheap_recipe.py': '7f60d0e23d1005eed14e42dfac91ae8bd5d5ba136023703d5ccc9a3225afa18f', 'scripts/rev4_featpot/e13_teacher.py': 'dbd6b7533a42753825dbaa31b838ff346a01a75b332b94168b5c7c59d8ba9116', 'scripts/rev4_featpot/e24_rev5.py': '5d447061d0ec8d460c467a9230e95424e28daba67f37a54f7f34906574b6d76e', 'scripts/rev4_featpot/external_sets.py': '078f98b3ad02703acd60c165f44c2902bfd0fbd3ff61946dad5a61322df9c85a', 'benchmarks/costset2_2026-10-03.candidate_ids.json': '0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17', 'scripts/rev4_featpot/v2_smoke_contract.py': 'c74d967c69d0d8562fd1264f8da95213289eb277366706092e2fc0c0591a4d3f', 'benchmarks/shippath_qualified_fit_contract_2026-10-07.json': '49de01953a55f3a02eb831efb55459f213bbe8f4c66c8ff01753837502dac30d', 'scripts/rev4_featpot/e29_consensus.py': '4ea97ebb4dbd0c6559d6d34b8115139b2d7238c54685ab2557850af0f23ab262', 'benchmarks/e29_fit_contract_2026-10-07.json': '68f4a71d4bd5c95dedd0d6797c52a8ebbd08e5f5d7345e2e4b057ee2ee43e33a', 'benchmarks/e29_four_source_amendment_2026-10-07.md': '8c5b0c775450aea45214769845bb6cb85b046b384f548a6b95f0b533909c6aa0', 'scripts/hdr/hdr_route_panel.py': '70fabc8f39c109cd8b27d7e71547979aca21c8043b568bf2b131653c74da6eb9'}

BINARIES = ("zensim_mlp_train", "bake_dial_refit", "panel")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_file(tar: tarfile.TarFile, path: Path, name: str) -> None:
    member = tar.gettarinfo(str(path), arcname=name)
    member.mtime = 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    with path.open("rb") as stream:
        tar.addfile(member, stream)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--executor", type=Path, required=True)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--build-meta", type=Path, required=True)
    p.add_argument("--profile", choices=("p0", "p2d2", "v2", "v2r5", "v2r5hdr", "v2r5hdr27", "v2r5s2recipe28", "v2d1", "v2e29"), default="p0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", action="append", default=[], metavar="NAME=PATH", help="a pinned V2_DATA file (profile v2)")
    args = p.parse_args()
    files = {}
    for name, expected in {"p0": SCRIPTS, "p2d2": P2D2_SCRIPTS, "v2": V2_SCRIPTS, "v2r5": V2_SCRIPTS, "v2r5hdr": V2R5HDR_SCRIPTS, "v2r5hdr27": V2R5HDR27_SCRIPTS, "v2r5s2recipe28": V2R5S2RECIPE28_SCRIPTS, "v2d1": D1_SCRIPTS, "v2e29": E29_SCRIPTS}[args.profile].items():
        path = args.source / name
        if digest(path) != expected:
            raise ValueError(f"preregistered fit source changed: {name}")
        files[name] = path
    given = dict(item.split("=", 1) for item in args.data)
    wanted = {"v2": V2_DATA, "v2r5": V2R5_DATA, "v2r5hdr": V2R5_DATA, "v2r5hdr27": V2R5_DATA, "v2r5s2recipe28": V2R5_DATA}.get(args.profile, {})
    if sorted(given) != sorted(wanted):
        raise ValueError(f"--data must name exactly {sorted(wanted)}")
    for name, expected in wanted.items():
        path = Path(given[name])
        if digest(path) != expected:
            raise ValueError(f"pinned data file changed: {name}")
        files[name] = path
    files["fit_cell_exec.py"] = args.executor
    files["fit_paths.py"] = args.executor.with_name("fit_paths.py")
    if args.profile in ("v2d1", "v2e29"):
        files["bin/inspect_qualified_checkpoint"] = args.bin_dir / "inspect_qualified_checkpoint"
        files["qualified_fit_contract.py"] = args.executor.with_name("qualified_fit_contract.py")
        files["harvest_fit_cells.py"] = args.executor.with_name("harvest_fit_cells.py")
    for name in BINARIES:
        path = args.bin_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"bin/{name}"] = path
    metadata = json.loads(args.build_meta.read_text())
    if metadata.get("schema") != "fleet-fits-build-v1":
        raise ValueError("build metadata missing or wrong schema")
    inventory = {name: digest(path) for name, path in sorted(files.items())}
    metadata = {**metadata, "files": inventory}
    meta_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode() + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            info = tarfile.TarInfo("build_meta.json")
            info.size = len(meta_bytes)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(meta_bytes))
            for name, path in sorted(files.items()):
                add_file(tar, path, name)
    print(json.dumps({"sha256": digest(args.out), "bytes": args.out.stat().st_size,
                      "files": len(files), "out": str(args.out)}))


if __name__ == "__main__":
    main()
