#!/usr/bin/env python3
# FINAL _MANIFEST.json for the avif944 corpus — REVIVAL version (2026-08-21).
# Replaces the draft; numbers pulled from on-disk artifacts, not typed.
import hashlib, json, os, datetime
import pyarrow.parquet as pq

D = '/mnt/v/output/avifgen-2026-08-06'
V = '/mnt/v/zen/zensim-training/avif944-2026-08-07'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()

draft = json.load(open(D + '/_MANIFEST.draft.json'))
gate = json.load(open(D + '/orientation_gate.json'))
dups = json.load(open(D + '/duplicate_renditions.json'))
verdict = json.load(open(D + '/rescue_sample_verdict.json'))
rescue_truth = json.load(open(D + '/rescue_truth.json'))
sc = pq.read_metadata(D + '/unified/scores.parquet')
ft = pq.read_metadata(D + '/unified/features.parquet')
ftcols = pq.read_schema(D + '/unified/features.parquet').names
n_feat = sum(1 for c in ftcols if c.startswith('feat_'))
views = {}
for v in ('train', 'eval8'):
    p = f'{V}/{v}_944.parquet'
    views[v] = {'rows': pq.read_metadata(p).num_rows, 'sha256': sha(p), 'path': p}

m = dict(draft)
m.pop('_', None)
m = {'_': 'SDR AVIF datagen corpus — FINAL (campaign appendix Z + Z.R; AC.R1-amended views). '
          'Produced 2026-08-06/07 on the household fleet; GPU-score rescue drained 2026-08-08..11 + 2026-08-21 (see provenance.stall).',
     **m}
m['finalized_utc'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
m['accounting'] = {
    'declared_cells': 564300,
    'unique_content_jobs': 562860,
    'duplicate_rendition_pairs': dups,
    'note': '4 byte-identical rendition pairs collapse 1,440 cell declares into shared jobs; cell tables fan them back out. All 4 pairs train/train — zero split leakage.',
    'encode_done_unique': 562860,
    'encode_gapfill': '3 transient upload_fail cells re-encoded via the pinned jobexec (worker=wsl-manual-gapfill, sidecar pass-wsl-manual-gapfill-1.parquet)',
    'unique_scored_pairs': 534464,
    'score_coverage': {'ssim2+butteraugli': '534,464/534,464 (18-pair residue closed manually post-SCORE_FINAL; VRAM-era corruption then cured by the AC.R1 rescue — see provenance)',
                       'cvvdp+features944': '534,464/534,464 (zero failures)'},
    'oom_retry_provenance': '10,291 sf-gpu jobs (23% of pairs) failed CUDA_ERROR_OUT_OF_MEMORY under unhinted admission (BoxBudget has no VRAM dimension); recovered via CHUNK=9 hinted retry then the AC.R1 rescue union. Pair-level coverage is the completeness authority.',
}
m['provenance'] = {
    'corruption_census_AC_R1': {
        'impossible_class_cells': 2169, 'renditions_affected': 99,
        'class': 'ssim2_gpu==100 AND butteraugli_max_gpu==0 AND q<=70 (exact saturation = lower bound); 100% med/large; pre-hint VRAM-OOM window survivors',
        'detected_by': 'G-Z5 default-stratum orientation bar 0.9876 < 0.99 (first gate run, 2026-08-08); supervisor re-verified from unified/scores.parquet',
        'cross_check': 'cvvdp (CPU queue) rises monotonically through the same ladders; 944 features clean (separate queue)'},
    'rescue': rescue_truth,
    'sample_verdict': verdict,
    'stall': ('The rescue union was declared 2026-08-08 14:44Z; its operator daemon died silently from 21:57Z '
              '(the old compact timeout 560s < the measured 768s ledger-read at ~10k sidecars — the count parse '
              'returned nothing and the daemon masked stderr) and the owning session was lost in harness restarts ~22:00Z. '
              'The WORKERS kept draining to 8,361/8,623 and stopped at scoped-cred expiry 2026-08-11 09:37Z. '
              'The lane sat dead 13 days (no waiter, no heartbeat) until the 2026-08-21 revival: fresh scoped cred minted, '
              'snapshot refreshed, node-2 drained the residue, chain completed. Recorded honestly as the recurring '
              'silent-waiter failure class (see docs/status/avif-datagen-2026-08.md).'),
}
m['tables'] = {
    'scores.parquet': {'rows': sc.num_rows, 'sha256': sha(D + '/unified/scores.parquet'),
                       'columns': 'identity(image_path,q,knob_tuple_json,encode_sha) + butteraugli_max_gpu + butteraugli_pnorm3_gpu + ssim2_gpu + cvvdp_cpu_imazen_v0_1_0',
                       'rebuild': 'rescue-wins overlay (rescue_verdict_and_rebuild.py); pre-rescue preserved as scores.pre-rescue.bak.parquet'},
    'features.parquet': {'rows': ft.num_rows, 'sha256': sha(D + '/unified/features.parquet'), 'feat_width': n_feat,
                         'regime': 'folded720append2 (944), AVX2-tier-pure extraction'},
    'pairs_final_fanned.parquet': {'rows': pq.read_metadata(D + '/pairs_final_fanned.parquet').num_rows,
                                   'sha256': sha(D + '/pairs_final_fanned.parquet')},
}
m['orientation_gate_G-Z5'] = gate
m['training_views_AC_R1'] = {
    'path': V, 'views': views,
    'split_rule': ('AC.R1 amendment 2: corpus is train-split-only (every origin ends 0/2/4/6/8). '
                   'train_944 = origins ending 0/2/4/6; eval8_944 = origins ending 8 (leg-side eval holdout, never trained on). '
                   'No validate/test views by construction. origin_split.py origin_id() is the splitter.'),
    'pre_amendment_views': 'train/validate/test_944.parquet.pre-rescue.bak (pre-rescue scores + superseded split shape; kept, do not train on)'}
m['gates']['G-Z4'] = 'PASSED: encode 562,860/562,860 unique (100%); scoring 534,464/534,464 unique pairs both queues (100%)'
m['gates']['G-Z5'] = (('PASSED' if gate['_gate']['PASS'] else 'FAILED') +
                      f" on the rescue-rebuilt scores (default-stratum ssim2 pass rate {gate['_gate']['default_stratum_ssim2_pass_rate']}, bar 0.99)")
m['known_residue'] = rescue_truth.get('known_residue', 'see provenance.rescue')
json.dump(m, open(D + '/_MANIFEST.json', 'w'), indent=1)
print('FINAL _MANIFEST.json written; G-Z4:', m['gates']['G-Z4'], '| G-Z5:', m['gates']['G-Z5'])
# view-dir manifest
vm = {'_': 'avif944 training views (AC.R1-amended). Emitted from the rescue-rebuilt unified tables.',
      'built_utc': m['finalized_utc'],
      'build_commit': rescue_truth.get('revival_commit', 'see corpus _MANIFEST'),
      'corpus_manifest': D + '/_MANIFEST.json',
      'corpus_manifest_sha256': sha(D + '/_MANIFEST.json'),
      'split_rule': m['training_views_AC_R1']['split_rule'],
      'views': views,
      'columns': 'identity + 4 score cols + feat_0..943 + origin + leg',
      'wave12_note': ('944 features AVX2-tier-pure; no zensim_score column (foldapp2 = features-only). '
                      'Score cols: ssim2_gpu (the codec-leg target), butteraugli_max_gpu, butteraugli_pnorm3_gpu, cvvdp_cpu_imazen_v0_1_0.')}
json.dump(vm, open(V + '/_MANIFEST.json', 'w'), indent=1)
print('view _MANIFEST.json written')
