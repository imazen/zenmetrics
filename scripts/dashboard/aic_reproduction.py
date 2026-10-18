"""Current column qualification, independent of the superseded coverage audit."""
import csv
import json
import re

from catalog import ROOT, md_tables, plain, rows as catalog_rows

INVENTORY = 'benchmarks/dashboard/aic2026/columns.tsv'
LEDGER = 'docs/AIC2026_METRICS_AND_FITTING.md'
STATUS = 'docs/AIC2026_REPRODUCTION_STATUS.md'
NATIVE = 'benchmarks/aic_refusals_2026-10-18/summary.tsv'


def canonical(name):
    return re.sub(r' \((?:excluded|dataset).*?\)', '', name.split(' / ', 1)[0]).strip()


def qualifications():
    result = {}
    for table in md_tables(ROOT / LEDGER):
        if table[0] != ['AIC-4 column', 'Actual implementation (identified)', 'Ours', 'Qualification']:
            continue
        for column, identified, ours, qualification in table[1:]:
            names = re.findall(r'`([^`]+)`', column)
            if names and names[0] == 'PSNR-HVS':
                names = ['PSNR-HVS', 'PSNR-HVS-Y', 'PSNR-HVS-Cb', 'PSNR-HVS-Cr']
            for name in names:
                result[name] = (plain(ours), plain(qualification), plain(identified))
    if not result:
        raise ValueError('current naming ledger was not parsed')
    return result


def rows(cards, commit):
    mapping = json.loads((ROOT / 'scripts/complexity/aic2026_columns.json').read_text())['map']
    current = qualifications()
    status_text = (ROOT / STATUS).read_text()
    reproduced = set(re.findall(r'`([^`]+)`', status_text.split('## Reproduced\n', 1)[1].split('\n## ', 1)[0]))
    private = json.loads((ROOT / 'benchmarks/dashboard/aic2026/private-aggregates.json').read_text())['models']
    by_private = {p['published_column']: p for p in private}
    cli_names={c['metric'] for c in cards} or {c['metric'] for c in catalog_rows()}
    retained={r['column']:r for r in csv.DictReader((ROOT/'benchmarks/dashboard/aic2026/retained-comparisons.tsv').read_text().splitlines(),delimiter='\t')}
    result = []
    for entry in csv.DictReader((ROOT / INVENTORY).read_text().splitlines(), delimiter='\t'):
        name = entry['column']; key = canonical(name)
        ours, qualification, identified = current.get(key, ('', '', ''))
        implementation = mapping.get(key, ours) or (key.lower() if key.lower() in cli_names else '')
        row = dict(column=name, availability=entry['availability'], implementation=implementation or 'not identified',
                   n='', median='', maximum='', max_jnd='', status='current comparison not recorded',
                   note=qualification, evidence=LEDGER, comparison_scope='Current naming and qualification ledger')
        if key in reproduced or (key.startswith('PSNR-HVS') and 'Daala' in identified):
            row['status'] = 'reproduced; stated residual'
        elif implementation:
            row['status'] = 'implemented; stated qualification'
        if name in retained:
            fact=retained[name]
            row.update({k:fact[k] for k in ('n','median','maximum','max_jnd')})
            row.update(status='reproduced; retained comparison panel',comparison_scope='Retained native-versus-published panel; measurement scope in the current qualification ledger')
        if key in ('Butteraugli-max','Butteraugli-3norm'):
            row['note']='libjxl Butteraugli, '+('maximum' if key=='Butteraugli-max' else 'pnorm3')+' pooling; distinct from the original Google implementation.'
        if key.startswith('PSNR-HVS') and '-vmaf' in key:
            row.update(implementation='psnrhvs-daala (combined and per-plane outputs)',
                       note='The dataset PSNR-HVS family uses the Daala implementation; the current port retains combined and Y/Cb/Cr outputs.')
        elif key.startswith('VIF-vmaf-scale'):
            row.update(implementation='vmaf crate, per-scale VIF features',status='implemented; site-column residual',
                       note='Four libvmaf VIF scales exist as crate features. The newer site recomputation differs from the dataset sum; a feature API is distinct from a standalone CLI metric.',evidence=STATUS)
        elif key.startswith(('VMAF v0','VMAF-neg v0','VMAF v1')):
            row.update(implementation='vmaf-v1' if key.startswith('VMAF v1') else 'vmaf-neg-studio601' if key.startswith('VMAF-neg') else 'vmaf-studio601',
                       status='implemented; site-column convention distinct',
                       note='Current CLI routes exist. The newer site VMAF recomputations are distinct from the older dataset VMAF/NEG columns; the dataset reproduction claim is not transferred to these columns.', evidence=STATUS)
        elif key=='VMAF_420':
            row.update(implementation='vmaf (YUV420 input)',status='implemented; published values unavailable',note='CLI YUV420 route exists; this excluded site alias has no published per-image values.')
        elif key=='pyiqa-GMSD-materialized-y':
            row.update(implementation='gmsd; exact alias not confirmed',status='implemented family; published values unavailable',note='GMSD is implemented; no per-image values establish this site-only alias.')
        elif key.startswith('PSNR-HVS-ref'):
            row.update(implementation='psnrhvs / psnrhvs-y (Ponomarenko); alias unconfirmed',status='implemented family; published values unavailable',note='The Ponomarenko reference family exists separately from Daala; this site-only alias has no per-image comparison values.')
        # An absent published table is not an absent implementation.
        variants = re.findall(r'(?:ssimc?|ms-ssim|fsimc?|vsi)(?:-pyiqa)?(?:-f32)?(?:-k[2-6]|-nodownscale)?', key.lower())
        if entry['availability'] == 'neither' and variants:
            row.update(implementation=implementation or ', '.join(variants), status='implemented; published values unavailable',
                       note='Fixed-factor/f32 implementations exist; the authors have not published the per-image values needed for column parity.', evidence=STATUS)
        if key.startswith(('HaarXYB', 'HaarY-', 'HaarYCoCg')):
            row.update(implementation='unidentified experiment', status='MISSING: identity and published values',
                       note='Completed method search identifies 0 of 11 variants; no candidate fit qualified.',
                       evidence='benchmarks/dashboard/aic2026/haar-search.json')
        elif key == 'AIMOS':
            row.update(status='MISSING: unidentified', note='16 of 34 NR candidates scored; no match; best hyperiqa ratio Spearman 0.82.', evidence=STATUS)
        elif '-vqmt' in key:
            row.update(status='MISSING: published values', note='MSU VQMT convention; per-image values unavailable.', evidence=STATUS)
        elif key.startswith('proposal-DVIFM'):
            row.update(implementation='dvifmish released-build port presets (private benchmark adapters)',
                       status='implemented; published columns unmatched',
                       note='Released v0.2/v0.3 presets and six benchmark routes exist. Released builds do not reproduce the historical columns; partial aggregate counts and qualified timings are available.',
                       evidence='benchmarks/aic_refusals_2026-10-18/README.md')
        elif key.startswith('WD_s'):
            row.update(implementation='wd (five outputs)', status='implemented; published columns unmatched',
                       note='Native upstream gate passes. Published sigma mapping unresolved; bounded search best median 0.13–0.20 site-JND, Spearman 0.98–0.99.', evidence=STATUS)
        elif key == 'A-FINE':
            row.update(implementation='afine', status='implemented; published column unmatched',
                       note='Upstream-verified afine_all. Published entry point and size handling unresolved; tested convention search best Spearman 0.94, median 0.11 site-JND.', evidence=STATUS)
        elif key == 'DMM':
            row.update(status='fails native gate', n='61', max_jnd='0.0024',
                       note='pyiqa 0.1.16 CUDA fp32 reference versus published: median 2.6e-4/max 2.4e-3 site-JND. Separate Rust dmm-pyiqa native-oracle failure: worst 0.071 JND, above 0.001 gate (SVD).', evidence=STATUS,
                       comparison_scope='pyiqa reference versus published; native failure stated separately')
        elif key in ('CW-SSIM', 'TOPIQ', 'TOPIQ-PIPAL', 'AHIQ', 'PieAPP', 'WaDIQaM'):
            row.update(status='implemented; native gate passes', note='Public non-commercial pyiqa-convention port. Native 0.001 JND gate passes; reference-versus-published comparison and native fixture scope are separate.',
                       evidence='crates/zenmetrics-pyiqa-ports/README.md')
        for published, record in by_private.items():
            if published in name or (key == 'MSSWD' and record['metric'].lower() == 'msswd'):
                row.update(implementation='private '+record['metric']+' port; aggregates only', status='private; aggregate match',
                           note=record['published_match']+' Native: '+record['native_gate'], evidence='benchmarks/dashboard/aic2026/private-aggregates.json')
        result.append(row)
    # Reference-versus-column figures must not be presented as native port errors.
    reference = next(t for t in md_tables(ROOT / STATUS) if t[0][:2] == ['Column', 'pyiqa metric'])
    for fields in reference[1:]:
        for row in result:
            if canonical(row['column']) == plain(fields[0]) and canonical(row['column']) != 'CKDN':
                row.update(n='61', median=plain(fields[2]), maximum=plain(fields[3]),
                           comparison_scope='pyiqa 0.1.16 reference versus published column; native gate stated separately')
                row['note'] += ' '+plain(fields[-1])
    more = json.loads((ROOT / 'benchmarks/aic_more_2026-10-15/summary.json').read_text())
    for row in result:
        key = canonical(row['column'])
        if key in ('Butteraugli-original', 'SSIMULACRA1', 'WS-HV'):
            record = more[{'Butteraugli-original':'butter_aic','SSIMULACRA1':'ssim1_aic','WS-HV':'ws_aic'}[key]]
            prefix = 'rgb_' if key == 'WS-HV' else ''
            row.update(n='61', median=str(record[prefix+'published_error']['median']), maximum=str(record[prefix+'published_error']['max']),
                       max_jnd=str(record[prefix+'published_jnd_error']['max']), evidence='benchmarks/aic_more_2026-10-15/summary.json',
                       comparison_scope='Native port versus published column, 61 pairs')
            if key == 'Butteraugli-original':
                row.update(implementation='butteraugli-original', status='reproduced; stated residual',
                           note='Distinct Google 2019 pre-libjxl port; native maximum 4.9473571777980396e-5 raw / 8.942517089938207e-6 site-JND. The old libjxl comparison does not describe this implementation.')
    native = {r['col']:r for r in csv.DictReader((ROOT / NATIVE).read_text().splitlines(), delimiter='\t')}
    qualification = {
        'HDR_VDP_3':('passes stated crop gate','Q output; 10-bit code quantization before 200-nit sRGB EOTF, D65 XYZ/OLED, 37.842505866 ppd.'),
        'HDR_VDP_2':('unreproduced','Q output; unrounded 200-nit sRGB, BT.709, 37.84 ppd. Native median 0.0277768 Q vs published; independent Octave median 0.02651 Q vs published, native-vs-Octave median 0.00869019 Q. RGB accuracy and setting residual remain separate open questions.'),
        'CAMBI':('passes stated crop median gate','Integer swscale-compatible luma; max raw 4.80485e-7. Near-zero mapping gives max site-JND 0.00212503; no maximum gate was specified.'),
        'proposal-Butteraugli':('fails crop median gate','Native 61-pair pnorm3 plus identified map: median 3.06732e-7 exceeds requested 3.2e-8 site-JND gate. Separate 9618-row published-column mapping is not 9618 native scores. Constants provenance unknown.')}
    for row in result:
        measured = native.get(canonical(row['column']))
        if measured:
            key=measured['col']; status,note=qualification[key]
            row.update(implementation=mapping[key], n=measured['n'], median=measured['med_abs'], maximum=measured['max_abs'], max_jnd=measured['max_jnd'],
                       status=status, note=note+' Median site-JND '+measured['med_jnd']+'; Spearman '+measured['spearman']+'.',
                       evidence='benchmarks/aic_refusals_2026-10-18/README.md', comparison_scope='Native port versus published, 61 crops')
    result.sort(key=lambda r:(0 if r['status'].startswith('MISSING') else 1 if 'unmatched' in r['status'] or 'fails' in r['status'] or r['status']=='unreproduced' else 2,r['column']))
    return result
