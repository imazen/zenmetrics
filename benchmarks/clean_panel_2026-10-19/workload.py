"""Fail-closed contract for the sole implemented paired-NR adapter."""
import json
import math
from pathlib import Path

FIELDS = ('metric_name', 'width', 'height', 'threads', 'mode', 'timing_phase', 'reference', 'distorted')
ADAPTER = 'brisque-paired-nr-v2'
VERSION_SUFFIX = ' brisque paired-NR v2'


def validate_contract(m):
    from panel import digest
    if m.get('schema') != 2 or m['metric'] != 'brisque':
        raise ValueError('unsupported adapter declaration')
    w = m['workload']
    expected = dict(metric_name='brisque', width=512, height=512, threads=1, mode='score', timing_phase='warm')
    if any(w.get(k) != v for k, v in expected.items()):
        raise ValueError('unsupported workload')
    inputs = {i['path']: i['sha256'] for i in m['inputs']}
    if set(inputs) != {w['reference'], w['distorted']}:
        raise ValueError('input path contract')
    bound = {a['path']: a['sha256'] for a in m['artifacts']}
    def require(path):
        if not Path(path).is_absolute() or path not in bound or digest(path) != bound[path]:
            raise ValueError('unbound executable/config/adapter: ' + path)
    for key in ('contention_script', 'containment_script', 'run_heavy', 'controller_python', 'taskset'):
        require(m[key])
    import sys
    if m['controller_python'] != str(Path(sys.executable).absolute()):
        raise ValueError('controller Python mismatch')
    for name in ('panel.py','workload.py','runtime_closure.py','bytecode_inventory.py'):
        require(str(Path(__file__).with_name(name).absolute()))
    if Path(m['containment_script']).resolve() != Path(__file__).with_name('containment.py').resolve():
        raise ValueError('unsupported containment helper')
    ours, canonical = m['arms']['ours'], m['arms']['canonical']
    if ours['protocol'] != 'iqa-row' or canonical['protocol'] != 'canonical-v2' or canonical['adapter'] != ADAPTER:
        raise ValueError('unsupported protocol/adapter')
    if len(ours['argv']) != 2 or len(canonical['argv']) != 5:
        raise ValueError('unsupported argument shape')
    for path in ours['argv'] + canonical['argv'][:2] + canonical['argv'][4:]:
        require(path)
    if Path(canonical['argv'][1]).resolve() != Path(__file__).with_name('brisque_oracle.py').resolve():
        raise ValueError('unsupported canonical adapter path')
    if canonical['argv'][2:4] != [w['reference'], w['distorted']]:
        raise ValueError('canonical input arguments')
    config = json.loads(Path(ours['argv'][1]).read_text())
    for k in FIELDS:
        if config.get('only_mode' if k == 'mode' else k) != w[k]:
            raise ValueError('frozen cell configuration mismatch: ' + k)
    if config.get('locked_process_accounting') is not True:
        raise ValueError('accounting required')
    if any(a['sample_count'] != 10 for a in m['arms'].values()):
        raise ValueError('unsupported sample declaration')
    if canonical['version'] != 'pyiqa-' + canonical['pyiqa_version'] + VERSION_SUFFIX:
        raise ValueError('canonical version declaration')


def one_json(text, prefix):
    rows = [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('missing or duplicate ' + prefix.strip())
    return rows[0]


def parse(text, name, m, clean):
    """Return observed metadata only after matching the emitted workload."""
    from panel import digest
    spec, w = m['arms'][name], m['workload']
    process = one_json(text, 'IQA_PROCESS ')
    if any(type(process.get(k)) is not int or process[k] <= 0 for k in ('pid', 'pgid')):
        raise ValueError('invalid process identity')
    if name == 'ours':
        row = one_json(text, 'IQA_ROW ')
        if row.get('status') != 'ok':
            raise ValueError('native status is not ok')
        for key in FIELDS:
            if row.get(key) != w[key]:
                raise ValueError('observed workload mismatch: ' + key)
        if row.get('only_mode') != 'score' or row.get('locked_process_accounting') is not True:
            raise ValueError('native mode/accounting mismatch')
        if row.get('declared_columns') != ['brisque','brisque_ref'] or row.get('headline_declared_column') != 'brisque':
            raise ValueError('native output column mismatch')
        if row.get('rayon_threads_observed',1) != 1:
            raise ValueError('native observed worker count mismatch')
        if 'input_sha256' in row and row['input_sha256'] != {k:digest(w[k]) for k in ('reference','distorted')}:
            raise ValueError('native emitted input hashes mismatch')
        if 'impl_binary_sha' in row and row['impl_binary_sha'] != digest(spec['argv'][0]):
            raise ValueError('native emitted executable hash mismatch')
        observed = {key: row[key] for key in FIELDS}
        raw = row['zenbench_samples']
        if row.get('n') != spec['sample_count'] or len(raw) != spec['sample_count']:
            raise ValueError('native sample count mismatch')
        samples = []
        for i, sample in enumerate(raw, 1):
            if sample['execution_order'] != [0] or len(sample['elapsed_ns']) != 1 or type(sample['elapsed_ns'][0]) is not int or sample['elapsed_ns'][0] <= 0 or type(sample['iterations']) is not int or sample['iterations'] <= 0 or type(sample['gate_clean']) is not bool:
                raise ValueError('native sample arm/order mismatch')
            samples.append(dict(sample_id=i, ns=sample['elapsed_ns'][0]/sample['iterations'], gate_clean=sample['gate_clean']))
        score = row['score']
        # IQA_ROW.ref_score is the first distorted timing result, NOT the NR reference.
        reference_score = None
    else:
        observed = one_json(text, 'WORKLOAD ')
        expected = dict(w, adapter=ADAPTER, adapter_sha256=digest(spec['argv'][1]),
                        weights_sha256=digest(spec['argv'][4]),
                        input_sha256={k: digest(w[k]) for k in ('reference', 'distorted')},
                        sample_count=spec['sample_count'])
        if observed != expected:
            raise ValueError('canonical observed workload mismatch')
        versions = [line[8:] for line in text.splitlines() if line.startswith('VERSION ')]
        if versions != [spec['version']]:
            raise ValueError('canonical VERSION mismatch')
        raw = [json.loads(line[7:]) for line in text.splitlines() if line.startswith('SAMPLE ')]
        if any(type(s['sample_id']) is not int for s in raw) or [s['sample_id'] for s in raw] != list(range(1, spec['sample_count']+1)):
            raise ValueError('duplicate/missing/out-of-order sample IDs')
        if any(type(s[k]) not in (int, float) or not math.isfinite(s[k]) for s in raw for k in ('score', 'reference_score', 'ns')):
            raise ValueError('invalid canonical sample')
        if len({s['score'] for s in raw}) != 1 or len({s['reference_score'] for s in raw}) != 1:
            raise ValueError('canonical score changed across samples')
        samples = [dict(sample_id=s['sample_id'], ns=s['ns'], gate_clean=clean) for s in raw]
        score, reference_score = raw[-1]['score'], raw[-1]['reference_score']
    return dict(observed_workload=observed, observed_process=process, observed_validated=True,
                samples=samples, score=score, reference_score=reference_score)
