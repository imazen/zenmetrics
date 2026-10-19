#!/usr/bin/env python3
"""Export durable attempts with separate timing and numerical qualification."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

FIELDS = ('metric_name width height size_name threads timing_phase cold_repeat only_mode device impl '
          'impl_binary_sha reference_sha256 distorted_sha256 median_ns mean_ns p10_ns p90_ns n '
          'score score_bits ref_score simd_tier input_method notes error round_means_ns '
          'raw_status gpu_score_repeatability_policy numerical_variation_accepted score_repeatability_qualified '
          'numerical_instability_flag numerical_qualification gpu_call_observation_count '
          'gpu_finite_observation_count gpu_nonfinite_observation_count gpu_call_bit_variation_count '
          'gpu_reference_score_bits gpu_min_score_bits gpu_max_score_bits gpu_max_abs_delta_bits '
          'gpu_max_abs_delta gpu_delta_overflow gpu_max_relative_delta gpu_one_different_score_bits').split()


def observed_attempts(path, report, cell, include_sealed):
    """Overlay only finalized, sealed evidence; never adopt/recover an attempt."""
    attempts = {a['attempt']:dict(a, evidence_origin='report') for a in cell.get('attempts', [])}
    if not include_sealed:
        return list(attempts.values()), {}
    import durable
    extras = {}
    config = cell['config']
    for result in sorted((path.parent/'cells'/config['durable_cell_id']).glob('attempt-*/result.json')):
        number = int(result.parent.name.removeprefix('attempt-'))
        if number in attempts:
            continue
        raw = result.read_bytes()
        packet = json.loads(raw)
        record = packet.get('record')
        if not isinstance(record,dict) or packet.get('sha256') != durable.digest(durable.encoded(record)):
            raise ValueError('sealed export result checksum mismatch: ' + str(result))
        if (record.get('identity') != report['identity'] or record.get('cell_id') != config['durable_cell_id']
                or record.get('attempt') != number):
            raise ValueError('sealed export identity mismatch: ' + str(result))
        for key, name in [('config_sha256','config.json'),('log_sha256','worker.log'),('intent_sha256','intent.json')]:
            target = result.parent/name
            actual = durable.file_hash(target) if target.exists() else None
            if record.get(key) != actual:
                raise ValueError('sealed export evidence changed: ' + str(target))
        if json.loads((result.parent/'config.json').read_bytes()) != config:
            raise ValueError('sealed export config differs from report')
        if record['status'] in ('ok','unsupported'):
            if record.get('cleanup_verified') is False or durable.validate_row(result.parent/'worker.log',config) != record.get('row'):
                raise ValueError('sealed export row/cleanup differs')
        if result.read_bytes() != raw:
            raise ValueError('sealed result changed during capture')
        checksum = hashlib.sha256(raw).hexdigest()
        attempts[number] = dict(record, evidence_origin='sealed-result-overlay', sealed_result_sha256=checksum)
        extras[str(result.relative_to(path.parent))] = checksum
    return [attempts[n] for n in sorted(attempts)], extras


def export(root, output, host, include_sealed=False, record_rejected_overlays=False):
    output.mkdir(parents=True, exist_ok=False)
    observations, sources = [], []
    captures = output.with_name(output.name + '-reports') if include_sealed else None
    if captures:
        captures.mkdir(exist_ok=False)
    for path in sorted(root.glob('**/report.json')):
        if not (path.parent/'identity.json').exists():
            continue
        raw = path.read_bytes()
        report = json.loads(raw)
        identity = json.loads((path.parent/'identity.json').read_bytes())
        source_id = hashlib.sha256(raw).hexdigest()
        if captures:
            captured = captures / (source_id + '.json')
            if captured.exists():
                if captured.read_bytes() != raw:
                    raise ValueError('source capture hash collision')
            else:
                captured.write_bytes(raw)
        sources.append(dict(id=source_id, report=str(path.relative_to(root)),
                            identity=report.get('identity'), frozen=identity.get('frozen', {}).get('manifest_sha256'),
                            policy=identity.get('policy'), complete=report.get('complete'), qualified=report.get('qualified')))
        extra_results, rejected = {}, []
        for cell in report['cells']:
            try:
                attempts, extras = observed_attempts(path, report, cell, include_sealed)
            except ValueError as error:
                if not record_rejected_overlays:
                    raise
                directory = path.parent/'cells'/cell['config']['durable_cell_id']
                rejected.append(dict(cell_id=cell['config']['durable_cell_id'], error=str(error),
                    artifacts={str(p.relative_to(path.parent)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                               for p in sorted(directory.glob('attempt-*/*')) if p.name in ('result.json','config.json','intent.json','worker.log')}))
                # Only reported history is retained. Rejected extras emit no
                # timing row and cannot contribute to qualification or caps.
                attempts, extras = observed_attempts(path, report, cell, False)
            extra_results.update(extras)
            config = cell['config']
            cell_status = cell['status']
            if extras:
                terminal = next((a['status'] for a in attempts if a['status'] in ('ok','unsupported')), None)
                cap = identity.get('policy',{}).get('max_attempts')
                cell_status = terminal or ('failed' if type(cap) is int and max(a['attempt'] for a in attempts) >= cap else 'pending')
            for attempt in attempts:
                row = attempt.get('row') or {}
                value = dict(host=host, source=source_id, cell_id=config['durable_cell_id'],
                             attempt=attempt['attempt'], status=attempt['status'],
                             cell_status=cell_status, report_cell_status=cell['status'], reason=attempt.get('reason'),
                             config_sha256=attempt.get('config_sha256'), log_sha256=attempt.get('log_sha256'),
                             evidence_origin=attempt['evidence_origin'], sealed_result_sha256=attempt.get('sealed_result_sha256'),
                             cleanup_verified=attempt.get('cleanup_verified'),
                             engine=config.get('worker_environment',{}).get('ZENMETRICS_NN_ENGINE'),
                             weights_consent=config.get('worker_environment',{}).get('ZENMETRICS_NN_ACCEPT_WEIGHTS'))
                value.update({k:row.get(k,config.get(k)) for k in FIELDS})
                samples = row.get('zenbench_samples', [])
                value['warm_rounds_clean'] = sum(s.get('gate_clean') is True for s in samples)
                value['warm_rounds_flagged'] = sum(s.get('gate_clean') is not True for s in samples)
                value['qualified_timing'] = attempt['status']=='ok' and (config['timing_phase']=='cold' or
                    (row.get('n',0)>=10 and len(samples)>=10 and all(s.get('gate_clean') is True for s in samples)))
                observations.append(value)
        if include_sealed:
            sources[-1].update(sealed_result_overlays=extra_results, rejected_overlay_cells=rejected)
    def save(name, value):
        (output/name).write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    save('sources.json', sources)
    with (output/'observations.jsonl').open('x') as stream:
        for value in observations:
            stream.write(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    save('snapshot.json',dict(schema=2 if include_sealed else 1,host=host,include_sealed_attempts=include_sealed,raw_report_captures=str(captures) if captures else None,exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         record_rejected_overlays=record_rejected_overlays,rejected_overlay_cells=sum(len(s.get('rejected_overlay_cells',[])) for s in sources),sources=len(sources),attempt_rows=len(observations),qualified_attempt_rows=sum(r['qualified_timing'] for r in observations),
         files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--host',required=True)
    parser.add_argument('--sealed-attempts',action='store_true',help='Verify and include finalized attempts not yet present in a running report; read-only')
    parser.add_argument('--record-rejected-overlays',action='store_true',help='Record invalid overlay artifacts as rejected diagnostics; never emit their timing rows. Default remains refusal.')
    args=parser.parse_args()
    export(args.root,args.output,args.host,args.sealed_attempts,args.record_rejected_overlays)
