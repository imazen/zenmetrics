"""Select current Rust CPU observations by exact resident input identity."""
import json,math,struct

def completed_warmup(value):
    """An actual returned score can compare algorithms without qualifying latency."""
    if not (value['impl']=='zenmetrics' and value['device']=='cpu' and value['mode']=='score'
            and value['status'].startswith('skipped:timeout_')):return None
    diagnostics=value.get('initialization_diagnostics',[])
    if len(diagnostics)!=1:return None
    observation=diagnostics[0]
    bits=observation.get('score_bits')
    if observation.get('completed') is not True or not isinstance(bits,str) or len(bits)!=16:return None
    try:score=struct.unpack('>d',bytes.fromhex(bits))[0]
    except (ValueError,struct.error):return None
    if not math.isfinite(score):return None
    if not value.get('log_sha256') or not value.get('log_path'):return None
    return value|dict(score=score,reference_kind='completed_untimed_warmup',
                      reference_source_status=value['status'])

def load_references(paths):
    references={}
    for path in paths:
        for line in path.read_text().splitlines():
            value=json.loads(line)
            if (value['impl']=='zenmetrics' and value['device']=='cpu' and value['mode']=='score'):
                if value['status']=='ok':
                    if value.get('n',0)<(value.get('required_samples',10) if value.get('coverage_first') else 10) or not isinstance(value.get('score'),(int,float)) or not math.isfinite(value['score']):continue
                else:
                    value=completed_warmup(value)
                    if value is None:continue
                key=(value['host'],value['os'],value['metric_name'],value['size_name'],value['threads'],value['pair_sha256'])
                # A later failed timing never replaces an already measured score.
                if key not in references or value['status']=='ok':references[key]=value
    return references

def same_pair_reference(references,row):
    return references.get((row['host'],row['os'],row['metric_name'].removesuffix('-gpu'),row['size_name'],
                           row['threads'],row['pair_sha256']))
