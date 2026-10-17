"""Select current Rust CPU observations by exact resident input identity."""
import json

def load_references(paths):
    references={}
    for path in paths:
        for line in path.read_text().splitlines():
            value=json.loads(line)
            if (value['impl']=='zenmetrics' and value['device']=='cpu'
                    and value['mode']=='score' and value['status']=='ok'):
                key=(value['host'],value['os'],value['metric_name'],value['size_name'],value['threads'],value['pair_sha256'])
                references[key]=value
    return references

def same_pair_reference(references,row):
    return references.get((row['host'],row['os'],row['metric_name'].removesuffix('-gpu'),row['size_name'],
                           row['threads'],row['pair_sha256']))
