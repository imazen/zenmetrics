"""Check returned cell exclusions against the independently pinned input root."""
import json


def check_linked_cell(dest, root, expected_sha, digest):
    if digest(root/'wide/frozen.json') != expected_sha:
        raise ValueError('linked input root differs from the registered pin')
    frozen = json.loads((root/'wide/frozen.json').read_text())
    packet = json.loads((root/'datasets/receipt.json').read_text())
    packet_sha = digest(root/'datasets/receipt.json')
    link = packet['linked_subset_omissions']
    relative = 'datasets/linked-subset-omissions.json'
    authority_sha = digest(root/relative)
    if (packet_sha != frozen['dataset_legs_sha256'] or link['path'] != relative
            or authority_sha != link['sha256'] or authority_sha != frozen['auxiliary_files'][relative]):
        raise ValueError('linked exclusion authority is not frozen')
    authority = json.loads((root/relative).read_text())
    if authority['labels_read'] is not False:
        raise ValueError('linked authority must be label-free')
    rec = json.loads((dest/'byc_record.json').read_text())
    result = json.loads((dest/'result.json').read_text())
    selected = json.loads((dest/'DATASET_GROUP_SELECTION.json').read_text())
    fold = rec['fold']
    rule = authority['folds'][fold]
    primary = rule['heldout_collections'] if fold in ('pipal','mos7','pairs') else []
    excluded = list(dict.fromkeys([*primary,*rule['excluded_subset_collections']]))
    collections = [r['collection'] for r in packet['groups']]
    omitted = [c for c in collections if c in excluded]
    retained = [c for c in collections if c not in excluded]
    if (rec.get('linked_subset_omission_sha256') != authority_sha
            or rec.get('dataset_exclusions') != excluded
            or result['frozen_sha256'] != expected_sha
            or selected['root_sha256'] != expected_sha or selected['receipt_sha256'] != packet_sha
            or selected['requested_exclusions'] != excluded
            or selected['omitted_collections'] != omitted
            or selected['retained_collections'] != retained):
        raise ValueError('returned cell does not enforce the frozen linked exclusions')
    return dict(root_sha256=expected_sha,authority_sha256=authority_sha,
                excluded_collections=excluded,retained_groups=len(retained))
