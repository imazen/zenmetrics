"""E36's exact family-mask derivation, before the unchanged checkpoint gate."""
import copy
import hashlib
import json

PACKAGE = 'benchmarks/e36_fit_contract_2026-10-17.json'
ARMS = {'tv0-c': 0., 'tv1-c': 1., 'tv10-c': 10.}


def bind(argv, contract, files):
    from qualified_fit_contract import argument
    e = contract['e36']
    if (argument(argv, '--e36-arm') != e['arm'] or e['arm'] not in ARMS
            or '--local-smoke-budget' in argv or contract['epochs'] != 120
            or contract['pairs_per_epoch'] != 50000 or contract['selected_epoch'] != 119
            or argument(argv, '--e36-tv-contract-sha256') != e['contract_sha256']
            or argument(argv, '--e36-base-filter-sha256') != e['base_filter_sha256']
            or files['bin/zensim_mlp_train'] != e['trainer_sha256']):
        raise ValueError('E36 full-budget arm/pool/mask/trainer binding differs')


def derive(result, decoded, expected):
    """Admit only the preregistered, unchanged-target row selection.

    Child target bytes are produced inside fit-cell-exec by the pinned existing
    write_curated owner. Here only its metadata projection is read. The generic
    gate still compares the actual checkpoint's input hashes, row counts,
    numerical weights, feature declarations, seeds and final epoch.
    """
    if expected.get('schema') != 'e36-research-fit-contract-v1':
        return expected
    e = expected['e36']
    tv = decoded['repro'].get('e36_tv', {})
    fixed = dict(schema='e36-effective-tv-v1', protocol_build='e36_tv_v1', cfg_e36_tv=True,
                 manifest_sha256=e['manifest_sha256'], contract_sha256=e['contract_sha256'],
                 binary_sha256=e['trainer_sha256'], lambda_=ARMS[e['arm']], raw_margin=.05,
                 every_primary_pairs=50, epoch_local_counter=True, batch=32,
                 pair_weight_class_values=[2, 1], mean_pool_weight=e['mean_pool_weight'],
                 tv_rng_seed=expected['sample_seed'] ^ 0xe3607476726e6700,
                 normalization='lambda*w/(32*mean_pool_weight)',
                 optimizer='pre-update gradient addition; one existing primary Adam boundary; no TV-only Adam')
    fixed['lambda'] = fixed.pop('lambda_')
    if any(tv.get(k) != v for k, v in fixed.items()) or tv.get('embedded_protocol_source_sha256') != e['protocol_source_sha256']:
        raise ValueError('E36 actual checkpoint TV configuration differs')
    proof = result.get('selection', {}).get('e36_base_filter', {})
    if (proof.get('schema') != 'e36-applied-base-filter-v1'
            or proof.get('plan_sha256') != e['base_filter_sha256']
            or proof.get('targets_changed') != 0 or proof.get('exact_group_weights_preserved') is not True):
        raise ValueError('E36 exact unchanged-target family mask proof missing')
    rows = proof.get('groups', [])
    parents = expected['table_admission']
    if len(rows) != len(parents) or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('E36 base group coverage differs')
    by_name = {r['name']: r for r in rows}
    changed = copy.deepcopy(expected)
    for table in changed['table_admission']:
        r = by_name.get(table['name'], {})
        if r.get('source_table_sha256') != table['table_sha256'] or r.get('targets_changed') != 0:
            raise ValueError('E36 base parent identity/targets differ')
        wanted = e['curated_tables'].get(table['table_sha256'])
        if wanted is None:
            if r.get('removed') != 0:
                raise ValueError('E36 unregistered row removal')
            continue
        declaration = r.get('stored_manifest', {})
        actual_sha = r.get('table_sha256')
        if not isinstance(actual_sha, str) or len(actual_sha) != 64:
            raise ValueError('E36 missing child table identity')
        exact = {**wanted['manifest'], 'table_sha256': actual_sha}
        raw = (json.dumps(exact, indent=2) + '\n').encode()
        if (r.get('removed') != wanted['removed'] or declaration != exact
                or r.get('manifest_sha256') != hashlib.sha256(raw).hexdigest()):
            raise ValueError('E36 curated declaration differs from exact frozen row selection')
        table.update({k: declaration[k] for k in ('table_sha256', 'keys_sha256', 'row_keys_sha256', 'row_selection_sha256')})
        table['manifest_sha256'] = r['manifest_sha256']
        changed['input_contracts'][expected['route']][table['name']]['rows'] = declaration['rows_kept']
    return changed
