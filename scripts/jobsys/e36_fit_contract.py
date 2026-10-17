"""E36's exact family-mask derivation, before the unchanged checkpoint gate."""
import copy
import hashlib
import json
from pathlib import Path

PACKAGE = 'benchmarks/e36_fit_contract_2026-10-17.json'
ARMS = {'tv0-c': 0., 'tv1-c': 1., 'tv10-c': 10.}
SMOKE = {'schema':'e36-memory-smoke-v1','epochs':2,'pairs_per_epoch':128,'wall_cap_sec':900,'installable':False,'pack_production':False}


def execution_environment(kind, program):
    """Export authorization only from the archive-bound native contract."""
    from qualified_fit_contract import trusted_contract
    expected=trusted_contract(kind,program.parent/'program.tar.gz',program/'bin/inspect_qualified_checkpoint')
    diagnostic=expected.get('execution_contract')=='local-smoke'
    if expected.get('schema')!='e36-research-fit-contract-v1':
        raise ValueError('native E36 execution contract required')
    if diagnostic:
        if (expected.get('e36_memory_smoke')!=SMOKE or expected.get('route')!='production'
            or expected.get('seed_index')!=0 or expected['e36']['arm']!='tv10-c'
            or [expected.get(k) for k in ['epochs','pairs_per_epoch','selected_epoch']]!=[2,128,1]):
            raise ValueError('exact archive-bound diagnostic authorization required')
    elif (expected.get('execution_contract')!='registered-fit' or expected.get('e36_memory_smoke') is not None
          or [expected.get(k) for k in ['epochs','pairs_per_epoch','selected_epoch']]!=[120,50000,119]):
        raise ValueError('unchanged full scientific budget required')
    return dict(ZEN_E36_EXECUTION_CONTRACT='local-smoke' if diagnostic else 'registered-fit',
                ZEN_E36_MEMORY_SMOKE='2:128' if diagnostic else '',ZEN_E36_ARGV_SHA=kind['argv_sha'])


def smoke_policy(argv,contract):
    """Only a distinct archive-bound diagnostic variant permits a smoke."""
    from qualified_fit_contract import argument
    policy=contract.get('e36_memory_smoke')
    requested='--local-smoke-budget' in argv
    if policy is None:
        if requested:raise ValueError('E36 full-budget variant refuses smoke')
        return None
    if policy!=SMOKE or not requested or argument(argv,'--local-smoke-budget')!='2:128' or any(flag in argv for flag in ['--pack-production','--e33-output-stage']):
        raise ValueError('E36 diagnostic smoke policy/budget differs')
    return policy


def bind(argv, contract, files):
    from qualified_fit_contract import argument
    e = contract['e36']
    smoke_policy(argv,contract)
    if (argument(argv, '--e36-arm') != e['arm'] or e['arm'] not in ARMS
            or contract['epochs'] != 120
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
        declarations = changed.get('table_declarations', {})
        if r['source_table_sha256'] in declarations:
            declarations[actual_sha] = copy.deepcopy(declarations[r['source_table_sha256']])
    return changed


def qualify_output(program, dest, kind, cell, result):
    """Run native decoded-checkpoint admission INSIDE fit-cell-exec."""
    from qualified_fit_contract import trusted_contract, verify_training, sha
    inspector = program / 'bin/inspect_qualified_checkpoint'
    expected = trusted_contract(kind, program.parent / 'program.tar.gz', inspector)
    checkpoint = Path(result['selected_bake'])
    if checkpoint.parent != dest/'refit' or sha(checkpoint) != result['selected_bake_sha256']:
        raise ValueError('E36 raw checkpoint path/hash differs')
    execution='local-smoke' if expected.get('e36_memory_smoke')==SMOKE else 'registered-fit'
    if verify_training(result, checkpoint, kind, expected, inspector) != execution:
        raise ValueError('E36 checkpoint qualification failed')
    binding_path = dest/'e36-model-binding.json'
    binding = json.loads(binding_path.read_text())
    packed_sha = result.get('packed_model_sha256')
    if expected['route'] == 'production' and execution=='registered-fit':
        packed = Path(result['packed_model'])
        if not packed.is_relative_to(dest) or sha(packed) != packed_sha:
            raise ValueError('E36 packed checkpoint path/hash differs')
    elif packed_sha is not None:
        raise ValueError('E36 withheld fit unexpectedly exports a packed pilot')
    if (binding['raw_checkpoint_sha256'] != sha(checkpoint) or binding['packed_model_sha256'] != packed_sha
            or binding['program_sha256'] != kind['program_sha']
            or binding['arm'] != expected['e36']['arm'] or binding['selected_epoch'] != (1 if execution=='local-smoke' else 119)):
        raise ValueError('E36 packed/raw projection differs from qualified checkpoint')
    receipt = dict(schema='e36-qualified-checkpoint-v1', execution_contract=execution,
        cell=cell, program_sha256=kind['program_sha'], data_sha256=kind['data_sha'], argv_sha256=kind['argv_sha'],
        result_sha256=sha(dest/'result.json'), raw_checkpoint_sha256=sha(checkpoint), packed_model_sha256=packed_sha,
        model_binding_sha256=sha(binding_path), inspector_sha256=sha(inspector),
        qualification_owner_sha256=sha(program/'qualified_fit_contract.py'),
        e36_owner_sha256=sha(program/'e36_fit_contract.py'), qualified=True,
        contains_labels=False, label_payload_read_by_projection=False)
    with (dest/'e36-qualified-checkpoint.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
