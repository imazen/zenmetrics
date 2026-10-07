"""Verify qualified training against a manifest-bound program contract and model loader."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

CONTRACT = 'benchmarks/shippath_qualified_fit_contract_2026-10-07.json'
E29_CONTRACT = 'benchmarks/e29_fit_contract_2026-10-07.json'
V40_CONTRACT = 'benchmarks/v40_fit_contract_2026-10-07.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def argument(argv, key):
    if argv.count(key) != 1:
        raise ValueError(f'trusted contract needs exactly one {key}')
    return argv[argv.index(key) + 1]


def trusted_contract(kind, program_archive, inspector):
    if program_archive is None or inspector is None:
        raise ValueError('unqualified/incomplete result: trusted program archive and checkpoint inspector required')
    if sha(program_archive) != kind['program_sha']:
        raise ValueError('trusted program SHA differs from manifest')
    with tarfile.open(program_archive, 'r:gz') as tar:
        meta = json.load(tar.extractfile('build_meta.json'))
        v40 = V40_CONTRACT in meta['files']
        if v40:
            raw = tar.extractfile(V40_CONTRACT).read()
            if hashlib.sha256(raw).hexdigest() != meta['files'][V40_CONTRACT]:
                raise ValueError('V40 contract bytes differ from program inventory')
            package = json.loads(raw)
            if package.get('schema') != 'v40-research-fit-package-v1':
                raise ValueError('unregistered V40 package')
            argv = kind['argv']
            matches = [c for c in package['variants'] if c['root'] == argument(argv, '--root')
                       and c['spec'] == argument(argv, '--spec') and c['data_sha'] == kind['data_sha']]
            if len(matches) != 1 or matches[0].get('launchable') is not True:
                raise ValueError('unregistered or owner-blocked V40 variant')
            contract = matches[0]
        e29 = not v40 and E29_CONTRACT in meta['files']
        if not v40:
            contract = json.load(tar.extractfile(E29_CONTRACT if e29 else CONTRACT))
        if e29:
            inherited = json.load(tar.extractfile(CONTRACT))
            contract = {**inherited, **contract}
    if (contract.get('schema') != ('v40-research-fit-contract-v1' if v40 else 'e29-research-fit-contract-v1' if e29 else 'shippath-qualified-fit-contract-v1')
            or contract.get('data_sha') != kind['data_sha']
            or sha(inspector) != meta['files']['bin/inspect_qualified_checkpoint']):
        raise ValueError('trusted data/inspector identity differs from manifest-bound program')
    argv = kind['argv']
    spec = argument(argv, '--spec')
    if e29:
        if spec not in contract['specs']:
            raise ValueError('unregistered E29 arm')
        contract['spec'] = spec
    if (argument(argv, '--spec') != contract['spec'] or argument(argv, '--head') != contract['head']
            or '--strict-admission' not in argv or '--train-only' not in argv):
        raise ValueError('strict training-only argv differs from trusted contract')
    actual_argv_sha = hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if actual_argv_sha != kind['argv_sha']:
        raise ValueError('trusted argv SHA differs from manifest')
    route = argument(argv, '--heldout') if argv[0] == 'v2_lodo_mlp.py' else 'production'
    if route not in contract['routes']:
        raise ValueError('unapproved population in trusted contract')
    tables = contract['routes'][route]
    if e29 and spec != contract['specs'][0]:
        tables = sorted([*tables, contract['hdr_tables'][spec.rsplit(':', 1)[1]]], key=lambda t:t['name'])
    contract = {**contract, 'table_admission': tables, 'route': route,
                'research_hdr': contract.get('research_hdr', False) if v40 else bool(e29 and spec != contract['specs'][0])}
    if contract['research_hdr'] or contract.get("research_upiq"):
        if contract.get("research_upiq"):
            if (argument(argv, "--upiq380-fit") != "/var/tmp/rev4-featpot/upiq380-fit/upiq380_fit.parquet"
                    or argument(argv, "--upiq-label-disposition") != "/var/tmp/rev4-featpot/upiq380-fit/owner_disposition.json"):
                raise ValueError("E31 native fit/owner decision binding differs")
        contract['feature_set_id'] = None
    contract['seed_index'] = int(argument(argv, '--seed-index'))
    if not 0 <= contract['seed_index'] < (3 if route == 'production' else 10):
        raise ValueError('seed outside registered grid')
    fold = 0 if route == 'production' else contract['fold_order'].index(route)
    contract['init_seed'] = contract['init_seeds'][contract['seed_index']]
    contract['sample_seed'] = contract['sample_seeds'][(contract['seed_index'] + fold) % 10]
    if [int(v) for v in argument(argv, '--columns').split(',')] != contract['columns']:
        raise ValueError('columns differ from pinned recipe')
    contract['execution_contract'] = 'registered-fit'
    if '--local-smoke-budget' in argv:
        epochs, pairs = map(int, argument(argv, '--local-smoke-budget').split(':'))
        if not (0 < epochs < contract['epochs'] and 0 < pairs < contract['pairs_per_epoch']):
            raise ValueError('invalid explicit local smoke budget')
        contract.update(epochs=epochs, pairs_per_epoch=pairs, selected_epoch=epochs-1,
                        execution_contract='local-smoke')
    return contract


def verify_training(result, checkpoint, kind, expected, inspector):
    def refuse(why):
        raise ValueError(f'unqualified/incomplete training-only result: {why}')
    if '--strict-admission' not in kind['argv'] or '--train-only' not in kind['argv']:
        refuse('strict training-only flags required')
    if result.get('training_only') is not True:
        refuse('registered training-only contract cannot use a prediction result')
    if result.get('schema') != ('e31-native-hdr-research-training-cell-v1' if expected.get('research_upiq') else 'e29-research-training-cell-v1' if expected.get('research_hdr') else 'rev5-qualified-training-cell-v1'):
        refuse('result schema')
    for key in ('epochs', 'pairs_per_epoch', 'seed_index', 'wide_receipt_sha256', 'frozen_sha256',
                'data_role_decision_sha256', 'execution_contract'):
        if result.get(key) != expected[key]:
            refuse(f'{key} differs from manifest-bound contract')
    selection = result.get('selection', {})
    if selection.get('epoch_rule') != 'last' or selection.get('selected_epoch') != expected['selected_epoch']:
        refuse('selected epoch differs from trusted budget')
    tables = selection.get('strict_table_admission', [])
    if sorted(tables, key=lambda t: t.get('name', '')) != expected['table_admission']:
        refuse('table admission differs from pinned inputs')
    decoded = json.loads(subprocess.run([str(inspector), str(checkpoint), *(['--e31-research'] if expected.get('research_upiq') else ['--e29-research'] if expected.get('research_hdr') else [])], check=True,
                                        capture_output=True, text=True).stdout)
    repro = decoded['repro']
    if (decoded['formula_revision'] != 5 or decoded['qualified_provenance'] is not (not (expected.get('research_hdr', False) or expected.get('research_upiq', False)))
            or decoded['feature_set_id'] != expected['feature_set_id']
            or repro.get('checkpoint_epoch') != f"{expected['selected_epoch']:03d}"
            or repro.get('epochs') != expected['epochs']
            or repro.get('requested_epochs', repro.get('epochs')) != expected['epochs']
            or repro.get('pairs_per_epoch') != expected['pairs_per_epoch']
            or repro.get('pair_sampling') != 'uniform'
            or repro.get('init_seed') != expected['init_seed']
            or repro.get('sample_seed') != expected['sample_seed']):
        refuse('decoded checkpoint budget/epoch/revision/sampling/seed differs')
    inputs = repro.get('inputs', [])
    if expected.get('schema') == 'v40-research-fit-contract-v1':
        actual_inputs = {t['name']: {k: t.get(k) for k in
            ('loss_mode', 'n_features', 'rows', 'train_w', 'val_w', 'within_ref')} for t in inputs}
        if (result.get('train_weights') != expected['train_weights'][expected['route']]
                or actual_inputs != expected['input_contracts'][expected['route']]
                or repro.get('effective_minibatch') != 1
                or repro.get('target_column') != 'human_score' or repro.get('target_scale') != 1.0):
            refuse('V40 input roles/weights/modes/rows or numerical recipe differs')
    pinned = {t['name']: t['table_sha256'] for t in expected['table_admission']}
    if len(inputs) != len(pinned) or {t['name']: t['sha256'] for t in inputs} != pinned:
        refuse('decoded checkpoint inputs differ from pinned table admission')
    admission = repro.get('table_admission', {})
    admitted = admission.get('tables', [])
    native = admission.get('upiq380') if expected.get('research_upiq') else None
    native_count = 1 if expected.get('research_upiq') else 0
    if (admission.get('qualified_provenance') is not (not (expected.get('research_hdr', False) or expected.get('research_upiq', False))) or admission.get('historical_replay') is not None
            or admission.get('formula_revision') != 5 or len(admitted) + native_count != len(pinned)):
        refuse('decoded per-table admission incomplete')
    # The executor binds this lexical root to the manifest's hash-named extraction.
    # Rust records lexical admission paths and canonical input paths separately.
    def transport_path(path):
        prefix = '/var/tmp/rev4-featpot/'
        if isinstance(path, str) and path.startswith(prefix):
            return '/scratch/fit-cell/' + kind['data_sha'] + '/rev4-featpot/' + path[len(prefix):]
        return path
    if expected.get('research_upiq'):
        wanted = expected['upiq_admission']
        for key in ('table_sha256', 'manifest_sha256', 'keys_sha256', 'label_source',
                    'label_disposition', 'label_disposition_sha256'):
            if not isinstance(native, dict) or native.get(key) != wanted[key]:
                refuse('decoded UPIQ TRAIN-fit/owner disposition differs: ' + key)
        if (native.get('selected_ids') != expected['columns']
                or native.get('native_width') != 1825 or native.get('logical_width') != 1853
                or native.get('input_contract') != 'upiq-exr-bt709-nits-v1'
                or selection.get('upiq380_fit_admission') != wanted):
            refuse('decoded native UPIQ projection/fit selection differs')
        native_input = [t for t in inputs if t['name'] == 'upiq380']
        if len(native_input) != 1 or transport_path(native['path']) != transport_path(native_input[0]['path']):
            refuse('decoded native UPIQ path does not cover exact fit input')
    input_paths = {transport_path(t['path']) for t in inputs if not (native_count and t['name'] == 'upiq380')}
    if {transport_path(t.get('path')) for t in admitted} != input_paths:
        refuse('decoded admissions do not cover exact training inputs')
    input_names = {transport_path(t['path']): t['name'] for t in inputs}
    for table in admitted:
        expected_d = expected.get('table_declarations', {}).get(
            pinned[input_names[transport_path(table['path'])]], expected.get('table_declaration'))
        d = table.get('stored_declarations', {})
        if expected.get('research_hdr') and d.get('study') == 'E29':
            arm = expected['spec'].rsplit(':', 1)[1]
            if (table.get('feature_set_id') is not None or table.get('inferred') is not False
                    or d != expected['hdr_declarations'][arm] or table.get('requested_ids') != expected['columns']
                    or repro.get('hdr_consensus_research') is not True):
                refuse('decoded native HDR subset differs from pinned research declaration')
            lists = repro.get('rank_pair_lists', [])
            if arm == 'hc4':
                if len(lists) != 1 or lists[0].get('group') != 'hdr' or lists[0].get('sha256') != d['pair_list_sha256']:
                    refuse('decoded agreement pair-list differs from pinned input')
            elif lists:
                refuse('Borda arm unexpectedly used an agreement pair-list')
            continue
        if (table.get('inferred') is not False or table.get('feature_set_id') != expected.get('sdr_feature_set_id', expected['feature_set_id'])
                or d.get('feature_set_id') != expected.get('sdr_feature_set_id', expected['feature_set_id'])
                or d != expected_d or table.get('requested_ids') != expected['columns']):
            refuse('decoded table feature/revision/decoder declaration differs')
    return expected['execution_contract']
