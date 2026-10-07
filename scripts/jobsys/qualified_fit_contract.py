"""Verify qualified training against a manifest-bound program contract and model loader."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

CONTRACT = 'benchmarks/shippath_qualified_fit_contract_2026-10-07.json'


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
        contract = json.load(tar.extractfile(CONTRACT))
        meta = json.load(tar.extractfile('build_meta.json'))
    if (contract.get('schema') != 'shippath-qualified-fit-contract-v1'
            or contract.get('data_sha') != kind['data_sha']
            or sha(inspector) != meta['files']['bin/inspect_qualified_checkpoint']):
        raise ValueError('trusted data/inspector identity differs from manifest-bound program')
    argv = kind['argv']
    if (argument(argv, '--spec') != contract['spec'] or argument(argv, '--head') != contract['head']
            or '--strict-admission' not in argv or '--train-only' not in argv):
        raise ValueError('strict training-only argv differs from trusted contract')
    actual_argv_sha = hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if actual_argv_sha != kind['argv_sha']:
        raise ValueError('trusted argv SHA differs from manifest')
    route = argument(argv, '--heldout') if argv[0] == 'v2_lodo_mlp.py' else 'production'
    if route not in contract['routes']:
        raise ValueError('unapproved population in trusted contract')
    contract = {**contract, 'table_admission': contract['routes'][route], 'route': route}
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
    if result.get('schema') != 'rev5-qualified-training-cell-v1':
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
    decoded = json.loads(subprocess.run([str(inspector), str(checkpoint)], check=True,
                                        capture_output=True, text=True).stdout)
    repro = decoded['repro']
    if (decoded['formula_revision'] != 5 or decoded['qualified_provenance'] is not True
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
    pinned = {t['name']: t['table_sha256'] for t in expected['table_admission']}
    if len(inputs) != len(pinned) or {t['name']: t['sha256'] for t in inputs} != pinned:
        refuse('decoded checkpoint inputs differ from pinned table admission')
    admission = repro.get('table_admission', {})
    admitted = admission.get('tables', [])
    if (admission.get('qualified_provenance') is not True or admission.get('historical_replay') is not None
            or admission.get('formula_revision') != 5 or len(admitted) != len(pinned)):
        refuse('decoded per-table admission incomplete')
    # The executor binds this lexical root to the manifest's hash-named extraction.
    # Rust records lexical admission paths and canonical input paths separately.
    def transport_path(path):
        prefix = '/var/tmp/rev4-featpot/'
        if isinstance(path, str) and path.startswith(prefix):
            return '/scratch/fit-cell/' + kind['data_sha'] + '/rev4-featpot/' + path[len(prefix):]
        return path
    input_paths = {transport_path(t['path']) for t in inputs}
    if {transport_path(t.get('path')) for t in admitted} != input_paths:
        refuse('decoded admissions do not cover exact training inputs')
    for table in admitted:
        d = table.get('stored_declarations', {})
        if (table.get('inferred') is not False or table.get('feature_set_id') != expected['feature_set_id']
                or d.get('feature_set_id') != expected['feature_set_id']
                or d != expected['table_declaration'] or table.get('requested_ids') != expected['columns']):
            refuse('decoded table feature/revision/decoder declaration differs')
    return expected['execution_contract']
