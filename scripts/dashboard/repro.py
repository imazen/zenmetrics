"""Reproduction commands for model cards, derived from the CLI source.

The `--features` a metric needs is read from its `disabled_msg(...)` arm in
crates/zenmetrics-cli/src/metrics/mod.rs and dropped when the CLI's default
feature closure (crates/zenmetrics-cli/Cargo.toml) already enables it.
"""
import re
import tomllib
from catalog import ROOT, compiled_groups, function

MOD = ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs'
MANIFEST = ROOT/'crates/zenmetrics-cli/Cargo.toml'


def _features():
    text = MOD.read_text()
    found = {}
    for metric, features in re.findall(r'disabled_msg\(\s*"([a-z0-9-]+)",\s*"([a-z0-9,-]+)', text):
        found.setdefault(metric, [f for f in features.split(',') if f])
    names=dict(re.findall(r'MetricKind::(\w+) => "([^"]+)"',function(text,'name')))
    dynamic=r'(MetricKind::\w+(?:\s*\|\s*MetricKind::\w+)*)\s*=>\s*(?:\{\s*)?Err\(disabled_msg\(kind.name\(\),\s*"([a-z0-9,-]+)"'
    for arm,features in re.findall(dynamic,text):
        for kind in re.findall(r'MetricKind::(\w+)',arm):
            found.setdefault(names[kind],features.split(','))
    for group,kinds in compiled_groups(text).items():
        feature={'nn':'nn','nr':'nr','pyiqa':'pyiqa-ports'}.get(group)
        if feature:
            for kind in kinds:found.setdefault(names[kind],[feature])
    return found


def _default_closure():
    table = tomllib.loads(MANIFEST.read_text())['features']
    seen, todo = set(), list(table.get('default', []))
    while todo:
        f = todo.pop()
        if f in seen or f.startswith('dep:') or '/' in f:
            continue
        seen.add(f)
        todo += table.get(f, [])
    return seen


FEATURES, DEFAULT = _features(), _default_closure()


def commands(metric, row, model):
    """Shell lines that score one pair with this metric from a fresh clone."""
    needed = [f for f in FEATURES.get(metric, []) if f not in DEFAULT]
    if row.get('nn') == 'true' and 'nn' not in needed and 'nn' not in DEFAULT:
        needed.append('nn')
    build = 'cargo build --release -p zenmetrics-cli' + (' --features ' + ','.join(needed) if needed else '')
    lines = ['git clone https://github.com/imazen/zenmetrics && cd zenmetrics',
             'bash scripts/ci/clone-siblings.sh', build]
    card_path=model.get('_card_source','') if model else ''
    has_files=row.get('nn') == 'true' or bool(card_path and metric!='cw-ssim-pyiqa')
    if has_files:
        card = model.get('_card_dir') if model else None
        card_path=card_path or 'crates/zenmetrics-nn/models/%s/card.toml' % (card or '<model>')
        lines.append('# export the model once (see %s for weight terms and export command)' % card_path)
        lines.append('export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models')
    score = f'target/release/zenmetrics score --metric {metric} --reference ref.png --distorted dist.png'
    if row.get('needs_display') == 'true':
        score += ' --display-model standard_fhd'
    lines.append(score)
    notes = []
    if model and model.get('_consent'):
        notes.append('After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS='+model['_consent']+'. Consent is not implied by building the CLI.')
    if row.get('nr') == 'true':
        notes.append('No-reference: the score describes the distorted image; the CLI still takes both paths.')
    if row.get('requires_gpu') == 'true':
        notes.append('Requires a GPU backend at run time.')
    if row.get('needs_display') == 'true':
        notes.append('Choose the display model that matches your viewing condition; standard_fhd is an example, not a default.')
    return {'lines': lines, 'features': needed, 'notes': notes,
            'derived_from': ['crates/zenmetrics-cli/src/metrics/mod.rs', 'crates/zenmetrics-cli/Cargo.toml']}
