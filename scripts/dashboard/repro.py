"""Reproduction commands for model cards, derived from the CLI source.

The `--features` a metric needs is read from its `disabled_msg(...)` arm in
crates/zenmetrics-cli/src/metrics/mod.rs and dropped when the CLI's default
feature closure (crates/zenmetrics-cli/Cargo.toml) already enables it.
"""
import re
import tomllib
from catalog import ROOT

MOD = ROOT/'crates/zenmetrics-cli/src/metrics/mod.rs'
MANIFEST = ROOT/'crates/zenmetrics-cli/Cargo.toml'


def _features():
    text = MOD.read_text()
    found = {}
    for metric, features in re.findall(r'disabled_msg\(\s*"([a-z0-9-]+)",\s*"([a-z0-9,-]+)', text):
        found.setdefault(metric, [f for f in features.split(',') if f])
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
    if row.get('nn') == 'true':
        card = model.get('_card_dir') if model else None
        lines.append('# export the model once (see crates/zenmetrics-nn/models/%s/card.toml for weight terms)' % (card or '<model>'))
        lines.append('export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models')
    score = f'target/release/zenmetrics score --metric {metric} --reference ref.png --distorted dist.png'
    if row.get('needs_display') == 'true':
        score += ' --display-model standard_fhd'
    lines.append(score)
    notes = []
    if row.get('nr') == 'true':
        notes.append('No-reference: the score describes the distorted image; the CLI still takes both paths.')
    if row.get('requires_gpu') == 'true':
        notes.append('Requires a GPU backend at run time.')
    if row.get('needs_display') == 'true':
        notes.append('Choose the display model that matches your viewing condition; standard_fhd is an example, not a default.')
    return {'lines': lines, 'features': needed, 'notes': notes,
            'derived_from': ['crates/zenmetrics-cli/src/metrics/mod.rs', 'crates/zenmetrics-cli/Cargo.toml']}
