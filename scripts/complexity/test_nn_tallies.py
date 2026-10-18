"""NN tally controls; frozen-graph checks are explicit and never skip missing files.

Standard-library controls:
  python3 scripts/complexity/test_nn_tallies.py
Frozen graph audit (requires onnx; does not infer shapes or execute a model):
  python3 scripts/complexity/test_nn_tallies.py --pair-graphs MODEL_ROOT --receipt OUT
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

from evaluate import evaluate, validate

ROOT = Path(__file__).resolve().parents[2]
TALLIES = ROOT / 'benchmarks/iqa_complexity_2026-10-17/tallies'
PAIR_GRAPHS = {
    'lpips-alex': 'lpips_alex_full.onnx',
    'lpips-vgg': 'lpips_vgg_full.onnx',
    'lpips-squeeze': 'lpips_squeeze_full.onnx',
    'stlpips-alex': 'stlpips_alex_full.onnx',
    'stlpips-vgg': 'stlpips_vgg_full.onnx',
    'dists': 'dists_full.onnx',
    'deepdc': 'deepdc_full.onnx',
}


def tally(metric):
    return json.loads((TALLIES / f'{metric}.swe2.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_pair_graphs(root):
    import onnx
    rows = []
    for metric, filename in PAIR_GRAPHS.items():
        path = root / filename
        before = digest(path)
        model = onnx.load(path, load_external_data=False)
        inputs = []
        for value in model.graph.input:
            dims = [d.dim_value if d.HasField('dim_value') else d.dim_param
                    for d in value.type.tensor_type.shape.dim]
            inputs.append({'name': value.name, 'shape': dims})
        expected = [{'name': side, 'shape': [1, 3, 'h', 'w']} for side in ('r', 'd')]
        if inputs != expected:
            raise AssertionError(f'{filename}: expected both RGB pair inputs, got {inputs}')
        stages = {s['id']: s for s in tally(metric)['stages']}
        multipliers = {sid: s['images'] for sid, s in stages.items() if sid.startswith('trunk_')}
        if not multipliers or set(multipliers.values()) != {1} or stages['ingress']['images'] != 2:
            raise AssertionError(f'{metric}: incorrect graph/ingress multipliers')
        if digest(path) != before:
            raise AssertionError(f'{filename}: graph changed during audit')
        rows.append({'metric': metric, 'file': filename, 'sha256': before,
                     'inputs': inputs, 'graph_stage_images': multipliers, 'ingress_images': 2})
        del model
    return {
        'schema': 'iqa-pair-graph-input-audit/1',
        'date': '2026-10-18', 'onnx_version': onnx.__version__,
        'scope': 'Pair-input identity and tally multipliers only; no shape inference or model execution.',
        'limitations': [
            'Existing 224x224 operation-density extrapolations and pointwise estimates remain provisional.',
            'MAC totals, dynamic shapes, reductions, traffic, halo, span and runtime parity are not qualified by this audit.',
            'External weight tensor payloads are not loaded or qualified; hashes identify the ONNX graph files.',
        ],
        'graphs': rows,
    }


class PairMultipliers(unittest.TestCase):
    def test_graph_counts_once_and_ingress_twice(self):
        for metric in PAIR_GRAPHS:
            with self.subTest(metric=metric):
                t = tally(metric)
                self.assertEqual(validate(t), [])
                graph = [s for s in t['stages'] if s['id'].startswith('trunk_')]
                self.assertTrue(graph)
                self.assertTrue(all(s['images'] == 1 for s in graph))
                self.assertEqual(next(s for s in t['stages'] if s['id'] == 'ingress')['images'], 2)
                # A pair graph's density is already per pair, at every evaluated shape.
                for w, h in [(256, 256), (1024, 1024), (1920, 1080), (3840, 2160)]:
                    r = evaluate(t, w, h)
                    graph_rows = [s for s in r['stages'] if s['id'].startswith('trunk_')]
                    self.assertTrue(all(s['elements_per_px'] == 1 for s in graph_rows))


if __name__ == '__main__':
    if '--pair-graphs' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--pair-graphs', type=Path, required=True)
        parser.add_argument('--receipt', type=Path, required=True)
        args = parser.parse_args()
        result = audit_pair_graphs(args.pair_graphs)
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(f"Verified {len(result['graphs'])}/7 pair graphs: {args.receipt}")
    else:
        unittest.main()
