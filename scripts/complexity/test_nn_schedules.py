"""Validate LIQE/MUSIQ schedules without running neural inference.

  python3 scripts/complexity/test_nn_schedules.py
  python3 scripts/complexity/test_nn_schedules.py --graphs MODEL_ROOT --receipt OUT

The explicit graph mode requires onnx and fails on missing/unresolved inputs.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import unittest

from evaluate import ev, shape_env, validate
from test_nn_tallies import ROOT, digest, tally


def constants(file, names):
    source = (ROOT / file).read_text()
    return {name: int(re.search(rf'const {name}: usize = (\d+);', source)[1]) for name in names}


def stage_work(t, sid, w=1024, h=1024):
    s = next(s for s in t['stages'] if s['id'] == sid)
    env = shape_env(t, w, h)
    n = ev(s['elements'], env) * s['images']
    return Counter({op: n * ev(value, env) for op, value in s['ops'].items()})


def graph_data(path, shape=None):
    import onnx
    model = onnx.load(path, load_external_data=False)
    if shape:
        for dim, value in zip(model.graph.input[0].type.tensor_type.shape.dim, shape):
            dim.dim_value = value
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    values = {v.name: v.type.tensor_type
              for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    weights = {i.name: list(i.dims) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for a in node.attribute:
                if a.name == 'value':
                    weights[node.output[0]] = list(a.t.dims)
    return model, values, weights


def logical_float_ops(model, values, weights, select=lambda n: True):
    """Fixed logical norm/softmax decomposition; constant payloads are never ops."""
    macs, ops = Counter(), Counter()

    def shape(name):
        if name in values:
            dims = [d.dim_value if d.HasField('dim_value') else None
                    for d in values[name].shape.dim]
        else:
            dims = weights[name]  # initializer/Constant dimensions need no tensor payload
        if not all(isinstance(d, int) and d > 0 for d in dims):
            raise AssertionError(f'unresolved tensor {name}: {dims}')
        return dims

    for node in model.graph.node:
        if not select(node) or node.output[0] not in values:
            continue
        if values[node.output[0]].elem_type not in (1, 11):  # float, double
            continue
        op = node.op_type
        if op in {'Constant', 'Reshape', 'Transpose', 'Concat', 'Gather', 'Expand',
                  'Squeeze', 'Unsqueeze', 'Cast'}:
            continue  # movement, constant tensors and shape bookkeeping
        out = shape(node.output[0])
        ins = shape(node.input[0])
        n = math.prod(out)
        if op == 'Conv':
            macs['conv'] += n * math.prod(weights[node.input[1]][1:])
        elif op == 'MatMul':
            macs['matmul'] += n * ins[-1]
        elif op == 'Gemm':
            macs['matmul'] += math.prod(ins[:-1]) * math.prod(weights[node.input[1]])
        elif op in {'Add', 'Sub', 'Mul', 'Div', 'Sqrt', 'Tanh'}:
            key = {'Add': 'add', 'Sub': 'add', 'Mul': 'mul', 'Div': 'div',
                   'Sqrt': 'sqrt', 'Tanh': 'trig'}[op]
            ops[key] += n
        elif op == 'LayerNormalization':
            d, rows = ins[-1], math.prod(ins[:-1])
            ops.update(add=rows*(4*d-1), mul=rows*2*d, div=rows*(d+2), sqrt=rows)
        elif op == 'Softmax':
            d, rows = ins[-1], n // ins[-1]
            ops.update(add=rows*(2*d-1), cmp=rows*(d-1), exp=n, div=n)
        elif op == 'Sigmoid':
            ops.update(add=n, mul=n, exp=n, div=n)
        elif op == 'ReduceL2':
            ops.update(mul=math.prod(ins), add=math.prod(ins)-n, sqrt=n)
        elif op == 'Clip':
            ops['cmp'] += n * sum(bool(x) for x in node.input[1:])
        else:
            raise AssertionError(f'unhandled floating operator: {node.name} {op}')
        if op in {'Conv', 'Gemm'} and len(node.input) > 2:
            ops['add'] += n
    return dict(macs), dict(ops)


def audit_graphs(root):
    import onnx
    receipt = {'schema': 'iqa-nn-schedule-audit/1', 'date': '2026-10-18',
               'onnx_version': onnx.__version__, 'graphs': [],
               'scope': 'Frozen graph dimensions and logical operation formulas; no inference or timings.',
               'limitations': [
                   'LIQE qualifies the CPU rten single-crop route, not the 15-crop ORT route.',
                   'MUSIQ stem geometry combines frozen weights/attributes with the committed exporter schedule.',
                   'Norm and softmax counts use the explicit logical decompositions in each tally, not engine kernels.',
                   'MUSIQ resampling/hash arithmetic, traffic and global halo/span remain uncounted.',
                   'External weight payloads, numerical parity and dynamic engine allocation are not qualified.',
               ]}
    receipt['sources_sha256'] = {name: digest(ROOT / name) for name in [
        'crates/zenmetrics-nn/src/liqe.rs', 'crates/zenmetrics-nn/src/musiq.rs',
        'crates/zenmetrics-nn/models/liqe/export.py',
        'crates/zenmetrics-nn/models/musiq/export.py',
        'scripts/complexity/test_nn_schedules.py',
        'benchmarks/iqa_complexity_2026-10-17/tallies/liqe.swe2.json',
        'benchmarks/iqa_complexity_2026-10-17/tallies/musiq.swe2.json',
    ]}
    path = root / 'liqe_trunk.onnx'
    identity = digest(path)
    model, values, weights = graph_data(path)
    macs, ops = logical_float_ops(model, values, weights)
    t = tally('liqe')
    assert stage_work(t, 'trunk_conv') == {'fma': macs['conv']}
    assert stage_work(t, 'trunk_matmul') == {'fma': macs['matmul']}
    assert stage_work(t, 'trunk_pointwise') == ops
    assert digest(path) == identity
    receipt['graphs'].append({'file': path.name, 'sha256': identity,
                              'trunk_macs': macs, 'trunk_pointwise': ops})
    del model

    path = root / 'musiq.onnx'
    identity = digest(path)
    t = tally('musiq')
    checks = []
    for w, h in [(256, 256), (1024, 1024), (1000, 750)]:
        env = shape_env(t, w, h)
        L, T = int(env['L']), int(env['T'])
        model, values, weights = graph_data(path, [1, L, 3075])
        nodes = {n.name: n for n in model.graph.node}
        # Source stem: explicit Pad(2,3,2,3), root Conv, then Pad(0,1,0,1), MaxPool.
        # Derive convolution/pool output geometry from the actual ONNX attributes.
        def output_axis(n, size):
            attrs = {a.name: list(a.ints) for a in n.attribute if a.ints}
            k = attrs['kernel_shape'][0]
            stride = attrs.get('strides', [1, 1])[0]
            pad = attrs.get('pads', [0, 0, 0, 0])
            dilation = attrs.get('dilations', [1, 1])[0]
            return (size + pad[0] + pad[2] - dilation*(k-1) - 1)//stride + 1
        root_conv = nodes['/conv_root/Conv']
        root_side = output_axis(root_conv, 32+2+3)
        pooled_side = output_axis(nodes['/MaxPool'], root_side+1)
        stem_macs = root_side**2 * math.prod(weights[root_conv.input[1]])
        for name in ('conv1', 'conv2', 'conv3', 'conv_proj'):
            node = nodes[f'/{name}/Conv']
            side = output_axis(node, pooled_side)
            assert side == pooled_side
            stem_macs += side**2 * math.prod(weights[node.input[1]])
        assert (root_side, pooled_side) == (16, 8)
        assert stage_work(t, 'stem_conv', w, h) == {'fma': L*stem_macs}
        embed = nodes['/embedding/Gemm']
        assert stage_work(t, 'embed', w, h)['fma'] == L*math.prod(weights[embed.input[1]])
        # Tensor shapes for transformer nodes resolve with a concrete L, including CLS.
        macs, ops = logical_float_ops(model, values, weights,
            lambda n: n.name.startswith(('/blocks.', '/enc_norm/', '/head/')))
        expected = Counter(ops)
        expected['fma'] = sum(macs.values())
        actual = Counter()
        for sid in ('blocks_linear', 'attention', 'softmax', 'layernorm',
                    'block_pointwise', 'gelu', 'head'):
            actual.update(stage_work(t, sid, w, h))
        assert actual == expected, (w, h, actual, expected)
        checks.append({'shape': [w, h], 'patch_rows': L, 'transformer_tokens': T,
                       'stem_macs_per_patch': stem_macs,
                       'transformer_and_head_macs': macs,
                       'transformer_and_head_pointwise': ops})
        del model
    assert digest(path) == identity
    receipt['graphs'].append({'file': path.name, 'sha256': identity, 'checks': checks})
    return receipt


class Schedules(unittest.TestCase):
    def test_liqe_crop_and_repeated_f64_head(self):
        t = tally('liqe')
        self.assertEqual(validate(t), [])
        c = constants('crates/zenmetrics-nn/src/liqe.rs', ['CROP', 'NUM_PATCH', 'DIM', 'NPROMPTS'])
        dot = stage_work(t, 'head_dot')
        n = c['NUM_PATCH']*c['DIM']*c['NPROMPTS']
        self.assertEqual(dot, {'mul': n, 'add': n, 'cvt': 2*n})
        self.assertEqual(next(s for s in t['stages'] if s['id'] == 'head_dot')['precision'], 'f64')
        # Changing the original image size cannot change the crop/head work.
        for w, h in [(224, 224), (1000, 750), (3840, 2160)]:
            self.assertEqual(stage_work(t, 'head_dot', w, h), dot)
            self.assertEqual(stage_work(t, 'crop', w, h)['load'], 3*c['CROP']**2)

    def test_musiq_class_token_and_rectangular_patch_grid(self):
        t = tally('musiq')
        self.assertEqual(validate(t), [])
        for w, h in [(32, 32), (256, 256), (1000, 750), (1920, 1080), (3840, 2160)]:
            L = 49+144+math.ceil(w/32)*math.ceil(h/32)
            env = shape_env(t, w, h)
            self.assertEqual((env['L'], env['T']), (L, L+1))
            self.assertEqual(stage_work(t, 'attention', w, h)['fma'], 14*2*384*(L+1)**2)
            self.assertEqual(stage_work(t, 'embed', w, h)['fma'], 16384*384*L)
        self.assertGreater(shape_env(t, 3840, 2160)['L'], 8192)


if __name__ == '__main__':
    if '--graphs' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--graphs', type=Path, required=True)
        parser.add_argument('--receipt', type=Path, required=True)
        args = parser.parse_args()
        result = audit_graphs(args.graphs)
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(f"Verified LIQE and MUSIQ graph schedules: {args.receipt}")
    else:
        unittest.main()
