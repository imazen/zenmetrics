"""Count multiply-accumulates and element operations of the exported graphs at one input size.

usage: onnx_ops.py MODEL_DIR W H [topiq_fr|topiq_fr_pipal ...]

Runs each graph once in onnxruntime (CPU) with every intermediate exposed as an output, so every
node's actual shape is known (no symbolic shape inference), then counts per node:
  Conv      MACs = out_elems * (C_in / groups) * kh * kw   (+ out_elems adds for a bias)
  MatMul    MACs = out_elems * K;  Gemm the same plus bias adds
  other     out_elems per op type (element work), movement ops listed separately.
Inputs: TOPIQ trunk gets a random [1,3,H,W] image; its head gets two trunk runs' taps plus the
pooling matrices and embeddings the Rust host builds (same rules as export.py). Prints JSON: per
graph, mac total, by_op.

ONNX_OPS_STATIC=1 propagates the concrete input shapes through the graph instead (static_shapes:
no graph run, no intermediates in memory; needed at 3840x2160, where exposing every TOPIQ
intermediate takes tens of GiB, and onnx's own shape inference loses the head's computed Slice
bounds). It counts the same nodes with the same rules; the two modes agree exactly where both run
(checked at 256x256, 840x944 and 1024x1024).
"""
import json, os, sys
import numpy as np, onnx, onnxruntime as ort

STATIC = os.environ.get('ONNX_OPS_STATIC') == '1'
MOVE = {'Reshape', 'Transpose', 'Concat', 'Slice', 'Split', 'Unsqueeze', 'Squeeze', 'Flatten',
        'Shape', 'Gather', 'Constant', 'Identity', 'Expand', 'Cast', 'ConstantOfShape', 'Range'}


SMALL = 4096   # tensors up to this many elements whose inputs are all known get computed exactly


def attrs(n):
    return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}


def bcast(*shapes):
    return tuple(np.broadcast_shapes(*shapes))


def conv_out(n, x, w, a):
    k = a['kernel_shape'] if 'kernel_shape' in a else w[2:]
    st, dl = a.get('strides', [1] * len(k)), a.get('dilations', [1] * len(k))
    pd = a.get('pads', [0] * 2 * len(k))
    sp = []
    for i, ki in enumerate(k):
        span = (ki - 1) * dl[i] + 1
        num = x[2 + i] + pd[i] + pd[i + len(k)] - span
        sp.append((-(-num // st[i]) if a.get('ceil_mode') else num // st[i]) + 1)
    return (x[0], w[0] if n.op_type == 'Conv' else x[1], *sp)


def static_shapes(m, feeds):
    """Every tensor's shape by propagating the inputs' shapes through the graph (no run, no
    intermediates): small integer tensors (shape arithmetic) are computed exactly with numpy,
    everything else only by shape. Covers the op types of the TOPIQ trunk and head graphs."""
    shp, val = {}, {}
    for i in m.graph.initializer:
        v = onnx.numpy_helper.to_array(i)
        shp[i.name] = v.shape
        if v.size <= SMALL:
            val[i.name] = v
    for k, f in feeds.items():
        shp[k] = tuple(f.shape)
    ew = {'Add': np.add, 'Sub': np.subtract, 'Mul': np.multiply, 'Div': None, 'Pow': np.power,
          'Erf': None, 'Sqrt': np.sqrt, 'Sigmoid': None, 'Relu': None, 'LeakyRelu': None, 'Where': None,
          'Greater': np.greater, 'Clip': None, 'Cast': None}
    for n in m.graph.node:
        a, op = attrs(n), n.op_type
        ins = [t for t in n.input]
        known = all((t in val) for t in ins if t)
        o = n.output[0]
        if op == 'Constant':
            v = onnx.numpy_helper.to_array(a['value'])
            shp[o], val[o] = v.shape, v
            continue
        if op == 'Shape':
            v = np.array(shp[ins[0]], np.int64)
            shp[o], val[o] = v.shape, v
            continue
        if known and op in ('Add', 'Sub', 'Mul', 'Div', 'Gather', 'Concat', 'Unsqueeze', 'Slice', 'Cast', 'Reshape'):
            x = [val[t] for t in ins if t]
            if op == 'Div':
                v = x[0] // x[1] if x[0].dtype.kind == 'i' else x[0] / x[1]
            elif op in ('Add', 'Sub', 'Mul'):
                v = ew[op](x[0], x[1])
            elif op == 'Gather':
                v = np.take(x[0], x[1], axis=a.get('axis', 0))
            elif op == 'Concat':
                v = np.concatenate(x, axis=a['axis'])
            elif op == 'Unsqueeze':
                v = np.expand_dims(x[0], tuple(int(i) for i in (x[1] if len(x) > 1 else a['axes'])))
            elif op == 'Cast':
                v = x[0].astype(onnx.helper.tensor_dtype_to_np_dtype(a['to']))
            elif op == 'Reshape':
                v = x[0].reshape(reshape_to(x[0].shape, x[1]))
            else:
                v = x[0][slices(x[0].shape, *x[1:])]
            v = np.asarray(v)
            shp[o] = v.shape
            if v.size <= SMALL:
                val[o] = v
            continue
        s0 = shp[ins[0]]
        if op in ew:
            shp[o] = bcast(*[shp[t] for t in ins if t])
        elif op in ('Softmax',):
            shp[o] = s0
        elif op == 'LayerNormalization':
            shp[o] = s0
            for extra in n.output[1:]:
                if extra:
                    ax = a.get('axis', -1) % len(s0)
                    shp[extra] = s0[:ax] + (1,) * (len(s0) - ax)
        elif op in ('Conv', 'MaxPool'):
            shp[o] = conv_out(n, s0, shp[ins[1]] if op == 'Conv' else None, a)
        elif op == 'MatMul':
            x, y = shp[ins[0]], shp[ins[1]]
            shp[o] = bcast(x[:-2], y[:-2]) + (x[-2], y[-1])
        elif op == 'Gemm':
            x, y = shp[ins[0]], shp[ins[1]]
            mm = x[1] if a.get('transA') else x[0]
            nn = y[0] if a.get('transB') else y[1]
            shp[o] = (mm, nn)
        elif op == 'Reshape':
            shp[o] = reshape_to(s0, val[ins[1]])
        elif op == 'Transpose':
            perm = a.get('perm', list(range(len(s0)))[::-1])
            shp[o] = tuple(s0[i] for i in perm)
        elif op == 'Slice':
            x = [val[t] for t in ins[1:] if t]
            shp[o] = np.empty(s0, np.int8)[slices(s0, *x)].shape if np.prod(s0) <= 1 << 26 else \
                slice_shape(s0, *x)
        elif op == 'Concat':
            ax = a['axis'] % len(s0)
            ss = [shp[t] for t in ins]
            shp[o] = s0[:ax] + (sum(t[ax] for t in ss),) + s0[ax + 1:]
        elif op == 'Gather':
            ax = a.get('axis', 0) % len(s0)
            shp[o] = s0[:ax] + tuple(shp[ins[1]]) + s0[ax + 1:]
        elif op == 'Unsqueeze':
            ax = sorted(int(i) for i in val[ins[1]])
            out = list(s0)
            for i in ax:
                out.insert(i if i >= 0 else len(out) + 1 + i, 1)
            shp[o] = tuple(out)
        elif op == 'ReduceMean':
            axes = a.get('axes') if 'axes' in a else (list(val[ins[1]]) if len(ins) > 1 else range(len(s0)))
            axes = {int(i) % len(s0) for i in axes}
            shp[o] = tuple((1 if a.get('keepdims', 1) else None) if i in axes else d for i, d in enumerate(s0))
            shp[o] = tuple(d for d in shp[o] if d is not None)
        else:
            raise SystemExit(f'static_shapes: no rule for {op}')
    return shp


def reshape_to(s, target):
    t = [int(v) for v in target]
    t = [s[i] if v == 0 else v for i, v in enumerate(t)]
    if -1 in t:
        known = int(np.prod([v for v in t if v != -1]))
        t[t.index(-1)] = int(np.prod(s)) // known
    return tuple(t)


def slices(s, starts, ends, axes=None, steps=None):
    sl = [slice(None)] * len(s)
    axes = range(len(starts)) if axes is None else axes
    steps = [1] * len(starts) if steps is None else steps
    for st, en, ax, sp in zip(starts, ends, axes, steps):
        sl[int(ax)] = slice(int(st), int(min(en, np.iinfo(np.int64).max)), int(sp))
    return tuple(sl)


def slice_shape(s, *x):
    return tuple(len(range(*sl.indices(d))) for sl, d in zip(slices(s, *x), s))


class Shaped:
    """Stands in for an array where only `.shape` is read."""
    def __init__(self, shape):
        self.shape = shape


def run_all(path, feeds):
    m = onnx.load(path)
    if STATIC:
        init = {i.name for i in m.graph.initializer}
        vals = {k: Shaped(v) for k, v in static_shapes(m, feeds).items() if k not in init}
        return count(m, vals)
    known = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in known:
                m.graph.output.append(onnx.ValueInfoProto(name=o))
                known.add(o)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
    names = [o.name for o in s.get_outputs()]
    vals = dict(zip(names, s.run(names, feeds)))
    vals.update(feeds)
    return count(m, vals)


def count(m, vals):
    init = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    shape = lambda t: (vals[t].shape if t in vals else init[t].shape)
    macs, by_op, moves = 0, {}, {}
    for n in m.graph.node:
        out = int(np.prod(shape(n.output[0]))) if n.output[0] in vals else 0
        if n.op_type == 'Conv':
            w = shape(n.input[1])
            k = int(np.prod(w[1:]))
            macs += out * k
            by_op['Conv_mac'] = by_op.get('Conv_mac', 0) + out * k
            if len(n.input) > 2 and n.input[2]:
                by_op['Add'] = by_op.get('Add', 0) + out
        elif n.op_type in ('MatMul', 'Gemm'):
            a = shape(n.input[0])
            trans_a = any(at.name == 'transA' and at.i for at in n.attribute)
            k = a[-2] if trans_a else a[-1]
            macs += out * k
            by_op[n.op_type + '_mac'] = by_op.get(n.op_type + '_mac', 0) + out * k
            if n.op_type == 'Gemm' and len(n.input) > 2:
                by_op['Add'] = by_op.get('Add', 0) + out
        elif n.op_type in MOVE:
            moves[n.op_type] = moves.get(n.op_type, 0) + out
        else:
            by_op[n.op_type] = by_op.get(n.op_type, 0) + out
    return {'macs': macs, 'by_op': by_op, 'movement_elems': moves}


def pool_matrix(n_in, n_out):
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        s, e = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)
        m[i, s:e] = np.float32(1.0) / np.float32(e - s)
    return m


def measure(d, w, h, models):
    """Counts for `models` at input W x H (the JSON main() prints)."""
    rng = np.random.default_rng(0)
    res = {'w': w, 'h': h}
    for col in models:
        tp = os.path.join(d, f'{col}_trunk.onnx')
        if STATIC:
            x = Shaped((1, 3, h, w))
            trunk = run_all(tp, {'x': x})
            m = onnx.load(tp)
            shapes = static_shapes(m, {'x': x})
            td = tr = [Shaped(shapes[o.name]) for o in m.graph.output]
        else:
            x = rng.random((1, 3, h, w), dtype=np.float32)
            trunk = run_all(tp, {'x': x})
            s = ort.InferenceSession(tp, providers=['CPUExecutionProvider'])
            td = s.run(None, {'x': x})
            tr = s.run(None, {'x': rng.random((1, 3, h, w), dtype=np.float32)})
        th, tw = td[4].shape[2:]
        ph, pw, pos = [], [], []
        for t in td:
            lh, lw = t.shape[2:]
            pooled = lh > th and lw > tw
            oh, ow = (th, tw) if pooled else (lh, lw)
            if STATIC:
                ph.append(Shaped((oh, lh)))
                pw.append(Shaped((lw, ow)))
                pos.append(Shaped((oh * ow, 1, 256)))
                continue
            ph.append(pool_matrix(lh, oh))
            pw.append(pool_matrix(lw, ow).T.copy())
            pos.append(rng.random((oh * ow, 1, 256), dtype=np.float32))
        names = [f'd{i}' for i in range(5)] + [f'r{i}' for i in range(5)] + [f'ph{i}' for i in range(5)] + \
            [f'pw{i}' for i in range(5)] + [f'pos{i}' for i in range(5)]
        head = run_all(os.path.join(d, f'{col}_head.onnx'), dict(zip(names, td + tr + ph + pw + pos)))
        res[col] = {'trunk_per_image': trunk, 'head': head,
                    'taps': [list(t.shape) for t in td]}
    return res


if __name__ == '__main__':
    print(json.dumps(measure(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]), indent=1))
