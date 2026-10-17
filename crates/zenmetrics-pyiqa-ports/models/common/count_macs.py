"""Conv / MatMul / Gemm multiply-accumulates of an ONNX graph at fixed input shapes.

usage: count_macs.py MODEL.onnx NAME=d0,d1,... [NAME=...]

Fixes each named input's shape, runs onnx shape inference, and counts per node: Conv
out_elems * (C_in / group) * prod(kernel); MatMul prod(out) * K (K = last dim of the first
input); Gemm M * N * K. Prints the total and the per-op-type split. Element-wise ops are not
counted. Reads graph structure only; never runs the model.
"""
import sys
from collections import defaultdict
import onnx
from onnx import shape_inference

m = onnx.load(sys.argv[1], load_external_data=False)
fixed = {a.split('=')[0]: [int(v) for v in a.split('=')[1].split(',')] for a in sys.argv[2:]}
for i in m.graph.input:
    if i.name in fixed:
        dims = i.type.tensor_type.shape.dim
        for d, v in zip(dims, fixed[i.name]):
            d.ClearField('dim_param'); d.dim_value = v
m = shape_inference.infer_shapes(m, data_prop=True)
shape = {}
for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
    dd = v.type.tensor_type.shape.dim
    shape[v.name] = [d.dim_value if d.HasField('dim_value') else None for d in dd]
for t in m.graph.initializer:
    shape[t.name] = list(t.dims)
tot = defaultdict(int)
bad = []
for n in m.graph.node:
    if n.op_type not in ('Conv', 'MatMul', 'Gemm'):
        continue
    out = shape.get(n.output[0]); a = shape.get(n.input[0]); b = shape.get(n.input[1])
    if not out or None in out or not a or None in a or not b or None in b:
        bad.append(n.name); continue
    po = 1
    for v in out: po *= v
    if n.op_type == 'Conv':
        g = next((x.i for x in n.attribute if x.name == 'group'), 1)
        k = 1
        for v in b[2:]: k *= v
        tot['Conv'] += po * (a[1] // g) * k
    elif n.op_type == 'MatMul':
        tot['MatMul'] += po * a[-1]
    else:
        ta = next((x.i for x in n.attribute if x.name == 'transA'), 0)
        tot['Gemm'] += po * (a[0] if ta else a[-1])
print('total_macs', sum(tot.values()), dict(tot), 'unresolved', len(bad))
