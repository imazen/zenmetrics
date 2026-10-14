# Interleaved CPU timing: torch eager vs onnxruntime vs tract (python bindings), same ONNX graphs.
import sys, time, statistics, numpy as np, torch, onnxruntime as ort, tract, onnx
from onnx import numpy_helper
models_dir, threads, rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(threads)
names = ["vgg16lpips_256", "vgg16lpips_512", "resnet50_224", "resnet50_384"]
sys.path.insert(0, "/home/lilith/tmp/nnrt-research/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("exp", "/home/lilith/tmp/nnrt-research/scripts/export_mods.py"); exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
print(f"# torch {torch.__version__} ort {ort.__version__} tract {tract.version()} threads={threads} rounds={rounds}")
for name in names:
    path = f"{models_dir}/{name}.onnx"
    hw = int(name.split("_")[1])
    x = np.random.default_rng(0).random((1, 3, hw, hw), dtype=np.float32)
    # torch: rebuild module and load weights from the ONNX initializers is awkward; instead run torch via onnx2torch-free path:
    tmod = exp.build(name); tmod.eval()
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0"); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    tm = tract.onnx().load(path)
    tm.set_input_fact(0, f"{hw if False else 1},3,{hw},{hw},f32")
    tr = tm.into_model().into_runnable()
    def run_torch():
        with torch.inference_mode(): return tmod(torch.from_numpy(x))
    def run_ort(): return sess.run(None, {iname: x})
    def run_tract(): return tr.run([x])
    fns = {"torch": run_torch, "ort": run_ort}
    if threads == 1: fns["tract"] = run_tract
    for f in fns.values(): f(); f()  # warm
    t = {k: [] for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            s = time.perf_counter(); f(); t[k].append((time.perf_counter() - s) * 1e3)
    line = "\t".join(f"{k} min={min(v):.1f} med={statistics.median(v):.1f}ms" for k, v in t.items())
    print(f"{name}\t{line}")
