from pathlib import Path
# GPU timing: torch eager (fp32 default cudnn [TF32 off], fp32+TF32, fp16 autocast, channels_last fp16) and ORT CUDA EP,
# timer = forward + device-side sum per output + scalar .item() (matches burnbench). fresh random input per iter.
import sys, time, statistics, importlib.util, numpy as np, torch
spec = importlib.util.spec_from_file_location("exp", Path(__file__).resolve().with_name("export_mods.py")); exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
import onnxruntime as ort
try: ort.preload_dlls()
except Exception as e: print("preload_dlls:", e)
models_dir, iters = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
print(f"# torch {torch.__version__} cudnn {torch.backends.cudnn.version()} ort {ort.__version__} gpu {torch.cuda.get_device_name()} iters={iters}")
torch.backends.cudnn.benchmark = True
def timeit(fn, hw):
    ts = []
    for i in range(iters + 5):
        x = torch.rand(1, 3, hw, hw, device=dev); torch.cuda.synchronize()
        s = time.perf_counter(); fn(x); 
        if i >= 5: ts.append((time.perf_counter() - s) * 1e3)
    return min(ts), statistics.median(ts)
for name in ["vgg16lpips_256", "vgg16lpips_512", "resnet50_224", "resnet50_384"]:
    hw = int(name.split("_")[1])
    m = exp.build(name).to(dev).eval()
    def red(out):
        outs = out if isinstance(out, tuple) else (out,)
        return [o.float().sum().item() for o in outs]
    res = {}
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        res["torch_fp32"] = timeit(lambda x: red(m(x)), hw)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        res["torch_tf32"] = timeit(lambda x: red(m(x)), hw)
        mh = exp.build(name).to(dev).eval().half().to(memory_format=torch.channels_last)
        res["torch_fp16_cl"] = timeit(lambda x: red(mh(x.half().contiguous(memory_format=torch.channels_last))), hw)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    so = ort.SessionOptions()
    sess = ort.InferenceSession(f"{models_dir}/{name}.onnx", so, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"})])
    iname = sess.get_inputs()[0].name
    def run_ort(x):
        io = sess.io_binding()
        io.bind_input(iname, "cuda", 0, np.float32, list(x.shape), x.data_ptr())
        for o in sess.get_outputs(): io.bind_output(o.name, "cuda")
        sess.run_with_iobinding(io)
        return [float(v.numpy().sum()) for v in io.get_outputs()]  # NOTE: full output readback (ORT has no device-side sum here)
    res["ort_cuda_fp32(full readback)"] = timeit(run_ort, hw)
    print(name + "\t" + "\t".join(f"{k} min={a:.2f} med={b:.2f}ms" for k, (a, b) in res.items()))
