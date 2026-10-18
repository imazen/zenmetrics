# Frozen Mac WebGPU capability audit — 2026-10-18

All four session constructions reproduce the strict CPU-assignment refusal.
They are capability refusals, not stalled workers. No inference, timing
qualification, fallback enablement, export change or production service change
was performed. The original queue continues independently.

The verbose provider distinguishes **no matching registered kernel for this
node** from **supported node deliberately preferred on CPU**. Both occur here.
A registry miss does not prove that every dtype/opset of an operator is
unsupported. In particular, CKDN's reported arithmetic follows Shape → Gather
→ Add → Div → Mul; these observations do not establish missing floating-point
Add/Div/Mul kernels. Tensor dtype/opset attribution remains unproven where the
frozen graph lacks intermediate type metadata.

| Graph | Unique registry-miss node/op observations | Registry-miss operator counts | Unique explicitly CPU-preferred nodes |
| --- | ---: | --- | ---: |
| A-FINE | 972 | Cast 24; Concat 224; ConstantOfShape 2; Div 24; Equal 29; Gather 229; Mul 48; Slice 98; Squeeze 24; Unsqueeze 270 | 72 |
| ARNIQA | 8 | ConcatFromSequence 2; Equal 2; Gather 1; SequenceAt 1; SequenceConstruct 1; Size 1 | 0 |
| CKDN | 10 | Add 2; Div 2; Gather 2; Mul 4 | 0 |
| DeepDC | 150 | Cast 20; Concat 40; ConstantOfShape 20; EyeLike 10; Gather 30; Mul 10; Unsqueeze 20 | 40 |

These are deduplicated observations across capability passes, including
optimizer-generated names. They are **not final CPU placement counts**: session
construction rejects before printing a complete final placement inventory.
They must not be summed into benchmark coverage or interpreted as a minimal
set of graph edits that would make a session load.

ARNIQA's exact names are `/ConcatFromSequence`, `/ConcatFromSequence_1`,
`/Equal`, `/Equal_1`, `/Gather`, `/SequenceAt`, `/SequenceConstruct`, `/Size`.
CKDN's are `/m/Gather`, `/m/Gather_1`, `/m/Add`, `/m/Add_1`, `/m/Div`,
`/m/Div_1`, `/m/Mul_1`, `/m/Mul_2`, `/m/Mul_3`, `/m/Mul_5`.
DeepDC includes `/m/EyeLike` and `/m/EyeLike_1` through `/m/EyeLike_9`.
The complete A-FINE/DeepDC lists and all CPU-preferred names are retained in
the hashed classification artifact below.

The provider's `ep.cc:91 GetCapabilityImpl` reports registry misses. Its
`get_capability_utils.h:204 GetCpuPreferredNodes` separately says a node is
capable of EP execution but CPU execution is deemed faster than the overhead.
A-FINE examples include `/Div_1` and `/Sqrt_36`; DeepDC includes `/m/Exp`,
`/m/Log`, `/m/Reciprocal` and their numbered variants. Therefore A-FINE and
DeepDC cannot accurately be described solely as unsupported-operator failures.
ARNIQA and CKDN emit no such CPU-preference message in this diagnostic; absence
of that message is not proof that every unreported node is GPU-assigned.

## Reproduction and provenance

The probe uses the ARM64 Python 3.12 ONNX Runtime 1.31.0 official wheel,
ONNX 1.23.2, and the **exact frozen** WebGPU plugin 0.4.0. The wheel's core dylib
hash equals the frozen core. Python uses its native binding, so this is a
matching-wheel session diagnostic, not a claim that the original Rust worker
was instrumented. Runtime build reports commit `402979a`.

- Frozen core SHA256: `ff52828ce7c6dd1aea99a5e9f5da9e2786e3d03e467fbf8b6893a4b624334920`.
- Frozen plugin SHA256: `894aa43c308c97afe58f6485b8411962181a39b27170f5f210cc7503adb52850`.
- A-FINE graph SHA256: `e9272d9c07c8b0f1b00ef7494cc217f476e65650ad72c5f30c127070c20d91f1`.
- ARNIQA graph SHA256: `9a518779087e94913b8da796bcd871752b2aafcf41e76b7508e610d333e03e29`.
- CKDN graph SHA256: `d6d6b6e8c41b365ea26811613317ef1b5b31b090a03773cf00e4af955193b875`.
- DeepDC graph SHA256: `3a537e819c4a8895d68b777259fb4cab82dc37ac8315bdd5873264f4bf781603`.

All four model hashes match before and after inspection. Both
`session.disable_cpu_ep_fallback=1` and Python `enable_fallback=False` remain
explicit. Single intra/inter-op thread, deterministic compute, Metal GPU device,
verbose node logging; no inference is executed. Each child holds the shared
Mac heavy lock through the coordinator and uses the 12 GiB monitored RSS guard,
8-job policy. No compiler was invoked. Resource receipts: A-FINE 2s/1.38 GiB
peak RSS; ARNIQA 1s/0.29 GiB; CKDN 1s/0.21 GiB; DeepDC 1s/0.25 GiB.
These are diagnostic resource observations, not metric performance results.

Frozen input: `~/benchmark-bundles/iqa-mac-20261018-coverage-v2/512-nn-metal` on mac.
Raw logs, graph inventories, original refusal configs and wheel setup receipt:
`~/tmp/iqa-mac-webgpu-audit.noindex` on mac; coordinator copy
`/home/lilith/benchmark-artifacts/iqa-mac-webgpu-audit-20261018` on i265.
`classification-v1.json` SHA256:
`a92b3d109f275a989b25aac8b6f7982437a2637ae5c15b9ec42bf5ee299433e2`.
It binds every source log/receipt hash and lists exact node names separately
for the two classifications. Generated raw data stays outside git.

From the existing matching-wheel environment, a fresh output directory:

```sh
just iqa-webgpu-node-audit "$HOME/tmp/iqa-mac-webgpu-audit.noindex/venv/bin/python" "$HOME/benchmark-bundles/iqa-mac-20261018-coverage-v2/512-nn-metal" "$HOME/tmp/iqa-mac-webgpu-audit.noindex/nodes-fresh"
just iqa-webgpu-node-classify "$HOME/tmp/iqa-mac-webgpu-audit.noindex/nodes-fresh" "$HOME/tmp/iqa-mac-webgpu-audit.noindex/classification-fresh.json"
```

The first command waits on the shared Mac lock. Both commands refuse to replace
existing output. Parser controls verify duplicate observations collapse, the
CPU-preference class stays separate, and a generic refusal invents no nodes.

Provider/API references: [WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html),
[plugin EP registration](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html),
[Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).
The classifications above come from the retained native logs, rather than an
operator support table or an inference from the generic refusal alone.
