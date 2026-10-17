#!/usr/bin/env bash
# Same models, same inputs (1024x1024 corpus pair), same thread budget, forced engine per pass:
# rten (CPU) is the committed i265_nn_1024 run; this adds ort-cuda and ort-webgpu (RTX 4080).
cd $HOME/work/zen/zenmetrics--dash
export LOCK_SNAP_DIR=$HOME/tmp/iqa-cx-lead/snap
SITE=$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages
export ORT_DYLIB_PATH=$SITE/onnxruntime/capi/libonnxruntime.so.1.31.0
export LD_LIBRARY_PATH=$(printf '%s:' $SITE/nvidia/*/lib)${LD_LIBRARY_PATH:-}
export ZENMETRICS_NN_WEBGPU_EP_LIB=$HOME/tmp/webgpu-ep/wheel/onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.so
export ZENMETRICS_NN_MODEL_DIR=$HOME/tmp/iqa-cx-lead/nn-models
export ZENMETRICS_NN_ACCEPT_WEIGHTS=ckdn,afine,dreamsim,liqe,hyperiqa,paq2piq,pieapp
export ORT_DISABLE_TELEMETRY=1
echo "start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
export ZENMETRICS_NN_WEBGPU_DEVICE=1
M="afine,arniqa,ckdn,deepdc,dists,dmm,dreamsim,hyperiqa,liqe,lpips-alex,lpips-squeeze,lpips-vgg,maniqa,musiq,paq2piq,pieapp,stlpips-alex,stlpips-vgg,wadiqam-fr-live,wadiqam-fr-tid,wadiqam-nr-live,wadiqam-nr-tid,wd"
ZENMETRICS_NN_MODEL_DIR=$HOME/tmp/nn-phase0/models ZENMETRICS_NN_ENGINE=ort-cuda python3 scripts/complexity/perf_deltas.py --bin ~/tmp/iqa-cx-lead/bin/complexity-reps-ort --out ~/tmp/iqa-cx-lead/perf/i265_nn_ort-cuda-dreamsim_1024_adaptive.tsv --shapes 1024x1024 --metrics dreamsim --repeats 3 --extra 0 --target-s 3 --cpu 0-7 --threads 8 --timeout-s 900
ZENMETRICS_NN_MODEL_DIR=$HOME/tmp/nn-phase0/models ZENMETRICS_NN_ENGINE=ort-webgpu python3 scripts/complexity/perf_deltas.py --bin ~/tmp/iqa-cx-lead/bin/complexity-reps-ort --out ~/tmp/iqa-cx-lead/perf/i265_nn_ort-webgpu-dreamsim_1024_adaptive.tsv --shapes 1024x1024 --metrics dreamsim --repeats 3 --extra 0 --target-s 3 --cpu 0-7 --threads 8 --timeout-s 900
for e in ort-cuda ort-webgpu; do
  ZENMETRICS_NN_ENGINE=$e python3 scripts/complexity/perf_deltas.py --bin ~/tmp/iqa-cx-lead/bin/complexity-reps-ort --out ~/tmp/iqa-cx-lead/perf/i265_nn_${e}_1024_adaptive.tsv --shapes 1024x1024 --metrics "${M/dreamsim,/}" --repeats 3 --extra 0 --target-s 3 --cpu 0-7 --threads 8 --timeout-s 900
done
echo "end_utc=$(date -u +%FT%TZ)"
