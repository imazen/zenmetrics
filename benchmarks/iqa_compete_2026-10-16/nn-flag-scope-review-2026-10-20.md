# Final review — 3/3 detailed routes complete

Requested by Codex root coordinator, 2026-10-20. No tracked source changed; no model/runtime imports, inference, compiler, tests, CI or other-agent prompts.

Missing/unresolved: canonical runtime contamination and actual library scheduling/utilization remain unverified. The cached-rlib first-probe receipt binds helper/rlib bytes but is not exhaustive transitive-source rebuild proof. No accepted-regression claim is made. No concrete measurement-invalidating clock-scope defect found in these three selected rows. No bounds or retained observations changed.

Reviewed selected 512-photo rows: deepdc-first-pair-cid22-t1, lpips-vgg-first-pair-cid22-t1 and paq2piq-original-cid22-t8. Other rows in their plans are outside this review. Both first plans have SHA256 030d3458e55722c6b687471494f0c543ca4c7b8cccd9347a43c6f5c9358975f0; PaQ confirmation plan SHA256 f943334a59b65e735b6fef626411bb0575caac74890c1d68ac38b51e8f730552. Existing bounds remain DeepDC 6.415300000000001e-05, VGG 0.00019392499999999998, PaQ 0.001229401. The first interval receipts each retain20 blocks; no pending-stage completion is claimed.

## First-evaluation common native boundary

Paths abbreviated here: ROOT=/home/lilith/tmp/local-canonical-20261019T2330; FIRST=ROOT/resident-nn-first-build-v2; SNAP=ROOT/build/work/zenmetrics; H=/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16. A source hash appendix below identifies the inspected bytes.

FIRST/resident_nn_first.rs:27–64 configures Rayon, reads/checks RGB8 and calls benchmark_prepare before cold_call. FIRST/original_nn.rs:249–254 loads the cached model without evaluating an image. Its :199–227 times the cache lock, RGB8 views, ordinary model.evaluate, output collection and headline selection. FIRST/timing.rs:8–14 starts before closure construction/evaluation and stops after scoring returns. The closure contains references; resident model destruction is outside timing. Scoring-local containers release before evaluation returns. Probe :66–71 performs a second exact-bit-check evaluation after the timed call; it does not prime the first evaluation. H/compare_first_evaluation.py:23–30 requests one canonical sample and consumes the COLD record, excluding its subsequent SAMPLE from this comparison. The helper SHA is bound by both interval helper_sha256 fields, rather than a plan binding-map entry.

The first build receipt records source efae6b8fed9e57346157d92dffc3cdb57c44560e and binary aba1c690560092c8c28414db5236a2199bd3ee1827ec7f64633a6bb9abd58565, with probe/timer/stock NN module/cached-rlib identities. This is first explicit evaluation after model construction, not startup, OS-cache-cold or warm throughput. Constructors and imports may initialize runtime state even without scoring inputs. No binary/model asset rehash was performed here.

## DeepDC first

Native SNAP/crates/zenmetrics-nn/src/deepdc.rs:61–80 loads metadata and RTEN trunk during preparation; :141–155 converts both RGB8 images to planar f32 [0,1], runs both trunks and computes the host distance-correlation head inside scoring. :175–190 uses ordinary score_pair, not its separate warm_reference API. No selected adapter resize. ImageNet normalization belongs to the exported trunk convention.

Canonical H/canonical_original_model_cell.py:20,32–44 sets Torch budgets, constructs DeepDC/VGG19, enables the exact bound ImageNet input convention and reads raw files before timing. :37–42 executes only the AST-selected with_input_norm function from models/deepdc/export.py. It does not execute that export module's top-level model/export/golden code. :47–52 copies both arrays, creates contiguous CHW tensors, casts/divides by255, calls the model, extracts the scalar and deletes input tensors inside the clock. Model destruction remains outside.

Bound DeepDC_PyTorch/DeepDC.py:23–44,69–72 constructs/loads the feature model without an explicit input forward. :46–59,74–102 applies ImageNet normalization and both feature forwards/correlation during scoring. Its separate prepare_image resizing function is not called. The presence of a torchsummary source binding does not show a dummy forward on the selected constructor path.

## LPIPS-VGG first

Native SNAP/crates/zenmetrics-nn/src/lpips.rs:105–136 loads trunk and linear weights before timing. :201–230 performs both RGB8-to-planar [-1,1] conversions, trunk forwards, feature normalization and learned head inside scoring; :338–350 calls ordinary pair evaluation, not cached-reference evaluation. No adapter resize.

Canonical H/canonical_original_model_cell.py:21–23 constructs LPIPS v0.1/VGG and loads weights before timing. :47–52 includes both RGB copies/layout conversions, float/255 then *2−1, model call, scalar extraction and input release. Bound lpips/lpips.py:23–110 constructs scaling/VGG/linear layers and loads weights without an explicit input forward; :112–144 executes input scaling, both towers, feature normalization and pooling. Bound lpips/pretrained_networks.py:97–134 builds and applies five VGG16 slices. No resize or reference-feature precomputation on the selected route.

## PaQ2PiQ warm

ROOT/author-plans-1100/paq2piq.worker executes H/canonical_worker_env.py with paq2piq.settings.json: native RTEN, frozen model directory, named PaQ consent, ROOT/nn-target/release/iqa-compete. Wrapper :18–20 sets RTEN_NUM_THREADS from runner RAYON_NUM_THREADS. H/compare_canonical.py:64–82 supplies environment budgets, native warm configuration and canonical placeholders.

SNAP/benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs:98–149,345–379 builds the ordinary score closure and performs full untimed warmup. :476–507 configures fixed warm rounds and times score calls through b.iter. SNAP/crates/zenmetrics-cli/src/metrics/nn.rs:145–150,199–227 loads PairedNoReference<Paq2Piq> and calls ordinary evaluate, collecting both image outputs while selecting the distorted headline. SNAP/crates/zenmetrics-abstractions/src/no_reference.rs:77–87,109–122 scores reference then distorted on every call. Its distinct warm_reference path :125–159 is not selected.

Native paq2piq.rs:79–102 loads body/head graphs before warm measurements; :124–132,204–227 converts RGB8 to planar f32 [0,1], runs the body, builds401 ROIs, pools and runs the head per image inside evaluation. :248–260 dispatches that image scorer. No selected input resize or ImageNet normalization.

Canonical H/canonical_author_network_cell.py:62–65,150–155 configures Torch, reads raw inputs, defines PIL/ToTensor conversion, constructs the original model and loads state before timing. :156–158,165–174 builds the20×20 grid plus global ROI, converts and scores distorted then reference, and extracts both scalars inside each clock. COLD primes the model; warm comparisons consume SAMPLE. Bound author paq2piq/model.py:41–61 constructs body/head/ROI pool without explicit forward; :63–91 applies the body and all401 ROI/head outputs and constructs the grid. Both image evaluations are timed. Native/canonical image order differs, but neither selected path caches the reference score. Canonical model.rois survives until replacement next call; temporary conversion/forward outputs release on return, while model destruction remains outside both clocks.

## Thread-control and evidence limits

First comparator sets RTEN/RAYON/OMP/MKL/OPENBLAS budgets explicitly. PaQ outer runner sets RAYON/OMP/MKL/OPENBLAS and the wrapper propagates RTEN. Both canonical helpers set Torch intra-op=requested and inter-op=1 before model construction. Native engine.rs:126–133 reads RTEN workers. Paired trunks :189–219,315–334 split explicit pools across two forwards for budgets>=2 and run sequentially at one; the selected first rows request one. PaQ RTEN image :382–409 invokes Model.run(...,None), relying on RTEN default/global pool behavior; this review does not independently establish that dependency's effective scheduling. Host preprocessing/head worker caps are work-sized (preprocess.rs:9–54; par.rs:13–18). These settings are not actual thread-count, affinity, utilization or clean-runtime proof.

Native engine is RTEN from exact retained command/settings; canonical is original CPU PyTorch in inference mode. No GPU inference claim. Six inspected frozen NN sources (deepdc/lpips/paq2piq/engine/preprocess/par) match present primary bytes, but that is not a transitive source-to-rlib proof. Canonical helper/export/package hashes below match actual plan bindings. Source-level model constructor inspection cannot independently qualify all imported dependency startup behavior. Root owns later execution and disposition; no human waiver implied.


## Independently inspected small-source/configuration SHA256

/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16/canonical_original_model_cell.py
SHA256 8b38fb9db4e31db0f5ca104bfff96bf6094aabcfd3915029297d47035506e033

/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16/canonical_author_network_cell.py
SHA256 c6ebc05c2703d9cbfa45b80ebed23350ab992feb63e7f9a00b2e213a50f7cc30

/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16/canonical_worker_env.py
SHA256 2b4eb29c2ef3fe134eb4d8accb6387685ad94e42a3dc379933059ff50802969d

/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16/compare_canonical.py
SHA256 f7606eafb0534206b7e6ef0c4bd4d5a6206994ef65404124f658c7e2be89f31c

/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16/compare_first_evaluation.py
SHA256 cff48a43b79eb71e369d28671e368bfda7bf13765b14fac86a8636557802487c

/home/lilith/tmp/local-canonical-20261019T2330/resident-nn-first-build-v2/resident_nn_first.rs
SHA256 89464c53b5e86a32f49c982df4e4b40a2888c6327a50899ea226831856f525e3

/home/lilith/tmp/local-canonical-20261019T2330/resident-nn-first-build-v2/original_nn.rs
SHA256 a720b93ffa06c2699602bee8fbfaabfc1b0a33acfff4f082acc5b1081b1313c3

/home/lilith/tmp/local-canonical-20261019T2330/resident-nn-first-build-v2/timing.rs
SHA256 c3c99d189e537ebeded6e6d1ec54af6f341e9bfacac05626733c1f44ffc0040b

/home/lilith/tmp/local-canonical-20261019T2330/resident-nn-first-build-v2/build-receipt.json
SHA256 48a5aa8f464eb3282fca701af7e023573afcfaaea0e111a9458250249273a40b

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs
SHA256 2948d4cd37f8ca11d6dcac113513a858e3c30b8c0b01912539a96b78877ba585

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-cli/src/metrics/nn.rs
SHA256 9cbc96ae64ec00858a924048641d72cc7c039e0a98a5814cf614ffc127a01131

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-abstractions/src/no_reference.rs
SHA256 b552aab9f9de1865cc59c30b8da37230665b5fc615a7f9a45faf6a15e13b166c

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-nn/src/deepdc.rs
SHA256 b8dd78574f890a8f7617c9787e24c7675dcb90d290050eb867830a4caaada7fc

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-nn/src/lpips.rs
SHA256 ceba0fcfa4aab41ace64607686dd60c740aa4750c5346ab05e3210b84a495754

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-nn/src/paq2piq.rs
SHA256 2e62a55d60aaff170707a4edf83c51ab9dcc3ce6b8839f43d7065cd4309b43e6

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-nn/src/engine.rs
SHA256 a117334e4bb6895fe67bf040a14396eb4d9080a8cafd0d35c8bf333ba9206f8a

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-nn/src/preprocess.rs
SHA256 f481da797dd7d35f362cbc13b886b291065f6a155d96561eb9830cc0cd9373e7

/home/lilith/tmp/local-canonical-20261019T2330/build/work/zenmetrics/crates/zenmetrics-nn/src/par.rs
SHA256 638f96b0b58ede182fbcdd254cba8c269c86c82cfdbe63d61c7bc5956f8ce5eb

/home/lilith/work/zen/zenmetrics/crates/zenmetrics-nn/models/deepdc/export.py
SHA256 8ba3811c46224503553af0480767a01472b4dd7031feb28d53b807e2b10ee92b

/home/lilith/tmp/local-canonical-20261019T2330/original-nn-env/lib/python3.12/site-packages/lpips/pretrained_networks.py
SHA256 6a27f714c51796db466e86bebba6a617c1bfc4d566f3a1756497629c1248686e

/home/lilith/tmp/local-canonical-20261019T2330/original-nn-env/lib/python3.12/site-packages/lpips/lpips.py
SHA256 780d09b907cb9b661e0ae28b2d163ddfba92f9e870d7feba34d4790cc6590658

/home/lilith/tmp/local-canonical-20261019T2330/original-nn-env/lib/python3.12/site-packages/DeepDC_PyTorch/DeepDC.py
SHA256 806e82fa65a18a483c4ff7cba55336a5cc7c657b5cffcfff9d9c5f1652eed4d4

/home/lilith/tmp/local-canonical-20261019T2330/original-extra-sources/paq2piq/paq2piq/model.py
SHA256 7b9e46755223c3e7117fd400d3e9d2c8131a00baa6ddde1232faf6dbc079ecf2

/home/lilith/tmp/local-canonical-20261019T2330/author-plans-1100/paq2piq.worker
SHA256 8bf505f042fc3cf3d55638a9ddb121408e923c88d6977461a1e8b6de8a4583ab

/home/lilith/tmp/local-canonical-20261019T2330/author-plans-1100/paq2piq.settings.json
SHA256 2610df9b7e7a5d3dcfa80e12b77a48d745d6aedc8cc64bfce25bd38274880019

/home/lilith/tmp/local-canonical-20261019T2330/author-networks-config.json
SHA256 6cd26a22326da72685e4e3e09b2674d906de53574382187461011cd01d4c4c49

