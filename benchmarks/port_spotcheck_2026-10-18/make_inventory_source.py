#!/usr/bin/env python3
"""make_inventory_source.py — generate/verify inventory-source.* for this dir.

Source-only census: every CLI metric name vs its claimed canonical
implementation, role, version/rev, upstream URL, build/runner binding,
harness route (unmerged lane evidence), documented parity gate, and
availability. No executables, builds, imports, or metric calls.

Inputs (same owned dir):
  inventory-base-unmerged-3494d8af.tsv — UNMERGED lane snapshot of
    inventory.tsv @3494d8af (sha256 0382d8a9…); evidence, not a master
    artifact. Family/oracle-runner columns are inherited evidence.

Usage:
  make_inventory_source.py <dir>            emit shards + index + meta
  make_inventory_source.py <dir> --verify   re-emit, byte-compare, check
                                            caps/hashes/exact-once coverage
"""
import csv, hashlib, importlib.util, json, os, pathlib, sys

OUT = pathlib.Path(sys.argv[1])
BASE_TSV = OUT / 'inventory-base-unmerged-3494d8af.tsv'
BASE_SHA = '0382d8a90f273a59a21295b5c9dbc0c52379b0d9936ab4faf7367bec65421878'

# shared reviewed helper (FULL sibling path — not a basename collision):
# benchmarks/iqa_compete_2026-10-16/fixed_factor_oracle.py @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1
# sha256 7b19441363c66bcbd4dda9cf8c4959e7afd2b1aa8e13924d8afa01cfbc4fb339
_FF = (pathlib.Path(__file__).resolve().parent.parent
       / 'iqa_compete_2026-10-16' / 'fixed_factor_oracle.py')
_spec = importlib.util.spec_from_file_location('fixed_factor_oracle', _FF)
_ffo = importlib.util.module_from_spec(_spec)
sys.modules['fixed_factor_oracle'] = _ffo
_spec.loader.exec_module(_ffo)
PatchError, write_artifact, sha256_bytes = _ffo.PatchError, _ffo.write_artifact, _ffo.sha256_bytes

if sha256_bytes(BASE_TSV.read_bytes()) != BASE_SHA:
    raise PatchError(
        f'base table sha256 mismatch: {BASE_TSV} — expected {BASE_SHA}; '
        'refusing to emit claims over unverified evidence')

# 126 exact MetricKind names @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1, registry order
# mechanical extraction of #[value(name=…)] inside `pub enum MetricKind`
# (mod.rs:97–1362 ONLY; LumaIngress/ZensimFeatureRegime value names are
# not metrics). Set-verified EQUAL to retained list-metrics output
# ~/tmp/portcheck/cli-metrics.txt (sha256 e729cec8…, 126 names).
NAMES = '''vmaf vmaf-neg vmaf-4k cambi msssim ssim ssimc vsi fsim gmsd
ms-gmsd ms-gmsdc mdsi mad psnr psnr-y psnr-y601 psnr-y-studio601 psnr-cb
psnr-cr psnr-ycbcr611 psnrhvs psnrhvs-y psnrhvs-daala nlpd brisque niqe
piqe haarpsi iwssim flip flip-v17 pieapp pieapp-pyiqa lpips-alex lpips-vgg
lpips-squeeze dists stlpips-alex stlpips-vgg deepdc dreamsim afine
wadiqam-fr-live wadiqam-fr-tid wadiqam-nr-live wadiqam-nr-tid wd ws-hv
ciede2000 cvvdp cvvdp-gpu nlpd-iqa iwssim-piq iwssim-gpu hdrvdp hdrvdp3
hdrvdp-aic2026 hdrvdp3-aic2026 vmaf-v1 vmaf-studio601 vmaf-neg-studio601
cambi-aic2026 ssim-libvmaf msssim-libvmaf psnr-y-libvmaf butteraugli
butteraugli-gpu butteraugli-original ssim2 ssim2-gpu ssimulacra1 dssim
dssim-gpu mdctpsnr vif vifvec haarpsi-y fsim-y fsim-k2 fsim-k3 fsim-k4
fsim-nodownscale vsi-k2 vsi-k3 vsi-k4 vsi-nodownscale cw-ssim cw-ssim-pyiqa
dmm dmm-pyiqa ahiq ahiq-pyiqa arniqa maniqa liqe musiq hyperiqa paq2piq
ckdn topiq-fr topiq-fr-pipal wadiqam-pyiqa ssim-pyiqa ssim-pyiqa-f32
ssimc-pyiqa-f32 ms-ssim-pyiqa ms-ssim-pyiqa-f32 ssim-pyiqa-f32-k2
ssim-pyiqa-f32-k3 ssim-pyiqa-f32-k4 ssim-pyiqa-f32-k5 ssim-pyiqa-f32-k6
ssimc-pyiqa-f32-k2 ssimc-pyiqa-f32-k3 ssimc-pyiqa-f32-k4 ssimc-pyiqa-f32-k5
ssimc-pyiqa-f32-k6 ms-ssim-pyiqa-f32-k2 ms-ssim-pyiqa-f32-k3
ms-ssim-pyiqa-f32-k4 ms-ssim-pyiqa-f32-k5 ms-ssim-pyiqa-f32-k6 zensim
zensim-gpu proposal-butteraugli'''.split()

base = {}
with BASE_TSV.open() as f:
    for r in csv.DictReader(f, delimiter='\t'):
        base[r['metric']] = r

GH = 'https://github.com/'
SOURCE_REV = '27350d07f2c3692a9d0665a81a3d0beffaa8c0e1'
DOC = f'docs/METRIC_PROVENANCE.md @{SOURCE_REV}'
UNM = 'unmerged canonical_run.py/oracle_drivers.py @3494d8af (lane snapshot)'
FF_REV = ('benchmarks/iqa_compete_2026-10-16/fixed_factor_oracle.py @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1 '
          'sha256:7b19441363c66bcbd4dda9cf8c4959e7afd2b1aa8e13924d8afa01cfbc4fb339 '
          '(reviewed SOURCE patcher; accepts k∈1–4; runs NO metrics)')
FF_UNM = ('benchmarks/port_spotcheck_2026-10-18/fixed_factor_oracle.py '
          '(UNMERGED lane monkeypatch tooling — basename collision; not the same file)')
STAGED_REV = ('UNKNOWN — staged binary sha ≠ source revision '
              '(resolver: staging manifest in unmerged lane)')
ROUTE_ONLY = 'route-present (unmerged @3494d8af); runtime UNQUALIFIED'
NO_ROUTE = 'no harness route located (source-only census)'
NNRD = f'crates/zenmetrics-nn/README.md @{SOURCE_REV}'
NNGATE = f'1e-3 JND per column ({NNRD}:27; tests/parity.rs)'

def route(key, bound=None):
    b = f'; configured bound {bound} — harness default, NOT a documented gate' if bound else ''
    return f'route "{key}" ({UNM}{b})'

def lv(impl, bound):
    return dict(
        impl=f'libvmaf {impl}',
        rev=('vendored libvmaf (resolver: vmaf-head-sys/Cargo.lock pin); '
             'doc reference: libvmaf 3.2.1 FFI — claimed-vs-inspected binding UNVERIFIED'),
        url=GH + 'Netflix/vmaf',
        bind='vendored C lib + libvmaf-compute shim (build_libvmaf.sh, unmerged lane recipe)',
        rt=route('cmd:' + impl.split(' ')[0], bound),
        gate='UNKNOWN (no documented gate located)',
        avail=ROUTE_ONLY)

E = {}
def put(n, role, impl, rev, url, bind, rt, gate, avail, family=None):
    E[n] = dict(role=role, impl=impl, rev=rev, url=url, bind=bind, rt=rt,
                gate=gate, avail=avail, family=family)

# --- libvmaf family (claimed canonical = libvmaf itself) ---
for n, impl, bound in (('vmaf', 'vmaf_v0.6.1 model', '.02'),
                       ('vmaf-neg', 'vmaf_v0.6.1neg model', '.02'),
                       ('vmaf-4k', 'vmaf_4k_v0.6.1 model', '.02'),
                       ('vmaf-v1', 'vmaf_v1.0.16_3d0h model (legacy v1 scorer)', '.02'),
                       ('vmaf-studio601', 'vmaf_v0.6.1, studio-601 ingress', '.02'),
                       ('vmaf-neg-studio601', 'vmaf_v0.6.1neg, studio-601 ingress', '.02')):
    d = lv(impl, bound)
    if n in ('vmaf', 'vmaf-neg'):
        d['gate'] = f"{DOC} 'vmaf' row: feats 1e-4, score 0.02 (doc reproduction col covers neg)"
    put(n, 'primary', **d)
put('cambi', 'primary', **lv('cambi', '1e-4'))
d = lv('cambi (aic2026 ingress parity-check config — different luma derivation)', '1e-4')
put('cambi-aic2026', 'diagnostic (ingress-variant config)', **d)
# note: lv() gate default is UNKNOWN — cambi carries no doc gate row; no inference
for n, impl in (('ssim-libvmaf', 'float_ssim @f85a8536'),
                ('msssim-libvmaf', 'float_ms_ssim @f85a8536'),
                ('psnr-y-libvmaf', 'float_psnr')):
    d = lv(impl, '2e-4' if 'ssim' in n else '1e-10')
    d['gate'] = f'{DOC}: ≤2e-4 vs vendored f85a8536' if 'ssim' in n else 'UNKNOWN (no documented gate located)'
    put(n, 'primary', **d)
d = lv('psnr_hvs (Daala/Xiph integer bin-DCT variant — distinct algorithm from Ponomarenko PSNR-HVS-M)', '2e-4')
d['gate'] = f'{DOC}: ≤2e-4 dB (420+444 cases)'
put('psnrhvs-daala', 'diagnostic (distinct Daala variant algorithm)', **d)

# --- libjxl family ---
JXL_G = 'UNKNOWN (no numeric gate in doc; doc :40 names libjxl butteraugli v0.9.2/0.9.4 as canonical)'
for n, twin in (('butteraugli', ''), ('butteraugli-gpu', ' (CPU twin scores GPU impl)')):
    put(n, 'primary', 'libjxl butteraugli' + twin,
        'harness shim libjxl 0.12.0; doc canonical v0.9.2/0.9.4',
        GH + 'libjxl/libjxl', 'vendored libjxl + iqa-jxl shim (unmerged lane recipe)',
        route('butteraugli-gpu' if twin else 'butteraugli', '0 rel .001'), JXL_G, ROUTE_ONLY)
for n, twin in (('ssim2', ''), ('ssim2-gpu', ' (CPU twin scores GPU impl)')):
    put(n, 'diagnostic alternative (standalone Cloudinary CMS lineage not yet established)',
        'libjxl ComputeSSIMULACRA2' + twin +
        ' — C++ scope exclusions recorded (BufferToImageBundle, ThreadPoolInternal, metadata, ImageBundle lifetime); scope verdict NOT fully matched (python-jxl audit)',
        'libjxl 0.12.0 (route impl); doc canonical: Cloudinary ssimulacra2 C++ v2 lineage',
        GH + 'libjxl/libjxl', 'iqa-jxl shim (unmerged lane recipe)',
        route('ssim2-gpu' if twin else 'ssim2', '0 rel 0'),
        'UNKNOWN (doc "bit-parity SIMD tiers" is intra-crate parity, not a cross-backend numeric gate)', ROUTE_ONLY)

# --- other C/C++ ---
put('ssimulacra1', 'primary', 'cloudinary/ssimulacra C++', STAGED_REV,
    GH + 'cloudinary/ssimulacra', 'ssimulacra-canonical C++ shim (unmerged staging)',
    route('ssimulacra1', '0'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
for n, twin in (('dssim', ''), ('dssim-gpu', ' (CPU twin scores GPU impl)')):
    put(n, 'primary', 'dssim-core (kornelski)' + twin,
        'UNKNOWN — crate dep `dssim-core ^3.4` (resolver: Cargo.lock resolved version)',
        'https://crates.io/crates/dssim-core (per METRIC_PROVENANCE:38 "crates.io dssim-core ^3.4"; upstream repo binding per crates.io metadata — not independently verified → origin UNKNOWN)',
        'dssim-canonical C-API shim (unmerged staging)',
        route('dssim-gpu' if twin else 'dssim', '0'),
        f'{DOC} evidence summary: "upstream" — not an acceptance threshold; numeric threshold UNKNOWN', ROUTE_ONLY)
put('butteraugli-original', 'primary', 'google/butteraugli', '@71b18b6',
    GH + 'google/butteraugli', 'butteraugli-orig C++ shim (unmerged staging)',
    route('butteraugli-original', '0'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('mdctpsnr', 'primary',
    "author's official C++ thorfdbg/mDCTpsnr — doc parity target is the COMPILED binary's op order (GCC -O3 -ffast-math AVX2 + libmvec), not source semantics",
    STAGED_REV, GH + 'thorfdbg/mDCTpsnr',
    'dctpsnr -CC{t} process-per-sample (unmerged staging; full process per timed sample)',
    route('py=dctpsnr', '2e-4'),
    f'{DOC} mdctpsnr row: ≤2e-4 dB x86_64+GNU goldens, 1e-2 off-platform', ROUTE_ONLY)

# --- octave/metrix ---
IQT = GH + 'sattarab/image-quality-tools'
put('ssim', 'primary', 'ssim_index.m (Wang) via metrix_mux', 'image-quality-tools @163fff12',
    IQT, 'octave 6.4 + image pkg; rgb2gray prep (unmerged octave overlay recipe)',
    route('octave:ssim', '5e-5'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('msssim', 'primary', 'mssim_index.m metrix_mux — open note: verify equals author ssim_mscale_new.m semantics',
    'image-quality-tools @163fff12', IQT, 'octave 6.4 + image pkg (unmerged recipe)',
    route('octave:msssim', '1e-4'), 'crates/msssim/src/tests.rs:145-163 assert |s−golden| < 1e-4 (absolute scalar, stored goldens); doc evidence "15 Octave goldens"', ROUTE_ONLY)
put('vif', 'primary', 'vifp_mscale.m (Sheikh/Bovik); libvmaf float_vif variant note',
    'image-quality-tools @163fff12 / libvmaf 3.2.0', IQT,
    'two unmerged routes: octave:vifp + libvmaf-compute vif',
    route('vif / vif-octave', '5e-13'), 'crates/vif/src/tests.rs:207 assert |s−golden| < 1e-9 (absolute scalar, stored f64 golden); doc evidence "goldens f64"', ROUTE_ONLY)
put('vifvec', 'primary', 'vifvec.m + matlabPyrTools sp5Filters/buildSpyr chain — different algorithm from vifp',
    'image-quality-tools @163fff12 + matlabPyrTools @f02ab059',
    IQT + ' + ' + GH + 'LabForComputationalVision/matlabPyrTools',
    'octave:vifvec (unmerged recipe)', route('octave:vifvec', '1e-9'),
    f'{DOC}: 10 goldens <1e-9', ROUTE_ONLY)
put('haarpsi', 'primary', "authors' HaarPSI.m (vendored in crate)", 'vendored (crate @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1)',
    GH + 'rgcda/haarpsi', 'octave:haarpsi (unmerged recipe)',
    route('octave:haarpsi', '5e-5'), f'{DOC}: 5e-5', ROUTE_ONLY)
put('haarpsi-y', 'diagnostic (luma route vs RGB oracle — indicative only)', 'HaarPSI.m',
    'vendored (crate @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1)', GH + 'rgcda/haarpsi', 'octave:haarpsi (unmerged recipe)',
    route('haarpsi-y', '5e-5'), f'{DOC}: 5e-5', ROUTE_ONLY)
put('psnrhvs', 'primary', "authors' psnrhvsm.m (Ponomarenko PSNR-HVS-M)",
    'file sha256 4a373c54… (site live; earlier 404 was wrong-path probe)',
    'https://www.ponomarenko.info/psnrhvsm.m',
    'octave route NOT wired; needs dct2 (octave-signal 1.4.1 or crate shim)',
    'no route in unmerged harness @3494d8af', f'{DOC}: 1e-3 dB', NO_ROUTE)
put('psnrhvs-y', 'diagnostic (luma variant of psnrhvs)', "authors' psnrhvsm.m",
    'file sha256 4a373c54…', 'https://www.ponomarenko.info/psnrhvsm.m',
    'same unexecuted route status as psnrhvs', 'no route in unmerged harness',
    f'{DOC}: 1e-3 dB', NO_ROUTE)

# --- fetched author source; harness currently uses a substitute impl ---
SUB = 'substitute (impl stands in for fetched author source — author source available, unexecuted)'
put('fsim', SUB, "authors' FR_FSIMc.m (doc canonical); mirrored FeatureSIM.m v1.0 returns [FSIM,FSIMc], embeds phasecong2/lowpassfilter",
    'file sha256 4b23dc28… (mirror manifest)', 'https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m',
    "author .m needs fspecial('average') + conv2 + rgb2gray (crate fspecial.m shim covers avg); route uses piq fsim instead",
    route('piq:fsim', '5e-5'), f'{DOC}: 5e-5', ROUTE_ONLY)
put('fsim-y', 'diagnostic (luma route vs RGB oracle — indicative)',
    "authors' FR_FSIMc.m", 'file sha256 4b23dc28…',
    'https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m',
    'same substitute basis as fsim (piq fsim luma route)',
    route('fsim-y', '5e-5'), f'{DOC}: 5e-5 (fsim/-y row)', ROUTE_ONLY)
for n, k in (('fsim-k2', 2), ('fsim-k3', 3), ('fsim-k4', 4), ('fsim-nodownscale', 1)):
    put(n, 'owner-variant (owner kernel_size semantics on substitute base)',
        f"fixed-factor variant of FSIM (kernel_size={k}; owner semantics — no author release)",
        'basis file sha256 4b23dc28…',
        'https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m',
        f'{FF_REV} + {FF_UNM}', route(f'ffactor:fsim:{k}', '5e-5'),
        f'{DOC}: base gate 5e-5 inherited', ROUTE_ONLY)
put('vsi', SUB, "authors' VSI.m (embeds SDSP)", 'file sha256 5939b905…',
    'https://cslinzhang.github.io/home/VSI/Files/VSI.m',
    'author .m needs fspecial + imresize bilinear (crate imresize_check.m verified Octave semantics); route uses piq vsi instead',
    route('piq:vsi', '1e-4'), f'{DOC}: 1e-4', ROUTE_ONLY)
for n, k in (('vsi-k2', 2), ('vsi-k3', 3), ('vsi-k4', 4), ('vsi-nodownscale', 1)):
    put(n, 'owner-variant (owner kernel_size semantics on substitute base)',
        f'fixed-factor variant of VSI (kernel_size={k})', 'basis file sha256 5939b905…',
        'https://cslinzhang.github.io/home/VSI/Files/VSI.m',
        f'{FF_REV} + {FF_UNM}', route(f'ffactor:vsi:{k}', '1e-4'),
        f'{DOC}: base gate 1e-4 inherited', ROUTE_ONLY)
put('gmsd', 'substitute (pyiqa gmsd stands in for crate-declared reference)',
    'crate-declared reference: libgmsd C @de646c9a (MIT); author GMSD.m (T=170) sha256 52cdecbf… mirrored',
    'libgmsd zip sha256 0da2b55b…', GH + 'clunietp/libgmsd + http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.m',
    'C source present — no completed runtime control on this host; route uses pyiqa gmsd',
    route('pyiqa:gmsd', '1e-14'), f'{DOC}: 1e-12 rel (map) / 1e-14 mean; crate test crates/gmsd/src/tests.rs:114 |mean−ref|<1e-14 abs', ROUTE_ONLY)
for n in ('ms-gmsd', 'ms-gmsdc'):
    put(n, 'substitute (claimed canonical UNKNOWN)',
        'piq multi_scale_gmsd; no author multi-scale release located',
        'UNKNOWN (resolver: upstream search not exhausted — not proven absent)',
        'https://github.com/photosynthesis-team/piq', 'pyiqa-env torch route (unmerged lane)',
        route('piq:multi_scale_gmsd', '0'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('mdsi', SUB, "authors' MDSI.m (Nafchi 2016, FEX 59809, BSD license.txt; C1=140 matches crate MDSI_CHOICES.md)",
    'zip sha256 8b78597d…', 'https://www.mathworks.com/matlabcentral/fileexchange/59809',
    'RGB input; toolbox-level calls only — octave route not wired; route uses piq mdsi; crate is clean-room from paper',
    route('piq:mdsi', '0'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('mad', SUB,
    "official STMAD_2011 hi_index.m+lo_index.m @229f2478 (+ical MEX); still-fusion derived JEI-2010 Eq.12/13 — NO author still wrapper in mirrored release (doc: official MATLAB release, larschandler.com)",
    'files sha256 e9a78615…/0aa62d07…; ical_std orig segfaults x86 Octave 6.4 (:159 tail-called mxDestroyArray) → reviewed compat copy sha256 79fb753e role compatibility-reference; ical_stat orig sha256 f6d6e305…',
    GH + 'Netflix/vmaf/tree/229f2478ab53ae0fd6b07328802294ba15fd0cdd/matlab/STMAD_2011_MatlabCode',
    'helper-arm decision open (compat vs imazen shim); imfilter (image pkg) needed; octave route proposed, unlanded; current route uses pyiqa mad',
    route('pyiqa:mad', '0'), 'crates/mad-iqa/src/tests.rs:206-208 assert d_hi,d_lo,d_mad < 1e-5 relative (denom e.max(1.0)), author goldens; doc evidence "13 goldens"', ROUTE_ONLY)
put('cw-ssim', SUB, "authors' cwssim_index.m (Sampat/Wang/Bovik, FEX 43017)",
    'zip sha256 eb2a6f93…', 'https://www.mathworks.com/matlabcentral/fileexchange/43017',
    "needs matlabPyrTools buildSCFpyr chain (mirrored complete) + fspecial('gaussian') — image pkg required; route uses pyiqa cw_ssim",
    route('pyiqa:cw_ssim', '0'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('cw-ssim-pyiqa', 'primary', 'pyiqa cw_ssim (claimed impl per unmerged inventory)',
    'pyiqa 0.1.16 (venv pin, unmerged lane)', GH + 'chaofengc/IQA-PyTorch',
    'pyiqa-env (unmerged lane recipe)', route('pyiqa:cw_ssim', '0'),
    'UNKNOWN (no documented gate located)', ROUTE_ONLY)

# --- python oracles (claimed impl = named package/release) ---
put('flip', 'primary', 'NVlabs FLIP (flip_evaluator pkg)', 'flip-evaluator 1.7 (uv.lock)',
    GH + 'NVlabs/flip', 'pyiqa-env torch (unmerged lane recipe)',
    route('flip-eval', '0'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('flip-v17', 'diagnostic (v1.7 pooling config; f64-vs-f32 accumulation note)',
    'flip_evaluator v1.7 pooling', '1.7 (uv.lock)', GH + 'NVlabs/flip',
    'pyiqa-env torch (unmerged lane)', route('flip-eval', '0'),
    'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('hdrvdp3-aic2026', 'primary',
    'jpeg-ai-qaf VDP3 numpy port — CONFIGURED reference for this route; full author-MATLAB comparison UNVERIFIED (coordinator owns current HDR work); doc also lists official HDR-VDP 3.0.7 MATLAB as canonical',
    'jpeg-ai-qaf @0628a6b', GH + 'jpeg-ai/jpeg-ai-qaf',
    'PORTCHECK_PYTHON env (unmerged lane recipe)', route('hdrvdp3-aic2026', '1e-3'),
    f'{DOC} evidence summary: "21-case golden corpus" — not an acceptance threshold; numeric threshold UNKNOWN', ROUTE_ONLY)
put('hdrvdp-aic2026', 'diagnostic (VDP2 port; MATLAB-2.2.2 fidelity UNVERIFIED)',
    'jpeg-ai-qaf VDP2 numpy port', '@0628a6b', GH + 'jpeg-ai/jpeg-ai-qaf',
    'PORTCHECK_PYTHON env (unmerged lane)', route('hdrvdp-aic2026', '1e-3'),
    'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('hdrvdp', 'primary', 'official HDR-VDP 2.2.2 MATLAB release (ISC)',
    'UNKNOWN — official release not mirrored (resolver: upstream release mirror)',
    'https://sourceforge.net/projects/hdrvdp/',
    'MATLAB/Octave + absolute-nits inputs — no runnable route on this harness convention',
    'no route in unmerged harness', f'{DOC} hdrvdp row: P_det 1e-4; JOD ~1e-2 cases', NO_ROUTE)
put('hdrvdp3', 'primary', 'official HDR-VDP 3.0.7 MATLAB release',
    'UNKNOWN — official release not mirrored (resolver: upstream release mirror)',
    'https://sourceforge.net/projects/hdrvdp/',
    'MATLAB/Octave + abs-nits inputs; oracle exists only as -aic2026 config row',
    'no direct route (config twin only)', f'{DOC} evidence summary: "21-case golden corpus" — not an acceptance threshold; numeric threshold UNKNOWN', NO_ROUTE)
put('iwssim', 'primary',
    "Python-IW-SSIM — authors'-page-linked Python port; second-order oracle (authors' MATLAB release not yet compared)",
    '@f9de37c', GH + 'Jack-guo-xy/Python-IW-SSIM',
    'torch env, u8-rounded BT.601 gray (unmerged lane recipe)', route('iwssim', '1e-4'),
    f'{DOC}: ~1e-4', ROUTE_ONLY)
put('iwssim-piq', 'primary', 'jpeg-ai-qaf IW_SSIM_PyTorch (torch port of Jack-guo ref) on unrounded 0–255 Y',
    '@0628a6b', GH + 'jpeg-ai/jpeg-ai-qaf', 'piq information_weighted_ssim route (unmerged lane)',
    route('iwssim-piq', '1e-4'), 'crates/iwssim README:25-26: identical ≤1e-5, distorted ≤5e-3 (shared code path test bounds)', ROUTE_ONLY)
put('iwssim-gpu', 'primary (CPU twin scores GPU impl)', 'Python-IW-SSIM', '@f9de37c',
    GH + 'Jack-guo-xy/Python-IW-SSIM', 'same env as iwssim', route('iwssim-gpu', '1e-4'),
    f'{DOC}: ~1e-4', ROUTE_ONLY)
put('nlpd', 'primary', "Valerolaparra/NLPD_Pytorch (authors' PyTorch; PyTorch 2.5.1 oracle per doc)",
    'staged upstream (unmerged lane)', GH + 'Valerolaparra/NLPD_Pytorch',
    'torch env (unmerged lane)', route('nlpd', '1e-5'), f'{DOC}: 1e-5 golden (97×99 pair)', ROUTE_ONLY)
put('nlpd-iqa', 'primary', 'IQA_pytorch NLPD(channels=1) — BT.709 pad-2/mean-RMS config',
    'dingkeyan93 staged (unmerged lane)', GH + 'dingkeyan93/IQA-optimization',
    'torch env (unmerged lane)', route('nlpd-iqa', '1e-4'), f'{DOC}: 1e-4 golden', ROUTE_ONLY)
put('wd', 'primary', 'Cool-Chic wasserstein.py WassersteinLoss VGG16',
    'main; file sha256 ed9de16b… (per METRIC_PROVENANCE)',
    GH + 'Orange-OpenSource/Cool-Chic/blob/main/coolchic/training/metrics/wasserstein.py',
    'torch env (unmerged lane)', route('wd', '0'),
    f'1e-3 JND per column (wd_s1…s5) (crates/zenmetrics-nn/README.md @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1:27; tests/parity.rs)', ROUTE_ONLY)
put('ws-hv', 'primary', 'WS_HV.ipynb upstream notebook (cv2+pywt)',
    'UNKNOWN (resolver: SOURCES/WS_HV origin)',
    'UNKNOWN (upstream notebook URL — resolver: SOURCES/WS_HV origin)',
    'torch env (unmerged lane)', route('ws-hv', '0'),
    'UNKNOWN (no documented gate located)', ROUTE_ONLY)
put('ciede2000', 'definition', 'scikit-image rgb2lab + deltaE_ciede2000 — definition-level reference',
    'skimage 0.26 (uv.lock)', 'https://scikit-image.org/', 'py env (unmerged lane)',
    route('ciede2000', '1e-3'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
for n, twin in (('cvvdp', ''), ('cvvdp-gpu', ' (CPU twin scores GPU impl)')):
    put(n, 'primary', 'pycvvdp standard_4k' + twin, '0.5.7 (uv.lock)',
        GH + 'gfxdisp/pycvvdp', 'py env (unmerged lane)',
        route('cvvdp-gpu' if twin else 'cvvdp', '1e-3'), f'{DOC}: ≤1e-3 JOD', ROUTE_ONLY)
put('brisque', 'primary (declared PIQ lineage per unmerged inventory; canonical MATLAB BRISQUE+LIBSVM unavailable)',
    'piq brisque (PIQ lineage port of MATLAB BRISQUE+LIBSVM)', 'piq 0.8.0 (uv.lock)',
    'https://github.com/photosynthesis-team/piq', 'pyiqa-env (unmerged lane)',
    route('piq:brisque', '15.138428e-3'),
    'JND/1000 = 0.015138428 absolute scalar (crates/brisque README:68, tests/parity.rs:3, src/lib.rs:673)', ROUTE_ONLY)
put('niqe', 'primary',
    'BasicSR niqe (crate parity oracle, parity.rs:3 "matches_basicsr_goldens"); unmerged inventory claimed pyiqa — binding unverified',
    'BasicSR goldens tsv (crate tests)', 'UNKNOWN (resolver: BasicSR repo pin)',
    'pyiqa-env (unmerged lane)', route('pyiqa:niqe', '0'),
    'JND/1000 = 6.02329e-4 absolute scalar (crates/niqe tests/parity.rs:3, src/lib.rs:94 JND=0.602329)', ROUTE_ONLY)
put('piqe', 'primary',
    'pypiqe 1.2 — Python conversion of MATLAB piqe (crate README:7-9); prior pyiqa claim superseded',
    '1.2', GH + 'michael-rutherford/pypiqe',
    'pyiqa-env (unmerged lane)', route('pyiqa:piqe', '0'),
    'JND/1000 = 0.012486448 absolute scalar (crates/piqe tests/parity.rs:1-5, src/lib.rs:104 JND=12.486448)', ROUTE_ONLY)
put('ssimc', 'primary', 'pyiqa ssim on RGB (chroma-mean claim; claimed impl = pyiqa per unmerged inventory)',
    'pyiqa 0.1.16 (venv pin, unmerged lane)', GH + 'chaofengc/IQA-PyTorch',
    'pyiqa-env (unmerged lane)', route('pyiqa:ssim', '5e-5'),
    'UNKNOWN (no documented gate located)', ROUTE_ONLY)
for n in ('dmm', 'dmm-pyiqa'):
    put(n, 'substitute (pyiqa route stands in for documented upstream)' if n == 'dmm' else 'primary',
        ('Baoliang93/DMM 0.1.2 — L2-pooled VGG16 relu3_3/relu4_3 + per-patch SVD head '
         f'({NNRD}:17)' + (' — pyiqa is the unmerged harness substitute' if n == 'dmm'
         else ' — declared impl for this *-pyiqa name is pyiqa; documented upstream recorded separately')),
        'Baoliang93/DMM 0.1.2 (nn README)', GH + 'Baoliang93/DMM',
        'pyiqa-env (unmerged lane)', route('pyiqa:dmm', '0'),
        f'{NNGATE} (JND=0.191420); gated vs netlib-sgesdd goldens, not MKL (nn README:17)', ROUTE_ONLY)

# --- pyiqa-declared names (claimed impl = pyiqa) ---
NN_UP = {  # documented upstream + JND from nn README rows :9-:27
 'ahiq': ('IIGROUP/AHIQ epoch-32 ckpt (:16)', 'UNKNOWN-JND'),
 'arniqa': ('miccunifi/ARNIQA single_image_inference.py (:18)', 'UNKNOWN-JND'),
 'maniqa': ('IIGROUP/MANIQA koniq10k eval (:19)', 'UNKNOWN-JND'),
 'liqe': ('zwx8981/LIQE LIQE.forward 495-prompt (:20)', '0.3938'),
 'musiq': ('google-research/musiq run_predict_image.py (:21)', 'UNKNOWN-JND'),
 'hyperiqa': ('SSL92/hyperIQA demo.py (:22)', 'UNKNOWN-JND'),
 'paq2piq': ('baidut/paq2piq InferenceModel.predict (:23)', '1.229'),
 'ckdn': ('researchmm/CKDN epoch-14 (:13)', 'UNKNOWN-JND'),
 'wadiqam-pyiqa': ('dmaniry/deepIQA evaluate.py CPU fp32 (:25)', 'UNKNOWN-JND'),
}
for n, (up, jnd) in NN_UP.items():
    repo = up.split(' ')[0]
    gate = f'{NNGATE}' + (f' (JND={jnd})' if jnd != 'UNKNOWN-JND' else '')
    put(n, 'substitute (pyiqa route stands in for documented upstream)',
        f'{up} — claimed canonical per {NNRD}; pyiqa route is the unmerged harness substitute',
        'see upstream (nn README row)', GH + repo,
        'pyiqa-env (unmerged lane)', route(f'pyiqa:{n.replace("-", "_")}', '0'),
        gate, ROUTE_ONLY)
for n in ('ahiq-pyiqa', 'topiq-fr', 'topiq-fr-pipal',
          'ssim-pyiqa', 'ssim-pyiqa-f32', 'ssimc-pyiqa-f32',
          'ms-ssim-pyiqa', 'ms-ssim-pyiqa-f32'):
    up = ' — documented upstream: IIGROUP/AHIQ epoch-32 (nn README:16)' if n == 'ahiq-pyiqa' else ''
    put(n, 'primary', f'pyiqa {n.replace("-", "_")} (claimed impl = pyiqa per unmerged inventory){up}',
        'pyiqa 0.1.16 (venv pin, unmerged lane)', GH + 'chaofengc/IQA-PyTorch',
        'pyiqa-env (unmerged lane)', route(f'pyiqa:{n.replace("-", "_")}', '0'),
        'UNKNOWN (no documented gate located)', ROUTE_ONLY)
# owner-variant downsample rows: tooling supports k1–4; k5/k6 recipes invalid
for fam, bound in (('ssim-pyiqa', '7.5e-7'), ('ssimc-pyiqa', '7.5e-7'), ('ms-ssim-pyiqa', '1.9e-6')):
    for k in (2, 3, 4, 5, 6):
        n = f'{fam}-f32-k{k}'
        qualified = k <= 4
        bind = (f'{FF_REV} + {FF_UNM}' if qualified else
                f'UNKNOWN — {FF_REV} accepts k∈1–4 only; k={k} recipe currently INVALID/unqualified')
        put(n, 'owner-variant (declared impl = pyiqa f32 convention + owner kernel_size)',
            f'owner-defined downsample variant (downsample=True + kernel_size={k})',
            'pyiqa 0.1.16 + fixed_factor_oracle tooling', GH + 'chaofengc/IQA-PyTorch',
            bind, route(f'ffactor:{fam}:{k}', bound),
            'UNKNOWN (no documented gate located)', ROUTE_ONLY)

# --- NN package oracles (claimed impl = named upstream package) ---
NN_PKGS = (
    ('lpips-alex', 'lpips alex', '0.1.4 (uv.lock)', GH + 'richzhang/PerceptualSimilarity'),
    ('lpips-vgg', 'lpips vgg', '0.1.4 (uv.lock)', GH + 'richzhang/PerceptualSimilarity'),
    ('lpips-squeeze', 'lpips squeeze', '0.1.4 (uv.lock)', GH + 'richzhang/PerceptualSimilarity'),
    ('dists', 'DISTS_pytorch', 'DISTS-pytorch 0.1 (uv.lock)', GH + 'dingkeyan93/DISTS'),
    ('stlpips-alex', 'stlpips_pytorch alex lpips=False',
     '0.0.2 sdist sha256:29b1d4615e18851571d3872c509e9df3f415a5ba406fd6ac757356c7767fa8a6 (uv.lock pin; origin repo binding UNKNOWN)',
     'https://pypi.org/project/stlpips-pytorch/0.0.2/ (verified pin source; origin repo UNKNOWN)'),
    ('stlpips-vgg', 'stlpips_pytorch vgg',
     '0.0.2 sdist sha256:29b1d4615e18851571d3872c509e9df3f415a5ba406fd6ac757356c7767fa8a6 (uv.lock pin; origin repo binding UNKNOWN)',
     'https://pypi.org/project/stlpips-pytorch/0.0.2/ (verified pin source; origin repo UNKNOWN)'),
    ('deepdc', 'DeepDC_PyTorch (imp-compat shim)',
     '0.2 sdist sha256:162faa3e0cee26cf4f0c7c89211dc2ded6605b7c02187d25a08f4e5ce82426d7 (uv.lock pin; origin repo binding UNKNOWN)',
     'https://pypi.org/project/DeepDC-PyTorch/0.2/ (verified pin source; origin repo UNKNOWN)'),
    ('dreamsim', 'dreamsim pip ensemble (DINO+CLIP+OpenCLIP)', 'pip (version per unmerged lane env)',
     GH + 'ssundaram21/dreamsim'),
    ('afine', 'ChrisDud0257/AFINE (A-FINE FR)', 'repo (unmerged lane staging)', GH + 'ChrisDud0257/AFINE'),
    ('wadiqam-fr-live', 'dmaniry/deepIQA chainer live-weights', '@51dece99 (unmerged lane)', GH + 'dmaniry/deepIQA'),
    ('wadiqam-fr-tid', 'dmaniry/deepIQA tid-weights', '@51dece99 (unmerged lane)', GH + 'dmaniry/deepIQA'),
    ('wadiqam-nr-live', 'dmaniry/deepIQA NR live', '@51dece99 (unmerged lane)', GH + 'dmaniry/deepIQA'),
    ('wadiqam-nr-tid', 'dmaniry/deepIQA NR tid', '@51dece99 (unmerged lane)', GH + 'dmaniry/deepIQA'),
)
NN_ROW = {'lpips-alex': 'lpips 0.1.4 CPU fp32 (:9, JND 0.065)',
          'lpips-vgg': 'lpips 0.1.4 CPU fp32 (:9, JND 0.19)',
          'lpips-squeeze': 'lpips 0.1.4 CPU fp32 (:9)',
          'dists': 'DISTS_pytorch 0.1 CPU fp32 no-resize (:10, JND 0.094)',
          'stlpips-alex': 'stlpips_pytorch 0.0.2 (:11, JND 0.023)',
          'stlpips-vgg': 'stlpips_pytorch 0.0.2 (:11, JND 0.066)',
          'deepdc': 'DeepDC_PyTorch 0.2 ImageNet-norm no-resize (:12, JND 0.064)',
          'dreamsim': 'dreamsim 0.21.2 ensemble LoRA-folded (:15)',
          'afine': 'ChrisDud0257/AFINE afine.pth + OpenAI ViT-B-32 (:14, JND 0.507)',
          'wadiqam-fr-live': 'dmaniry/deepIQA evaluate.py CPU fp32 (:25)',
          'wadiqam-fr-tid': 'dmaniry/deepIQA evaluate.py CPU fp32 (:25)',
          'wadiqam-nr-live': 'dmaniry/deepIQA evaluate.py CPU fp32 (:25)',
          'wadiqam-nr-tid': 'dmaniry/deepIQA evaluate.py CPU fp32 (:25)'}
for n, impl, rev, url in NN_PKGS:
    doc = NN_ROW.get(n)
    put(n, 'primary',
        impl + (f' — documented upstream per {NNRD} row {doc}' if doc else ' (claimed impl per unmerged inventory)'),
        rev, url, 'torch/pkg env (unmerged lane)', route(n, '0'),
        NNGATE if doc else 'UNKNOWN (no documented gate located)', ROUTE_ONLY)

# --- definition-level psnr family ---
put('psnr', 'definition', 'definition: RGB MSE — skimage peak_signal_noise_ratio',
    'skimage 0.26 (uv.lock)', 'https://scikit-image.org/', 'py env (unmerged lane)',
    route('psnr', '1e-10'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)
for n, v in (('psnr-y', 'BT.709 luma (PSNR_VARIANT=y709)'), ('psnr-y601', 'studio-601 Y'),
             ('psnr-y-studio601', 'studio-601 Y'), ('psnr-cb', 'studio-601 Cb 444'),
             ('psnr-cr', 'studio-601 Cr 444'), ('psnr-ycbcr611', 'studio-601 6:1:1 combined')):
    put(n, 'definition', f'definition-level {v} numpy', 'n/a (definition)',
        'n/a (in-tree definition — no upstream)', 'psnr-studio601 shared oracle env variant (unmerged lane)',
        route('psnr-studio601', '1e-3'), 'UNKNOWN (no documented gate located)', ROUTE_ONLY)

# --- in-house / excluded ---
for n in ('zensim', 'zensim-gpu'):
    put(n, 'self', 'in-house ML metric (zensim crate) — not a reproduction target',
        'in-tree @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1', 'in-house (no upstream)', 'in-tree',
        'no external oracle route', 'UNKNOWN (no external oracle; no gate)', 'in-tree self')
put('proposal-butteraugli', 'self', 'in-house AIC-4 submission remap — no external canonical',
    'in-tree @27350d07f2c3692a9d0665a81a3d0beffaa8c0e1', 'in-house (no upstream)', 'in-tree',
    'no external oracle route', 'UNKNOWN (no external oracle; no gate)', 'in-tree self')

EXCLUDED_CLI = ('pieapp', 'pieapp-pyiqa')
EXTRA_OUTPUTS = ('fsimc',)

# Primary-reference language and harness implementation are separate facts.
# Inherited `language` describes the unmerged runner, not the claimed primary.
MATLAB_PRIMARY = {'ssim', 'msssim', 'vif', 'vifvec', 'haarpsi', 'haarpsi-y',
                  'psnrhvs', 'psnrhvs-y', 'fsim', 'fsim-y', 'mdsi', 'mad',
                  'vsi', 'cw-ssim', 'hdrvdp', 'hdrvdp3'}
GATE_SOURCES = (
    'docs/METRIC_PROVENANCE.md', 'crates/mad-iqa/src/tests.rs',
    'crates/vif/src/tests.rs', 'crates/msssim/src/tests.rs',
    'crates/brisque/README.md', 'crates/brisque/tests/parity.rs',
    'crates/brisque/src/lib.rs', 'crates/niqe/tests/parity.rs',
    'crates/niqe/src/lib.rs', 'crates/piqe/README.md',
    'crates/piqe/tests/parity.rs', 'crates/piqe/src/lib.rs',
    'crates/gmsd/src/tests.rs', 'crates/iwssim/README.md',
    'crates/zenmetrics-nn/README.md', 'crates/zenmetrics-nn/tests/parity.rs',
)

def identities(n, e, b):
    """Preserve unbound joins; language never establishes source equivalence."""
    language = b.get('language', 'UNKNOWN')
    primary = language
    if n in MATLAB_PRIMARY or n.startswith(('fsim-k', 'vsi-k')) or n in ('fsim-nodownscale', 'vsi-nodownscale'):
        primary = 'MATLAB'
    if n == 'gmsd':
        primary = 'C (libgmsd); MATLAB (author)'
    if n in ('ms-gmsd', 'ms-gmsdc'):
        primary = 'UNKNOWN (primary unbound)'
    if n in ('dssim', 'dssim-gpu'):
        primary, language = 'Rust', 'Rust + C API shim'
    if n.startswith('wadiqam-') and n != 'wadiqam-pyiqa':
        primary, language = 'Python (Chainer)', 'Python (PyIQA substitute)'
        e['role'] = 'claimed primary: deepIQA; harness: PyIQA substitute'
        e['bind'] = 'PyIQA wadiqam_fr/nr (unmerged); Chainer evaluate.py is the primary, source/weights join UNKNOWN'
    elif n == 'piqe':
        primary, language = 'Python (pypiqe)', 'Python (PyIQA substitute)'
        e['role'] = 'claimed primary: pypiqe 1.2; harness: PyIQA substitute'
    elif n == 'niqe':
        primary, language = 'Python (BasicSR)', 'Python (PyIQA substitute)'
        e['role'] = 'claimed primary: BasicSR; harness: PyIQA substitute'
    elif n == 'iwssim-piq':
        e['role'] = 'claimed primary: jpeg-ai-qaf; harness: PIQ substitute (join UNKNOWN)'
    elif n == 'wadiqam-pyiqa':
        primary, language = 'Python (PyIQA)', 'Python (PyIQA)'
        e['role'] = 'primary (declared PyIQA variant; deepIQA upstream separate)'
        e['impl'] = 'PyIQA wadiqam_fr; documented upstream dmaniry/deepIQA evaluate.py (Chainer) is separate'
    elif n in ('hdrvdp-aic2026', 'hdrvdp3-aic2026'):
        primary = 'Python (configured port); MATLAB (author comparison unverified)'
    elif e['avail'] == NO_ROUTE:
        language = 'n/a (no harness route)'
    elif e['role'] == 'self':
        primary, language = 'Rust (in-house)', 'n/a (no external oracle)'
    if n in ('flip', 'flip-v17'):
        primary = 'C++ (Python bindings)'
    if primary in ('python', 'numpy', 'python(torch)', 'python(cv2,pywt)'):
        primary = 'Python'
    if language in ('python', 'numpy', 'python(torch)', 'python(cv2,pywt)'):
        language = 'Python'
    if primary == 'octave':
        primary = 'MATLAB (Octave runner)'
    if language == 'octave':
        language = 'MATLAB via Octave'
    return primary, language

def pinned_gate(gate):
    # Pin path:line references as well as the shared DOC/NNRD labels.
    for path in GATE_SOURCES:
        gate = gate.replace(path + ':', path + ' @' + SOURCE_REV + ':')
    # Compact legacy path spellings while retaining their exact source join.
    for crate in ('brisque', 'niqe', 'piqe', 'iwssim'):
        gate = gate.replace('crates/' + crate + ' ', 'crates/' + crate + ' @' + SOURCE_REV + ' ')
    return gate

def excl(n, why):
    return [n, '—', 'excluded', why, 'n/a (excluded)', 'n/a (excluded)',
            'n/a (excluded)', 'n/a (excluded)', 'n/a (excluded)', 'n/a (excluded)',
            'n/a (excluded)', 'excluded']
excl_rows = {
    'aimos': excl('aimos', 'excluded: proprietary, unreleased — owner clarification 2026-10-19; never described or released'),
    'pieapp': excl('pieapp', 'excluded: all PieAPP aliases excluded by owner rule (in CLI registry)'),
    'pieapp-pyiqa': excl('pieapp-pyiqa', 'excluded: all PieAPP aliases excluded by owner rule (in CLI registry)'),
}

rows, missing = [], []
for n in NAMES:
    if n in EXCLUDED_CLI:
        rows.append(excl_rows[n]); continue
    b = base.get(n, {})
    e = E.get(n)
    if e is None:
        missing.append(n); continue
    primary_language, harness_language = identities(n, e, b)
    rows.append([n, b.get('group', 'UNKNOWN'), e['role'], e['impl'], primary_language, harness_language, e['rev'],
                 e['url'], e['bind'],
                 e['rt'] + f" [runner: {b.get('oracle_runner', 'UNKNOWN')}]",
                 pinned_gate(e['gate']), e['avail']])
rows.append(excl_rows['aimos'])

R = {
    'raw_value_enum_names_in_file': 133,
    'raw_split': {'MetricKind (mod.rs:97-1362)': 126,
                  'LumaIngress (mod.rs:1363+)': ['house', 'yuv601-studio'],
                  'ZensimFeatureRegime/other': ['v2-ab', 'folded720append',
                      'folded720append2', 'folded720append2carriers',
                      'folded720append2pools']},
    'cli_names': 126,
    'data_rows': 124,
    'excluded_cli_records': list(EXCLUDED_CLI),
    'excluded_non_cli_records': ['aimos'],
    'extra_outputs_not_names': list(EXTRA_OUTPUTS),
    'total_rows': 127, 'missing': missing,
}
sys.stderr.write(
    f"raw value-names={R['raw_value_enum_names_in_file']} "
    f"(126 MetricKind + 2 LumaIngress + 5 other-enum) | "
    f"cli=126 -> 124 data + {R['excluded_cli_records']} + ['aimos'] = {R['total_rows']} rows | "
    f"extra_outputs_not_names={R['extra_outputs_not_names']} missing={missing}\n")
assert not missing, f'unmapped names: {missing}'

HDR = ('metric', 'family', 'role', 'claimed_canonical', 'claimed_reference_language',
       'harness_language_unmerged_3494d8af', 'version_rev_sha',
       'upstream_url', 'build_or_runner_binding', 'harness_route_unmerged_3494d8af',
       'parity_gate_documented', 'availability_status')

def emit():
    hdr = '\t'.join(HDR) + '\n'
    CAP, TARGET = 30000, 29000
    shards, buf, first = [], hdr, None
    for r in rows:
        line = '\t'.join(r) + '\n'
        name = r[0]
        if first is None:
            first = name
        if len((buf + line).encode()) > TARGET:
            shards.append((first, buf.split('\n')[-2].split('\t')[0], buf))
            buf, first = hdr, name
        buf += line
    if buf != hdr:
        shards.append((first, buf.split('\n')[-2].split('\t')[0], buf))
    return shards

shards = emit()
assert all(len(b.encode()) <= 30000 for _, _, b in shards)
_listing0 = [(f'inventory-source-{i:02d}.tsv', f0, l0, len(b.encode()),
              hashlib.sha256(b.encode()).hexdigest()) for i, (f0, l0, b) in enumerate(shards)]

def _meta(listing):
    return {
        'schema': 'inventory-source/4',
        'source_revision': SOURCE_REV,
        'gate_sources': [{'path': path, 'revision': SOURCE_REV} for path in GATE_SOURCES],
        'generated_by': 'benchmarks/port_spotcheck_2026-10-18/make_inventory_source.py (committed)',
        'shard_cap_bytes': 30000, 'shard_fill_target': 29000,
        'reconciliation': R,
        'registry': {'file': 'crates/zenmetrics-cli/src/metrics/mod.rs',
                     'block': 'pub enum MetricKind, mod.rs:97-1362',
                     'revision': '27350d07f2c3692a9d0665a81a3d0beffaa8c0e1',
                     'names': 126,
                     'sha256': 'c8f1a2cc4c289102939082ab754f352131676edcf0306b7c083c2b42baa3149a'},
        'crosscheck': {'retained_list_metrics': '~/tmp/portcheck/cli-metrics.txt',
                       'sha256': 'e729cec8161671d82f2692bf0db19484efb1dc7451a2259c5131375285ab0453',
                       'result': 'name-set EQUAL (126)'},
        'base_table': {'file': 'inventory-base-unmerged-3494d8af.tsv (committed evidence copy)',
                       'source': 'benchmarks/port_spotcheck_2026-10-18/inventory.tsv',
                       'revision': '3494d8af34839c21ff0b97f787ffd15c8a4d4ac3 (UNMERGED lane — NOT ancestor of master)',
                       'sha256': BASE_SHA, 'status': 'unmerged source snapshot, not master coverage'},
        'harness_refs': {'canonical_run.py @3494d8af blob': '0675a14f…',
                         'oracle_drivers.py @3494d8af blob': '0e99966b…',
                         'note': 'unmerged lane; configured bounds are harness defaults, not documented gates'},
        'gate_doc': DOC + ' (tolerance table ~:103-126); see gate_sources for crate tests/README',
        'shards': {n: {'first_row': f, 'last_row': l, 'bytes': z, 'sha256': h}
                   for n, f, l, z, h in listing},
        'no_metric_calls': True, 'timing_qualification': '0/19',
    }

def _meta_bytes(listing):
    return (json.dumps(_meta(listing), indent=2, sort_keys=True) + '\n').encode()

if '--verify' in sys.argv:
    ok = True
    listing = []
    for i, (f0, l0, b) in enumerate(shards):
        p = OUT / f'inventory-source-{i:02d}.tsv'
        listing.append((p.name, f0, l0, len(b.encode()), hashlib.sha256(b.encode()).hexdigest()))
        if not p.exists() or p.read_bytes() != b.encode():
            ok = False; print(f'MISMATCH {p}')
        elif len(b.encode()) > 30000:
            ok = False; print(f'OVERSIZE {p}')
    expected = {f'inventory-source-{i:02d}.tsv' for i in range(len(shards))}
    stray = {p.name for p in OUT.glob('inventory-source-*.tsv')} - expected
    if stray:
        ok = False; print(f'STRAY shard files: {sorted(stray)}')
    want_idx = ('# shards (deterministic, registry order, each <=30000 bytes)\n'
                'shard\tfirst_row\tlast_row\tbytes\tsha256\n' +
                ''.join(f'{n}\t{f}\t{l}\t{z}\t{h}\n' for n, f, l, z, h in listing))
    if (OUT / 'inventory-source.tsv').read_text() != want_idx:
        ok = False; print('INDEX field mismatch (regenerated index != committed)')
    mp = OUT / 'inventory-source.meta'
    if not mp.exists() or mp.read_bytes() != _meta_bytes(listing):
        ok = False; print('META field mismatch (regenerated meta != committed)')
    emitted = [r[0] for r in rows]
    cli_hit = [n for n in emitted if n in NAMES]
    assert sorted(cli_hit) == sorted(NAMES)
    assert emitted.count('aimos') == 1
    assert all(c != '—' and not c.startswith('as ') and c != 'same'
               for r in rows for c in r[2:]), 'placeholder cell found'
    print(f'VERIFY {"PASS" if ok else "FAIL"}: {len(shards)} shards ≤30000B, '
          f'{R["data_rows"]} data + {len(EXCLUDED_CLI)} excluded-cli + 1 aimos = '
          f'{R["total_rows"]} rows, all {R["cli_names"]} CLI names exactly once, no placeholders')
    sys.exit(0 if ok else 1)

OUT.mkdir(parents=True, exist_ok=True)
expected = {f'inventory-source-{i:02d}.tsv' for i in range(len(shards))}
stray = {p.name for p in OUT.glob('inventory-source-*.tsv')} - expected
if stray:
    raise PatchError(f'pre-existing shard files not in expected set (preserved, refusing to delete): {sorted(stray)}')
listing = _listing0
for name, _, _, _, _ in listing:
    i = int(name.split('-')[-1].split('.')[0])
    write_artifact(OUT / name, shards[i][2].encode())
write_artifact(OUT / 'inventory-source.tsv', (
    '# shards (deterministic, registry order, each <=30000 bytes)\n'
    'shard\tfirst_row\tlast_row\tbytes\tsha256\n' +
    ''.join(f'{n}\t{f}\t{l}\t{z}\t{h}\n' for n, f, l, z, h in listing)).encode())
write_artifact(OUT / 'inventory-source.meta', _meta_bytes(listing))
print(json.dumps({'rows': R['total_rows'],
                  'shards': {n: z for n, _, _, z, _ in listing}}))
