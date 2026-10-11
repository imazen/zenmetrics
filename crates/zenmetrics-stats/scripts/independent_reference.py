#!/usr/bin/env python3
"""Independent reference values for zenmetrics-stats' evaluation modules.

Computes, with scipy / scikit-learn / statsmodels (never zenmetrics-stats code), the
quantities that `tests/independent_reference.rs` compares against, on a
deterministic synthetic dataset, and writes them as Rust constants to
`tests/independent_reference_data.rs`.

    uv run --with numpy --with scipy --with scikit-learn --with statsmodels \
        crates/zenmetrics-stats/scripts/independent_reference.py

The data are synthetic: no human labels from any study are read.
"""
import os
import subprocess
import numpy as np
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng(20261011)

# 6 sources x 5 codecs x 8 rungs; quality-shaped score, JND-like target
rows = []
for src in range(6):
    src_off = rng.normal(0, 2.0)
    for codec in range(5):
        codec_off = rng.normal(0, 1.0)
        for rung in range(8):
            truth = 0.45 * rung + 0.1 * codec  # JND distortion
            score = 93.0 - (truth / 0.09) ** (1 / 0.94) + src_off + codec_off + rng.normal(0, 1.5)
            sigma = 0.01 + 0.03 * truth
            target = truth + rng.normal(0, sigma)
            rows.append((src, codec, score, target, sigma))
src = np.array([r[0] for r in rows])
codec = np.array([r[1] for r in rows])
x = np.array([r[2] for r in rows])
y = np.array([r[3] for r in rows])
sig = np.array([r[4] for r in rows])

# 1. knee power: y = a*max(0, b - x)^c, least squares, many starts
def knee(xv, a, b, c):
    return a * np.maximum(0.0, b - xv) ** c

best = None
for b0 in [x.max(), x.max() + 1, x.max() + 5, x.max() + 20]:
    for c0 in [0.5, 1.0, 2.0]:
        try:
            p, _ = optimize.curve_fit(knee, x, y, p0=[0.1, b0, c0], maxfev=200000,
                                      bounds=([0, -np.inf, 1e-6], [np.inf, np.inf, 50]))
        except RuntimeError:
            continue
        sse = float(np.sum((knee(x, *p) - y) ** 2))
        if best is None or sse < best[1]:
            best = (p, sse)
knee_params, knee_sse = best

# 1b. knee power with the knee fixed at 100 (pixel identity for a 0-100 quality
#     score): y = a*max(0, 100 - x)^c, a and c free; full data and
#     leave-one-source-out
KNEE_FIXED = 100.0


def knee_fixed_fit(xv, yv):
    best_f = None
    for a0 in [0.01, 0.1, 1.0]:
        for c0 in [0.5, 1.0, 2.0]:
            try:
                p, _ = optimize.curve_fit(lambda t, a, c: knee(t, a, KNEE_FIXED, c), xv, yv,
                                          p0=[a0, c0], maxfev=200000,
                                          bounds=([0, 1e-6], [np.inf, 50]),
                                          xtol=1e-15, ftol=1e-15, gtol=1e-15)
            except RuntimeError:
                continue
            sse = float(np.sum((knee(xv, p[0], KNEE_FIXED, p[1]) - yv) ** 2))
            if best_f is None or sse < best_f[1]:
                best_f = (p, sse)
    return best_f


(kf_a, kf_c), knee_fixed_sse = knee_fixed_fit(x, y)
knee_fixed_params = [kf_a, KNEE_FIXED, kf_c]
oof_knee_fixed = np.empty_like(y)
for g in np.unique(src):
    m = src != g
    (fa, fc), _ = knee_fixed_fit(x[m], y[m])
    oof_knee_fixed[~m] = knee(x[~m], fa, KNEE_FIXED, fc)

# 2. isotonic, decreasing in x (quality score vs distortion)
iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
probe = np.linspace(x.min() - 3, x.max() + 3, 41)
iso_at_probe = iso.predict(probe)

# 3. leave-one-source-out linear predictions
oof = np.empty_like(y)
for g in np.unique(src):
    m = src != g
    slope, intercept = np.polyfit(x[m], y[m], 1)
    oof[~m] = intercept + slope * x[~m]

# 4. tau-hat ML on the OOF residuals: r ~ N(0, sigma^2 + tau^2)
r = oof - y
def nll(t2):
    v = sig ** 2 + t2
    return 0.5 * np.sum(np.log(v) + r ** 2 / v)
res = optimize.minimize_scalar(nll, bounds=(0.0, 10.0), method="bounded",
                               options={"xatol": 1e-14, "maxiter": 10000})
tau_ml = float(np.sqrt(res.x))
tau_moment = float(np.sqrt(max(0.0, np.mean(r ** 2) - np.mean(sig ** 2))))

# 5. Holm
pvals = np.array([0.001, 0.04, 0.03, 0.2, 0.012, 0.5, 0.049])
holm = multipletests(pvals, method="holm")[1]

# 6. expected observer agreement, JND_75, metric = -x (distance-shaped flip handled by orientation)
k = stats.norm.ppf(0.75)
acc = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0]}
for i in range(len(y)):
    for j in range(i + 1, len(y)):
        cls = 2 if src[i] != src[j] else (1 if codec[i] != codec[j] else 0)
        dt = y[i] - y[j]
        dp = -(x[i] - x[j])  # metric oriented with the target
        pr = stats.norm.cdf(k * abs(dt))
        pw = stats.norm.cdf(-k * abs(dt))
        if dp == 0 or dt == 0:
            m = 0.5
        elif (dp > 0) == (dt > 0):
            m = pr
        else:
            m = pw
        a = acc[cls]
        a[0] += 1
        a[1] += m
        a[2] += pr
exp_agree = [(acc[c][1] / acc[c][0], acc[c][2] / acc[c][0]) for c in (0, 1, 2)]


# 7. small-G Student interval half-width t_{(1+level)/2}(G-1) * se * sqrt(G/(G-1)),
#    for se = 1, at the cluster counts the protocols meet
student_groups = [2, 3, 4, 6, 12, 31, 70]
student_levels = [0.95, 0.99]
student_half = [stats.t.ppf(0.5 * (1 + lv), g - 1) * np.sqrt(g / (g - 1))
                for lv in student_levels for g in student_groups]


def arr(name, a):
    body = ", ".join(repr(float(v)) for v in a)
    return f"pub(crate) const {name}: [f64; {len(a)}] = [{body}];\n"


def arr_u32(name, a):
    body = ", ".join(str(int(v)) for v in a)
    return f"pub(crate) const {name}: [u32; {len(a)}] = [{body}];\n"


out = ["// @generated by crates/zenmetrics-stats/scripts/independent_reference.py; do not edit.\n",
       "#![allow(dead_code, clippy::approx_constant, clippy::unreadable_literal)]\n"]
out += [arr_u32("SOURCE", src), arr_u32("CODEC", codec), arr("X", x), arr("Y", y), arr("SIGMA", sig)]
out += [arr("KNEE_PARAMS", knee_params), f"pub(crate) const KNEE_SSE: f64 = {knee_sse!r};\n"]
out += [arr("KNEE_FIXED_PARAMS", knee_fixed_params),
        f"pub(crate) const KNEE_FIXED_SSE: f64 = {knee_fixed_sse!r};\n",
        arr("OOF_KNEE_FIXED", oof_knee_fixed)]
out += [arr("ISO_PROBE", probe), arr("ISO_AT_PROBE", iso_at_probe)]
out += [arr("OOF_LINEAR", oof)]
out += [f"pub(crate) const TAU_ML: f64 = {tau_ml!r};\n", f"pub(crate) const TAU_MOMENT: f64 = {tau_moment!r};\n"]
out += [arr("HOLM_P", pvals), arr("HOLM_ADJ", holm)]
out += [arr("EXPECTED_AGREEMENT", [v for pair in exp_agree for v in pair])]
out += [arr_u32("STUDENT_GROUPS", student_groups), arr("STUDENT_LEVELS", student_levels),
        arr("STUDENT_HALF", student_half)]
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "..", "tests", "independent_reference_data.rs")
with open(path, "w") as f:
    f.write("".join(out))
# keep the generated file stable under `cargo fmt --check`
subprocess.run(["rustfmt", "--edition", "2024", path], check=True)
print("wrote", os.path.normpath(path), "rows", len(y), "knee", knee_params, "sse", knee_sse, "tau", tau_ml, tau_moment)
