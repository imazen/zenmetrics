#!/usr/bin/env python3
"""Does butteraugli side with ssim2 (DISFAVORS jxl) or with zensim (FAVORS jxl)?

The canonical picker corpus has zensim+ssim2 but NO butteraugli. The 2026-06-24
6-metric data has butteraugli (max + 3-norm) + zensim for webp/avif/jpeg but NO
jxl. This fills the jxl gap: butteraugli-gpu scored on box-21's jxl-lossy variants
(reusing the existing sweep run `jxl-lossy-vardct-1782609551`), joined to the
canonical jxl `score_zensim`.

Bias = butteraugli achieved at a given zensim, minus the cross-codec mean at that
zensim. Butteraugli is a DISTORTION metric (lower = better), so:
    bias < 0  ⇒ LOWER butteraugli at matched zensim ⇒ BETTER ⇒ butteraugli FAVORS that codec
    bias > 0  ⇒ HIGHER butteraugli at matched zensim ⇒ WORSE ⇒ butteraugli DISFAVORS that codec
(Opposite sign convention from metric_bias.png, which used ssim2 where higher=better.)

Headline: if jxl's butteraugli bias is NEGATIVE (jxl achieves lower butteraugli at
matched zensim, like its zensim home-court) → butteraugli SIDES WITH zensim (favors jxl).
If POSITIVE (worse butteraugli at matched zensim, like ssim2) → SIDES WITH ssim2 (disfavors jxl).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANON = "/mnt/v/output/canonical-picker-2026-06-27"
SIX = "/mnt/v/zen/zensim-training/2026-06-24/unified"
OUT = "/mnt/v/output/picker-metric-investigation"
JXL_SIDECAR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/butter_jxl_sidecar.parquet"

COL = {"jpeg": "#888888", "webp": "#2a9d2a", "jxl": "#1f6fd0", "avif": "#d62728"}
MAXC, P3C = "butteraugli_max_gpu", "butteraugli_pnorm3_gpu"


def load_jxl():
    """jxl: butteraugli from the new sidecar joined to canonical score_zensim.

    The canonical zenjxl_lossy split distributes box-21's cells across train/validate/test
    by origin — concatenating all three joins ALL 59,220 scored cells (vs 38,430 in train alone).
    """
    bd = pd.read_parquet(JXL_SIDECAR,
                         columns=["image_path", "codec", "q", "knob_tuple_json", MAXC, P3C])
    cols = ["image_path", "codec", "q", "knob_tuple_json", "score_zensim"]
    cd = pd.concat([pd.read_parquet(f"{CANON}/zenjxl_lossy/{s}.parquet", columns=cols)
                    for s in ("train", "validate", "test")], ignore_index=True)
    # q dtype align (canonical q is float; sidecar q from score-pairs is int)
    bd["q"] = bd["q"].astype(float)
    cd["q"] = cd["q"].astype(float)
    m = bd.merge(cd, on=["image_path", "codec", "q", "knob_tuple_json"], how="inner")
    m = m.rename(columns={"score_zensim": "zensim", MAXC: "bmax", P3C: "b3"})
    print(f"  [jxl join] sidecar={len(bd):,} cells → joined={len(m):,} to canonical score_zensim")
    return m[["zensim", "bmax", "b3"]].dropna()


def load_six(codec_dir, is_jpeg=False):
    """webp/avif/jpeg: butteraugli + zensim from the 2026-06-24 6-metric data."""
    if is_jpeg:
        b = pd.read_parquet(f"{SIX}/{codec_dir}/sidecars/butteraugli-gpu.parquet",
                            columns=["image_path", "codec", "q", "knob_tuple_json", MAXC, P3C])
        # jpeg's zensim sidecar names the column `zensim_gpu` (webp/avif scores.parquet use `zensim_score`).
        z = pd.read_parquet(f"{SIX}/{codec_dir}/sidecars/zensim-gpu.parquet",
                            columns=["image_path", "codec", "q", "knob_tuple_json", "zensim_gpu"])
        m = b.merge(z, on=["image_path", "codec", "q", "knob_tuple_json"], how="inner")
        m = m.rename(columns={"zensim_gpu": "zensim_score"})
    else:
        m = pd.read_parquet(f"{SIX}/{codec_dir}/scores.parquet",
                            columns=[MAXC, P3C, "zensim_score"])
    m = m.rename(columns={"zensim_score": "zensim", MAXC: "bmax", P3C: "b3"})
    return m[["zensim", "bmax", "b3"]].dropna()


def main():
    parts = {
        "jxl": load_jxl(),
        "webp": load_six("zenwebp"),
        "avif": load_six("zenavif"),
        "jpeg": load_six("zenjpeg", is_jpeg=True),
    }
    for f, d in parts.items():
        zz = d.zensim
        print(f"{f:>5}: n={len(d):>7,}  zensim[{zz.min():.1f},{zz.max():.1f}] "
              f"median bmax={d.bmax.median():.3f} b3={d.b3.median():.3f}")

    # Bias curves: at each zensim bin, median butteraugli per codec minus cross-codec mean.
    # Restrict to the zensim range all codecs cover with mass (jxl swept to ~q90; the
    # ssim2-investigation used 46..92). Use 46..90 in 2-pt bins.
    QB = np.arange(46, 89, 2)
    fams = ["jpeg", "webp", "jxl", "avif"]

    def bias(metric):
        med = {f: np.array([np.median(parts[f][metric][(parts[f].zensim >= b) & (parts[f].zensim < b + 2)])
                            if ((parts[f].zensim >= b) & (parts[f].zensim < b + 2)).sum() >= 8 else np.nan
                            for b in QB]) for f in fams}
        mean_all = np.nanmean(np.array([med[f] for f in fams]), axis=0)
        return {f: med[f] - mean_all for f in fams}, med

    bmax_bias, bmax_med = bias("bmax")
    b3_bias, b3_med = bias("b3")

    # Companion RAW curves: per-codec median butteraugli vs zensim (absolute, not
    # mean-subtracted). More robust to the cross-corpus caveat than the bias view —
    # it shows each codec's own butteraugli-at-matched-zensim directly. Lower = better.
    rfig, raxes = plt.subplots(1, 2, figsize=(16, 6))
    for rax, (med, title) in zip(raxes, [(bmax_med, "butteraugli max-norm"),
                                         (b3_med, "butteraugli 3-norm")]):
        for f in fams:
            rax.plot(QB + 1, med[f], color=COL[f], lw=2.4, label=f, marker="o", ms=3)
        rax.set_xlabel("zensim (matched quality)")
        rax.set_ylabel(f"median {title}  (lower = less distortion = BETTER)")
        rax.set_title(f"Per-codec {title} vs zensim (raw, absolute)\n"
                      "lower curve at a given zensim ⇒ that codec is FAVORED by butteraugli")
        rax.grid(alpha=.3)
        rax.legend()
        rax.invert_yaxis()  # so 'up' = better, matching intuition
    rfig.suptitle("Raw butteraugli vs zensim per codec (cross-corpus; jxl=box-21 clean-picker, "
                  "webp/avif/jpeg=2026-06-24 imazen-26)", fontsize=11)
    rfig.savefig(f"{OUT}/butter_jxl_rd.png", dpi=110, bbox_inches="tight")
    plt.close(rfig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (bias_d, title) in zip(axes, [(bmax_bias, "butteraugli max-norm"),
                                          (b3_bias, "butteraugli 3-norm")]):
        for f in fams:
            ax.plot(QB + 1, bias_d[f], color=COL[f], lw=2.4, label=f, marker="o", ms=3)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel("zensim (matched quality)")
        ax.set_ylabel(f"{title} bias  (achieved − cross-codec mean, at that zensim)")
        ax.set_title(f"Per-codec {title} bias vs zensim\n"
                     "butteraugli is DISTORTION: <0 ⇒ LOWER ⇒ BETTER ⇒ FAVORED")
        ax.grid(alpha=.3)
        ax.legend()
        # annotate jxl direction
        jxl_mean = np.nanmean(bias_d["jxl"])
        verdict = ("FAVORS jxl (sides with zensim)" if jxl_mean < 0
                   else "DISFAVORS jxl (sides with ssim2)")
        ax.text(0.02, 0.02, f"jxl mean bias = {jxl_mean:+.3f} ⇒ {verdict}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="#ffffcc", alpha=.8))

    fig.suptitle("Does butteraugli favor or disfavor JXL at matched zensim? "
                 "(jxl: box-21 of run jxl-lossy-vardct-1782609551)", fontsize=12)
    fig.savefig(f"{OUT}/butter_jxl_bias.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Numeric summary table
    print("\nzensim  | jpeg_max webp_max jxl_max avif_max | jpeg_3 webp_3 jxl_3 avif_3   (bias; <0=favored)")
    for i, b in enumerate(QB):
        row = f"{b+1:>5.0f}   | "
        row += " ".join(f"{bmax_bias[f][i]:+7.3f}" for f in fams) + " | "
        row += " ".join(f"{b3_bias[f][i]:+6.3f}" for f in fams)
        print(row)

    # Raw absolute medians per zensim bin (companion view, robust to cross-corpus mean)
    print("\nRAW median butteraugli per zensim bin (lower=better):")
    print("zensim  | jpeg_max webp_max jxl_max avif_max | jpeg_3 webp_3 jxl_3 avif_3")
    for i, b in enumerate(QB):
        row = f"{b+1:>5.0f}   | "
        row += " ".join(f"{bmax_med[f][i]:7.3f}" if not np.isnan(bmax_med[f][i]) else "    nan" for f in fams) + " | "
        row += " ".join(f"{b3_med[f][i]:6.3f}" if not np.isnan(b3_med[f][i]) else "   nan" for f in fams)
        print(row)

    print("\n=== HEADLINE ===")
    print("CAVEAT: cross-corpus comparison — jxl from box-21 (clean-picker o_95xx renditions),")
    print("webp/avif/jpeg from the 2026-06-24 imazen-26 data; 0 shared renditions. The matched-")
    print("zensim normalization controls for quality but the cross-codec mean mixes content.")
    print("The SIGN (favor/disfavor) is robust when large+consistent across the zensim band.\n")
    for metric, bd, md in [("max-norm", bmax_bias, bmax_med), ("3-norm", b3_bias, b3_med)]:
        jm = np.nanmean(bd["jxl"])
        side = "ZENSIM (favors jxl)" if jm < 0 else "SSIM2 (disfavors jxl)"
        # raw rank of jxl among 4 codecs (avg over bins): 1=lowest butteraugli=best
        ranks = []
        for i in range(len(QB)):
            vals = {f: md[f][i] for f in fams if not np.isnan(md[f][i])}
            if "jxl" in vals and len(vals) >= 3:
                srt = sorted(vals, key=vals.get)
                ranks.append(srt.index("jxl") + 1)
        avg_rank = np.mean(ranks) if ranks else float("nan")
        print(f"butteraugli {metric}: jxl mean bias {jm:+.3f}  ⇒ sides with {side}"
              f"  | jxl avg raw rank {avg_rank:.2f}/4 (1=best/lowest)")
    print(f"\ngraphs:\n  http://172.23.240.1:3300/picker-metric-investigation/butter_jxl_bias.png"
          f"\n  http://172.23.240.1:3300/picker-metric-investigation/butter_jxl_rd.png")


if __name__ == "__main__":
    main()
