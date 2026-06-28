#!/usr/bin/env python3
"""Interpretable CART vs full GBDT/MLP on the EXACT picker dataset dumped by
picker_tree_ab (same cells / reach / oracle, so overhead is directly
comparable). Also codegens the CART to a standalone Rust heuristic and emits
byte-exact verification cases.

Usage:
  cart_analysis.py --dump-dir DIR --codec-tag TAG --eval-split val
                   [--codegen-depth D --codegen-out FILE --verify-n N]
"""
import sys, os, json, csv, argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--dump-dir", required=True)
ap.add_argument("--codec-tag", required=True)
ap.add_argument("--eval-split", default="val")
ap.add_argument("--train-mode", default="subsample", choices=["subsample", "full"],
                help="subsample = same in_train rows the Rust MLP used (apples-to-apples); full = all train rows")
ap.add_argument("--depths", default="3,4,5,6,8,10,12,16")
ap.add_argument("--codegen-depth", type=int, default=0)
ap.add_argument("--codegen-out", default="")
ap.add_argument("--verify-n", type=int, default=256)
args = ap.parse_args()

D = args.dump_dir
TAG = args.codec_tag
meta = json.load(open(f"{D}/{TAG}_meta.json"))
n_rows = meta["n_rows"]; n_in = meta["n_in"]; n_cells = meta["n_cells"]
n_imgf = meta["n_image_feats"]
feat_names = meta["feature_names"]; cell_labels = meta["cell_labels"]

X = np.fromfile(f"{D}/{TAG}_X.f32", dtype=np.float32).reshape(n_rows, n_in)
B = np.fromfile(f"{D}/{TAG}_byteslog.f32", dtype=np.float32).reshape(n_rows, n_cells)
reach = np.isfinite(B)

# Per-row split + oracle (oracle computed in Rust -> identical to the A/B).
split = np.empty(n_rows, dtype=object)
oracle = np.full(n_rows, -1, dtype=np.int64)
in_train = np.zeros(n_rows, dtype=np.int8)
with open(f"{D}/{TAG}_rows.tsv") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        i = int(row["idx"]); split[i] = row["split"]; oracle[i] = int(row["oracle_cell"])
        in_train[i] = int(row.get("in_train", 1))

if args.train_mode == "subsample" and in_train.sum() > 0:
    tr = np.where((split == "train") & (oracle >= 0) & (in_train == 1))[0]
else:
    tr = np.where((split == "train") & (oracle >= 0))[0]
ev = np.where((split == args.eval_split) & (oracle >= 0))[0]
print(f"[{TAG}] n_rows={n_rows} n_in={n_in} n_cells={n_cells} | train({args.train_mode})={len(tr)} eval({args.eval_split})={len(ev)}", flush=True)

# Global cell preference (by train oracle frequency) for the deployable fallback
# when a predicted cell is unreachable for an eval row.
counts = np.bincount(oracle[tr], minlength=n_cells)
pref_order = list(np.argsort(-counts))

def summarize(ovs):
    if len(ovs) == 0:
        return (float("nan"),)*5
    v = np.sort(np.asarray(ovs))
    pct = lambda p: float(v[min(int(round((len(v)-1)*p)), len(v)-1)])
    return (float(v.mean()), pct(.50), pct(.90), pct(.99), float(v[-1]))

def overhead_for_picks(picks):
    """Overhead per eval row given a predicted cell each; applies the
    reachable fallback when the pick is unreachable. Returns (ovs, acc,
    unreach_rate_before_fallback)."""
    ovs = []; hits = 0; unreach = 0
    for k, i in enumerate(ev):
        pk = picks[k]
        if not reach[i, pk]:
            unreach += 1
            for c in pref_order:
                if reach[i, c]:
                    pk = c; break
        orc = oracle[i]
        ov = float(np.exp(B[i, pk] - B[i, orc]) - 1.0)
        ovs.append(ov)
        if picks[k] == orc:
            hits += 1
    return ovs, hits/len(ev), unreach/len(ev)

# ---- Depth -> overhead curve (cost of interpretability) ----
depths = [int(d) for d in args.depths.split(",")]
print(f"\n=== {TAG}: CART DecisionTreeClassifier depth -> overhead (eval={args.eval_split}) ===")
print("depth  leaves  argmin_acc  ov_mean  ov_p50  ov_p90  ov_p99  ov_WORST  unreach%")
results = {}
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X[tr], oracle[tr])
    picks = clf.predict(X[ev])
    ovs, acc, unr = overhead_for_picks(picks)
    m, p50, p90, p99, worst = summarize(ovs)
    leaves = clf.get_n_leaves()
    results[d] = dict(clf=clf, acc=acc, mean=m, p50=p50, p90=p90, p99=p99, worst=worst, unreach=unr, leaves=leaves)
    print(f"{d:>5}  {leaves:>6}  {acc:.4f}     {m:.4f}   {p50:.4f}  {p90:.4f}  {p99:.4f}  {worst:.4f}   {unr*100:.2f}", flush=True)

# Unpruned reference.
clf_full = DecisionTreeClassifier(random_state=0).fit(X[tr], oracle[tr])
ovs, acc, unr = overhead_for_picks(clf_full.predict(X[ev]))
m, p50, p90, p99, worst = summarize(ovs)
print(f" full  {clf_full.get_n_leaves():>6}  {acc:.4f}     {m:.4f}   {p50:.4f}  {p90:.4f}  {p99:.4f}  {worst:.4f}   {unr*100:.2f}  (depth={clf_full.get_depth()})", flush=True)

# ---- Tail of the dumped GBDT/MLP/RF per-row overheads (cross-check Task 4) ----
print(f"\n=== {TAG}: GBDT/MLP/RF per-row tail (from picker_tree_ab dump, eval={args.eval_split}) ===")
print("model  n   ov_mean  ov_p50  ov_p90  ov_p99  ov_WORST  argmin_acc")
for model in ["mlp", "gbdt", "rf"]:
    p = f"{D}/{TAG}_perrow_{model}.tsv"
    if not os.path.exists(p):
        continue
    ovs = []; hits = 0; n = 0
    with open(p) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ovs.append(float(row["overhead"])); hits += int(row["hit"]); n += 1
    m, p50, p90, p99, worst = summarize(ovs)
    print(f"{model:>5}  {n}  {m:.4f}   {p50:.4f}  {p90:.4f}  {p99:.4f}  {worst:.4f}   {hits/max(n,1):.4f}", flush=True)

# ---- Rust codegen + byte-exact verification cases ----
def fval_expr(fi):
    """Rust expression for tree feature index fi (image feats f32->f64; last = zq_norm)."""
    if fi < n_imgf:
        return f"(feats[{fi}] as f64)"   # feat name: {feat_names[fi]}
    return "zq_norm"

def codegen(clf, out_path, verify_n):
    t = clf.tree_
    classes = clf.classes_
    fn = f"pick_{TAG}_heuristic"
    lines = []
    lines.append("// AUTO-GENERATED by cart_analysis.py — interpretable CART code-heuristic.")
    lines.append(f"// codec={TAG}  depth={clf.get_depth()}  leaves={clf.get_n_leaves()}  n_image_feats={n_imgf}")
    lines.append(f"// feats[] are the zenanalyze image features in meta.feature_names order; zq is target quality 0..100.")
    lines.append(f"// Returns the cell id (index into cell_labels). Comparison done in f64 to match sklearn exactly.")
    lines.append("#[rustfmt::skip]")
    lines.append(f"pub fn {fn}(feats: &[f32], zq: f32) -> u16 {{")
    lines.append("    let zq_norm = (zq as f64) / 100.0;")
    def emit(node, depth):
        ind = "    " * (depth + 1)
        if t.children_left[node] == t.children_right[node]:  # leaf
            cls = int(classes[int(np.argmax(t.value[node][0]))])
            lines.append(f"{ind}return {cls};")
            return
        fi = int(t.feature[node]); thr = float(t.threshold[node])
        lines.append(f"{ind}if {fval_expr(fi)} <= {thr!r}f64 {{")
        emit(int(t.children_left[node]), depth + 1)
        lines.append(f"{ind}}} else {{")
        emit(int(t.children_right[node]), depth + 1)
        lines.append(f"{ind}}}")
    emit(0, 0)
    lines.append("}")
    lines.append("")
    # Standalone verifier: reads verify_cases.bin (N records of n_imgf f32
    # feats + 1 f32 zq + 1 f32 expected-cell). Run: rustc -O file.rs && ./file
    binname = os.path.basename(out_path).replace(".rs", "_cases.bin")
    # Standalone verifier main(): rustc -O file.rs && ./file [cases.bin]
    lines.append(f"const N_IMGF: usize = {n_imgf};")
    lines.append("fn main() {")
    lines.append(f"    let path = std::env::args().nth(1).unwrap_or_else(|| \"{binname}\".to_string());")
    lines.append("    let raw = std::fs::read(&path).expect(\"read verify cases bin\");")
    lines.append("    let rec = (N_IMGF + 2) * 4;")
    lines.append("    let n = raw.len() / rec;")
    lines.append("    let mut ok = 0usize;")
    lines.append("    for k in 0..n {")
    lines.append("        let base = k * rec;")
    lines.append("        let mut feats = vec![0f32; N_IMGF];")
    lines.append("        for i in 0..N_IMGF {")
    lines.append("            let o = base + i*4;")
    lines.append("            feats[i] = f32::from_le_bytes([raw[o],raw[o+1],raw[o+2],raw[o+3]]);")
    lines.append("        }")
    lines.append("        let o = base + N_IMGF*4;")
    lines.append("        let zq = f32::from_le_bytes([raw[o],raw[o+1],raw[o+2],raw[o+3]]);")
    lines.append("        let o2 = base + (N_IMGF+1)*4;")
    lines.append("        let exp = f32::from_le_bytes([raw[o2],raw[o2+1],raw[o2+2],raw[o2+3]]) as u16;")
    lines.append(f"        let got = {fn}(&feats, zq);")
    lines.append("        if got == exp { ok += 1; } else { eprintln!(\"MISMATCH case {k}: got {got} exp {exp}\"); }")
    lines.append("    }")
    lines.append("    println!(\"VERIFIED {}/{} cases byte-exact\", ok, n);")
    lines.append("    if ok != n { std::process::exit(1); }")
    lines.append("}")
    open(out_path, "w").write("\n".join(lines) + "\n")

    # Verification cases: random eval rows, exact f32 feats fed to sklearn.
    rng = np.random.default_rng(0)
    sel = rng.choice(ev, size=min(verify_n, len(ev)), replace=False)
    Xs = X[sel]                      # f32 image feats + zq_norm
    exp = clf.predict(Xs).astype(np.float32)
    buf = bytearray()
    for r in range(len(sel)):
        imgf = Xs[r, :n_imgf].astype(np.float32)
        zq = np.float32(Xs[r, n_imgf] * 100.0)    # zq_norm -> zq
        buf += imgf.tobytes()
        buf += np.float32(zq).tobytes()
        buf += np.float32(exp[r]).tobytes()
    binpath = out_path.replace(".rs", "_cases.bin")
    open(binpath, "wb").write(buf)
    print(f"\ncodegen -> {out_path} ({clf.get_n_leaves()} leaves, depth {clf.get_depth()}); {len(sel)} verify cases -> {binpath}", flush=True)

if args.codegen_depth and args.codegen_out:
    codegen(results[args.codegen_depth]["clf"], args.codegen_out, args.verify_n)
