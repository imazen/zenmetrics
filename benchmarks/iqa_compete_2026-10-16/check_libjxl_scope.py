#!/usr/bin/env python3
"""Source/CLI controls for the libjxl adapter's opt-in rgb-input timing scope.

Read-only over the adapter source; never runs a metric. CLI controls (with
--binary) exercise argument/scope validation and the failed-open path on
nonexistent inputs — every invocation exits without a successful read or
any metric API call. See native-rgb-scope.md.
"""
import argparse, subprocess, sys

def fail(msg): print(f"FAIL {msg}", file=sys.stderr); sys.exit(1)

def block(src, start_marker):
    """Return (start_index, end_index_exclusive) of the {...} region that
    follows start_marker, by brace matching."""
    i = src.index(start_marker)
    return match_block(src, src.index("{", i))

def block_before(src, marker):
    """Return the {...} region whose opening '{' precedes the marker text
    (marker sits inside the block, e.g. a comment on its first line)."""
    i = src.index(marker)
    return match_block(src, src.rindex("{", 0, i))

def match_block(src, j):
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{": depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return j, k + 1
    fail(f"unbalanced braces at offset {j}")

def check_source(src):
    c = []
    def ok(name, cond, detail=""):
        c.append((name, cond, detail))
    ok("argc accepts exactly 9 or 10", 'argc!=9 && argc!=10' in src)
    ok("default scope is native-buffer", 'argc==10?argv[9]:"native-buffer"' in src)
    ok("unknown scope rejected", '"unknown scope"' in src and src.index('"unknown scope"') < src.index("read(argv[4]"))
    ok("single VERSION literal", src.count('"VERSION libjxl-v0.12.0\\n"') == 1)
    ok("SCOPE literal emitted once", src.count('"SCOPE rgb-input\\n"') == 1)
    eh_start, eh_end = block(src, "auto emit_header=")
    eh = src[eh_start:eh_end]
    ok("emit_header owns the single VERSION emission", '"VERSION libjxl-v0.12.0' in eh and eh.count("VERSION") == 1)
    ok("emit_header gates SCOPE on rgb-input", 'if(scope=="rgb-input")' in eh and '"SCOPE rgb-input' in eh)
    ok("intensity unchanged (ssim2 255 / butteraugli 80)", 'metric=="ssim2"?255.f:80.f' in src)
    ok("butteraugli params unchanged", 'hf_asymmetry=1.f' in src and 'xmul=1.f' in src and 'intensity_target=80.f' in src)
    ok("single ComputeScore call site (shared)", src.count("jxl::ComputeScore(") == 1)
    ok("single ssim_score call site (shared)", src.count("ssim_score(r,d,&score)") == 1)
    ok("same ComputeScore args", 'mode=="score+map"?&map:nullptr,pool.get(),false' in src)
    ok("map release kept (zenbench iter() match)", src.count("map=jxl::ImageF(); // Include native output release, matching zenbench iter().") == 2)

    # --- old-scope containment: staging before the loop, clock inside it ---
    nb_start, nb_end = block(src, 'if(scope=="native-buffer")')
    nb = src[nb_start:nb_end]
    loop_i = nb.index("for(unsigned i=0;i<=samples;++i)")
    ok("native-buffer stages bundles before loop", nb.index("ImageBundle r(manager,&rm)") < loop_i)
    ok("native-buffer stages comparator before loop", nb.index("JxlButteraugliComparator comparator") < loop_i)
    ok("native-buffer: no per-call ImageBundle inside loop", "ImageBundle r(manager" not in nb[loop_i:])
    ok("native-buffer: emit_header after staging, before loop",
       nb.index("JxlButteraugliComparator comparator") < nb.index("emit_header()") < loop_i)
    ok("native-buffer: no direct cout before emit_header", 'std::cout<<' not in nb[:nb.index("emit_header()")])

    # --- rgb-input timed-region containment via brace scanning ---
    ri_start, ri_end = block(src, "} else {")
    ri = src[ri_start:ri_end]
    ok("rgb-input: emit_header before loop", ri.index("emit_header()") < ri.index("for(unsigned"))
    ok("rgb-input: no direct SCOPE emission in branch", '"SCOPE rgb-input' not in ri)
    t0, t1 = block_before(ri, "// Timed RAII scope")
    timed = ri[t0:t1]
    start_i = ri.index("steady_clock::now()")
    elap_i = ri.index("elapsed=std::chrono")
    ok("rgb-input: clock starts before timed scope", start_i < t0)
    ok("rgb-input: elapsed computed after scope closes", elap_i > t1)
    for token, name in [
        ("ImageMetadata rm,dm", "metadata created"),
        ("ImageBundle r(manager,&rm),d(manager,&dm)", "bundles created"),
        ("convert(rb,r)", "ref conversion"),
        ("convert(db,d)", "dist conversion"),
        ("if(metric==\"butteraugli\")", "comparator gated on need"),
        ("JxlButteraugliComparator comparator(params,cms)", "per-call comparator"),
        ("run_metric(r,d", "metric call"),
        ("map=jxl::ImageF()", "map release"),
    ]:
        ok(f"rgb-input: {name} inside timed scope", token in timed)
    for token, name in [("read(", "RGB file read"), ("ThreadPoolInternal", "worker pool")]:
        ok(f"rgb-input: {name} stays outside timed scope", token not in timed)
    ok("rgb-input: no second clock inside scope", timed.count("steady_clock::now()") == 0)
    ok("rgb-input: emit_header before timed work", ri.index("emit_header()") < t0)

    # --- both loops share identical timing arithmetic and print contract ---
    ok("same elapsed expression in both loops", src.count("duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count()") == 2)
    ok("warmup suppression identical (i gate)", src.count('if(i)std::cout<<"SAMPLE "') == 2)
    return c

def check_cli(binary):
    """Argument-validation controls; every invocation must exit before any
    file read or metric call (all inputs are nonexistent paths)."""
    c = []
    def run(args):
        p = subprocess.run([binary] + args, capture_output=True, text=True, timeout=30)
        return p.returncode, p.stdout + p.stderr
    rc, out = run(["butteraugli"])
    c.append(("argc!=9,10 rejected", rc == 1 and "mode samples" in out, out.strip()[:80]))
    rc, out = run(["butteraugli", "4", "4", "/nonexistent/a.rgb", "/nonexistent/b.rgb", "1", "score", "1", "bogus-scope"])
    c.append(("unknown scope rejected before file reads", rc == 1 and "unknown scope" in out and "open RGB" not in out, out.strip()[:80]))
    rc, out = run(["butteraugli", "4", "4", "/nonexistent/a.rgb", "/nonexistent/b.rgb", "1", "score", "1"])
    c.append(("argc-9 default accepted: attempts file open, fails — no successful read or metric API", rc == 1 and "open RGB input" in out, out.strip()[:80]))
    rc, out = run(["ssim2", "4", "4", "/nonexistent/a.rgb", "/nonexistent/b.rgb", "1", "score+map", "1", "rgb-input"])
    c.append(("ssim2 score+map still UNSUPPORTED_MAP in rgb-input", rc == 0 and "UNSUPPORTED_MAP" in out, out.strip()[:80]))
    rc, out = run(["butteraugli", "4", "4", "/nonexistent/a.rgb", "/nonexistent/b.rgb", "1", "score", "1", "native-buffer"])
    c.append(("explicit native-buffer arg accepted: attempts file open, fails — no successful read or metric API", rc == 1 and "open RGB input" in out and "unknown scope" not in out, out.strip()[:80]))
    return c

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="path to libjxl_compute.cpp")
    ap.add_argument("--binary", help="built iqa-jxl for CLI controls (optional)")
    a = ap.parse_args()
    src = open(a.source).read()
    results = check_source(src)
    if a.binary:
        results += check_cli(a.binary)
    bad = [r for r in results if not r[1]]
    for name, passed, detail in results:
        print(("PASS" if passed else "FAIL"), name, ("| " + detail) if detail and not passed else "")
    print(f"{len(results) - len(bad)}/{len(results)} controls pass")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
