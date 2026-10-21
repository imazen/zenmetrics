"""Check clock/lifetime ordering with a synthetic C API, including a negative control.

This never executes a real metric. Caller supplies heavy admission.
"""
import argparse
import json
from pathlib import Path
import subprocess
from compare_canonical import sha
HEADER = r"""
typedef struct Dssim Dssim;
typedef struct DssimImage DssimImage;
Dssim *dssim_new(void);
DssimImage *dssim_create_image_rgb(Dssim *, const unsigned char *, unsigned, unsigned);
double dssim_compare(Dssim *, DssimImage *, DssimImage *);
void dssim_free_image(DssimImage *);
void dssim_free(Dssim *);
"""
STUB = r"""
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dssim.h"
struct Dssim { int unused; };
struct DssimImage { int unused; };
static struct Dssim context;
static struct DssimImage images[2];
static int live_context, live_images, comparisons, clocks;
static void require(int ok, const char *why) {
    if (!ok) { fprintf(stderr, "LIFETIME_REJECT %s\n", why); exit(42); }
}
int clock_gettime(clockid_t id, struct timespec *value) {
    require(id == CLOCK_MONOTONIC, "monotonic clock required");
    if (clocks == 0 || clocks == 2)
        require(!live_context && !live_images, "new observation starts after prior release");
    else if (clocks == 1)
        require(live_context == 1 && live_images == 2 && comparisons == 1,
                "COLD context/images must remain alive at stop");
    else if (clocks == 3) {
        require(!live_context && !live_images && comparisons == 2,
                "SAMPLE includes image/context release");
        fprintf(stderr, "LIFETIME_SEQUENCE_OK\n");
    } else require(0, "unexpected clock call");
    value->tv_sec = 1;
    value->tv_nsec = (++clocks) * 1000000;
    return 0;
}
Dssim *dssim_new(void) {
    require(!live_context && !live_images, "no stale context");
    live_context = 1; return &context;
}
DssimImage *dssim_create_image_rgb(Dssim *c, const unsigned char *rgb, unsigned w, unsigned h) {
    require(c == &context && live_context && rgb && w == 1 && h == 1 && live_images < 2,
            "valid image construction");
    return &images[live_images++];
}
double dssim_compare(Dssim *c, DssimImage *a, DssimImage *b) {
    require(c == &context && a == &images[0] && b == &images[1] && live_images == 2,
            "both images alive for evaluation");
    ++comparisons; return 0.25;
}
void dssim_free_image(DssimImage *image) {
    require(live_images > 0 && (image == &images[0] || image == &images[1]), "valid image release");
    --live_images;
}
void dssim_free(Dssim *c) {
    require(c == &context && live_context && !live_images, "images release before context");
    live_context = 0;
}
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compiler', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir()
    (root / 'dssim.h').write_text(HEADER)
    (root / 'stub.c').write_text(STUB)
    for name in ['reference.rgb', 'distorted.rgb']:
        (root / name).write_bytes(bytes([37, 71, 109]))
    sources = [Path(__file__).with_name('dssim_first_reference.c'), Path(__file__).with_name('dssim_fullcall_reference.c')]
    observed = []
    for (number, source) in enumerate(sources):
        binary = root / f'control-{number}'
        command = [str(args.compiler.resolve()), '-O2', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-I', str(root), str(source.resolve()), str(root / 'stub.c'), '-o', str(binary)]
        with (root / f'compile-{number}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
        result = subprocess.run([str(binary), '1', '1', str(root / 'reference.rgb'), str(root / 'distorted.rgb'), '1', '1'], capture_output=True, text=True, timeout=5)
        (root / f'control-{number}.stdout').write_text(result.stdout)
        (root / f'control-{number}.stderr').write_text(result.stderr)
        if number == 0:
            if result.returncode or 'LIFETIME_SEQUENCE_OK' not in result.stderr:
                raise ValueError('first-call lifetime boundary failed')
            rows = [line.split() for line in result.stdout.splitlines() if line.startswith(('COLD ', 'SAMPLE '))]
            if len(rows) != 2 or [r[0] for r in rows] != ['COLD', 'SAMPLE']:
                raise ValueError('first and subsequent observation coverage missing')
        elif result.returncode != 42 or 'COLD context/images must remain alive at stop' not in result.stderr:
            raise ValueError('full-call negative control did not exercise the distinct boundary')
        observed.append(dict(source=str(source), source_sha256=sha(source), command=command, returncode=result.returncode, stdout_sha256=sha(root / f'control-{number}.stdout'), stderr_sha256=sha(root / f'control-{number}.stderr')))
    with (root / 'result.json').open('x') as stream:
        json.dump(dict(passed=True, synthetic_lifetime_control=True, real_reference_calls=0, native_metric_calls=0, compiler_sha256=sha(args.compiler), source_sha256=sha(__file__), controls=observed, qualification='Synthetic call/clock ordering only. The actual first-call adapter must retain context/images at COLD stop and free them before SAMPLE stop; the unchanged full-call adapter is rejected by that distinct contract. This is not an accuracy or timing observation.'), stream, indent=2)
    print(json.dumps(dict(passed=True, synthetic_lifetime_control=True, real_reference_calls=0)))
if __name__ == '__main__':
    main()
