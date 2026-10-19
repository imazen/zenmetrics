/* Benchmark glue for the original dssim-core C API. Both RGB8 image builds,
 * comparison and image destruction are inside each warm sample. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dssim.h"

static unsigned char *read_rgb(const char *path, size_t bytes) {
    FILE *f = fopen(path, "rb");
    unsigned char *p = malloc(bytes);
    if (!f || !p || fread(p, 1, bytes, f) != bytes || fgetc(f) != EOF) {
        fprintf(stderr, "RGB read failed: %s\n", path); exit(2);
    }
    fclose(f); return p;
}
static long long now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}
int main(int argc, char **argv) {
    if (argc != 7) return 2;
    const unsigned w = strtoul(argv[1], NULL, 10), h = strtoul(argv[2], NULL, 10);
    const unsigned samples = strtoul(argv[6], NULL, 10);
    if (!w || !h || !samples) return 2;
    unsigned char *r = read_rgb(argv[3], (size_t)w*h*3);
    unsigned char *d = read_rgb(argv[4], (size_t)w*h*3);
    Dssim *metric = dssim_new();
    if (!metric) return 2;
    for (unsigned i = 0; i <= samples; ++i) {
        long long start = now_ns();
        DssimImage *a = dssim_create_image_rgb(metric, r, w, h);
        DssimImage *b = dssim_create_image_rgb(metric, d, w, h);
        if (!a || !b) return 2;
        double score = dssim_compare(metric, a, b);
        dssim_free_image(a); dssim_free_image(b);
        long long elapsed = now_ns()-start;
        printf("%s %lld %.17g\n", i ? "SAMPLE" : "COLD", elapsed, score);
        fflush(stdout);
    }
    dssim_free(metric); free(r); free(d); return 0;
}
