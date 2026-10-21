/* Benchmark-only full-call adapter for the unchanged author dssim-core C API.
 * Raw files and thread environment are prepared before any library call.
 * COLD includes context/image creation and comparison, with free after clock.
 * Warm SAMPLE retains context/image creation, comparison and free.
 * No scorer call precedes the COLD observation. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dssim.h"

static int positive_u32(const char *text, uint32_t *value) {
    if (!text[0]) return 0;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return 0;
    errno = 0;
    char *end = NULL;
    uintmax_t parsed = strtoumax(text, &end, 10);
    if (errno || *end || !parsed || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static unsigned char *read_rgb(const char *path, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "RGB open failed: %s\n", path);
        return NULL;
    }
    unsigned char *pixels = malloc(bytes);
    int ok = pixels && fread(pixels, 1, bytes, file) == bytes;
    if (ok) ok = fgetc(file) == EOF && !ferror(file);
    if (fclose(file)) ok = 0;
    if (!ok) {
        fprintf(stderr, "RGB length/read/allocation failed: %s\n", path);
        free(pixels);
        return NULL;
    }
    return pixels;
}

static int now_ns(int64_t *value) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) {
        perror("clock_gettime");
        return 0;
    }
    if (t.tv_sec < 0 || (uintmax_t)t.tv_sec > (INT64_MAX - 999999999) / 1000000000) {
        fprintf(stderr, "monotonic clock outside int64 range\n");
        return 0;
    }
    *value = (int64_t)t.tv_sec * INT64_C(1000000000) + t.tv_nsec;
    return 1;
}

static int set_thread_budget(uint32_t threads) {
    char value[32];
    if (snprintf(value, sizeof(value), "%" PRIu32, threads) <= 0) return 0;
    const char *names[] = {"RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
        if (setenv(names[i], value, 1)) return 0;
    return !setenv("OMP_DYNAMIC", "FALSE", 1) && !setenv("MKL_DYNAMIC", "FALSE", 1);
}

int main(int argc, char **argv) {
    uint32_t width, height, threads, samples;
    if (argc != 7 || !positive_u32(argv[1], &width) ||
        !positive_u32(argv[2], &height) || !positive_u32(argv[5], &threads) ||
        !positive_u32(argv[6], &samples)) {
        fprintf(stderr, "usage: width height reference.rgb distorted.rgb threads samples (positive integers)\n");
        return 2;
    }
    const size_t row_bytes = (size_t)width * 3;
    if (row_bytes / 3 != width || (size_t)height > SIZE_MAX / row_bytes) {
        fprintf(stderr, "RGB size overflow\n");
        return 2;
    }
    if (!setlocale(LC_NUMERIC, "C") || !set_thread_budget(threads)) {
        fprintf(stderr, "locale/thread environment setup failed\n");
        return 2;
    }
    const size_t bytes = (size_t)width * height * 3;
    unsigned char *reference = read_rgb(argv[3], bytes);
    if (!reference) return 2;
    unsigned char *distorted = read_rgb(argv[4], bytes);
    if (!distorted) { free(reference); return 2; }
    printf("VERSION dssim-core-3.5.1 constructor-first-v1\nSCOPE first-context-images-compare;warm-includes-free\n"
           "THREADS requested=%" PRIu32 " environment=explicit effective=unverified\n", threads);
    fflush(stdout);
    int rc = 0;
    for (uint64_t i = 0; i <= (uint64_t)samples; ++i) {
        int64_t start, stop;
        if (!now_ns(&start)) { rc = 2; break; }
        Dssim *context = dssim_new();
        if (!context) {
            fprintf(stderr, "dssim_new failed at observation %" PRIu64 "\n", i);
            rc = 2; break;
        }
        DssimImage *a = dssim_create_image_rgb(context, reference, width, height);
        DssimImage *b = dssim_create_image_rgb(context, distorted, width, height);
        if (!a || !b) {
            if (a) dssim_free_image(a);
            if (b) dssim_free_image(b);
            dssim_free(context);
            fprintf(stderr, "dssim image construction refused %" PRIu32 "x%" PRIu32
                            " at observation %" PRIu64 "\n", width, height, i);
            rc = 2; break;
        }
        double score = dssim_compare(context, a, b);
        /* The first observation keeps the scorer/images alive past elapsed
         * capture, matching the native CUDA first-call lifetime boundary.
         * Subsequent SAMPLE observations retain the full-call free boundary. */
        int clock_ok = i ? 1 : now_ns(&stop);
        dssim_free_image(a);
        dssim_free_image(b);
        dssim_free(context);
        if (i) clock_ok = now_ns(&stop);
        if (!clock_ok) { rc = 2; break; }
        if (stop < start || !isfinite(score)) {
            fprintf(stderr, "invalid elapsed time/nonfinite DSSIM at observation %" PRIu64 "\n", i);
            rc = 2; break;
        }
        /* Scalar validation and stdout transport follow the sample clock. */
        printf("%s %" PRId64 " %.17g\n", i ? "SAMPLE" : "COLD", stop - start, score);
        if (fflush(stdout)) { rc = 2; break; }
    }
    free(reference);
    free(distorted);
    return rc;
}
