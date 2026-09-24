/*
 * bench.c — fcvvdp metric-only benchmark harness.
 *
 * Mirrors zenmetrics' crates/cvvdp/examples/video_vs_ssim2.rs: same
 * deterministic clip (bit-identical frames), same median-of-reps
 * methodology, same /proc jiffies + VmHWM columns. Frame synthesis
 * runs inside the timed loop like the Rust side; a `gen` mode
 * measures synthesis alone so wall − gen ≈ metric-only time.
 *
 * usage: bench <fcvvdp|gen> <w> <h> <n_frames> [fps=30] [reps=3] [threads=0]
 *
 * stdout TSV row matches video_vs_ssim2:
 *   metric  w  h  n  fps  threads  reps  wall_ms  user_ms  sys_ms  score  vmhwm_kb
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "../src/cvvdp.h"

/* --- identical port of make_frame/distort_frame from
 * video_vs_ssim2.rs (wrapping u32 LCG, truncating casts, u8
 * saturation on the gx ramp) --- */

static inline int clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

static void make_frame(int w, int h, int t, uint32_t seed, uint8_t* out) {
    uint32_t s = seed + (uint32_t)t;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const size_t i = (size_t)(y * w + x) * 3;
            int gx = (int)(((float)(x + t * 3) / (float)w) * 255.0f);
            if (gx > 255) gx = 255;
            int gy = (int)((float)y / (float)h * 200.0f + 30.0f);
            s *= 48271u;
            const int noise = ((int)(s >> 24) - 128) / 16;
            out[i]     = (uint8_t)clamp255(gx + noise);
            out[i + 1] = (uint8_t)clamp255(gy + noise / 2);
            out[i + 2] = (uint8_t)clamp255((gx + gy) / 2 + noise / 4);
        }
    }
}

static void distort_frame(const uint8_t* f, size_t n, int t,
                          uint32_t seed, uint8_t* out) {
    uint32_t s = seed + (uint32_t)t * 7919u;
    const float gain = 1.0f + 0.06f * ((float)(t % 4) - 1.5f);
    for (size_t i = 0; i < n; i++) {
        s *= 48271u;
        const int d = ((int)(s >> 24) - 128) / 8;
        out[i] = (uint8_t)clamp255((int)((float)f[i] * gain) + d);
    }
}

/* (utime, stime) jiffies from /proc/self/stat, USER_HZ=100 */
static void cpu_jiffies(uint64_t* u, uint64_t* s) {
    *u = *s = 0;
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return;
    char buf[4096];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    /* fields after last ')' start at field 3; utime=14 → idx 11 */
    char* p = strrchr(buf, ')');
    if (!p) return;
    p += 2;
    int idx = 3;
    char* tok = strtok(p, " ");
    while (tok) {
        if (idx == 14) *u = strtoull(tok, NULL, 10);
        if (idx == 15) { *s = strtoull(tok, NULL, 10); break; }
        tok = strtok(NULL, " ");
        idx++;
    }
}

static uint64_t vmhwm_kb(void) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t v = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM", 5) == 0) {
            sscanf(line + 5, ": %llu", (unsigned long long*)&v);
            break;
        }
    }
    fclose(f);
    return v;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static double median(double* v, int n) {
    qsort(v, n, sizeof(double), cmp_d);
    return v[n / 2];
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
            "usage: %s <fcvvdp|gen> <w> <h> <n_frames> [fps=30] [reps=3] [threads=0]\n",
            argv[0]);
        return 2;
    }
    const char* metric = argv[1];
    const int w = atoi(argv[2]), h = atoi(argv[3]), n = atoi(argv[4]);
    const float fps = argc > 5 ? (float)atof(argv[5]) : 30.0f;
    const int reps = argc > 6 ? atoi(argv[6]) : 3;
    const unsigned threads = argc > 7 ? (unsigned)atoi(argv[7]) : 0;
    const size_t frame_bytes = (size_t)w * h * 3;

    uint8_t* rf = malloc(frame_bytes);
    uint8_t* df = malloc(frame_bytes);
    double* walls = malloc(sizeof(double) * reps);
    double* users = malloc(sizeof(double) * reps);
    double* syss = malloc(sizeof(double) * reps);
    double score = 0.0;

    for (int r = 0; r < reps; r++) {
        uint64_t u0, s0, u1, s1;
        cpu_jiffies(&u0, &s0);
        const double t0 = now_ms();

        if (strcmp(metric, "gen") == 0) {
            uint64_t acc = 0;
            for (int t = 0; t < n; t++) {
                make_frame(w, h, t, 1234, rf);
                distort_frame(rf, frame_bytes, t, 9876, df);
                acc += rf[0] + df[0];
            }
            score = (double)acc;
        } else {
            FcvvdpCtx* c = NULL;
            FcvvdpError err = cvvdp_create(w, h, fps,
                CVVDP_DISPLAY_STANDARD_4K, threads, NULL, &c);
            if (err != CVVDP_OK || !c) {
                fprintf(stderr, "cvvdp_create: %s\n",
                        cvvdp_error_string(err));
                return 1;
            }
            FcvvdpImage ref = {
                .width = w, .height = h, .stride = 0,
                .data = rf,
                .format = CVVDP_PIXEL_FORMAT_RGB_UINT8,
                .colorspace = CVVDP_COLORSPACE_SRGB,
            };
            FcvvdpImage dst = ref;
            dst.data = df;
            for (int t = 0; t < n; t++) {
                make_frame(w, h, t, 1234, rf);
                distort_frame(rf, frame_bytes, t, 9876, df);
                FcvvdpResult res;
                err = cvvdp_process_frame(c, &ref, &dst, &res);
                if (err != CVVDP_OK) {
                    fprintf(stderr, "process_frame t=%d: %s\n", t,
                            cvvdp_error_string(err));
                    cvvdp_destroy(c);
                    return 1;
                }
                score = res.jod;
            }
            cvvdp_destroy(c);
        }

        walls[r] = now_ms() - t0;
        cpu_jiffies(&u1, &s1);
        users[r] = (double)(u1 - u0) * 10.0;
        syss[r] = (double)(s1 - s0) * 10.0;
    }

    printf("%s\t%d\t%d\t%d\t%g\t%u\t%d\t%.3f\t%.3f\t%.3f\t%.6f\t%llu\n",
           metric, w, h, n, fps, threads, reps,
           median(walls, reps), median(users, reps), median(syss, reps),
           score, (unsigned long long)vmhwm_kb());
    return 0;
}
