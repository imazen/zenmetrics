#include "VshipAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Compare two doubles, qsort callback
static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    int W = argc > 1 ? atoi(argv[1]) : 4000;
    int H = argc > 2 ? atoi(argv[2]) : 3000;
    int N_warm = 5;
    int N_iter = 20;

    // sRGB RGB planar at full range, BT709 primaries
    Vship_Colorspace_t cs = {
        .width = W,
        .height = H,
        .target_width = -1,
        .target_height = -1,
        .sample = Vship_SampleUINT8,
        .range = Vship_RangeFull,
        .subsampling = {0, 0},
        .chromaLocation = Vship_ChromaLoc_Left,
        .colorFamily = Vship_ColorRGB,
        .YUVMatrix = Vship_MATRIX_RGB,
        .transferFunction = Vship_TRC_sRGB,
        .primaries = Vship_PRIMARIES_BT709,
        .crop = {0, 0, 0, 0},
    };

    Vship_CVVDPHandler h;
    Vship_Exception e = Vship_CVVDPInit3(&h, cs, cs, 30.0f, false, "standard_4k", NULL, 0);
    if (e != Vship_NoError) {
        char msg[1024] = {0};
        Vship_GetDetailedLastError(msg, sizeof(msg));
        fprintf(stderr, "Vship_CVVDPInit3 failed err=%d: %s\n", e, msg);
        return 1;
    }
    fprintf(stderr, "CVVDP handler created. W=%d H=%d MP=%.2f\n", W, H, (double)(W*H)/1e6);

    // Pinned allocate per-plane buffers (3 planes per side)
    uint8_t *ref_p[3], *dst_p[3];
    int64_t stride[3] = { W, W, W };
    size_t plane_bytes = (size_t)W * (size_t)H;
    for (int i = 0; i < 3; i++) {
        e = Vship_PinnedMalloc((void**)&ref_p[i], plane_bytes);
        if (e != Vship_NoError) { fprintf(stderr, "pin ref %d failed\n", i); return 1; }
        e = Vship_PinnedMalloc((void**)&dst_p[i], plane_bytes);
        if (e != Vship_NoError) { fprintf(stderr, "pin dst %d failed\n", i); return 1; }
        // Fill with deterministic noise so ref != dst
        srand(0x1234 + i);
        for (size_t k = 0; k < plane_bytes; k++) ref_p[i][k] = (uint8_t)(rand() & 0xff);
        srand(0xabcd + i);
        for (size_t k = 0; k < plane_bytes; k++) dst_p[i][k] = (uint8_t)(rand() & 0xff);
    }

    // Warmup
    for (int i = 0; i < N_warm; i++) {
        double score = 0.0;
        Vship_ResetCVVDP(h);
        Vship_ResetScoreCVVDP(h);
        e = Vship_ComputeCVVDP(h, &score, NULL, 0,
            (const uint8_t**)ref_p, (const uint8_t**)dst_p, stride, stride);
        if (e != Vship_NoError) {
            char msg[1024] = {0};
            Vship_CVVDPGetDetailedLastError(h, msg, sizeof(msg));
            fprintf(stderr, "ComputeCVVDP warm err=%d: %s\n", e, msg);
            return 1;
        }
    }

    // Measure
    double times[N_iter];
    double scores[N_iter];
    for (int i = 0; i < N_iter; i++) {
        Vship_ResetCVVDP(h);
        Vship_ResetScoreCVVDP(h);
        double t0 = now_sec();
        double score = 0.0;
        e = Vship_ComputeCVVDP(h, &score, NULL, 0,
            (const uint8_t**)ref_p, (const uint8_t**)dst_p, stride, stride);
        double t1 = now_sec();
        if (e != Vship_NoError) { fprintf(stderr, "ComputeCVVDP iter %d err=%d\n", i, e); return 1; }
        times[i] = t1 - t0;
        scores[i] = score;
    }

    qsort(times, N_iter, sizeof(double), cmp_double);
    double median = times[N_iter/2];
    double p25 = times[N_iter/4];
    double p75 = times[(3*N_iter)/4];
    double sum = 0;
    for (int i = 0; i < N_iter; i++) sum += times[i];
    double mean = sum / N_iter;

    double npx = (double)W * (double)H;
    printf("VSHIP CVVDP  W=%d H=%d  MP=%.2f\n", W, H, npx/1e6);
    printf("  iters: %d warmup, %d measured\n", N_warm, N_iter);
    printf("  per-call:  median %.3f ms  mean %.3f ms  p25 %.3f  p75 %.3f\n",
        median*1000, mean*1000, p25*1000, p75*1000);
    printf("  per-pixel: median %.2f ns/px  mean %.2f ns/px\n",
        median*1e9/npx, mean*1e9/npx);
    printf("  example score (random uint8 noise): %.4f\n", scores[0]);

    // Cleanup
    for (int i = 0; i < 3; i++) {
        Vship_PinnedFree(ref_p[i]);
        Vship_PinnedFree(dst_p[i]);
    }
    Vship_CVVDPFree(h);
    return 0;
}
