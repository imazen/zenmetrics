/*
 * cuda_dlsym_stub.c — LD_PRELOAD shim that stubs out cudarc 0.19.4's
 * lookup of CUDA coredump-callback symbols that were removed from
 * libcuda.so in CUDA 13.x driver releases.
 *
 * Symbols intercepted (all return a no-op function pointer):
 *   - cuCoredumpDeregisterCompleteCallback  (observed on driver 580.142)
 *   - cuCoredumpDeregisterStartCallback     (observed on driver 580.x
 *                                            during EXP-MULTI-CODEC
 *                                            smoke 2026-05-18; the
 *                                            first stub revision only
 *                                            handled the Complete
 *                                            variant, leaving the
 *                                            Start variant uncovered.)
 *
 * Without this shim, every cubecl-cuda device init panics with
 *   "Expected symbol in library: DlSym { source: ... }"
 * and the GPU dispatcher dies, causing every subsequent metric call
 * to fail with RecvError on the closed channel. Observed on driver
 * 580.142 (instance 37035295) and 570.x (instance 37041995); fix
 * unblocks the entire vast.ai offer pool.
 *
 * Safety: the Register/Deregister Start/Complete callbacks are part
 * of CUDA's coredump-callback teardown API — only invoked during
 * process shutdown if the application has registered a callback
 * (cuCoredumpRegisterStartCallback / cuCoredumpRegisterCompleteCallback).
 * zen-metrics never does. No-op stubs are therefore safe: cudarc's
 * static lookup succeeds, the dispatcher stays alive, and the
 * function pointers are never actually called at runtime.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o /usr/local/lib/cuda_dlsym_stub.so \
 *       cuda_dlsym_stub.c -ldl
 * Use:
 *   ENV LD_PRELOAD=/usr/local/lib/cuda_dlsym_stub.so
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

static void cu_coredump_callback_noop(void) {
    /* CUDA coredump callback teardown — no-op. Returning silently
     * is the documented behavior when no callback was registered. */
}

typedef void *(*dlsym_fn)(void *, const char *);

void *dlsym(void *handle, const char *symbol) {
    /* Bootstrap real_dlsym lazily. dlvsym lets us name a specific
     * glibc version so we resolve the underlying symbol rather than
     * recursing into our own override. GLIBC_2.2.5 has been the
     * dlsym ABI since 2002 — safe across every glibc the vast.ai
     * fleet ships. */
    static dlsym_fn real_dlsym = NULL;
    if (!real_dlsym) {
        real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    /* Prefix-match the entire cuCoredump* family. cudarc 0.19.4 also
     * looks up cuCoredumpGetAttribute, cuCoredumpSetAttributeGlobal,
     * etc., all removed from libcuda 13.x. Returning the same no-op
     * stub for all is safe because none of them are invoked unless
     * the application explicitly registers a coredump callback first
     * (and zen-metrics doesn't). */
    if (symbol != NULL && strncmp(symbol, "cuCoredump", 10) == 0) {
        return (void *)cu_coredump_callback_noop;
    }
    return real_dlsym(handle, symbol);
}
