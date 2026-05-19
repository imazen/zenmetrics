/*
 * cuda_dlsym_stub.c — LD_PRELOAD shim that stubs out cudarc 0.19.4's
 * lookup of `cuCoredumpDeregisterCompleteCallback`, removed from
 * libcuda.so in CUDA 13.x driver releases.
 *
 * Without this shim, every cubecl-cuda device init panics with
 *   "Expected symbol in library: DlSym { source: ... }"
 * and the GPU dispatcher dies, causing every subsequent metric call
 * to fail with RecvError on the closed channel. Observed on driver
 * 580.142 (instance 37035295) and 570.x (instance 37041995); fix
 * unblocks the entire vast.ai offer pool.
 *
 * Safety: cuCoredumpDeregisterCompleteCallback is for CUDA's
 * coredump-completion-callback teardown — only invoked during
 * process shutdown if the application has registered a coredump
 * callback (cuCoredumpRegisterCompleteCallback). zen-metrics never
 * does. A no-op stub is therefore safe: cudarc's static lookup
 * succeeds, the dispatcher stays alive, and the function pointer
 * is never actually called at runtime.
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

static void cu_coredump_deregister_complete_callback_stub(void) {
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
    if (symbol != NULL
        && strcmp(symbol, "cuCoredumpDeregisterCompleteCallback") == 0) {
        return (void *)cu_coredump_deregister_complete_callback_stub;
    }
    return real_dlsym(handle, symbol);
}
