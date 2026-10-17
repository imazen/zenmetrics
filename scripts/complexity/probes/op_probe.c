/* op_probe.c — measured per-instruction latency / reciprocal-throughput probes
 * for the IQA_COMPLEXITY operation-cost weights (lane swe2, host i265).
 *
 * Build:   gcc -O2 -static -o op_probe op_probe.c -lm
 * Usage:   ./op_probe --list                    dump probe table (tsv)
 *          ./op_probe <name> lat <iters>        run latency probe
 *          ./op_probe <name> tp  <iters>        run throughput probe
 *
 * A latency probe is one loop-carried dependent chain of the instruction;
 * a throughput probe is NACC independent accumulators, NACC instructions
 * per loop iteration. Cycles are counted externally with perf stat; the
 * harness subtracts a low-iteration run of the same probe (slope method),
 * so process startup and probe init cancel out.
 *
 * Conventions / caveats (see report):
 *  - vmovaps_ld_*: chain is vmovq (xmm->gpr) feeding the next load address;
 *    per-iter = load latency + vmovq xfer. vmovq_rt probe measures the
 *    xmm<->gpr round trip used to decompose it.
 *  - vmovaps_stlf_* / mov_stlf: store->load forwarding round trip per iter.
 *  - vgatherdps_*_lat: mask is regenerated with vpcmpeqd (input-independent,
 *    off the index-register critical path); per-iter ~= gather latency.
 *  - libm *_lat feeds the result back through a range-keeping op (documented
 *    per probe); *_tp uses 8 independent accumulators, scalar calls only.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* data pools                                                          */
/* ------------------------------------------------------------------ */

static float initf[16 * 8] __attribute__((aligned(64)));
static double initd[16 * 8] __attribute__((aligned(64)));
static int32_t initi[16 * 8] __attribute__((aligned(64)));

/* f32 const lanes: 0:2.0 4:0.5 8:1.0 12:1.0000001 16:0.7 20:1e-4 */
static const float cf[] __attribute__((aligned(64))) = {
    2.f, .5f, 1.f, 1.0000001f, .7f, 1e-4f, 0.f, 0.f};
/* f64 const lanes: 0:2.0 8:0.5 16:1.0 24:1+2^-51 32:0.7 40:1e-4 */
static const double cd[] __attribute__((aligned(64))) = {
    2., .5, 1., 1.0000000000000002, .7, 1e-4, 0., 0.};
/* i32 const rows (32B each):
 *   0   : eight x 1        (vpaddd const)
 *   32  : eight x 3        (vpmulld const)
 *   64  : eight x -1       (all-ones)
 *   96  : 0..7             (gather index init)
 *   128 : bytes 0..15,0..15 (vpshufb identity, in-lane)
 */
static int32_t ci[8 * 8] __attribute__((aligned(64)));

static uint8_t lbuf[8192] __attribute__((aligned(64))); /* chase + stlf buf */
static int32_t gtbl[64] __attribute__((aligned(64)));   /* gather table */

static void init_pools(void) {
  for (int i = 0; i < 16 * 8; i++) {
    initf[i] = 0.5f + 0.013f * (float)i;          /* 0.5 .. 1.16 */
    initd[i] = 0.5 + 0.013 * (double)i;
    initi[i] = (int32_t)(i * 2654435761u >> 20);  /* varied small ints */
  }
  for (int j = 0; j < 8; j++) {
    ci[0 * 8 + j] = 1;
    ci[1 * 8 + j] = 3;
    ci[2 * 8 + j] = -1;
    ci[3 * 8 + j] = j;
  }
  /* vpshufb identity mask: lane bytes 0..15 in each 128-bit lane */
  for (int j = 0; j < 32; j++) ((uint8_t *)ci)[128 + j] = (uint8_t)(j & 15);
  /* pointer-chase offsets: line i (32B) -> line (i+37)%128 */
  memset(lbuf, 0, sizeof(lbuf));
  for (int i = 0; i < 128; i++)
    *(uint64_t *)(lbuf + 32 * i) = (uint64_t)(((i + 37) & 127) * 32);
  /* gather table: rotation permutation, indices stay in 0..7 */
  for (int j = 0; j < 64; j++) gtbl[j] = 0;
  for (int j = 0; j < 8; j++) gtbl[j] = (j + 1) & 7;
}

/* ------------------------------------------------------------------ */
/* asm skeletons                                                       */
/* ------------------------------------------------------------------ */

#define CLOB                                                               \
  "rax", "rcx", "cc", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",    \
      "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12",   \
      "ymm13", "ymm14", "ymm15"

#define DEFP(fn, INIT, BODY)                                                \
  static void fn(uint64_t n) {                                              \
    __asm__ volatile(INIT                                                   \
                     "mov %[nn], %%rax\n\t"                                 \
                     ".p2align 6\n\t"                                       \
                     "1:\n\t" BODY "dec %%rax\n\t"                          \
                     "jnz 1b\n\t" ::[nn] "r"(n),                            \
                         [If] "r"(initf),                                   \
                     [Id] "r"(initd), [Ii] "r"(initi), [Cf] "r"(cf),        \
                     [Cd] "r"(cd), [Ci] "r"(ci), [Lb] "r"(lbuf),            \
                     [Gt] "r"(gtbl)                                         \
                     : CLOB);                                               \
  }

/* repetition over accumulator index i (last macro arg) */
#define REP8(m, a, b)                                                      \
  m(a, b, 0) m(a, b, 1) m(a, b, 2) m(a, b, 3) m(a, b, 4) m(a, b, 5)        \
      m(a, b, 6) m(a, b, 7)
#define REP12(m, a, b)                                                     \
  REP8(m, a, b) m(a, b, 8) m(a, b, 9) m(a, b, 10) m(a, b, 11)
#define REP13(m, a, b) REP12(m, a, b) m(a, b, 12)
#define REP14(m, a, b) REP13(m, a, b) m(a, b, 13)
#define REP15(m, a, b) REP14(m, a, b) m(a, b, 14)

/* per-accumulator body templates, i = accumulator register index */
#define T3O(px, ins, i)  ins " %%" px "15, %%" px #i ", %%" px #i "\n\t" /* acc op= c15 */
#define T3R(px, ins, i)  ins " %%" px #i ", %%" px "15, %%" px #i "\n\t" /* acc op= c15/acc */
#define TU3(px, ins, i)  ins " %%" px #i ", %%" px #i ", %%" px #i "\n\t"
#define TU2(px, ins, i)  ins " %%" px #i ", %%" px #i "\n\t"
#define TFMA(px, ins, i) ins " %%" px "14, %%" px "15, %%" px #i "\n\t" /* acc = acc*c15 + c14 */
#define TBL(px, ins, i)                                                   \
  ins " %%" px "15, %%" px "14, %%" px #i ", %%" px #i "\n\t" /* blend(acc,14,mask15) */
#define TCVXY(px, ins, i) ins " %%xmm" #i ", %%" px #i "\n\t" /* px dst, xmm src */
#define TCVYX(px, ins, i) ins " %%ymm" #i ", %%xmm" #i "\n\t" /* xmm dst, ymm src */
#define TLD(px, ins, i)   ins " " #i "*32(%[Lb]), %%" px #i "\n\t"
#define TST(px, ins, i)   ins " %%" px #i ", " #i "*32(%[Lb])\n\t"

/* init helpers (px and r are string literals, e.g. BCF(20, "xmm", "15")) */
#define ACCI(pool, px, i) "vmovdqa " #i "*32(%[" pool "]), %%" px #i "\n\t"
#define BCF(off, px, r)   "vbroadcastss " #off "(%[Cf]), %%" px r "\n\t"
#define BCD(off, px, r)   "vmovddup " #off "(%[Cd]), %%" px r "\n\t"
#define BCI(off, px, r)   "vmovdqa " #off "(%[Ci]), %%" px r "\n\t"

/* ------------------------------------------------------------------ */
/* probe instantiation                                                 */
/* ------------------------------------------------------------------ */

/* scalar ss/sd and packed x/y, 3-operand: acc OP= acc,c15 */
#define LAT_BIN(fn, ins, px, accinit, cst)                                \
  DEFP(fn, accinit cst("15"), OP3L(ins, px, 0, 0, 15))
#define OP3L(ins, px, d, a, b) ins " %%" px #b ", %%" px #a ", %%" px #d "\n\t"

/* init rows: acc <- pool row0; consts below */
#define IF0(px) "vmovdqa (%[If]), %%" px "0\n\t"
#define ID0(px) "vmovdqa (%[Id]), %%" px "0\n\t"
#define II0(px) "vmovdqa (%[Ii]), %%" px "0\n\t"
#define IFn(px, n) REP##n(ACCI, "If", px)
#define IDn(px, n) REP##n(ACCI, "Id", px)
#define IIn(px, n) REP##n(ACCI, "Ii", px)

/* float const operands (cf offsets) */
#define F2(px)   BCF(0, px, "15")
#define FH(px)   BCF(4, px, "15")
#define F1(px)   BCF(8, px, "15")
#define FMUL(px) BCF(12, px, "15")
#define FMM(px)  BCF(16, px, "15")
#define FEPS(px) BCF(20, px, "15")
/* fma addend reg 14 = 1.0 */
#define FFAC(px) BCF(4, px, "14")
#define FA1(px)  BCF(8, px, "15")
#define D2(px)   BCD(0, px, "15")
#define DH(px)   BCD(4 * 2, px, "15")
#define D1(px)   BCD(8 * 2, px, "15")
#define DMUL(px) BCD(12 * 2, px, "15")
#define DMM(px)  BCD(16 * 2, px, "15")
#define DEPS(px) BCD(20 * 2, px, "15")
#define DFAC(px) BCD(4 * 2, px, "14")
#define DA1(px)   BCD(8 * 2, px, "15")
#define IONE(px) BCI(0, px, "15")
#define I3(px)   BCI(32, px, "15")
#define IONES(px) BCI(64, px, "15")
#define PSHUFID(px) BCI(128, px, "15")

/* ---- add ---- */
DEFP(lat_vaddss, IF0("xmm") FEPS("xmm"), OP3L("vaddss", "xmm", 0, 0, 15))
DEFP(lat_vaddsd, ID0("xmm") DEPS("xmm"), OP3L("vaddsd", "xmm", 0, 0, 15))
DEFP(lat_vaddps_x, IF0("xmm") FEPS("xmm"), OP3L("vaddps", "xmm", 0, 0, 15))
DEFP(lat_vaddps_y, IF0("ymm") FEPS("ymm"), OP3L("vaddps", "ymm", 0, 0, 15))
DEFP(lat_vaddpd_x, ID0("xmm") DEPS("xmm"), OP3L("vaddpd", "xmm", 0, 0, 15))
DEFP(lat_vaddpd_y, ID0("ymm") DEPS("ymm"), OP3L("vaddpd", "ymm", 0, 0, 15))
DEFP(tp_vaddss, IFn("xmm", 12) FEPS("xmm"), REP12(T3O, "xmm", "vaddss"))
DEFP(tp_vaddsd, IDn("xmm", 12) DEPS("xmm"), REP12(T3O, "xmm", "vaddsd"))
DEFP(tp_vaddps_x, IFn("xmm", 12) FEPS("xmm"), REP12(T3O, "xmm", "vaddps"))
DEFP(tp_vaddps_y, IFn("ymm", 12) FEPS("ymm"), REP12(T3O, "ymm", "vaddps"))
DEFP(tp_vaddps_y15, IFn("ymm", 15) FEPS("ymm"), REP15(T3O, "ymm", "vaddps"))
DEFP(tp_vaddpd_x, IDn("xmm", 12) DEPS("xmm"), REP12(T3O, "xmm", "vaddpd"))
DEFP(tp_vaddpd_y, IDn("ymm", 12) DEPS("ymm"), REP12(T3O, "ymm", "vaddpd"))

/* ---- sub ---- */
DEFP(lat_vsubss, IF0("xmm") FEPS("xmm"), OP3L("vsubss", "xmm", 0, 0, 15))
DEFP(lat_vsubsd, ID0("xmm") DEPS("xmm"), OP3L("vsubsd", "xmm", 0, 0, 15))
DEFP(lat_vsubps_x, IF0("xmm") FEPS("xmm"), OP3L("vsubps", "xmm", 0, 0, 15))
DEFP(lat_vsubps_y, IF0("ymm") FEPS("ymm"), OP3L("vsubps", "ymm", 0, 0, 15))
DEFP(lat_vsubpd_x, ID0("xmm") DEPS("xmm"), OP3L("vsubpd", "xmm", 0, 0, 15))
DEFP(lat_vsubpd_y, ID0("ymm") DEPS("ymm"), OP3L("vsubpd", "ymm", 0, 0, 15))
DEFP(tp_vsubss, IFn("xmm", 12) FEPS("xmm"), REP12(T3O, "xmm", "vsubss"))
DEFP(tp_vsubsd, IDn("xmm", 12) DEPS("xmm"), REP12(T3O, "xmm", "vsubsd"))
DEFP(tp_vsubps_x, IFn("xmm", 12) FEPS("xmm"), REP12(T3O, "xmm", "vsubps"))
DEFP(tp_vsubps_y, IFn("ymm", 12) FEPS("ymm"), REP12(T3O, "ymm", "vsubps"))
DEFP(tp_vsubpd_x, IDn("xmm", 12) DEPS("xmm"), REP12(T3O, "xmm", "vsubpd"))
DEFP(tp_vsubpd_y, IDn("ymm", 12) DEPS("ymm"), REP12(T3O, "ymm", "vsubpd"))

/* ---- mul ---- */
DEFP(lat_vmulss, IF0("xmm") FMUL("xmm"), OP3L("vmulss", "xmm", 0, 0, 15))
DEFP(lat_vmulsd, ID0("xmm") DMUL("xmm"), OP3L("vmulsd", "xmm", 0, 0, 15))
DEFP(lat_vmulps_x, IF0("xmm") FMUL("xmm"), OP3L("vmulps", "xmm", 0, 0, 15))
DEFP(lat_vmulps_y, IF0("ymm") FMUL("ymm"), OP3L("vmulps", "ymm", 0, 0, 15))
DEFP(lat_vmulpd_x, ID0("xmm") DMUL("xmm"), OP3L("vmulpd", "xmm", 0, 0, 15))
DEFP(lat_vmulpd_y, ID0("ymm") DMUL("ymm"), OP3L("vmulpd", "ymm", 0, 0, 15))
DEFP(tp_vmulss, IFn("xmm", 12) FMUL("xmm"), REP12(T3O, "xmm", "vmulss"))
DEFP(tp_vmulsd, IDn("xmm", 12) DMUL("xmm"), REP12(T3O, "xmm", "vmulsd"))
DEFP(tp_vmulps_x, IFn("xmm", 12) FMUL("xmm"), REP12(T3O, "xmm", "vmulps"))
DEFP(tp_vmulps_y, IFn("ymm", 12) FMUL("ymm"), REP12(T3O, "ymm", "vmulps"))
DEFP(tp_vmulps_y15, IFn("ymm", 15) FMUL("ymm"), REP15(T3O, "ymm", "vmulps"))
DEFP(tp_vmulpd_x, IDn("xmm", 12) DMUL("xmm"), REP12(T3O, "xmm", "vmulpd"))
DEFP(tp_vmulpd_y, IDn("ymm", 12) DMUL("ymm"), REP12(T3O, "ymm", "vmulpd"))

/* ---- fma: acc = acc*cf + addend ---- */
DEFP(lat_vfmadd231ss, IF0("xmm") FFAC("xmm") FA1("xmm"),
     OP3L("vfmadd231ss", "xmm", 0, 15, 14))
DEFP(lat_vfmadd231sd, ID0("xmm") DFAC("xmm") DA1("xmm"),
     OP3L("vfmadd231sd", "xmm", 0, 15, 14))
DEFP(lat_vfmadd231ps_x, IF0("xmm") FFAC("xmm") FA1("xmm"),
     OP3L("vfmadd231ps", "xmm", 0, 15, 14))
DEFP(lat_vfmadd231ps_y, IF0("ymm") FFAC("ymm") FA1("ymm"),
     OP3L("vfmadd231ps", "ymm", 0, 15, 14))
DEFP(lat_vfmadd231pd_x, ID0("xmm") DFAC("xmm") DA1("xmm"),
     OP3L("vfmadd231pd", "xmm", 0, 15, 14))
DEFP(lat_vfmadd231pd_y, ID0("ymm") DFAC("ymm") DA1("ymm"),
     OP3L("vfmadd231pd", "ymm", 0, 15, 14))
DEFP(tp_vfmadd231ss, IFn("xmm", 12) FFAC("xmm") FA1("xmm"),
     REP12(TFMA, "xmm", "vfmadd231ss"))
DEFP(tp_vfmadd231sd, IDn("xmm", 12) DFAC("xmm") DA1("xmm"),
     REP12(TFMA, "xmm", "vfmadd231sd"))
DEFP(tp_vfmadd231ps_x, IFn("xmm", 12) FFAC("xmm") FA1("xmm"),
     REP12(TFMA, "xmm", "vfmadd231ps"))
DEFP(tp_vfmadd231ps_y, IFn("ymm", 12) FFAC("ymm") FA1("ymm"),
     REP12(TFMA, "ymm", "vfmadd231ps"))
DEFP(tp_vfmadd231ps_y14, IFn("ymm", 14) FFAC("ymm") FA1("ymm"),
     REP14(TFMA, "ymm", "vfmadd231ps"))
DEFP(tp_vfmadd231pd_x, IDn("xmm", 12) DFAC("xmm") DA1("xmm"),
     REP12(TFMA, "xmm", "vfmadd231pd"))
DEFP(tp_vfmadd231pd_y, IDn("ymm", 12) DFAC("ymm") DA1("ymm"),
     REP12(TFMA, "ymm", "vfmadd231pd"))

/* ---- div: acc = 2.0/acc (oscillates 1<->2, never denormal) ---- */
DEFP(lat_vdivss, IF0("xmm") F2("xmm"), OP3L("vdivss", "xmm", 0, 15, 0))
DEFP(lat_vdivsd, ID0("xmm") D2("xmm"), OP3L("vdivsd", "xmm", 0, 15, 0))
DEFP(lat_vdivps_x, IF0("xmm") F2("xmm"), OP3L("vdivps", "xmm", 0, 15, 0))
DEFP(lat_vdivps_y, IF0("ymm") F2("ymm"), OP3L("vdivps", "ymm", 0, 15, 0))
DEFP(lat_vdivpd_x, ID0("xmm") D2("xmm"), OP3L("vdivpd", "xmm", 0, 15, 0))
DEFP(lat_vdivpd_y, ID0("ymm") D2("ymm"), OP3L("vdivpd", "ymm", 0, 15, 0))
DEFP(tp_vdivss, IFn("xmm", 12) F2("xmm"), REP12(T3R, "xmm", "vdivss"))
DEFP(tp_vdivsd, IDn("xmm", 12) D2("xmm"), REP12(T3R, "xmm", "vdivsd"))
DEFP(tp_vdivps_x, IFn("xmm", 12) F2("xmm"), REP12(T3R, "xmm", "vdivps"))
DEFP(tp_vdivps_y, IFn("ymm", 12) F2("ymm"), REP12(T3R, "ymm", "vdivps"))
DEFP(tp_vdivps_y15, IFn("ymm", 15) F2("ymm"), REP15(T3R, "ymm", "vdivps"))
DEFP(tp_vdivpd_x, IDn("xmm", 12) D2("xmm"), REP12(T3R, "xmm", "vdivpd"))
DEFP(tp_vdivpd_y, IDn("ymm", 12) D2("ymm"), REP12(T3R, "ymm", "vdivpd"))

/* ---- sqrt: acc = sqrt(acc) -> 1 ---- */
DEFP(lat_vsqrtss, IF0("xmm"), TU3("xmm", "vsqrtss", 0))
DEFP(lat_vsqrtsd, ID0("xmm"), TU3("xmm", "vsqrtsd", 0))
DEFP(lat_vsqrtps_x, IF0("xmm"), TU2("xmm", "vsqrtps", 0))
DEFP(lat_vsqrtps_y, IF0("ymm"), TU2("ymm", "vsqrtps", 0))
DEFP(lat_vsqrtpd_x, ID0("xmm"), TU2("xmm", "vsqrtpd", 0))
DEFP(lat_vsqrtpd_y, ID0("ymm"), TU2("ymm", "vsqrtpd", 0))
DEFP(tp_vsqrtss, IFn("xmm", 12), REP12(TU3, "xmm", "vsqrtss"))
DEFP(tp_vsqrtsd, IDn("xmm", 12), REP12(TU3, "xmm", "vsqrtsd"))
DEFP(tp_vsqrtps_x, IFn("xmm", 12), REP12(TU2, "xmm", "vsqrtps"))
DEFP(tp_vsqrtps_y, IFn("ymm", 12), REP12(TU2, "ymm", "vsqrtps"))
DEFP(tp_vsqrtpd_x, IDn("xmm", 12), REP12(TU2, "xmm", "vsqrtpd"))
DEFP(tp_vsqrtpd_y, IDn("ymm", 12), REP12(TU2, "ymm", "vsqrtpd"))

/* ---- rcp / rsqrt (approx; ps only) ---- */
DEFP(lat_vrcpss, IF0("xmm"), TU3("xmm", "vrcpss", 0))
DEFP(lat_vrcpps_x, IF0("xmm"), TU2("xmm", "vrcpps", 0))
DEFP(lat_vrcpps_y, IF0("ymm"), TU2("ymm", "vrcpps", 0))
DEFP(tp_vrcpss, IFn("xmm", 12), REP12(TU3, "xmm", "vrcpss"))
DEFP(tp_vrcpps_x, IFn("xmm", 12), REP12(TU2, "xmm", "vrcpps"))
DEFP(tp_vrcpps_y, IFn("ymm", 12), REP12(TU2, "ymm", "vrcpps"))
DEFP(lat_vrsqrtss, IF0("xmm"), TU3("xmm", "vrsqrtss", 0))
DEFP(lat_vrsqrtps_x, IF0("xmm"), TU2("xmm", "vrsqrtps", 0))
DEFP(lat_vrsqrtps_y, IF0("ymm"), TU2("ymm", "vrsqrtps", 0))
DEFP(tp_vrsqrtss, IFn("xmm", 12), REP12(TU3, "xmm", "vrsqrtss"))
DEFP(tp_vrsqrtps_x, IFn("xmm", 12), REP12(TU2, "xmm", "vrsqrtps"))
DEFP(tp_vrsqrtps_y, IFn("ymm", 12), REP12(TU2, "ymm", "vrsqrtps"))

/* ---- min / max ---- */
DEFP(lat_vminss, IF0("xmm") FMM("xmm"), OP3L("vminss", "xmm", 0, 0, 15))
DEFP(lat_vminsd, ID0("xmm") DMM("xmm"), OP3L("vminsd", "xmm", 0, 0, 15))
DEFP(lat_vminps_x, IF0("xmm") FMM("xmm"), OP3L("vminps", "xmm", 0, 0, 15))
DEFP(lat_vminps_y, IF0("ymm") FMM("ymm"), OP3L("vminps", "ymm", 0, 0, 15))
DEFP(lat_vminpd_x, ID0("xmm") DMM("xmm"), OP3L("vminpd", "xmm", 0, 0, 15))
DEFP(lat_vminpd_y, ID0("ymm") DMM("ymm"), OP3L("vminpd", "ymm", 0, 0, 15))
DEFP(tp_vminss, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vminss"))
DEFP(tp_vminsd, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vminsd"))
DEFP(tp_vminps_x, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vminps"))
DEFP(tp_vminps_y, IFn("ymm", 12) FMM("ymm"), REP12(T3O, "ymm", "vminps"))
DEFP(tp_vminpd_x, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vminpd"))
DEFP(tp_vminpd_y, IDn("ymm", 12) DMM("ymm"), REP12(T3O, "ymm", "vminpd"))
DEFP(lat_vmaxss, IF0("xmm") FMM("xmm"), OP3L("vmaxss", "xmm", 0, 0, 15))
DEFP(lat_vmaxsd, ID0("xmm") DMM("xmm"), OP3L("vmaxsd", "xmm", 0, 0, 15))
DEFP(lat_vmaxps_x, IF0("xmm") FMM("xmm"), OP3L("vmaxps", "xmm", 0, 0, 15))
DEFP(lat_vmaxps_y, IF0("ymm") FMM("ymm"), OP3L("vmaxps", "ymm", 0, 0, 15))
DEFP(lat_vmaxpd_x, ID0("xmm") DMM("xmm"), OP3L("vmaxpd", "xmm", 0, 0, 15))
DEFP(lat_vmaxpd_y, ID0("ymm") DMM("ymm"), OP3L("vmaxpd", "ymm", 0, 0, 15))
DEFP(tp_vmaxss, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vmaxss"))
DEFP(tp_vmaxsd, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vmaxsd"))
DEFP(tp_vmaxps_x, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vmaxps"))
DEFP(tp_vmaxps_y, IFn("ymm", 12) FMM("ymm"), REP12(T3O, "ymm", "vmaxps"))
DEFP(tp_vmaxpd_x, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vmaxpd"))
DEFP(tp_vmaxpd_y, IDn("ymm", 12) DMM("ymm"), REP12(T3O, "ymm", "vmaxpd"))

/* ---- cmp (lt) ---- */
DEFP(lat_vcmpltss, IF0("xmm") FMM("xmm"), OP3L("vcmpltss", "xmm", 0, 0, 15))
DEFP(lat_vcmpltsd, ID0("xmm") DMM("xmm"), OP3L("vcmpltsd", "xmm", 0, 0, 15))
DEFP(lat_vcmpltps_x, IF0("xmm") FMM("xmm"), OP3L("vcmpltps", "xmm", 0, 0, 15))
DEFP(lat_vcmpltps_y, IF0("ymm") FMM("ymm"), OP3L("vcmpltps", "ymm", 0, 0, 15))
DEFP(lat_vcmpltpd_x, ID0("xmm") DMM("xmm"), OP3L("vcmpltpd", "xmm", 0, 0, 15))
DEFP(lat_vcmpltpd_y, ID0("ymm") DMM("ymm"), OP3L("vcmpltpd", "ymm", 0, 0, 15))
DEFP(tp_vcmpltss, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vcmpltss"))
DEFP(tp_vcmpltsd, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vcmpltsd"))
DEFP(tp_vcmpltps_x, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vcmpltps"))
DEFP(tp_vcmpltps_y, IFn("ymm", 12) FMM("ymm"), REP12(T3O, "ymm", "vcmpltps"))
DEFP(tp_vcmpltpd_x, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vcmpltpd"))
DEFP(tp_vcmpltpd_y, IDn("ymm", 12) DMM("ymm"), REP12(T3O, "ymm", "vcmpltpd"))

/* ---- and (abs mask) ---- */
DEFP(lat_vandps_x, IF0("xmm") FMM("xmm"), OP3L("vandps", "xmm", 0, 0, 15))
DEFP(lat_vandps_y, IF0("ymm") FMM("ymm"), OP3L("vandps", "ymm", 0, 0, 15))
DEFP(lat_vandpd_x, ID0("xmm") DMM("xmm"), OP3L("vandpd", "xmm", 0, 0, 15))
DEFP(lat_vandpd_y, ID0("ymm") DMM("ymm"), OP3L("vandpd", "ymm", 0, 0, 15))
DEFP(tp_vandps_x, IFn("xmm", 12) FMM("xmm"), REP12(T3O, "xmm", "vandps"))
DEFP(tp_vandps_y, IFn("ymm", 12) FMM("ymm"), REP12(T3O, "ymm", "vandps"))
DEFP(tp_vandpd_x, IDn("xmm", 12) DMM("xmm"), REP12(T3O, "xmm", "vandpd"))
DEFP(tp_vandpd_y, IDn("ymm", 12) DMM("ymm"), REP12(T3O, "ymm", "vandpd"))

/* ---- blendv ---- */
DEFP(lat_vblendvps_x, IF0("xmm") FMM("xmm") ACCI("If", "xmm", 14),
     "vblendvps %%xmm15, %%xmm14, %%xmm0, %%xmm0\n\t")
DEFP(lat_vblendvps_y, IF0("ymm") FMM("ymm") ACCI("If", "ymm", 14),
     "vblendvps %%ymm15, %%ymm14, %%ymm0, %%ymm0\n\t")
DEFP(lat_vblendvpd_x, ID0("xmm") DMM("xmm") ACCI("Id", "xmm", 14),
     "vblendvpd %%xmm15, %%xmm14, %%xmm0, %%xmm0\n\t")
DEFP(lat_vblendvpd_y, ID0("ymm") DMM("ymm") ACCI("Id", "ymm", 14),
     "vblendvpd %%ymm15, %%ymm14, %%ymm0, %%ymm0\n\t")
DEFP(tp_vblendvps_x, IFn("xmm", 12) FMM("xmm") ACCI("If", "xmm", 14),
     REP12(TBL, "xmm", "vblendvps"))
DEFP(tp_vblendvps_y, IFn("ymm", 12) FMM("ymm") ACCI("If", "ymm", 14),
     REP12(TBL, "ymm", "vblendvps"))
DEFP(tp_vblendvpd_x, IDn("xmm", 12) DMM("xmm") ACCI("Id", "xmm", 14),
     REP12(TBL, "xmm", "vblendvpd"))
DEFP(tp_vblendvpd_y, IDn("ymm", 12) DMM("ymm") ACCI("Id", "ymm", 14),
     REP12(TBL, "ymm", "vblendvpd"))

/* ---- cvt ---- */
DEFP(lat_vcvtdq2ps_x, II0("xmm"), TU2("xmm", "vcvtdq2ps", 0))
DEFP(lat_vcvtdq2ps_y, II0("ymm"), TU2("ymm", "vcvtdq2ps", 0))
DEFP(tp_vcvtdq2ps_x, IIn("xmm", 12), REP12(TU2, "xmm", "vcvtdq2ps"))
DEFP(tp_vcvtdq2ps_y, IIn("ymm", 12), REP12(TU2, "ymm", "vcvtdq2ps"))
DEFP(lat_vcvttps2dq_x, IF0("xmm"), TU2("xmm", "vcvttps2dq", 0))
DEFP(lat_vcvttps2dq_y, IF0("ymm"), TU2("ymm", "vcvttps2dq", 0))
DEFP(tp_vcvttps2dq_x, IIn("xmm", 12), REP12(TU2, "xmm", "vcvttps2dq"))
DEFP(tp_vcvttps2dq_y, IIn("ymm", 12), REP12(TU2, "ymm", "vcvttps2dq"))
DEFP(lat_vcvtps2dq_x, IF0("xmm"), TU2("xmm", "vcvtps2dq", 0))
DEFP(lat_vcvtps2dq_y, IF0("ymm"), TU2("ymm", "vcvtps2dq", 0))
DEFP(tp_vcvtps2dq_x, IIn("xmm", 12), REP12(TU2, "xmm", "vcvtps2dq"))
DEFP(tp_vcvtps2dq_y, IIn("ymm", 12), REP12(TU2, "ymm", "vcvtps2dq"))
/* ps -> pd */
DEFP(lat_vcvtps2pd_xx, IF0("xmm"), "vcvtps2pd %%xmm0, %%xmm0\n\t")
DEFP(lat_vcvtps2pd_xy, IF0("xmm"), "vcvtps2pd %%xmm0, %%ymm0\n\t")
DEFP(tp_vcvtps2pd_xx, IIn("xmm", 12), REP12(TU2, "xmm", "vcvtps2pd"))
DEFP(tp_vcvtps2pd_xy, IIn("xmm", 12), REP12(TCVXY, "ymm", "vcvtps2pd"))
/* pd -> ps */
DEFP(lat_vcvtpd2ps_xx, ID0("xmm"), "vcvtpd2ps %%xmm0, %%xmm0\n\t")
DEFP(lat_vcvtpd2ps_yx, ID0("ymm"), "vcvtpd2ps %%ymm0, %%xmm0\n\t")
DEFP(tp_vcvtpd2ps_xx, IDn("xmm", 12), REP12(TU2, "xmm", "vcvtpd2ps"))
DEFP(tp_vcvtpd2ps_yx, IDn("ymm", 12), REP12(TCVYX, "", "vcvtpd2ps"))
/* dq -> pd */
DEFP(lat_vcvtdq2pd_xy, II0("xmm"), "vcvtdq2pd %%xmm0, %%ymm0\n\t")
DEFP(tp_vcvtdq2pd_xy, IIn("xmm", 12), REP12(TCVXY, "ymm", "vcvtdq2pd"))
/* pd -> dq */
DEFP(lat_vcvtpd2dq_xx, ID0("xmm"), "vcvtpd2dq %%xmm0, %%xmm0\n\t")
DEFP(lat_vcvtpd2dq_yx, ID0("ymm"), "vcvtpd2dq %%ymm0, %%xmm0\n\t")
DEFP(tp_vcvtpd2dq_xx, IDn("xmm", 12), REP12(TU2, "xmm", "vcvtpd2dq"))
DEFP(tp_vcvtpd2dq_yx, IDn("ymm", 12), REP12(TCVYX, "", "vcvtpd2dq"))
/* scalar ss<->sd */
DEFP(lat_vcvtss2sd, IF0("xmm"), TU3("xmm", "vcvtss2sd", 0))
DEFP(lat_vcvtsd2ss, ID0("xmm"), TU3("xmm", "vcvtsd2ss", 0))
DEFP(tp_vcvtss2sd, IFn("xmm", 12), REP12(TU3, "xmm", "vcvtss2sd"))
DEFP(tp_vcvtsd2ss, IDn("xmm", 12), REP12(TU3, "xmm", "vcvtsd2ss"))
/* pmovzxbd */
DEFP(lat_vpmovzxbd_xx, II0("xmm"), "vpmovzxbd %%xmm0, %%xmm0\n\t")
DEFP(lat_vpmovzxbd_xy, II0("xmm"), "vpmovzxbd %%xmm0, %%ymm0\n\t")
DEFP(tp_vpmovzxbd_xx, IIn("xmm", 12), REP12(TU2, "xmm", "vpmovzxbd"))
DEFP(tp_vpmovzxbd_xy, IIn("xmm", 12), REP12(TCVXY, "ymm", "vpmovzxbd"))

/* ---- integer ---- */
DEFP(lat_vpaddd_x, II0("xmm") IONE("xmm"), OP3L("vpaddd", "xmm", 0, 0, 15))
DEFP(lat_vpaddd_y, II0("ymm") IONE("ymm"), OP3L("vpaddd", "ymm", 0, 0, 15))
DEFP(tp_vpaddd_x, IIn("xmm", 12) IONE("xmm"), REP12(T3O, "xmm", "vpaddd"))
DEFP(tp_vpaddd_y, IIn("ymm", 12) IONE("ymm"), REP12(T3O, "ymm", "vpaddd"))
DEFP(lat_vpmulld_x, II0("xmm") I3("xmm"), OP3L("vpmulld", "xmm", 0, 0, 15))
DEFP(lat_vpmulld_y, II0("ymm") I3("ymm"), OP3L("vpmulld", "ymm", 0, 0, 15))
DEFP(tp_vpmulld_x, IIn("xmm", 12) I3("xmm"), REP12(T3O, "xmm", "vpmulld"))
DEFP(tp_vpmulld_y, IIn("ymm", 12) I3("ymm"), REP12(T3O, "ymm", "vpmulld"))
DEFP(tp_vpmulld_y15, IIn("ymm", 15) I3("ymm"), REP15(T3O, "ymm", "vpmulld"))

/* ---- shufb ---- */
DEFP(lat_vpshufb_x, II0("xmm") PSHUFID("xmm"), OP3L("vpshufb", "xmm", 0, 0, 15))
DEFP(lat_vpshufb_y, II0("ymm") PSHUFID("ymm"), OP3L("vpshufb", "ymm", 0, 0, 15))
DEFP(tp_vpshufb_x, IIn("xmm", 12) PSHUFID("xmm"), REP12(T3O, "xmm", "vpshufb"))
DEFP(tp_vpshufb_y, IIn("ymm", 12) PSHUFID("ymm"), REP12(T3O, "ymm", "vpshufb"))

/* ---- gather (8-lane dps) ----
 * vgatherdps dest, mem, mask requires dest != index and dest != mask.
 * Latency: ping-pong two gathers (idx chain ymm0 -> ymm1 -> ymm0), so the
 * reported per-iteration value is TWO gather latencies; lat_div=2 in the
 * table and the runner divides. The mask is regenerated with
 * vpcmpeqd ymm15,ymm15,ymm14 -- reads constant ymm15 only, so it is never
 * on the index dependency chain and mask registers may be freely reused.
 * Throughput: N independent index registers, one shared dead dest ymm13
 * (WAW-renamed away), shared mask ymm14 re-materialized per gather. */
DEFP(lat_vgatherdps_y,
     "vmovdqa 96(%[Ci]), %%ymm0\n\tvmovdqa 96(%[Ci]), %%ymm1\n\t"
     "vmovdqa 64(%[Ci]), %%ymm15\n\t",
     "vpcmpeqd %%ymm15, %%ymm15, %%ymm14\n\t"
     "vgatherdps %%ymm14, (%[Gt], %%ymm0, 4), %%ymm1\n\t"
     "vpcmpeqd %%ymm15, %%ymm15, %%ymm14\n\t"
     "vgatherdps %%ymm14, (%[Gt], %%ymm1, 4), %%ymm0\n\t")
#define GATH_TP(i)                                                        \
  "vpcmpeqd %%ymm15, %%ymm15, %%ymm14\n\t"                                 \
  "vgatherdps %%ymm14, (%[Gt], %%ymm" #i ", 4), %%ymm13\n\t"
#define GINIT(i, m) "vmovdqa 96(%[Ci]), %%ymm" #i "\n\t"
DEFP(tp_vgatherdps_y,
     GINIT(0, 0) GINIT(1, 0) GINIT(2, 0) GINIT(3, 0)
     GINIT(4, 0) GINIT(5, 0) GINIT(6, 0) GINIT(7, 0)
     "vmovdqa 64(%[Ci]), %%ymm15\n\t",
     GATH_TP(0) GATH_TP(1) GATH_TP(2) GATH_TP(3)
     GATH_TP(4) GATH_TP(5) GATH_TP(6) GATH_TP(7))
DEFP(tp_vgatherdps_y12,
     GINIT(0, 0) GINIT(1, 0) GINIT(2, 0) GINIT(3, 0)
     GINIT(4, 0) GINIT(5, 0) GINIT(6, 0) GINIT(7, 0)
     GINIT(8, 0) GINIT(9, 0) GINIT(10, 0) GINIT(11, 0)
     "vmovdqa 64(%[Ci]), %%ymm15\n\t",
     GATH_TP(0) GATH_TP(1) GATH_TP(2) GATH_TP(3)
     GATH_TP(4) GATH_TP(5) GATH_TP(6) GATH_TP(7)
     GATH_TP(8) GATH_TP(9) GATH_TP(10) GATH_TP(11))

/* ---- load / store (vmovaps, L1-resident 8KB buffer) ---- */
/* latency: vmovq xmm0->rcx then vmovaps [lbuf+rcx] -> ymm0 (index chase).
 * per-iter = load_lat + vmovq xfer; decompose with vmovq_rt below. */
DEFP(lat_vmovaps_ld_y,
     "vxorps %%ymm0, %%ymm0, %%ymm0\n\txor %%rcx, %%rcx\n\t",
     "vmovq %%xmm0, %%rcx\n\t"
     "vmovaps (%[Lb], %%rcx, 1), %%ymm0\n\t")
DEFP(lat_vmovaps_ld_x,
     "vxorps %%xmm0, %%xmm0, %%xmm0\n\txor %%rcx, %%rcx\n\t",
     "vmovq %%xmm0, %%rcx\n\t"
     "vmovaps (%[Lb], %%rcx, 1), %%xmm0\n\t")
/* scalar reference: mov (%Lb,%rcx),%rcx — pure L1 pointer chase */
DEFP(lat_movq_ld, "xor %%rcx, %%rcx\n\t",
     "mov (%[Lb], %%rcx, 1), %%rcx\n\t")
/* vmovq xmm<->gpr round trip (2 transfers per iter) */
DEFP(lat_vmovq_rt, "vmovdqa (%[If]), %%xmm0\n\t",
     "vmovq %%xmm0, %%rcx\n\t"
     "vmovq %%rcx, %%xmm0\n\t")
/* store->load forwarding round trip: st [L],acc ; ld acc,[L] */
DEFP(lat_vmovaps_stlf_y, "vmovdqa (%[If]), %%ymm0\n\t",
     "vmovaps %%ymm0, (%[Lb])\n\t"
     "vmovaps (%[Lb]), %%ymm0\n\t")
DEFP(lat_vmovaps_stlf_x, "vmovdqa (%[If]), %%xmm0\n\t",
     "vmovaps %%xmm0, (%[Lb])\n\t"
     "vmovaps (%[Lb]), %%xmm0\n\t")
DEFP(lat_mov_stlf, "mov $1234, %%rcx\n\t",
     "mov %%rcx, (%[Lb])\n\t"
     "mov (%[Lb]), %%rcx\n\t")
/* throughput: independent loads / stores */
DEFP(tp_vmovaps_ld_x, "", REP12(TLD, "xmm", "vmovaps"))
DEFP(tp_vmovaps_ld_y, "", REP12(TLD, "ymm", "vmovaps"))
DEFP(tp_vmovaps_ld_y15, "", REP15(TLD, "ymm", "vmovaps"))
DEFP(tp_vmovaps_st_x, IFn("xmm", 12), REP12(TST, "xmm", "vmovaps"))
DEFP(tp_vmovaps_st_y, IFn("ymm", 12), REP12(TST, "ymm", "vmovaps"))
DEFP(tp_vmovaps_st_y15, IFn("ymm", 15), REP15(TST, "ymm", "vmovaps"))

/* ---- empty-loop reference ---- */
DEFP(lat_loop_empty, "", "")
DEFP(tp_loop_empty, "", "")

/* ------------------------------------------------------------------ */
/* libm scalar calls (what Rust std / system libm executes)            */
/* ------------------------------------------------------------------ */

static volatile float snk_float;
static volatile double snk_double;

/* latency: out feeds next arg, kept in range (see header/report) */
#define DEFLIBM_LAT(fn, T, EX)                                            \
  static void fn(uint64_t n) {                                            \
    T x = (T)0.9;                                                       \
    for (uint64_t i = 0; i < n; i++) x = (EX);                          \
    snk_##T = x;                                                        \
  }

DEFLIBM_LAT(lat_expf, float, -expf(x))          /* -> -0.567 */
DEFLIBM_LAT(lat_logf, float, logf(x + 2.0f))    /* -> ~1.146 */
DEFLIBM_LAT(lat_log10f, float, log10f(x + 2.0f))
DEFLIBM_LAT(lat_powf, float, powf(0.7f, x))     /* x = exponent, -> ~0.77 */
DEFLIBM_LAT(lat_cbrtf, float, cbrtf(x))         /* -> 1 */
DEFLIBM_LAT(lat_cosf, float, cosf(x))           /* -> 0.739 */
DEFLIBM_LAT(lat_atan2f, float, atan2f(x, 1.3f) + 0.4f) /* -> ~1.1, args nonzero */
DEFLIBM_LAT(lat_exp, double, -exp(x))
DEFLIBM_LAT(lat_log, double, log(x + 2.0))
DEFLIBM_LAT(lat_log10, double, log10(x + 2.0))
DEFLIBM_LAT(lat_pow, double, pow(0.7, x))
DEFLIBM_LAT(lat_cbrt, double, cbrt(x))

/* throughput: 8 independent accumulators, scalar calls; EX applied per acc */
#define DEFLIBM_TP(fn, T, EX)                                              \
  static void fn(uint64_t n) {                                             \
    T a0 = (T).51, a1 = (T).55, a2 = (T).59, a3 = (T).63, a4 = (T).67,     \
      a5 = (T).71, a6 = (T).73, a7 = (T).79;                               \
    for (uint64_t i = 0; i < n; i++) {                                     \
      a0 = EX(a0); a1 = EX(a1); a2 = EX(a2); a3 = EX(a3);                  \
      a4 = EX(a4); a5 = EX(a5); a6 = EX(a6); a7 = EX(a7);                  \
    }                                                                      \
    snk_##T = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;                       \
  }
#define EX_NEG_EXPF(x) (-expf(x))
#define EX_LOGF(x)     (logf((x) + 2.0f))
#define EX_LOG10F(x)   (log10f((x) + 2.0f))
#define EX_POWF(x)     (powf(0.7f, (x)))
#define EX_CBRTF(x)    (cbrtf(x))
#define EX_COSF(x)     (cosf(x))
#define EX_ATAN2F(x)   (atan2f((x), 1.3f) + 0.4f)
#define EX_NEG_EXP(x)  (-exp(x))
#define EX_LOG(x)      (log((x) + 2.0))
#define EX_LOG10(x)    (log10((x) + 2.0))
#define EX_POW(x)      (pow(0.7, (x)))
#define EX_CBRT(x)     (cbrt(x))

DEFLIBM_TP(tp_expf, float, EX_NEG_EXPF)
DEFLIBM_TP(tp_logf, float, EX_LOGF)
DEFLIBM_TP(tp_log10f, float, EX_LOG10F)
DEFLIBM_TP(tp_powf, float, EX_POWF)
DEFLIBM_TP(tp_cbrtf, float, EX_CBRTF)
DEFLIBM_TP(tp_cosf, float, EX_COSF)
DEFLIBM_TP(tp_atan2f, float, EX_ATAN2F)
DEFLIBM_TP(tp_exp, double, EX_NEG_EXP)
DEFLIBM_TP(tp_log, double, EX_LOG)
DEFLIBM_TP(tp_log10, double, EX_LOG10)
DEFLIBM_TP(tp_pow, double, EX_POW)
DEFLIBM_TP(tp_cbrt, double, EX_CBRT)

/* ------------------------------------------------------------------ */
/* dispatch table                                                      */
/* ------------------------------------------------------------------ */

typedef void (*probe_fn)(uint64_t);

struct probe {
  const char *name; /* probe id */
  const char *inst; /* mnemonic for llvm-mca join */
  const char *form; /* ss sd ps pd i32 mem libm */
  int width;        /* bits: element width for scalar, vector width else */
  const char *prec;
  probe_fn lat, tp; /* NULL = not measured */
  int lat_ops;      /* instructions in the latency chain per iteration */
  int tp_n;         /* ops per tp iteration (accumulators) */
  int cls;          /* 0 normal, 1 heavy(gather), 2 libm, 3 mem */
};

#define P(n, inst, form, w, prec, l, t, tp_n, cls)                         \
  P2(n, inst, form, w, prec, l, t, 1, tp_n, cls)
#define P2(n, inst, form, w, prec, l, t, lat_ops, tp_n, cls)               \
  {                                                                      \
    n, inst, form, w, prec, l, t, lat_ops, tp_n, cls                       \
  }

static const struct probe probes[] = {
    /* add */
    P("vaddss", "vaddss", "ss", 32, "f32", lat_vaddss, tp_vaddss, 12, 0),
    P("vaddsd", "vaddsd", "sd", 64, "f64", lat_vaddsd, tp_vaddsd, 12, 0),
    P("vaddps_x", "vaddps", "ps", 128, "f32", lat_vaddps_x, tp_vaddps_x, 12, 0),
    P("vaddps_y", "vaddps", "ps", 256, "f32", lat_vaddps_y, tp_vaddps_y, 12, 0),
    P("vaddps_y15", "vaddps", "ps", 256, "f32", NULL, tp_vaddps_y15, 15, 0),
    P("vaddpd_x", "vaddpd", "pd", 128, "f64", lat_vaddpd_x, tp_vaddpd_x, 12, 0),
    P("vaddpd_y", "vaddpd", "pd", 256, "f64", lat_vaddpd_y, tp_vaddpd_y, 12, 0),
    /* sub */
    P("vsubss", "vsubss", "ss", 32, "f32", lat_vsubss, tp_vsubss, 12, 0),
    P("vsubsd", "vsubsd", "sd", 64, "f64", lat_vsubsd, tp_vsubsd, 12, 0),
    P("vsubps_x", "vsubps", "ps", 128, "f32", lat_vsubps_x, tp_vsubps_x, 12, 0),
    P("vsubps_y", "vsubps", "ps", 256, "f32", lat_vsubps_y, tp_vsubps_y, 12, 0),
    P("vsubpd_x", "vsubpd", "pd", 128, "f64", lat_vsubpd_x, tp_vsubpd_x, 12, 0),
    P("vsubpd_y", "vsubpd", "pd", 256, "f64", lat_vsubpd_y, tp_vsubpd_y, 12, 0),
    /* mul */
    P("vmulss", "vmulss", "ss", 32, "f32", lat_vmulss, tp_vmulss, 12, 0),
    P("vmulsd", "vmulsd", "sd", 64, "f64", lat_vmulsd, tp_vmulsd, 12, 0),
    P("vmulps_x", "vmulps", "ps", 128, "f32", lat_vmulps_x, tp_vmulps_x, 12, 0),
    P("vmulps_y", "vmulps", "ps", 256, "f32", lat_vmulps_y, tp_vmulps_y, 12, 0),
    P("vmulps_y15", "vmulps", "ps", 256, "f32", NULL, tp_vmulps_y15, 15, 0),
    P("vmulpd_x", "vmulpd", "pd", 128, "f64", lat_vmulpd_x, tp_vmulpd_x, 12, 0),
    P("vmulpd_y", "vmulpd", "pd", 256, "f64", lat_vmulpd_y, tp_vmulpd_y, 12, 0),
    /* fma */
    P("vfmadd231ss", "vfmadd231ss", "ss", 32, "f32", lat_vfmadd231ss,
      tp_vfmadd231ss, 12, 0),
    P("vfmadd231sd", "vfmadd231sd", "sd", 64, "f64", lat_vfmadd231sd,
      tp_vfmadd231sd, 12, 0),
    P("vfmadd231ps_x", "vfmadd231ps", "ps", 128, "f32", lat_vfmadd231ps_x,
      tp_vfmadd231ps_x, 12, 0),
    P("vfmadd231ps_y", "vfmadd231ps", "ps", 256, "f32", lat_vfmadd231ps_y,
      tp_vfmadd231ps_y, 12, 0),
    P("vfmadd231ps_y14", "vfmadd231ps", "ps", 256, "f32", NULL,
      tp_vfmadd231ps_y14, 14, 0),
    P("vfmadd231pd_x", "vfmadd231pd", "pd", 128, "f64", lat_vfmadd231pd_x,
      tp_vfmadd231pd_x, 12, 0),
    P("vfmadd231pd_y", "vfmadd231pd", "pd", 256, "f64", lat_vfmadd231pd_y,
      tp_vfmadd231pd_y, 12, 0),
    /* div */
    P("vdivss", "vdivss", "ss", 32, "f32", lat_vdivss, tp_vdivss, 12, 0),
    P("vdivsd", "vdivsd", "sd", 64, "f64", lat_vdivsd, tp_vdivsd, 12, 0),
    P("vdivps_x", "vdivps", "ps", 128, "f32", lat_vdivps_x, tp_vdivps_x, 12, 0),
    P("vdivps_y", "vdivps", "ps", 256, "f32", lat_vdivps_y, tp_vdivps_y, 12, 0),
    P("vdivps_y15", "vdivps", "ps", 256, "f32", NULL, tp_vdivps_y15, 15, 0),
    P("vdivpd_x", "vdivpd", "pd", 128, "f64", lat_vdivpd_x, tp_vdivpd_x, 12, 0),
    P("vdivpd_y", "vdivpd", "pd", 256, "f64", lat_vdivpd_y, tp_vdivpd_y, 12, 0),
    /* sqrt */
    P("vsqrtss", "vsqrtss", "ss", 32, "f32", lat_vsqrtss, tp_vsqrtss, 12, 0),
    P("vsqrtsd", "vsqrtsd", "sd", 64, "f64", lat_vsqrtsd, tp_vsqrtsd, 12, 0),
    P("vsqrtps_x", "vsqrtps", "ps", 128, "f32", lat_vsqrtps_x, tp_vsqrtps_x,
      12, 0),
    P("vsqrtps_y", "vsqrtps", "ps", 256, "f32", lat_vsqrtps_y, tp_vsqrtps_y,
      12, 0),
    P("vsqrtpd_x", "vsqrtpd", "pd", 128, "f64", lat_vsqrtpd_x, tp_vsqrtpd_x,
      12, 0),
    P("vsqrtpd_y", "vsqrtpd", "pd", 256, "f64", lat_vsqrtpd_y, tp_vsqrtpd_y,
      12, 0),
    /* rcp / rsqrt (approx) */
    P("vrcpss", "vrcpss", "ss", 32, "f32", lat_vrcpss, tp_vrcpss, 12, 0),
    P("vrcpps_x", "vrcpps", "ps", 128, "f32", lat_vrcpps_x, tp_vrcpps_x, 12, 0),
    P("vrcpps_y", "vrcpps", "ps", 256, "f32", lat_vrcpps_y, tp_vrcpps_y, 12, 0),
    P("vrsqrtss", "vrsqrtss", "ss", 32, "f32", lat_vrsqrtss, tp_vrsqrtss, 12,
      0),
    P("vrsqrtps_x", "vrsqrtps", "ps", 128, "f32", lat_vrsqrtps_x,
      tp_vrsqrtps_x, 12, 0),
    P("vrsqrtps_y", "vrsqrtps", "ps", 256, "f32", lat_vrsqrtps_y,
      tp_vrsqrtps_y, 12, 0),
    /* min / max */
    P("vminss", "vminss", "ss", 32, "f32", lat_vminss, tp_vminss, 12, 0),
    P("vminsd", "vminsd", "sd", 64, "f64", lat_vminsd, tp_vminsd, 12, 0),
    P("vminps_x", "vminps", "ps", 128, "f32", lat_vminps_x, tp_vminps_x, 12, 0),
    P("vminps_y", "vminps", "ps", 256, "f32", lat_vminps_y, tp_vminps_y, 12, 0),
    P("vminpd_x", "vminpd", "pd", 128, "f64", lat_vminpd_x, tp_vminpd_x, 12, 0),
    P("vminpd_y", "vminpd", "pd", 256, "f64", lat_vminpd_y, tp_vminpd_y, 12, 0),
    P("vmaxss", "vmaxss", "ss", 32, "f32", lat_vmaxss, tp_vmaxss, 12, 0),
    P("vmaxsd", "vmaxsd", "sd", 64, "f64", lat_vmaxsd, tp_vmaxsd, 12, 0),
    P("vmaxps_x", "vmaxps", "ps", 128, "f32", lat_vmaxps_x, tp_vmaxps_x, 12, 0),
    P("vmaxps_y", "vmaxps", "ps", 256, "f32", lat_vmaxps_y, tp_vmaxps_y, 12, 0),
    P("vmaxpd_x", "vmaxpd", "pd", 128, "f64", lat_vmaxpd_x, tp_vmaxpd_x, 12, 0),
    P("vmaxpd_y", "vmaxpd", "pd", 256, "f64", lat_vmaxpd_y, tp_vmaxpd_y, 12, 0),
    /* cmp lt */
    P("vcmpltss", "vcmpltss", "ss", 32, "f32", lat_vcmpltss, tp_vcmpltss, 12,
      0),
    P("vcmpltsd", "vcmpltsd", "sd", 64, "f64", lat_vcmpltsd, tp_vcmpltsd, 12,
      0),
    P("vcmpltps_x", "vcmpltps", "ps", 128, "f32", lat_vcmpltps_x,
      tp_vcmpltps_x, 12, 0),
    P("vcmpltps_y", "vcmpltps", "ps", 256, "f32", lat_vcmpltps_y,
      tp_vcmpltps_y, 12, 0),
    P("vcmpltpd_x", "vcmpltpd", "pd", 128, "f64", lat_vcmpltpd_x,
      tp_vcmpltpd_x, 12, 0),
    P("vcmpltpd_y", "vcmpltpd", "pd", 256, "f64", lat_vcmpltpd_y,
      tp_vcmpltpd_y, 12, 0),
    /* and */
    P("vandps_x", "vandps", "ps", 128, "f32", lat_vandps_x, tp_vandps_x, 12, 0),
    P("vandps_y", "vandps", "ps", 256, "f32", lat_vandps_y, tp_vandps_y, 12, 0),
    P("vandpd_x", "vandpd", "pd", 128, "f64", lat_vandpd_x, tp_vandpd_x, 12, 0),
    P("vandpd_y", "vandpd", "pd", 256, "f64", lat_vandpd_y, tp_vandpd_y, 12, 0),
    /* blendv */
    P("vblendvps_x", "vblendvps", "ps", 128, "f32", lat_vblendvps_x,
      tp_vblendvps_x, 12, 0),
    P("vblendvps_y", "vblendvps", "ps", 256, "f32", lat_vblendvps_y,
      tp_vblendvps_y, 12, 0),
    P("vblendvpd_x", "vblendvpd", "pd", 128, "f64", lat_vblendvpd_x,
      tp_vblendvpd_x, 12, 0),
    P("vblendvpd_y", "vblendvpd", "pd", 256, "f64", lat_vblendvpd_y,
      tp_vblendvpd_y, 12, 0),
    /* cvt */
    P("vcvtdq2ps_x", "vcvtdq2ps", "ps", 128, "i32>f32", lat_vcvtdq2ps_x,
      tp_vcvtdq2ps_x, 12, 0),
    P("vcvtdq2ps_y", "vcvtdq2ps", "ps", 256, "i32>f32", lat_vcvtdq2ps_y,
      tp_vcvtdq2ps_y, 12, 0),
    P("vcvttps2dq_x", "vcvttps2dq", "ps", 128, "f32>i32", lat_vcvttps2dq_x,
      tp_vcvttps2dq_x, 12, 0),
    P("vcvttps2dq_y", "vcvttps2dq", "ps", 256, "f32>i32", lat_vcvttps2dq_y,
      tp_vcvttps2dq_y, 12, 0),
    P("vcvtps2dq_x", "vcvtps2dq", "ps", 128, "f32>i32", lat_vcvtps2dq_x,
      tp_vcvtps2dq_x, 12, 0),
    P("vcvtps2dq_y", "vcvtps2dq", "ps", 256, "f32>i32", lat_vcvtps2dq_y,
      tp_vcvtps2dq_y, 12, 0),
    P("vcvtps2pd_xx", "vcvtps2pd", "ps>pd", 128, "f32>f64", lat_vcvtps2pd_xx,
      tp_vcvtps2pd_xx, 12, 0),
    P("vcvtps2pd_xy", "vcvtps2pd", "ps>pd", 256, "f32>f64", lat_vcvtps2pd_xy,
      tp_vcvtps2pd_xy, 12, 0),
    P("vcvtpd2ps_xx", "vcvtpd2ps", "pd>ps", 128, "f64>f32", lat_vcvtpd2ps_xx,
      tp_vcvtpd2ps_xx, 12, 0),
    P("vcvtpd2ps_yx", "vcvtpd2ps", "pd>ps", 256, "f64>f32", lat_vcvtpd2ps_yx,
      tp_vcvtpd2ps_yx, 12, 0),
    P("vcvtdq2pd_xy", "vcvtdq2pd", "i32>pd", 256, "i32>f64", lat_vcvtdq2pd_xy,
      tp_vcvtdq2pd_xy, 12, 0),
    P("vcvtpd2dq_xx", "vcvtpd2dq", "pd>i32", 128, "f64>i32", lat_vcvtpd2dq_xx,
      tp_vcvtpd2dq_xx, 12, 0),
    P("vcvtpd2dq_yx", "vcvtpd2dq", "pd>i32", 256, "f64>i32", lat_vcvtpd2dq_yx,
      tp_vcvtpd2dq_yx, 12, 0),
    P("vcvtss2sd", "vcvtss2sd", "ss>sd", 64, "f32>f64", lat_vcvtss2sd,
      tp_vcvtss2sd, 12, 0),
    P("vcvtsd2ss", "vcvtsd2ss", "sd>ss", 32, "f64>f32", lat_vcvtsd2ss,
      tp_vcvtsd2ss, 12, 0),
    P("vpmovzxbd_xx", "vpmovzxbd", "u8>i32", 128, "u8>i32", lat_vpmovzxbd_xx,
      tp_vpmovzxbd_xx, 12, 0),
    P("vpmovzxbd_xy", "vpmovzxbd", "u8>i32", 256, "u8>i32", lat_vpmovzxbd_xy,
      tp_vpmovzxbd_xy, 12, 0),
    /* int */
    P("vpaddd_x", "vpaddd", "i32", 128, "i32", lat_vpaddd_x, tp_vpaddd_x, 12,
      0),
    P("vpaddd_y", "vpaddd", "i32", 256, "i32", lat_vpaddd_y, tp_vpaddd_y, 12,
      0),
    P("vpmulld_x", "vpmulld", "i32", 128, "i32", lat_vpmulld_x, tp_vpmulld_x,
      12, 0),
    P("vpmulld_y", "vpmulld", "i32", 256, "i32", lat_vpmulld_y, tp_vpmulld_y,
      12, 0),
    P("vpmulld_y15", "vpmulld", "i32", 256, "i32", NULL, tp_vpmulld_y15, 15, 0),
    /* shuf */
    P("vpshufb_x", "vpshufb", "ps", 128, "i8", lat_vpshufb_x, tp_vpshufb_x, 12,
      0),
    P("vpshufb_y", "vpshufb", "ps", 256, "i8", lat_vpshufb_y, tp_vpshufb_y, 12,
      0),
    /* gather: latency chain is a ping-pong pair, 2 gathers per iter */
    P2("vgatherdps_y", "vgatherdps", "ps", 256, "f32", lat_vgatherdps_y,
       tp_vgatherdps_y, 2, 8, 1),
    P("vgatherdps_y12", "vgatherdps", "ps", 256, "f32", NULL,
      tp_vgatherdps_y12, 12, 1),
    /* load / store; latency chains have 2 insns per iter (documented) */
    P2("vmovaps_ld_x", "vmovaps", "mem", 128, "f32", lat_vmovaps_ld_x,
       tp_vmovaps_ld_x, 2, 12, 3),
    P2("vmovaps_ld_y", "vmovaps", "mem", 256, "f32", lat_vmovaps_ld_y,
       tp_vmovaps_ld_y, 2, 12, 3),
    P2("vmovaps_ld_y15", "vmovaps", "mem", 256, "f32", NULL,
       tp_vmovaps_ld_y15, 1, 15, 3),
    P2("vmovaps_st_x", "vmovaps", "mem", 128, "f32", NULL, tp_vmovaps_st_x, 1,
       12, 3),
    P2("vmovaps_st_y", "vmovaps", "mem", 256, "f32", NULL, tp_vmovaps_st_y, 1,
       12, 3),
    P2("vmovaps_st_y15", "vmovaps", "mem", 256, "f32", NULL,
       tp_vmovaps_st_y15, 1, 15, 3),
    P2("vmovaps_stlf_x", "vmovaps_stlf", "mem", 128, "f32",
       lat_vmovaps_stlf_x, NULL, 2, 0, 3),
    P2("vmovaps_stlf_y", "vmovaps_stlf", "mem", 256, "f32",
       lat_vmovaps_stlf_y, NULL, 2, 0, 3),
    P2("mov_ld", "movq", "mem", 64, "i64", lat_movq_ld, NULL, 1, 0, 3),
    P2("mov_stlf", "mov_stlf", "mem", 64, "i64", lat_mov_stlf, NULL, 2, 0, 3),
    P2("vmovq_rt", "vmovq", "mem", 64, "i64", lat_vmovq_rt, NULL, 2, 0, 3),
    /* libm f32 */
    P("expf", "libm_expf", "libm", 32, "f32", lat_expf, tp_expf, 8, 2),
    P("logf", "libm_logf", "libm", 32, "f32", lat_logf, tp_logf, 8, 2),
    P("log10f", "libm_log10f", "libm", 32, "f32", lat_log10f, tp_log10f, 8, 2),
    P("powf", "libm_powf", "libm", 32, "f32", lat_powf, tp_powf, 8, 2),
    P("cbrtf", "libm_cbrtf", "libm", 32, "f32", lat_cbrtf, tp_cbrtf, 8, 2),
    P("cosf", "libm_cosf", "libm", 32, "f32", lat_cosf, tp_cosf, 8, 2),
    P("atan2f", "libm_atan2f", "libm", 32, "f32", lat_atan2f, tp_atan2f, 8, 2),
    /* libm f64 */
    P("exp", "libm_exp", "libm", 64, "f64", lat_exp, tp_exp, 8, 2),
    P("log", "libm_log", "libm", 64, "f64", lat_log, tp_log, 8, 2),
    P("log10", "libm_log10", "libm", 64, "f64", lat_log10, tp_log10, 8, 2),
    P("pow", "libm_pow", "libm", 64, "f64", lat_pow, tp_pow, 8, 2),
    P("cbrt", "libm_cbrt", "libm", 64, "f64", lat_cbrt, tp_cbrt, 8, 2),
    /* reference */
    P("loop_empty", "loop_empty", "ref", 0, "na", lat_loop_empty,
      tp_loop_empty, 12, 0),
};

int main(int argc, char **argv) {
  init_pools();
  if (argc >= 2 && !strcmp(argv[1], "--list")) {
    printf("probe\tinst\tform\twidth\tprec\tlat_ops\ttp_n\tcls\thas_lat\thas_"
           "tp\n");
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++)
      printf("%s\t%s\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\n", probes[i].name,
             probes[i].inst, probes[i].form, probes[i].width, probes[i].prec,
             probes[i].lat_ops, probes[i].tp_n, probes[i].cls,
             probes[i].lat != NULL, probes[i].tp != NULL);
    return 0;
  }
  if (argc != 4) {
    fprintf(stderr, "usage: %s <probe|--list> <lat|tp> <iters>\n", argv[0]);
    return 2;
  }
  const char *name = argv[1], *kind = argv[2];
  uint64_t iters = strtoull(argv[3], NULL, 0);
  for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
    if (strcmp(probes[i].name, name)) continue;
    probe_fn f = !strcmp(kind, "lat") ? probes[i].lat
             : !strcmp(kind, "tp")  ? probes[i].tp
                                    : NULL;
    if (!f) {
      fprintf(stderr, "probe %s has no %s\n", name, kind);
      return 2;
    }
    f(iters);
    return 0;
  }
  fprintf(stderr, "unknown probe %s\n", name);
  return 2;
}
