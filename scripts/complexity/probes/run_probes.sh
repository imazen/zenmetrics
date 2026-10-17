#!/usr/bin/env bash
# run_probes.sh — measured operation-cost weights for docs/IQA_COMPLEXITY.md
# (lane swe2). Builds scripts/complexity/probes/op_probe.c into $SCRATCH
# (default ~/tmp/iqa-cx-swe2), then:
#
#   1. runs each probe under 5 separate `perf stat -x,` invocations on the
#      P-core (taskset -c 2,
#      event cpu_core/cycles/) and the E-core (taskset -c 12,
#      cpu_atom/cycles/), at a high and a low iteration count; per-iteration
#      cycles come from the slope between the two (cancels startup/init/exit).
#   2. writes $OUT/probes_i265.tsv (per-op cycles) and probes_i265.meta.
#   3. runs llvm-mca -mcpu=arrowlake and -mcpu=znver5 on one loop per
#      instruction and writes llvm_mca_arrowlake.tsv / llvm_mca_znver5.tsv.
#
# Conventions:
#   - latency probes: cycles = slope/iters, NOT divided or empty-subtracted:
#     the dec/jnz loop overhead executes off the dependent chain.
#   - throughput probes: cycles = (probe_slope - empty_loop_slope)/tp_n; the
#     empty loop (dec/jnz only) is measured at the same iters on the same
#     core. Residual over/under-subtraction is <= ~1 cycle/iter / tp_n.
#   - probes with lat_ops=2 (gather ping-pong, vmovq round trip, ld chase,
#     stlf) divide the slope by 2; see report for the decomposition.
#
# Usage: scripts/complexity/probes/run_probes.sh [all|probes|mca]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRATCH="${SCRATCH:-$HOME/tmp/iqa-cx-swe2}"
OUT="${OUT:-$ROOT/benchmarks/iqa_complexity_2026-10-17/weights}"
MCA="${MCA:-/usr/lib/llvm-22/bin/llvm-mca}"
REPS=5
mkdir -p "$SCRATCH" "$OUT"

PCPU=2; PEV="cpu_core/cycles/"
ECPU=12; EEV="cpu_atom/cycles/"

MODE="${1:-all}"

# ---------------------------------------------------------------- build
if [[ "$MODE" == "all" || "$MODE" == "probes" || "$MODE" == "mca" ]]; then
  gcc -O2 -static -o "$SCRATCH/op_probe" \
      "$ROOT/scripts/complexity/probes/op_probe.c" -lm
  "$SCRATCH/op_probe" --list > "$SCRATCH/probe_list.tsv"
fi

# ------------------------------------------------------------- mca files
# asm text per probe row; %PG = "gather-friendly" registers.
declare -A MCA_ASM=(
  [vaddss]='vaddss %xmm0,%xmm0,%xmm0'
  [vaddsd]='vaddsd %xmm0,%xmm0,%xmm0'
  [vaddps_x]='vaddps %xmm0,%xmm0,%xmm0'
  [vaddps_y]='vaddps %ymm0,%ymm0,%ymm0'
  [vaddpd_x]='vaddpd %xmm0,%xmm0,%xmm0'
  [vaddpd_y]='vaddpd %ymm0,%ymm0,%ymm0'
  [vsubss]='vsubss %xmm0,%xmm0,%xmm0'
  [vsubsd]='vsubsd %xmm0,%xmm0,%xmm0'
  [vsubps_x]='vsubps %xmm0,%xmm0,%xmm0'
  [vsubps_y]='vsubps %ymm0,%ymm0,%ymm0'
  [vsubpd_x]='vsubpd %xmm0,%xmm0,%xmm0'
  [vsubpd_y]='vsubpd %ymm0,%ymm0,%ymm0'
  [vmulss]='vmulss %xmm0,%xmm0,%xmm0'
  [vmulsd]='vmulsd %xmm0,%xmm0,%xmm0'
  [vmulps_x]='vmulps %xmm0,%xmm0,%xmm0'
  [vmulps_y]='vmulps %ymm0,%ymm0,%ymm0'
  [vmulpd_x]='vmulpd %xmm0,%xmm0,%xmm0'
  [vmulpd_y]='vmulpd %ymm0,%ymm0,%ymm0'
  [vfmadd231ss]='vfmadd231ss %xmm2,%xmm1,%xmm0'
  [vfmadd231sd]='vfmadd231sd %xmm2,%xmm1,%xmm0'
  [vfmadd231ps_x]='vfmadd231ps %xmm2,%xmm1,%xmm0'
  [vfmadd231ps_y]='vfmadd231ps %ymm2,%ymm1,%ymm0'
  [vfmadd231pd_x]='vfmadd231pd %xmm2,%xmm1,%xmm0'
  [vfmadd231pd_y]='vfmadd231pd %ymm2,%ymm1,%ymm0'
  [vdivss]='vdivss %xmm0,%xmm1,%xmm0'
  [vdivsd]='vdivsd %xmm0,%xmm1,%xmm0'
  [vdivps_x]='vdivps %xmm0,%xmm1,%xmm0'
  [vdivps_y]='vdivps %ymm0,%ymm1,%ymm0'
  [vdivpd_x]='vdivpd %xmm0,%xmm1,%xmm0'
  [vdivpd_y]='vdivpd %ymm0,%ymm1,%ymm0'
  [vsqrtss]='vsqrtss %xmm0,%xmm0,%xmm0'
  [vsqrtsd]='vsqrtsd %xmm0,%xmm0,%xmm0'
  [vsqrtps_x]='vsqrtps %xmm0,%xmm0'
  [vsqrtps_y]='vsqrtps %ymm0,%ymm0'
  [vsqrtpd_x]='vsqrtpd %xmm0,%xmm0'
  [vsqrtpd_y]='vsqrtpd %ymm0,%ymm0'
  [vrcpss]='vrcpss %xmm0,%xmm0,%xmm0'
  [vrcpps_x]='vrcpps %xmm0,%xmm0'
  [vrcpps_y]='vrcpps %ymm0,%ymm0'
  [vrsqrtss]='vrsqrtss %xmm0,%xmm0,%xmm0'
  [vrsqrtps_x]='vrsqrtps %xmm0,%xmm0'
  [vrsqrtps_y]='vrsqrtps %ymm0,%ymm0'
  [vminss]='vminss %xmm1,%xmm0,%xmm0'
  [vminsd]='vminsd %xmm1,%xmm0,%xmm0'
  [vminps_x]='vminps %xmm1,%xmm0,%xmm0'
  [vminps_y]='vminps %ymm1,%ymm0,%ymm0'
  [vminpd_x]='vminpd %xmm1,%xmm0,%xmm0'
  [vminpd_y]='vminpd %ymm1,%ymm0,%ymm0'
  [vmaxss]='vmaxss %xmm1,%xmm0,%xmm0'
  [vmaxsd]='vmaxsd %xmm1,%xmm0,%xmm0'
  [vmaxps_x]='vmaxps %xmm1,%xmm0,%xmm0'
  [vmaxps_y]='vmaxps %ymm1,%ymm0,%ymm0'
  [vmaxpd_x]='vmaxpd %xmm1,%xmm0,%xmm0'
  [vmaxpd_y]='vmaxpd %ymm0,%ymm0,%ymm0'
  [vcmpltss]='vcmpltss %xmm1,%xmm0,%xmm0'
  [vcmpltsd]='vcmpltsd %xmm1,%xmm0,%xmm0'
  [vcmpltps_x]='vcmpltps %xmm1,%xmm0,%xmm0'
  [vcmpltps_y]='vcmpltps %ymm1,%ymm0,%ymm0'
  [vcmpltpd_x]='vcmpltpd %xmm1,%xmm0,%xmm0'
  [vcmpltpd_y]='vcmpltpd %ymm1,%ymm0,%ymm0'
  [vandps_x]='vandps %xmm1,%xmm0,%xmm0'
  [vandps_y]='vandps %ymm1,%ymm0,%ymm0'
  [vandpd_x]='vandpd %xmm1,%xmm0,%xmm0'
  [vandpd_y]='vandpd %ymm1,%ymm0,%ymm0'
  [vblendvps_x]='vblendvps %xmm3,%xmm2,%xmm1,%xmm0'
  [vblendvps_y]='vblendvps %ymm3,%ymm2,%ymm1,%ymm0'
  [vblendvpd_x]='vblendvpd %xmm3,%xmm2,%xmm1,%xmm0'
  [vblendvpd_y]='vblendvpd %ymm3,%ymm2,%ymm1,%ymm0'
  [vcvtdq2ps_x]='vcvtdq2ps %xmm0,%xmm0'
  [vcvtdq2ps_y]='vcvtdq2ps %ymm0,%ymm0'
  [vcvttps2dq_x]='vcvttps2dq %xmm0,%xmm0'
  [vcvttps2dq_y]='vcvttps2dq %ymm0,%ymm0'
  [vcvtps2dq_x]='vcvtps2dq %xmm0,%xmm0'
  [vcvtps2dq_y]='vcvtps2dq %ymm0,%ymm0'
  [vcvtps2pd_xx]='vcvtps2pd %xmm0,%xmm0'
  [vcvtps2pd_xy]='vcvtps2pd %xmm0,%ymm0'
  [vcvtpd2ps_xx]='vcvtpd2ps %xmm0,%xmm0'
  [vcvtpd2ps_yx]='vcvtpd2ps %ymm0,%xmm0'
  [vcvtdq2pd_xy]='vcvtdq2pd %xmm0,%ymm0'
  [vcvtpd2dq_xx]='vcvtpd2dq %xmm0,%xmm0'
  [vcvtpd2dq_yx]='vcvtpd2dq %ymm0,%xmm0'
  [vcvtss2sd]='vcvtss2sd %xmm0,%xmm0,%xmm0'
  [vcvtsd2ss]='vcvtsd2ss %xmm0,%xmm0,%xmm0'
  [vpmovzxbd_xx]='vpmovzxbd %xmm0,%xmm0'
  [vpmovzxbd_xy]='vpmovzxbd %xmm0,%ymm0'
  [vpaddd_x]='vpaddd %xmm1,%xmm0,%xmm0'
  [vpaddd_y]='vpaddd %ymm1,%ymm0,%ymm0'
  [vpmulld_x]='vpmulld %xmm1,%xmm0,%xmm0'
  [vpmulld_y]='vpmulld %ymm1,%ymm0,%ymm0'
  [vpshufb_x]='vpshufb %xmm1,%xmm0,%xmm0'
  [vpshufb_y]='vpshufb %ymm1,%ymm0,%ymm0'
  [vgatherdps_y]='vpcmpeqd %ymm2,%ymm2,%ymm2
vgatherdps %ymm2,(%rbx,%ymm0,4),%ymm1'
  [vmovaps_ld_x]='vmovq %xmm0,%rcx
vmovaps (%rbx,%rcx,1),%xmm0'
  [vmovaps_ld_y]='vmovq %xmm0,%rcx
vmovaps (%rbx,%rcx,1),%ymm0'
  [vmovaps_st_x]='vmovaps %xmm0,(%rbx)'
  [vmovaps_st_y]='vmovaps %ymm0,(%rbx)'
  [vmovaps_stlf_x]='vmovaps %xmm0,(%rbx)
vmovaps (%rbx),%xmm0'
  [vmovaps_stlf_y]='vmovaps %ymm0,(%rbx)
vmovaps (%rbx),%ymm0'
  [mov_ld]='mov (%rbx,%rcx,1),%rcx'
  [mov_stlf]='mov %rcx,(%rbx)
mov (%rbx),%rcx'
  [vmovq_rt]='vmovq %xmm0,%rcx
vmovq %rcx,%xmm0'
)

run_mca() { # $1 = cpu, $2 = outfile tag
  local cpu="$1"
  local tsv="$OUT/llvm_mca_${2}.tsv"
  {
    printf 'instruction\tform\twidth_bits\tprecision\tlatency\trthroughput\tsched_model\n'
    tail -n +2 "$SCRATCH/probe_list.tsv" | while IFS=$'\t' read -r name inst form w prec lat_ops tp_n cls hl ht; do
      [[ "$inst" == libm_* || "$inst" == loop_empty ]] && continue
      asm="${MCA_ASM[$name]:-}"
      [[ -z "$asm" ]] && continue
      printf '.loop:\n%s\n\tdec %%rax\n\tjnz .loop\n' "$asm" > "$SCRATCH/mca_$name.s"
      out="$("$MCA" -mcpu="$cpu" "$SCRATCH/mca_$name.s" 2>/dev/null)"
      # instruction-info rows, then pick the row matching the probe mnemonic
      rows="$(printf '%s\n' "$out" | awk '
        /Instructions:$/{seen=1; next}
        seen && $1 ~ /^[0-9]+$/ && NF >= 3 {
          s=$0; sub(/^[0-9]+ +[0-9]+(\.[0-9]+)? +[0-9]+(\.[0-9]+)? +[0-9]+/,"",s);
          print $1"\t"$2"\t"$3"\t"s}')"
      info="$(printf '%s\n' "$rows" | grep -F "$inst" | head -1 || true)"
      [[ -z "$info" ]] && info="$(printf '%s\n' "$rows" | head -1)"
      brt="$(printf '%s\n' "$out" | awk '/^Block RThroughput:/{print $3}')"
      lat="$(printf '%s' "$info" | cut -f2)"
      rt="$(printf '%s' "$info" | cut -f3)"
      model="$(printf '%s\n' "$out" | grep -oE 'ADLPPort|Zn4AGU|Zn5AGU|SKLPort|GLCPort' | head -1 || true)"
      case "$model" in
        ADLPPort) model="${cpu}/ADLP";;
        Zn4*)     model="${cpu}/Zn4";;
        Zn5*)     model="${cpu}/Zn5";;
        SKLPort)  model="${cpu}/SKL";;
        GLCPort)  model="${cpu}/GLC";;
        *)        model="${cpu}/?";;
      esac
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$inst" "$form" "$w" "$prec" "${lat:-NA}" "${rt:-NA}" "$model"
      # keep block rt for the report
      printf '%s\t%s\n' "$name" "${brt:-NA}" >> "$SCRATCH/mca_blockrt_${2}.tsv"
    done
  } > "$tsv"
}

if [[ "$MODE" == "all" || "$MODE" == "mca" ]]; then
  rm -f "$SCRATCH/mca_blockrt_arrowlake.tsv" "$SCRATCH/mca_blockrt_znver5.tsv"
  run_mca arrowlake arrowlake
  run_mca znver5 znver5
fi

# ---------------------------------------------------------------- probes
if [[ "$MODE" == "all" || "$MODE" == "probes" ]]; then
  # iteration counts per class: hi/lo (slope endpoints)
  declare -A LAT_HI=([0]=4000000 [1]=500000 [2]=1000000 [3]=2000000)
  declare -A LAT_LO=([0]=250000  [1]=31250  [2]=62500   [3]=125000)
  declare -A TP_HI=([0]=2000000  [1]=200000 [2]=250000  [3]=2000000)
  declare -A TP_LO=([0]=125000   [1]=12500  [2]=15625   [3]=125000)

  # REPS separate perf stat invocations per point (perf -r in -x, mode prints
  # only the aggregated mean, not per-rep lines). The counter runtime-% field
  # ($4) is logged when <95% so PMU multiplexing shows up in the report.
  PCTLOG="$SCRATCH/perf_low_runtime.log"; : > "$PCTLOG"
  run_one() { # cpu event probe kind iters -> REPS cycle samples on stdout
    local i
    for i in $(seq "$REPS"); do
      taskset -c "$1" perf stat -x, -e "$2" -- \
        "$SCRATCH/op_probe" "$3" "$4" "$5" 2>&1 |
        awk -F, -v tag="$3/$4/$5" '
          /^[0-9]+(\.[0-9]+)?,.*cycles/{
            p=$4; gsub(/%/,"",p);
            if (p+0 > 0 && p+0 < 95)
              printf "low-runtime %s pct=%s\n", tag, p >> "'"$PCTLOG"'";
            print $1}'
    done
  }

  med_min_max() { # samples on stdin -> "med min max"
    sort -n | awk '{a[NR]=$1} END{
      n=NR; m=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
      printf "%.6f\t%.6f\t%.6f\n", m, a[1], a[n]}'
  }

  TSV="$OUT/probes_i265.tsv"
  printf 'probe\tform\twidth_bits\tprecision\tcore_type\tcpu\tkind\tcycles_median\tcycles_min\tcycles_max\trepeats\titerations\n' > "$TSV"

  for core in P E; do
    if [[ $core == P ]]; then cpu=$PCPU; ev=$PEV; else cpu=$ECPU; ev=$EEV; fi
    # empty-loop slopes per class (tp baseline) at hi and lo iters
    declare -A EMPS=()
    for cls in 0 1 2 3; do
      hi="${TP_HI[$cls]}"; lo="${TP_LO[$cls]}"
      h="$(run_one "$cpu" "$ev" loop_empty tp "$hi" | med_min_max | cut -f1)"
      l="$(run_one "$cpu" "$ev" loop_empty tp "$lo" | med_min_max | cut -f1)"
      EMPS[$cls]="$(awk -v h="$h" -v l="$l" -v d="$((hi-lo))" \
                      'BEGIN{printf "%.6f",(h-l)/d}')"
    done

    tail -n +2 "$SCRATCH/probe_list.tsv" | \
    while IFS=$'\t' read -r name inst form w prec lat_ops tp_n cls hl ht; do
      [[ "$name" == loop_empty ]] && continue
      for kind in lat tp; do
        [[ $kind == lat && $hl != 1 ]] && continue
        [[ $kind == tp && $ht != 1 ]] && continue
        if [[ $kind == lat ]]; then
          hi=${LAT_HI[$cls]}; lo=${LAT_LO[$cls]}; div=$lat_ops
        else
          hi=${TP_HI[$cls]}; lo=${TP_LO[$cls]}; div=$tp_n
        fi
        hs="$(run_one "$cpu" "$ev" "$name" "$kind" "$hi")"
        ls="$(run_one "$cpu" "$ev" "$name" "$kind" "$lo")"
        # per-rep slope -> per-op cycles; median/min/max over slope pairs
        read -r med mn mx < <(paste <(printf '%s\n' "$hs") \
            <(printf '%s\n' "$ls") | awk -v d="$((hi-lo))" \
            '{print ($1-$2)/d}' | med_min_max)
        if [[ $kind == tp ]]; then
          med="$(awk -v m="$med" -v e="${EMPS[$cls]}" -v n="$div" \
                 'BEGIN{v=(m-e)/n; printf "%.4f",(v>0)?v:0}')"
          mn="$(awk -v m="$mn" -v e="${EMPS[$cls]}" -v n="$div" \
                'BEGIN{v=(m-e)/n; printf "%.4f",(v>0)?v:0}')"
          mx="$(awk -v m="$mx" -v e="${EMPS[$cls]}" -v n="$div" \
                'BEGIN{v=(m-e)/n; printf "%.4f",(v>0)?v:0}')"
        else
          med="$(awk -v m="$med" -v n="$div" 'BEGIN{printf "%.4f",m/n}')"
          mn="$(awk -v m="$mn" -v n="$div" 'BEGIN{printf "%.4f",m/n}')"
          mx="$(awk -v m="$mx" -v n="$div" 'BEGIN{printf "%.4f",m/n}')"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$name" "$form" "$w" "$prec" "$core" "$cpu" \
          "$([ $kind == lat ] && echo latency || echo rthroughput)" \
          "$med" "$mn" "$mx" "$REPS" "$hi" >> "$TSV"
      done
    done
    unset EMPS; declare -A EMPS=()
  done
fi

# ---------------------------------------------------------------- meta
if [[ "$MODE" == "all" || "$MODE" == "probes" ]]; then
  M_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  M_HOST="$(hostname)"
  M_COMMIT="$(cd "$ROOT" && git log -1 --format=%H 2>/dev/null || true)"
  [[ -z "$M_COMMIT" ]] && M_COMMIT="$(cd "$ROOT" && jj log -r '@-' --no-graph -T commit_id 2>/dev/null || true)"
  M_LSCPU="$(lscpu | awk -F: '/Model name/{sub(/^ +/,"",$2); print $2}')"
  M_CORES="$(cat /sys/devices/cpu_core/cpus)"
  M_ECORES="$(cat /sys/devices/cpu_atom/cpus)"
  M_KERNEL="$(uname -r)"
  M_GCC="$(gcc --version | head -1)"
  M_MCA="$("$MCA" --version | head -2 | tr '\n' ' ')"
  M_LIBC="$(ldd --version | head -1)"
  M_PERF="$(perf --version)"
  M_PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
  M_LOAD="$(cat /proc/loadavg)"
  M_FREE="$(free -h | awk 'NR==2{print "avail=" $7 " used=" $3}')"
  {
    echo "# probes_i265.meta"
    echo "date_utc: $M_DATE"
    echo "host: $M_HOST"
    echo "workspace_commit: $M_COMMIT"
    echo "lscpu: $M_LSCPU"
    echo "cpus: P-core=$PCPU (cpu_core, cpus $M_CORES), E-core=$ECPU (cpu_atom, cpus $M_ECORES)"
    echo "kernel: $M_KERNEL"
    echo "gcc: $M_GCC"
    echo "llvm-mca: $M_MCA"
    echo "libm/libc: $M_LIBC"
    echo "perf: $M_PERF"
    echo "perf_event_paranoid: $M_PARANOID"
    echo "loadavg_at_start: $M_LOAD"
    echo "free_at_start: $M_FREE"
    echo "build: gcc -O2 -static -o $SCRATCH/op_probe scripts/complexity/probes/op_probe.c -lm"
    echo "reps: $REPS separate perf stat invocations per (probe,kind,cpu,iter-point); per-iter cycles = slope(hi,lo)/(hi-lo); tp additionally subtracts loop_empty slope and divides by tp_n accumulators"
    echo "iters: see TSV iterations column (hi point; lo = hi/16)"
    echo "exact commands: taskset -c <cpu> perf stat -x, -e <event> -- op_probe <probe> <lat|tp> <iters>   (x$REPS per point)"
    echo "events: P-core cpu_core/cycles/, E-core cpu_atom/cycles/"
  } > "$OUT/probes_i265.meta"
fi

echo "done: $OUT"
