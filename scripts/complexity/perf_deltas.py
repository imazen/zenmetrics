#!/usr/bin/env python3
"""Per-call CPU counters for CPU metrics, excluding decode, setup and process start.

For each metric and shape, runs `complexity-reps <shape> <metric> <n>` under
`perf stat` with n = 1 and n = 1 + K, pinned to one core with one Rayon
thread, and reports (counters(1+K) - counters(1)) / K. Repeats each pair and
keeps every repeat; the median is the reported value. Method:
docs/IQA_COMPLEXITY.md ("Measured comparison").

  perf_deltas.py --bin target/release/complexity-reps --out FILE.tsv \
      --shapes 1024x1024 --metrics psnr,ssim,... [--extra 2] [--repeats 3] [--cpu 2]

Run it under run-heavy. On i265 (hybrid Arrow Lake) the events are the P-core
PMU's (`cpu_core/...`), so --cpu must be a P-core (0-7).
"""
import argparse, os, signal, statistics, subprocess, sys, time
from pathlib import Path

EVENTS = ['task-clock', 'cpu_core/cycles/', 'cpu_core/instructions/',
          'cpu_core/fp_arith_inst_retired.vector_128b/', 'cpu_core/fp_arith_inst_retired.vector_256b/',
          'cpu_core/mem_inst_retired.all_loads/', 'cpu_core/mem_inst_retired.all_stores/',
          'cpu_core/arith.fpdiv_active/']
NAMES = ['task_clock_ms', 'cycles', 'instructions', 'fp_vec128_inst', 'fp_vec256_inst', 'loads', 'stores',
         'fpdiv_active_cycles']
COLS = ['metric', 'shape', 'threads', 'repeat', 'extra_reps'] + [n + '_per_call' for n in NAMES + ['wall_ms']] + ['score']


def prepare_output(path):
    """Add a missing header, retaining every original byte in a sibling backup."""
    path = Path(path)
    original = path.read_bytes() if path.exists() else b''
    header = '\t'.join(COLS).encode() + b'\n'
    if any(line == header.rstrip(b'\n') for line in original.splitlines()):
        return
    if original:
        backup = path.with_name(path.name + '.headerless.bak')
        # Never replace a previous backup; refuse conflicting recovery attempts.
        with backup.open('xb') as saved:
            saved.write(original)
    path.write_bytes(header + original)


def append_error(path, metric, threads, extra, reason, shape='3840x2160'):
    prepare_output(path)
    with open(path, 'a') as out:
        out.write('\t'.join([metric, shape, str(threads), '0', str(extra)]
                            + ['error'] * (len(NAMES) + 1) + [reason]) + '\n')


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def run(bin_, shape, metric, reps, cpu, threads=1, timeout=None):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), RTEN_NUM_THREADS=str(threads))
    cmd = ['perf', 'stat', '-x', ',', '-e', ','.join(EVENTS), 'taskset', '-c', str(cpu), bin_, shape, metric, str(reps)]
    t0 = time.monotonic()
    # Own process group, so a timeout kills perf and the metric process under it.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException as error:
        # An outer hold timeout also reaches here via main's signal handler.
        # Only this Popen's session is ours; never scan other benchmark PIDs.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise RuntimeError(f'{metric} {shape} reps={reps}: timeout after {timeout} s') from error
        raise
    p = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    wall = time.monotonic() - t0
    if p.returncode:
        raise RuntimeError(f'{metric} {shape} reps={reps}: rc={p.returncode}\n{p.stderr[-800:]}')
    values, score = {}, None
    for line in p.stderr.splitlines():
        parts = line.split(',')
        if len(parts) > 3 and parts[2] in EVENTS:
            if parts[0] in ('<not counted>', '<not supported>', ''):
                raise RuntimeError(f'{metric}: event {parts[2]} {parts[0]}')
            values[NAMES[EVENTS.index(parts[2])]] = float(parts[0])
        elif line.startswith(metric + ' '):
            score = line.split('score=')[-1]
    values['wall_ms'] = wall * 1e3
    return values, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shapes', default='1024x1024')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--extra', type=int, default=0, help='fixed extra reps; 0 = adaptive (delta covers >= --target-s)')
    ap.add_argument('--target-s', type=float, default=0.3)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--cpu', default='2', help='taskset CPU list, P-cores only on i265 (0-7)')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--timeout-s', type=float, default=None, help='skip a metric (recorded as error) when one probe run exceeds this')
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    prepare_output(a.out)
    with open(a.out, 'a') as out:
        for shape in a.shapes.split(','):
            for metric in a.metrics.split(','):
                extra = a.extra
                if not extra:
                    try:
                        one, _ = run(a.bin, shape, metric, 1, a.cpu, a.threads, a.timeout_s)
                        four, _ = run(a.bin, shape, metric, 4, a.cpu, a.threads, a.timeout_s and 4 * a.timeout_s)
                    except RuntimeError as e:
                        print('FAIL', e, file=sys.stderr, flush=True)
                        out.write('\t'.join([metric, shape, str(a.threads), '0', '0'] + ['error'] * (len(NAMES) + 1) + [str(e).splitlines()[0]]) + '\n')
                        out.flush()
                        continue
                    per = max((four['task_clock_ms'] - one['task_clock_ms']) / 3, 1e-3)
                    extra = int(min(5000, max(2, -(-a.target_s * 1e3 // per))))
                for rep in range(a.repeats):
                    try:
                        one, s1 = run(a.bin, shape, metric, 1, a.cpu, a.threads, a.timeout_s)
                        many, s2 = run(a.bin, shape, metric, 1 + extra, a.cpu, a.threads, a.timeout_s and (1 + extra) * a.timeout_s)
                    except RuntimeError as e:
                        print('FAIL', e, file=sys.stderr, flush=True)
                        out.write('\t'.join([metric, shape, str(a.threads), str(rep), str(extra)] + ['error'] * (len(NAMES) + 1) + [str(e).splitlines()[0]]) + '\n')
                        out.flush()
                        break
                    if s1 != s2:
                        print(f'WARN {metric}: score differs between runs {s1} vs {s2}', file=sys.stderr)
                    delta = [(many[n] - one[n]) / extra for n in NAMES + ['wall_ms']]
                    out.write('\t'.join([metric, shape, str(a.threads), str(rep), str(extra)] + [f'{v:.6g}' for v in delta] + [str(s2)]) + '\n')
                    out.flush()
                    print(metric, shape, rep, f'task_clock {delta[0]:.2f} ms/call', flush=True)


if __name__ == '__main__':
    main()
