"""Benchmark scheduling policy; explicit metric names always remain available."""
import sys

OPT_IN = frozenset({'pieapp', 'pieapp-pyiqa'})


def select_metrics(request, candidates, *, learned=()):
    """Exclude costly metrics only when expanding a default or group selection."""
    if request not in (None, 'all', 'learned'):
        return set(request.split(','))
    selected = set(learned if request == 'learned' else candidates)
    omitted = selected & OPT_IN
    if omitted:
        print('BENCHMARK OMITTED (slow; explicit selection required): '
              + ','.join(sorted(omitted)), file=sys.stderr, flush=True)
    return selected - OPT_IN
