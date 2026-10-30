#!/usr/bin/env python3
"""Write the `zenfleet-ctl declare-views` spec for trade view jobs (executor: evif_trade_exec.py).

Each job is one prepared view of the chroma-ladder Stage B grid. The partition rule is the EVIF one:
SHA-256 of "{width}x{height}:{SHA-256 of the original decoded pixels}", first eight bytes big-endian,
modulo five; zero is the gauntlet. Only TRAIN roles (fit, new-train) are declared unless asked.

The worker stages at most 1 GiB per job (512 objects, 32 MiB each), and a gauntlet view of about
0.9 MP writes 1.1-2.3 GB of EVIFs. `--split MIN_PIXELS:PARTS` therefore declares a view of at least
MIN_PIXELS as PARTS jobs, each running a disjoint, load-balanced subset of the trade families on the
same view. The part templates are the base template with `families` (and the byte controls of those
families) narrowed; they are written next to `--parts-dir` and pinned by hash like any template.
"""
import argparse
import hashlib
import json
from pathlib import Path

CLASS = {
    'cid22-train': 'photo', '1000-lilith-photos-general': 'photo', '1200-lilith-interiors': 'photo',
    '1400-lilith-nature': 'photo', '1600-lilith-food': 'photo', '2000-unsplash-people': 'photo',
    '2200-unsplash-renders': 'illustration_render', '2400-unsplash-textures': 'texture',
    '3000-art-institute-of-chicago-photos': 'photo', '3300-met-museum-photos': 'photo',
    '5000-national-park-service-brochures': 'lineart_scan', '5200-epa-climate-impact-2021-report': 'lineart_scan',
    '5300-noaa-hurricane-documents': 'lineart_scan', '6000-lilith-scans-public-patents': 'lineart_scan',
    '6600-ia-scans-manuscript-illustrations': 'lineart_scan', '6800-ia-scans-manuscript-text': 'lineart_scan',
    '7000-lilith-plots': 'screen_plot', '8000-lilith-mobile-screenshots': 'screen_plot',
    '8100-lilith-web-screenshots': 'screen_plot', '9000-lilith-ai-clipart': 'illustration_render',
    '9094-lilith-ai-illustrations': 'illustration_render', '9226-lilith-ai-products': 'photo',
    'safesyn-train': 'safesyn',
}


# Share of a view's output per trade family (percent of ledger rows that persist an EVIF, from the
# 68-view pilot). Only used to balance the parts and to order the jobs; it is not a bound.
WEIGHT = {'xyb-extended': 23, 'ycc444': 18, 'ycc420': 18, 'svt-420': 18, 'xyb-capped': 17}
BASELINE_WEIGHT = 6


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def split_families(families, parts):
    """`parts` disjoint family groups covering `families`: heaviest first into the lightest group."""
    if not 1 <= parts <= len(families):
        raise ValueError(f'cannot split {len(families)} families into {parts} parts')
    groups, load = [[] for _ in range(parts)], [0] * parts
    for family in sorted(families, key=lambda f: (-WEIGHT.get(f, 1), f)):
        k = min(range(parts), key=lambda i: (load[i], i))
        groups[k].append(family)
        load[k] += WEIGHT.get(family, 1)
    return [tuple(sorted(g, key=families.index)) for g in sorted(groups, key=lambda g: families.index(g[0]))]


def part_template(base, group):
    """The base template restricted to the trade families in `group`."""
    part = dict(base)
    part['families'] = list(group)
    part['byte_controls'] = [c for c in base.get('byte_controls', []) if c['family'] in group]
    return part


def parts_for(pixels, rules):
    """Number of parts for a view of `pixels`: the largest rule whose minimum it reaches, else one."""
    return max([n for minimum, n in rules if pixels >= minimum], default=1)


def partition(width, height, pixels_sha256):
    identity = f'{width}x{height}:{pixels_sha256}'
    return 'gauntlet' if int(sha(identity.encode())[:16], 16) % 5 == 0 else 'training-leg'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('views', type=Path, help='Stage B VIEWS.json')
    p.add_argument('sources', type=Path, help='Stage B SOURCES.json')
    p.add_argument('template', type=Path, help='trade template spec (no views; candidates name model files)')
    p.add_argument('output', type=Path, help='new declare-views spec')
    p.add_argument('--executor', type=Path, default=Path(__file__).with_name('evif_trade_exec.py'))
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--model', action='append', default=[], metavar='NAME=SHA256', required=True)
    p.add_argument('--roles', nargs='+', default=['fit', 'new-train'])
    p.add_argument('--partition', choices=['gauntlet', 'training-leg', 'all'], default='gauntlet')
    p.add_argument('--max-views', type=int)
    p.add_argument('--split', action='append', default=[], metavar='MIN_PIXELS:PARTS',
                   help='declare views of at least MIN_PIXELS as PARTS family-subset jobs (repeatable)')
    p.add_argument('--parts-dir', type=Path, help='where the part templates are written (a path the workers can read)')
    p.add_argument('--base-mem-bytes', type=int, default=400 * 1024 * 1024)
    p.add_argument('--mem-bytes-per-pixel', type=int, default=2048)
    a = p.parse_args()
    views = json.loads(a.views.read_text())
    sources = {s['source']['reference_group']: s for s in json.loads(a.sources.read_text())}
    template_bytes = a.template.read_bytes()
    base = json.loads(template_bytes)
    if base.get('views'):
        raise SystemExit('the template must not carry views')
    rules = sorted((int(m), int(n)) for m, n in (r.split(':') for r in a.split))
    if rules and not a.parts_dir:
        raise SystemExit('--split needs --parts-dir')
    templates = {tuple(base['families']): (a.template, template_bytes)}

    def template_for(group):
        if group not in templates:
            data = (canonical(part_template(base, group)) + '\n').encode()
            path = (a.parts_dir / f"{a.template.stem}-{'+'.join(group)}.json").resolve()
            if path.exists() and path.read_bytes() != data:
                raise SystemExit(f'{path} exists with different content')
            if not path.exists():
                path.write_bytes(data)
            templates[group] = (path, data)
        return templates[group]
    models = dict(m.split('=', 1) for m in a.model)
    executor = sha(a.executor.read_bytes())
    out = []
    for v in views:
        if v['mlic_role'] not in a.roles:
            continue
        s = sources[v['reference_group']]
        part = partition(s['width'], s['height'], s['source_pixels_sha256'])
        if a.partition != 'all' and part != a.partition:
            continue
        raw = Path(v['path']).read_bytes()
        if sha(raw) != v['sha256']:
            raise SystemExit(f"view bytes differ: {v['id']}")
        view = {'id': v['id'], 'source': v['reference_group'], 'source_path': s['source']['worker_raw'],
                'source_sha256': s['raw_sha256'], 'class': CLASS[v['content_class']],
                'content_class': v['content_class'], 'split': 'train', 'role': v['mlic_role'],
                'path': v['path'], 'sha256': v['sha256'], 'seeds': {}}
        pixels = v['width'] * v['height']
        groups = split_families(base['families'], parts_for(pixels, rules)) if rules else [tuple(base['families'])]
        for group in groups:
            path, data = template_for(group)
            knobs = {'template_path': str(path.resolve()), 'template_sha256': sha(data), 'view': view,
                     'binaries': {'evif-trade': a.binary_sha256}, 'models': models, 'executor_sha256': executor}
            inputs = [v['sha256'], sha(canonical(knobs).encode()), sha(data), executor, a.binary_sha256, *models.values()]
            work = pixels * (sum(WEIGHT.get(f, 1) for f in group) + BASELINE_WEIGHT)
            out.append({'view': 'evif-trade-view-v1',
                        'cell': {'image_path': v['path'], 'codec': 'evif:trade', 'q': 95, 'knob_tuple_json': canonical(knobs)},
                        'inputs': inputs, 'hint': {'peak_mem_bytes': a.base_mem_bytes + a.mem_bytes_per_pixel * pixels, 'threads': 1, 'vram_bytes': 0},
                        'requires': ['evif-trade-view-v1'], '_work': work})
    # Longest first: with ZEN_CLAIM_ORDER=manifest the fleet claims big jobs first and ends on short ones.
    out.sort(key=lambda j: (-j['_work'], j['cell']['image_path'], j['cell']['knob_tuple_json']))
    if a.max_views:
        out = out[:a.max_views]
    for job in out:
        del job['_work']
    with a.output.open('x') as f:
        json.dump({'schema': 1, 'views': out}, f)
    print(json.dumps({'jobs': len(out), 'views': len({j['cell']['image_path'] for j in out}),
                      'partition': a.partition, 'roles': a.roles, 'split': rules}))


if __name__ == '__main__':
    main()
