"""Identity, packing and limit controls for the trade view executor (a fake `evif-trade` binary)."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch
import evif_trade_exec as ex

FAKE = r'''#!/usr/bin/env bash
set -eu
[ "$1" = trade ] || exit 2
spec=$2; out=$3
view=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['views'][0]['id'])" "$spec")
mkdir -p "$out/$view" "$out/maps" "$out/bitstreams"
for i in 1 2 3; do head -c $((1000*i)) /dev/zero | tr '\0' 'x' > "$out/$view/cell-$i.evif"; done
printf 'jpeg-a' > "$out/bitstreams/aa.jpg"
printf 'map-a' > "$out/maps/m1.pfm"
printf '{"view":"%s","kind":"ladder"}\n{"view":"%s","kind":"de"}\n' "$view" "$view" > "$out/ledger.jsonl"
for f in SPEC JUDGES SOURCE_IDENTITIES CONTROL METRIC_CONTROL BYTE_CONTROL COMPLETE; do echo '{}' > "$out/$f.json"; done
cp "$spec" "$out/SPEC.json"
echo '{"ladder":1.5}' > "$out/$view/timing.json"
echo "env $EVIF_E_MODEL $EVIF_ECODEC_MODEL $RAYON_NUM_THREADS" >&2
'''


def png(width, height):
    return b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', width, height) + bytes([8, 2, 0, 0, 0])


class Fixture:
    def __init__(self, root, fake=FAKE):
        self.root = root
        self.bins = root / 'bin'
        self.models = root / 'models'
        self.scratch = root / 'scratch'
        for d in (self.bins, self.models, self.scratch):
            d.mkdir()
        (self.bins / 'evif-trade').write_text(fake)
        (self.bins / 'evif-trade').chmod(0o755)
        names = ['rev7_general_h128_2026-10-27.bin', 'rev7_codec_h128_2026-10-27.bin', 'F-general.bin']
        for n in names:
            (self.models / n).write_bytes(n.encode())
        self.view_png = root / 'v.png'
        self.view_png.write_bytes(png(64, 48) + b'tail')
        self.template = root / 'template.json'
        self.template.write_text(json.dumps(dict(families=['ycc420'], candidates=[
            dict(label='zensim-f', model='F-general.bin', sha256='00' * 32)], views=[])))
        view = dict(id='v-1', source='s', source_path=str(root / 'raw.png'), source_sha256='11' * 32, split='train',
                    role='fit', path=str(self.view_png), sha256=ex.sha(self.view_png.read_bytes()),
                    seeds={}, content_class='x')
        view['class'] = 'photo'
        self.knobs = dict(template_path=str(self.template), template_sha256=ex.sha(self.template.read_bytes()),
                          view=view, binaries={'evif-trade': ex.sha(FAKE.encode())},
                          models={n: ex.sha(n.encode()) for n in names},
                          executor_sha256=ex.sha(Path(ex.__file__).read_bytes()))
        self.env = dict(ZEN_EVIF_BIN_DIR=str(self.bins), ZEN_EVIF_MODEL_DIR=str(self.models), TMPDIR=str(self.scratch))

    def job(self, knobs=None):
        knobs = knobs or self.knobs
        inputs = [ex.sha(self.view_png.read_bytes()), ex.sha(ex.canonical(knobs).encode()), knobs['template_sha256'],
                  knobs['executor_sha256'], *knobs['binaries'].values(), *knobs['models'].values()]
        return dict(kind=ex.VIEW_KIND, inputs=inputs,
                    cell=dict(image_path=str(self.view_png), codec='evif:trade', q=95, knob_tuple_json=ex.canonical(knobs)))


class Executor(unittest.TestCase):
    def tmp(self):
        return tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])

    def test_every_output_is_packed_with_member_hashes_and_the_envelope_is_bounded(self):
        with self.tmp() as td:
            fx = Fixture(Path(td))
            with patch.dict(os.environ, fx.env):
                payload = ex.execute(fx.job())
            self.assertTrue(payload.startswith(b'ZFEVIFSTAGE1\n'))
            envelope = json.loads(payload[len(b'ZFEVIFSTAGE1\n'):])
            root = Path(envelope['root'])
            self.assertEqual(envelope['primary'], 'result.json')
            self.assertLessEqual(len(envelope['objects']) + 1, ex.MAX_OBJECTS)
            names = {o['path'] for o in envelope['objects']}
            self.assertTrue({'ledger.jsonl', 'SPEC.json', 'BYTE_CONTROL.json', 'timing.json'} <= names)
            for o in envelope['objects']:
                self.assertEqual(ex.sha((root / o['path']).read_bytes()), o['sha256'])
            manifest = json.loads((root / 'result.json').read_text())
            self.assertEqual(manifest['image_path'], str(fx.view_png))
            self.assertEqual(manifest['knob_tuple_json'], ex.canonical(fx.knobs))
            self.assertEqual(manifest['rows'], 2)
            kinds = {m['kind'] for m in manifest['members']}
            self.assertEqual(kinds, {'evif', 'bitstreams', 'maps'})
            # every member of every pack matches its recorded hash
            for m in manifest['members']:
                with tarfile.open(root / m['pack']) as tar:
                    self.assertEqual(ex.sha(tar.extractfile(m['name']).read()), m['sha256'])
            # the template named a model file; the spec the binary saw carries the resolved path
            spec = json.loads((root / 'SPEC.json').read_text())
            self.assertEqual(spec['views'], [fx.knobs['view']])
            self.assertTrue(spec['candidates'][0]['model'].endswith('/models/F-general.bin'))

    def test_packs_are_deterministic_and_split_at_the_object_limit(self):
        with self.tmp() as td:
            root = Path(td)
            files = []
            for i in range(5):
                p = root / f'f{i}.bin'
                p.write_bytes(bytes([i]) * 3000)
                files.append((f'v/f{i}.bin', p))
            a, ma = ex.pack(files, 'evif', root)
            first = [ex.sha(p.read_bytes()) for p in a]
            b, mb = ex.pack(list(reversed(files)), 'evif', root)
            self.assertEqual(first, [ex.sha(p.read_bytes()) for p in b])
            self.assertEqual(ma, mb)
            with patch.object(ex, 'PACK_BYTES', 8000):
                packs, members = ex.pack(files, 'small', root)
            self.assertGreater(len(packs), 1)
            self.assertEqual({m['pack'] for m in members}, {p.name for p in packs})
            self.assertTrue(all(p.stat().st_size < 8000 + 4096 for p in packs))

    def test_identity_and_pin_refusals(self):
        with self.tmp() as td:
            fx = Fixture(Path(td))
            base = fx.job()
            with patch.dict(os.environ, fx.env):
                for label in ['kind', 'split', 'template', 'view', 'unbound']:
                    job = copy.deepcopy(base)
                    if label == 'kind':
                        job['kind'] = dict(kind='metric', metric='x')
                    elif label == 'split':
                        knobs = copy.deepcopy(fx.knobs)
                        knobs['view']['split'] = 'validate'
                        job = fx.job(knobs)
                    elif label == 'template':
                        fx.template.write_text('{"views":[]}')
                    elif label == 'view':
                        fx.view_png.write_bytes(png(64, 48) + b'changed')
                    else:
                        job['inputs'] = job['inputs'][:-1]
                    with self.subTest(label=label), self.assertRaises(ValueError):
                        ex.validate(job)
                    fx.template.write_text(json.dumps(dict(families=['ycc420'], candidates=[
                        dict(label='zensim-f', model='F-general.bin', sha256='00' * 32)], views=[])))
                    fx.view_png.write_bytes(png(64, 48) + b'tail')
                # a changed binary or model is refused before anything runs
                (fx.bins / 'evif-trade').write_text(FAKE + '\n# changed')
                with self.assertRaises(ValueError):
                    ex.execute(base)
                (fx.bins / 'evif-trade').write_text(FAKE)
                (fx.models / 'F-general.bin').write_bytes(b'other')
                with self.assertRaises(ValueError):
                    ex.execute(base)

    def test_megapixel_boundary_and_oversize_output_are_refused(self):
        with self.tmp() as td:
            fx = Fixture(Path(td))
            fx.view_png.write_bytes(png(1000, 1000))
            knobs = copy.deepcopy(fx.knobs)
            knobs['view']['sha256'] = ex.sha(fx.view_png.read_bytes())
            with patch.dict(os.environ, fx.env), self.assertRaises(ValueError):
                ex.validate(fx.job(knobs))
        with self.tmp() as td:
            fx = Fixture(Path(td))
            with patch.dict(os.environ, dict(fx.env, ZEN_STAGED_MAX_TOTAL_BYTES='2048')), \
                    self.assertRaisesRegex(RuntimeError, 'exceeds the worker limits'):
                ex.execute(fx.job())
            self.assertEqual(list((fx.scratch / '.staging').glob('evif-trade-*')), [])      # no leaked stage

    def test_a_runaway_view_is_stopped_while_it_runs_and_leaves_nothing_behind(self):
        runaway = FAKE.replace('printf \'jpeg-a\' > "$out/bitstreams/aa.jpg"',
                               'head -c 5000 /dev/zero > "$out/$view/big.evif"; sleep 60')
        with self.tmp() as td:
            fx = Fixture(Path(td), fake=runaway)
            knobs = copy.deepcopy(fx.knobs)
            knobs['binaries']['evif-trade'] = ex.sha((fx.bins / 'evif-trade').read_bytes())
            with patch.dict(os.environ, dict(fx.env, ZEN_STAGED_MAX_TOTAL_BYTES='2048')), \
                    patch.object(ex, 'POLL_SECONDS', 0.05):
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, 'exceeds the worker limits'):
                    ex.execute(fx.job(knobs))
                self.assertLess(time.monotonic() - started, 30)       # stopped long before the 60 s sleep ends
            self.assertEqual(list((fx.scratch / '.staging').glob('evif-trade-*')), [])

    def test_tree_bytes_counts_every_file_below_a_directory(self):
        with self.tmp() as td:
            root = Path(td)
            (root / 'a').mkdir()
            (root / 'a' / 'x').write_bytes(b'12345')
            (root / 'y').write_bytes(b'67')
            self.assertEqual(ex.tree_bytes(root), 7)
            self.assertEqual(ex.tree_bytes(root / 'missing'), 0)

    def test_a_failed_trade_run_raises_and_retains_its_stage(self):
        with self.tmp() as td:
            fx = Fixture(Path(td), fake='#!/usr/bin/env bash\necho boom >&2\nexit 7\n')
            knobs = copy.deepcopy(fx.knobs)
            knobs['binaries']['evif-trade'] = ex.sha(Path(fx.bins / 'evif-trade').read_bytes())
            with patch.dict(os.environ, fx.env), self.assertRaisesRegex(RuntimeError, 'boom'):
                ex.execute(fx.job(knobs))
            stages = list((fx.scratch / '.staging').glob('evif-trade-*'))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / 'trade.stderr').exists())          # the evidence stays
            self.assertFalse((stages[0] / 'run').exists())                  # the partial output does not


if __name__ == '__main__':
    unittest.main()
