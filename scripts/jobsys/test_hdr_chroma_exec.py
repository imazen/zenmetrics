"""Refusal controls for scientific identity before any kernel is invoked."""
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import hdr_chroma_exec as hdr


class Identity(unittest.TestCase):
    def test_source_controls_and_binary_pins_are_required(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            root = Path(td)
            raw = b'\x89PNG\r\n\x1a\n'+struct.pack('>I', 13)+b'IHDR'+struct.pack('>II', 32, 32)+bytes([16, 2])
            source = root/'source.png'; source.write_bytes(raw)
            bins = {'chroma-hdr-render': hdr.sha(b'render'), 'chroma-natural-hdr': hdr.sha(b'teacher')}
            for name, data in [('chroma-hdr-render', b'render'), ('chroma-natural-hdr', b'teacher')]:
                (root/name).write_bytes(data)
            knobs = dict(control=False, width=32, height=32, source_sha256=hdr.sha(raw),
                         source_identity='committed-TRAIN/im1064', binaries=bins,
                         executor_sha256=hdr.sha(Path(hdr.__file__).read_bytes()))
            inputs = [hdr.sha(raw), hdr.sha(hdr.canonical(knobs).encode()), hdr.sha(knobs['source_identity'].encode()), knobs['executor_sha256'], *bins.values()]
            job = dict(kind=dict(kind='bake', view='hdr-chroma-view-v1'), inputs=inputs,
                       cell=dict(image_path=str(source), codec='hdr-chroma:avif-svt', q=-1, knob_tuple_json=hdr.canonical(knobs)))
            with patch.dict(os.environ, ZEN_HDR_BIN_DIR=str(root)):
                self.assertEqual(hdr.validate(job)[0], knobs)
                for label in ['source', 'controls', 'implementation', 'kind']:
                    forged = copy.deepcopy(job)
                    if label == 'source':
                        source.write_bytes(raw+b'changed')
                    elif label == 'controls':
                        changed = dict(knobs, control=True)
                        forged['cell']['knob_tuple_json'] = hdr.canonical(changed)
                    elif label == 'implementation':
                        (root/'chroma-natural-hdr').write_bytes(b'changed teacher')
                    else:
                        forged['kind'] = dict(kind='metric', metric='hdrvdp3')
                    with self.subTest(label=label), self.assertRaises(ValueError):
                        hdr.validate(forged)
                    source.write_bytes(raw)
                    (root/'chroma-natural-hdr').write_bytes(b'teacher')

    def test_unbound_dimensions_and_rgb8_are_refused(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            path = Path(td)/'bad.png'
            for width, depth in [(1000, 16), (32, 8)]:
                path.write_bytes(b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II', width, 1000)+bytes([depth,2]))
                knobs=dict(control=False,width=width,height=1000,source_sha256='',source_identity='',binaries={},executor_sha256='')
                job=dict(kind=dict(kind='bake',view='hdr-chroma-view-v1'),inputs=[],cell=dict(image_path=str(path),codec='hdr-chroma:avif-svt',q=-1,knob_tuple_json=json.dumps(knobs)))
                with self.assertRaises(ValueError):
                    hdr.validate(job)


if __name__ == '__main__':
    unittest.main()
