"""Feature transport must reject changed declarations before kernel invocation."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hdr_features_exec as hdr


class FeatureIdentity(unittest.TestCase):
    def test_controls_source_and_extractor_are_bound(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            root = Path(td)
            reference = root / 'reference.png'
            reference.write_bytes(b'native source identity fixture')
            binary = root / 'hdr944_extract'
            binary.write_bytes(b'pinned extractor identity fixture')
            controls = dict(reference_sha256=hdr.sha(reference.read_bytes()), candidates=[hdr.sha(b'candidate')],
                            extractor_sha256=hdr.sha(binary.read_bytes()), extractor_build_commit='build',
                            executor_sha256=hdr.sha(Path(hdr.__file__).read_bytes()),
                            input_contract='hdr-common-primaries-v2-cicp-pq10000', candidate_paths=[])
            job = dict(cell=dict(codec='hdr:features', image_path=str(reference), knob_tuple_json=hdr.canonical(controls)),
                       kind=dict(kind='bake', view=hdr.CAPABILITY),
                       inputs=[controls['reference_sha256'], *controls['candidates'], controls['extractor_sha256'],
                               controls['executor_sha256'], hdr.sha(hdr.canonical(controls).encode()), hdr.sha(b'build')])
            with patch.dict(os.environ, ZEN_HDR_BIN_DIR=str(root)):
                self.assertEqual(hdr.validate(job)[0], controls)
                for field in ['candidates', 'extractor_build_commit', 'reference_sha256']:
                    changed = copy.deepcopy(controls)
                    changed[field] = [hdr.sha(b'other')] if field == 'candidates' else hdr.sha(b'other')
                    forged = copy.deepcopy(job)
                    forged['cell']['knob_tuple_json'] = hdr.canonical(changed)
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        hdr.validate(forged)
                binary.write_bytes(b'changed implementation')
                with self.assertRaises(ValueError):
                    hdr.validate(job)
                binary.write_bytes(b'pinned extractor identity fixture')
                reference.write_bytes(b'changed source')
                with self.assertRaises(ValueError):
                    hdr.validate(job)

    def test_rgbe_cannot_be_staged_as_a_generated_png(self):
        controls = dict(reference_sha256='0'*64, candidates=['1'*64], extractor_sha256='2'*64,
                        extractor_build_commit='build', executor_sha256='3'*64,
                        input_contract='rousselot-registered-k179-as-is', candidate_paths=[])
        with self.assertRaisesRegex(ValueError, 'read in place'):
            hdr.validate(dict(cell=dict(codec='hdr:features', image_path='unused', knob_tuple_json=hdr.canonical(controls)),
                              kind=dict(kind='bake', view=hdr.CAPABILITY), inputs=[]))

    def test_actual_research_manifest_build_field_is_required(self):
        # Field spelling/types from the native Rust producer, not wrapper aliases.
        research = dict(feature_set_id=hdr.FEATURE_ID, layout_width=1825,
                        formula_revision='Rev7', zensim_build_commit='source-snapshot')
        controls = dict(extractor_build_commit='source-snapshot')
        hdr.verify_native_producer(dict(research=research), controls)
        broken = dict(research)
        broken['build_commit'] = broken.pop('zensim_build_commit')
        with self.assertRaises(ValueError):
            hdr.verify_native_producer(dict(research=broken), controls)
        with self.assertRaises(ValueError):
            hdr.verify_native_producer(dict(research=research), dict(extractor_build_commit='other'))

    def test_old_or_rgb8_capability_is_refused(self):
        for codec, view in [('evif:features', hdr.CAPABILITY), ('hdr:features', 'hdr-native-rev5-features')]:
            with self.subTest(codec=codec, view=view), self.assertRaises(ValueError):
                hdr.validate(dict(cell=dict(codec=codec), kind=dict(kind='bake', view=view)))

    def test_png_ladder_candidates_read_in_place_one_encoding_per_job(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            root = Path(td)
            lib = root / 'lib'
            lib.mkdir()
            binary = root / 'hdr944_extract'
            binary.write_bytes(b'pinned extractor identity fixture')
            reference = lib / 'ref.png'
            reference.write_bytes(b'ref')
            files = {n: lib / n for n in ['a.png', 'b.png', 'c.jxl']}
            for n, f in files.items():
                f.write_bytes(n.encode())
            outside = root / 'out.png'
            outside.write_bytes(b'out')

            def job(paths):
                controls = dict(reference_sha256=hdr.sha(b'ref'), candidates=[hdr.sha(p.read_bytes()) for p in paths],
                                extractor_sha256=hdr.sha(binary.read_bytes()), extractor_build_commit='build',
                                executor_sha256=hdr.sha(Path(hdr.__file__).read_bytes()),
                                input_contract='hdr-common-primaries-v2-cicp-pq10000', candidate_paths=[str(p) for p in paths])
                return dict(cell=dict(codec='hdr:features', image_path=str(reference), knob_tuple_json=hdr.canonical(controls)),
                            kind=dict(kind='bake', view=hdr.CAPABILITY),
                            inputs=[controls['reference_sha256'], *controls['candidates'], controls['extractor_sha256'],
                                    controls['executor_sha256'], hdr.sha(hdr.canonical(controls).encode()), hdr.sha(b'build')])
            with patch.dict(os.environ, ZEN_HDR_BIN_DIR=str(root), ZEN_HDR_LIBRARY_DIR=str(lib)):
                hdr.validate(job([files['a.png'], files['b.png']]))
                hdr.validate(job([files['c.jxl']]))
                for bad in ([files['a.png'], files['c.jxl']], [outside]):
                    with self.subTest(bad=[p.name for p in bad]), self.assertRaises(ValueError):
                        hdr.validate(job(bad))


if __name__ == '__main__':
    unittest.main()
