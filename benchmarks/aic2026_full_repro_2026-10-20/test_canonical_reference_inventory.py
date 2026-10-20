import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from canonical_reference_inventory import inventory, package_bindings


class ReferenceInventory(unittest.TestCase):
    def test_captured_package_and_expected_asset_bytes_share_one_binding_format(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            source, asset = root/'reference.py', root/'weights.bin'
            source.write_text('reference implementation\n')
            asset.write_bytes(b'fixture asset bytes')
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            observed = inventory({str(source): None, str(asset): digest})
            self.assertEqual(observed[str(source)], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(observed[str(asset)], digest)
            asset.write_bytes(b'changed asset')
            with self.assertRaisesRegex(ValueError, 'differs from expected bytes'):
                inventory({str(asset): digest})

    def test_relative_sources_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'must be absolute'):
            inventory({'relative/reference.py': None})

    def test_missing_source_is_not_skipped(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            with self.assertRaises(FileNotFoundError):
                inventory({str(Path(directory)/'missing.py'): None})

    def test_empty_inventory_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'must contain files'):
            inventory({})

    def test_distribution_metadata_and_sources_are_captured_without_importing_package(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root = Path(directory)
            package = root/'canonical_inventory_fixture'
            package.mkdir()
            (package/'__init__.py').write_text('raise AssertionError("must not import reference package")\n')
            (package/'libmath.so.1').write_bytes(b'fixture shared library')
            (package/'vocabulary.txt.gz').write_bytes(b'fixture vocabulary')
            info = root/'canonical_inventory_fixture-0.0.1.dist-info'
            info.mkdir()
            (info/'METADATA').write_text('Name: canonical-inventory-fixture\nVersion: 0.0.1\n'
                                       'Project-URL: Source, https://example.com/reference\n\n')
            (info/'RECORD').write_text('canonical_inventory_fixture/__init__.py,,\n'
                                     'canonical_inventory_fixture/libmath.so.1,,\n'
                                     'canonical_inventory_fixture/vocabulary.txt.gz,,\n'
                                     'canonical_inventory_fixture-0.0.1.dist-info/METADATA,,\n'
                                     'canonical_inventory_fixture-0.0.1.dist-info/RECORD,,\n')
            with patch.object(sys, 'path', [str(root), *sys.path]):
                bindings, packages = package_bindings(['canonical-inventory-fixture'])
            self.assertEqual(packages['canonical-inventory-fixture']['version'], '0.0.1')
            self.assertEqual(packages['canonical-inventory-fixture']['project_urls'],
                             ['Source, https://example.com/reference'])
            self.assertEqual(set(bindings), {str(package/'__init__.py'), str(package/'libmath.so.1'),
                                            str(package/'vocabulary.txt.gz'), str(info/'METADATA'), str(info/'RECORD')})
            self.assertNotIn('canonical_inventory_fixture', sys.modules)


if __name__ == '__main__':
    unittest.main()
