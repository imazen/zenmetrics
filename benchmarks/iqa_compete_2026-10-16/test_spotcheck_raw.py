import tempfile
from pathlib import Path
import unittest

from prepare_spotcheck_raw import crop_rgb8


class RawCrop(unittest.TestCase):
    def test_packed_and_strided_rows_select_the_same_active_pixels(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp', prefix='raw-crop-test-') as directory:
            base = Path(directory)
            # RGB values encode coordinates; 0xFF padding must never enter output.
            rows = [bytes(v for x in range(4) for v in (x, y, x + y)) for y in range(3)]
            packed = base / 'packed.rgb'
            strided = base / 'strided.rgb'
            packed.write_bytes(b''.join(rows))
            strided.write_bytes(b''.join(row + b'\xff' * 5 for row in rows))
            expected = bytes([1, 1, 2, 2, 1, 3, 1, 2, 3, 2, 2, 4])
            self.assertEqual(crop_rgb8(packed, 4, 3, 12, 1, 1, 2, 2), expected)
            self.assertEqual(crop_rgb8(strided, 4, 3, 17, 1, 1, 2, 2), expected)
            self.assertEqual(crop_rgb8(packed, 4, 3, 12, 0, 1, 4, 2), b''.join(rows[1:]))
            self.assertEqual(crop_rgb8(strided, 4, 3, 17, 0, 1, 4, 2), b''.join(rows[1:]))
            self.assertEqual(packed.read_bytes(), b''.join(rows))
            self.assertEqual(strided.read_bytes(), b''.join(row + b'\xff' * 5 for row in rows))

    def test_invalid_geometry_and_short_rows_fail(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp', prefix='raw-crop-test-') as directory:
            path = Path(directory) / 'short.rgb'
            path.write_bytes(bytes(35))
            with self.assertRaises(ValueError):
                crop_rgb8(path, 4, 3, 12, 0, 0, 4, 3)
            path.write_bytes(bytes(36))
            with self.assertRaises(ValueError):
                crop_rgb8(path, 4, 3, 11, 0, 0, 1, 1)
            with self.assertRaises(ValueError):
                crop_rgb8(path, 4, 3, 12, 3, 0, 2, 1)


if __name__ == '__main__':
    unittest.main()
