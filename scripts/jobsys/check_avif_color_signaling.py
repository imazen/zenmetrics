#!/usr/bin/env python3
"""External-decoder regression gate for full-range BT.601/sRGB AVIF sweep output.

Run explicitly via `just avif-color-regression FILE [--sha256 HASH]`.
Missing files/tools and decode errors fail; there are no skipped assertions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


class AvifColorRegression(unittest.TestCase):
    artifact = None
    expected_sha = None
    ffprobe = "ffprobe"
    ffmpeg = "ffmpeg"

    @classmethod
    def setUpClass(cls):
        data = cls.artifact.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if cls.expected_sha is not None and sha != cls.expected_sha:
            raise ValueError(f"fixture SHA256 mismatch: expected {cls.expected_sha}, got {sha}")
        print(f"artifact_sha256={sha}", flush=True)
        probe = subprocess.run(
            [cls.ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries",
             "-of", "json", str(cls.artifact)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        streams = json.loads(probe.stdout)["streams"]
        if len(streams) != 1 or streams[0].get("codec_name") != "av1":
            raise ValueError(f"expected one selected AV1 stream, got {streams}")
        cls.stream = streams[0]
        width, height = cls.stream["width"], cls.stream["height"]
        if not (0 < width * height <= 1_000_000):
            raise ValueError("this regression runner requires a positive sub-1MP image")
        cls.rgb_len = width * height * 3
        print(json.dumps(cls.stream, sort_keys=True), flush=True)

    def decode_rgb(self, force_declared_color):
        cmd = [self.ffmpeg, "-v", "error", "-nostdin", "-threads", "1",
               "-i", str(self.artifact), "-map", "0:v:0", "-filter_threads", "1"]
        if force_declared_color:
            cmd += ["-vf", "scale=in_range=pc:out_range=pc:in_color_matrix=bt601"]
        cmd += ["-frames:v", "1", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        self.assertEqual(len(result.stdout), self.rgb_len, "RGB output geometry changed")
        return result.stdout

    def test_av1_declares_input_color(self):
        # These are the input conversion and container declarations of the SDR arm.
        expected = dict(color_range="pc", color_space="smpte170m",
                        color_transfer="iec61966-2-1", color_primaries="bt709")
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(self.stream.get(key), value,
                                 f"AV1 {key} must describe the full-range BT.601/sRGB input")

    def test_normal_decode_matches_explicit_input_color(self):
        normal = self.decode_rgb(False)
        explicit = self.decode_rgb(True)
        changed = sum(a != b for a, b in zip(normal, explicit))
        maximum = max(abs(a - b) for a, b in zip(normal, explicit))
        print(f"different_RGB_components={changed}/{len(normal)} max_abs_delta={maximum}", flush=True)
        self.assertEqual(changed, 0,
                         f"normal vs explicit full-range BT.601 decode: {changed} RGB components differ; max delta {maximum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("avif", type=Path)
    parser.add_argument("--sha256", help="required identity for a historical witness, omitted for fresh outputs")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    AvifColorRegression.artifact = args.avif.resolve(strict=True)
    AvifColorRegression.expected_sha = args.sha256
    AvifColorRegression.ffprobe = args.ffprobe
    AvifColorRegression.ffmpeg = args.ffmpeg
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AvifColorRegression)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
