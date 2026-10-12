#!/usr/bin/env python3
"""Audit the stored libvmaf low-pass coefficients with exact rational math."""

import hashlib
import re
import struct
from fractions import Fraction
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def minor(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def main():
    source = Path(__file__).resolve().parents[2] / "crates/msssim/src/libvmaf.rs"
    raw = source.read_bytes()
    table = raw.decode().split("const LPF9:", 1)[1].split(" = [", 1)[1].split("\n];", 1)[0]
    values = [f32(float(v)) for v in re.findall(r"-?\d+\.\d+", table)]
    assert len(values) == 81, "LPF9 must be a 9x9 coefficient table"
    matrix = [[Fraction(v) for v in values[y * 9 : (y + 1) * 9]] for y in range(9)]
    # Negative control: an exactly separable constant box has a zero minor.
    assert minor([[Fraction(1, 4)] * 2 for _ in range(2)]) == 0
    determinant = minor(matrix)
    assert determinant != 0, "LPF9 changed: reassess the exact separability finding"
    original = f32(255 * values[0])
    factored = f32(255 * values[4] * values[36] / values[40])
    assert original != factored, "Impulse no longer distinguishes the proposed factorization"
    print(f"source: {source.relative_to(source.parents[3])}")
    print(f"source_sha256: {hashlib.sha256(raw).hexdigest()}")
    print(f"leading_2x2_minor_exact: {determinant}")
    print(f"leading_2x2_minor_f64: {float(determinant):.17g}")
    print(f"impulse255_original_f32: {original:.17g} bits={struct.pack('<f', original).hex()}")
    print(f"impulse255_center_factor_f32: {factored:.17g} bits={struct.pack('<f', factored).hex()}")
    print("verdict: stored LPF9 is not rank one; center-row factorization changes output bits")


if __name__ == "__main__":
    main()
