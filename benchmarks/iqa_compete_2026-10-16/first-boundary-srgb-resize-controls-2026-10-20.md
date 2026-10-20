# All-24 sRGB golden resize control

Prepared, not executed. No HDR inference, compilation, package installation,
Docker pull or active-runner edit occurred in this lane. Qualification remains
diagnostic until root's controls pass. Keep the base Octave 6.4/image 2.12
observations and failures alongside the separate overlay observations.

[first_boundary_hdrvdp_srgb_controls.py](first_boundary_hdrvdp_srgb_controls.py)
prepares two immutable source-bound entries:

- `system212.m`: unchanged official HDR-VDP-2.2.2 and system image 2.12.
- `resize2201.m`: the same runtime and original metric, adding only unchanged upstream `imresize.m` and `private/interp_method.m` from image 2.20.1.

Each entry uses the same twelve frozen RGB8 pairs for all 24 existing golden
cases: two geometries (128×128 and 160×96), six distortions, and 30/60 ppd.
The preparer reproduces the original generator's integer-only reference,
MINSTD noise, replicated integer blur, dark, bright and chroma operations.
Raw files and their SHA-256 identities are shared by both entries. They bypass
Octave RNG and image-package fixture creation. Root must independently confirm
these raw bytes against the original integer generator before qualification.

The original `gen_srgb_goldens.m` and golden TSV are frozen and hash checked;
expected scalar values are copied without alteration. Every observed original
Q, hex bits, expected Q, absolute residual and pass flag is printed. Each entry
retains the existing `5e-3` limit, attempts the complete 24-case panel and refuses
at completion if any numerical case fails. A failed base process must not
prevent root from executing the separate overlay process. This adds no v3 bound
and does not qualify native parity or timing.

The existing archive, not a new download, is used:

- Archive SHA-256: `b4b6865e7f65ee5b9c254a69128bf10e93b38bc27f5bc5b973fef300217fc725`.
- Exact `imresize.m`: `c358d5381f325c5ab58c7eb0c3ccc8acc2141bc70634c66cc516541b823a15c8`.
- Exact private `interp_method.m`: `bcd3156239d6d4dccb626108a642afead1b4c60b400fed8b11d73ef5edcc6beb`.
- Original generator: `907fab1cd50619ce930df4d63c9fa5fcb765f7b36899da78c4476de6dacdb071`.
- Original golden TSV: `323aa664d1227344d8ab6a0b1773dd51bb9792516730f4618324619c4dd74599`.

The two upstream M-files were verified byte-for-byte against their archive
members in the preceding source review. This is an explicit M-file overlay,
not a claim that the whole image 2.20.1 package supports Octave 6.4.
[Archive provenance](https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/image-2.20.1.tar.gz).

Root's execution command uses both shared locks, `run-heavy --mem 12G --jobs 1`,
an existing disk-backed scratch directory, one-thread OMP/OpenBLAS/MKL limits,
and `octave-cli --no-gui --no-init-file --no-history --quiet` followed by the
chosen frozen entry. Bind and verify core/image/shared-library runtime closure
before and after each process. Keep full stdout/stderr in a distinct result
directory. `control-plan.json` supplies exact argv, source/data hashes and
unexecuted status; no existing helper or plan needs modification.
