"""Invented source metadata only; no files, labels or jobs opened."""

import unittest
import hashlib
import json
import palette_plan as p


class PalettePlan(unittest.TestCase):
    def test_exact_profiles_and_dense_axes(self):
        profiles = p.profiles()
        self.assertEqual(len(profiles), 144)
        self.assertEqual(sum(not c["joint"] for c in profiles), 18)
        self.assertEqual(
            [c["target"] for c in profiles if c["target"] is not None],
            list(range(0, 101, 2)),
        )
        self.assertEqual(p.EFFORTS, (*range(32), 200))
        self.assertEqual(len(p.SIZES), 20)

    def test_geometry_and_no_upscale(self):
        self.assertEqual(p.geometry(4, 3, 2), (2, 2))
        self.assertEqual(p.geometry(9, 1, 2), (2, 1))
        self.assertIsNone(p.geometry(31, 8, 32))
        with self.assertRaises(ValueError):
            p.geometry(0, 8, 32)

    def test_hash_binds_every_parameter_and_all_efforts(self):
        r = dict(source="invented", shape=(32, 16), colors=2, profile=p.profiles()[0])
        resized = dict(
            source_id="invented",
            shape=[32, 16],
            kernel="mitchell-bc-1/3",
            colour="rgb8-srgb-opaque",
            image_path="synthetic.png",
            sha256="a" * 64,
        )
        items = list(p.encode_items(r, resized))
        self.assertEqual(len({p.digest(v) for v in items}), 33)
        self.assertTrue(all(v["q"] == 0 and "encode_fp" not in v for v in items))
        with self.assertRaises(ValueError):
            list(p.encode_items(r, resized | dict(shape=[32, 15])))

    def test_inventory_role_and_family_refusals(self):
        sources = [
            dict(
                source_id=f"{c}{i}",
                family=f"{c}{i}",
                **{"class": c},
                role="fit" if i < 50 else "dev",
                colour="rgb8-srgb-opaque",
                source_sha256="a" * 64,
                cluster="invented-cluster",
                width=32,
                height=32,
            )
            for c in p.CLASSES
            for i in range(64)
        ]
        p.check_sources(sources, pool_for(sources))
        sources[0]["role"] = "dev"
        with self.assertRaises(ValueError):
            p.check_sources(sources, pool_for(sources))
        sources[0]["role"] = "fit"
        sources[1]["family"] = sources[0]["family"]
        with self.assertRaises(ValueError):
            p.check_sources(sources, pool_for(sources))

    def test_short_classes_require_complete_certified_pool(self):
        sources = []
        for cls, n in zip(p.CLASSES, [64, 64, 4, 25, 61]):
            dev = max(1, min(n - 1, (14 * n + 32) // 64))
            sources += [
                dict(
                    source_id=f"{cls}{i}",
                    family=f"{cls}{i}",
                    **{"class": cls},
                    role="fit" if i < n - dev else "dev",
                    colour="rgb8-srgb-opaque",
                    source_sha256="a" * 64,
                    cluster="invented",
                    width=32,
                    height=32,
                )
                for i in range(n)
            ]
        pool = pool_for(sources)
        p.check_sources(sources, pool)
        with self.assertRaises(ValueError):
            p.check_sources(sources[:-1], pool)
        with self.assertRaises(ValueError):
            p.check_sources(sources, pool | dict(certified_clean=False))
        pool["available_families"]["screen"].append("unselected-clean-family")
        with self.assertRaises(ValueError):
            p.check_sources(sources, pool)


def pool_for(sources):
    pool = dict(
        schema="e35-palette-pool-v2",
        certified_clean=True,
        available_families={
            c: [s["family"] for s in sources if s["class"] == c] for c in p.CLASSES
        },
    )
    pool["authority_sha256"] = hashlib.sha256(
        json.dumps(pool, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return pool


if __name__ == "__main__":
    unittest.main()
