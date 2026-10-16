"""Invented source metadata only; no files, labels or jobs opened."""
import unittest
import palette_plan as p


class PalettePlan(unittest.TestCase):
    def test_exact_profiles_and_dense_axes(self):
        profiles = p.profiles()
        self.assertEqual(len(profiles), 144)
        self.assertEqual(sum(not c["joint"] for c in profiles), 18)
        self.assertEqual([c["target"] for c in profiles if c["target"] is not None], list(range(0, 101, 2)))
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
        resized = dict(source_id="invented", shape=[32, 16], kernel="mitchell-bc-1/3",
                       colour="rgb8-srgb-opaque", image_path="synthetic.png", sha256="a"*64)
        items = list(p.encode_items(r, resized))
        self.assertEqual(len({p.digest(v) for v in items}), 33)
        self.assertTrue(all(v["q"] == 0 and "encode_fp" not in v for v in items))
        with self.assertRaises(ValueError):
            list(p.encode_items(r, resized | dict(shape=[32, 15])))

    def test_inventory_role_and_family_refusals(self):
        sources = [dict(source_id=f"{c}{i}", family=f"{c}{i}", **{"class":c},
                        role="fit" if i < 50 else "dev", colour="rgb8-srgb-opaque",
                        source_sha256="a"*64, cluster="invented-cluster", width=32, height=32)
                   for c in p.CLASSES for i in range(64)]
        p.check_sources(sources)
        sources[0]["role"] = "dev"
        with self.assertRaises(ValueError):
            p.check_sources(sources)
        sources[0]["role"] = "fit"
        sources[1]["family"] = sources[0]["family"]
        with self.assertRaises(ValueError):
            p.check_sources(sources)


if __name__ == "__main__":
    unittest.main()
