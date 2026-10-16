import unittest
import budget_preflight as b
import palette_plan as p


class Preflight(unittest.TestCase):
    def sources(self, side=4096):
        return [dict(source_id=f"{c}{i}", family=f"{c}{i}", **{"class":c},
                     role="fit" if i < 50 else "dev", colour="rgb8-srgb-opaque",
                     source_sha256=f"{n*64+i:064x}", cluster="invented", width=side, height=side)
                for n,c in enumerate(p.CLASSES) for i in range(64)]

    def shapes(self):
        routes = {f"{f}/{r}/{a}":dict(groups=[dict(rows=2, feature_ids_sha256="a"*64, keys_sha256="b"*64)],
                  targets="deterministic-invented", palette_features="explicit-shape-fixture")
                  for f in ("A", "C") for r in ("d1", "full", "loso")
                  for a in (("h", "ph") if r == "loso" else ("control", "p", "h", "ph"))}
        return dict(schema="mock-fit-shapes-v1", labels_mounted=False, sha256="a"*64, routes=routes)

    def test_union_and_native_exclusions(self):
        result = b.generation(self.sources())
        self.assertEqual(len(result["tasks"]), 380)
        self.assertEqual(len({t["task_id"] for t in result["tasks"]}), 380)
        small = b.generation(self.sources(256))
        self.assertEqual(len(small["tasks"]), 260)
        self.assertEqual(len(small["unattainable"]), 120)

    def test_missing_budget_is_unready_and_no_host_reuse(self):
        hosts = ["fixture-1", "fixture-2", "fixture-3", "fixture-4"]
        result = b.manifest(self.sources(), hosts, self.shapes(), None, {}, hosts[0])
        self.assertEqual(result["status"], "UNREADY")
        self.assertEqual(result["generation_per_host"], 380)
        self.assertEqual(result["writes_per_host"], 12540)
        self.assertEqual(sum(t["kind"] == "fit" for t in result["tasks"]), 80)
        self.assertEqual(len({t["task_id"] for t in result["tasks"]}), len(result["tasks"]))
        self.assertFalse(result["executed"])
        self.assertFalse(result["measured"])
        self.assertEqual(b.measured_watchdog(1.01), 4)

    def test_shapes_and_label_mount_refuse(self):
        shape = self.shapes()
        shape["labels_mounted"] = True
        with self.assertRaises(ValueError):
            b.fits(shape)
        shape = self.shapes()
        del shape["routes"]["A/loso/h"]["groups"]
        with self.assertRaises(ValueError):
            b.fits(shape)


if __name__ == "__main__":
    unittest.main()
