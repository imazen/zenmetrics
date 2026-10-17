from palette_plan import CLASSES, check_sources as validate

import copy
import hashlib
import json
import unittest


def seal(pool):
    content = {k: v for k, v in pool.items() if k != "authority_sha256"}
    result = copy.deepcopy(pool)
    result["authority_sha256"] = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return result


class PoolAuthority(unittest.TestCase):
    def fixture(self):
        sources = []
        for cls, n in zip(CLASSES, (2, 3, 4, 25, 64)):
            dev = max(1, min(n - 1, (14 * n + 32) // 64))
            for i in range(n):
                sources.append(
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
                )
        pool = seal(
            dict(
                schema="e35-palette-pool-v2",
                certified_clean=True,
                available_families={
                    cls: [s["family"] for s in sources if s["class"] == cls]
                    for cls in CLASSES
                },
            )
        )
        return sources, pool

    def test_canonical_content_digest_and_all_fields(self):
        sources, pool = self.fixture()
        validate(sources, pool)
        extended = seal(pool | {"provenance": {"version": 1, "name": "caf\u00e9"}})
        validate(sources, extended)
        extended["provenance"]["version"] = 2
        with self.assertRaisesRegex(ValueError, "authority hash mismatch"):
            validate(sources, extended)

    def test_pruned_pool_with_old_hash_refuses(self):
        sources, pool = self.fixture()
        removed = next(
            s for s in sources if s["class"] == "screen" and s["role"] == "fit"
        )
        pruned = copy.deepcopy(pool)
        pruned["available_families"]["screen"].remove(removed["family"])
        remaining = [s for s in sources if s["source_id"] != removed["source_id"]]
        with self.assertRaisesRegex(ValueError, "authority hash mismatch"):
            validate(remaining, pruned)
        # Correctly hashed invented metadata exercises the three-family counts;
        # this is a test fixture, not a real source authority or admission.
        validate(remaining, seal(pruned))

    def test_large_pool_caps_at_sixty_four_and_short_pool_cannot_prune(self):
        sources, pool = self.fixture()
        pool["available_families"]["mixed"].extend(f"extra{i}" for i in range(6))
        validate(sources, seal(pool))
        pool["available_families"]["screen"].append("unselected-clean")
        with self.assertRaises(ValueError):
            validate(sources, seal(pool))

    def test_missing_or_invented_digest_refuses(self):
        sources, pool = self.fixture()
        for digest in ("0" * 64, "x" * 64, ""):
            with self.assertRaises(ValueError):
                validate(sources, pool | {"authority_sha256": digest})
        missing = dict(pool)
        del missing["authority_sha256"]
        with self.assertRaises(ValueError):
            validate(sources, missing)


if __name__ == "__main__":
    unittest.main()
