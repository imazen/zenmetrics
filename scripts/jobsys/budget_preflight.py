#!/usr/bin/env python3
"""Declare a label-free budget preflight. Never execute or contact a worker."""
import argparse
import json
import math
from pathlib import Path

from palette_plan import CLASSES, canonical, check_sources, digest, geometry, profiles


def generation(sources):
    check_sources(sources)
    chosen = [min((s for s in sources if s["class"] == cls),
                  key=lambda s: (-max(s["width"], s["height"]), s["source_sha256"]))
              for cls in CLASSES]
    photo = next(s for s in chosen if s["class"] == "photo")
    if max(photo["width"], photo["height"]) < 256:
        raise ValueError("photo cannot support native 256 preflight")
    all_profiles = profiles()
    core = [p for p in all_profiles[:72] if p["quality"] == "best" and (
        (p["format"] != "min-size" and p["dither"] == "adaptive")
        or (p["format"] == "min-size" and p["dither"] == "blue-noise"))]
    if len(core) != 4:
        raise ValueError("four-route generation inventory drift")
    union, omitted = {}, []

    def add(s, side, colors, profile):
        shape = geometry(s["width"], s["height"], side)
        if shape is None:
            omitted.append(dict(source=s["source_id"], side=side, colors=colors,
                                profile=profile, native=[s["width"], s["height"]]))
            return
        task = dict(source=s["source_id"], source_sha256=s["source_sha256"],
                    side=side, shape=shape, colors=colors, profile=profile,
                    efforts=[*range(32), 200], stages=["resample", "quantize", "encode", "decode",
                        "independent-decode", "features", "teacher", "diffmaps", "upload", "receipt"])
        union[digest(task)] = task

    for s in chosen:
        for side in (32, 256, 1024, 4096):
            for colors in (2, 32, 256):
                for profile in core:
                    add(s, side, colors, profile)
    for profile in all_profiles:
        add(photo, 256, 32, profile)
    return dict(tasks=[dict(task_id=k, **v) for k, v in sorted(union.items())],
                unattainable=omitted, chosen_sources=[s["source_id"] for s in chosen])


def fits(shape_inventory):
    """Require intended shapes instead of silently timing smaller substitutes."""
    if (shape_inventory.get("schema") != "mock-fit-shapes-v1"
            or shape_inventory.get("labels_mounted") is not False
            or len(shape_inventory.get("sha256", "")) != 64):
        raise ValueError("label-free intended-shape inventory required")
    out = []
    for family in ("A", "C"):
        for route, arms in (("d1", ("control", "p", "h", "ph")),
                            ("full", ("control", "p", "h", "ph")),
                            ("loso", ("h", "ph"))):
            for arm in arms:
                key = f"{family}/{route}/{arm}"
                shape = shape_inventory["routes"][key]
                if (not shape.get("groups") or any(type(g["rows"]) is not int or g["rows"] <= 0
                        or not g.get("feature_ids_sha256") or not g.get("keys_sha256") for g in shape["groups"])
                        or shape.get("targets") != "deterministic-invented"
                        or shape.get("palette_features") not in ("generator", "explicit-shape-fixture")):
                    raise ValueError("intended fit group/key/feature shapes missing")
                out.append(dict(route=key, shapes=shape, init_seed=1101, sample_seed=101,
                    epochs=120, pairs_per_epoch=50000, selected_epoch=119, tier="v3",
                    formula_revision=5, timing=["archive", "load", "first-epoch", "loop", "pack", "upload"],
                    cold_cache=True, fixture_only=True))
    return out


def manifest(sources, hosts, shapes, preparation_cap, watchdogs, supporting_host):
    if not hosts or len(hosts) != len(set(hosts)) or supporting_host not in hosts:
        raise ValueError("unique admitted hosts and supporting-stage placement required")
    gen, fit = generation(sources), fits(shapes)
    tasks = []
    missing = []
    for host in hosts:
        for kind, entries in (("generation", gen["tasks"]), ("fit", fit)):
            for entry in entries:
                # Host is part of both identity and namespace; no cache result can
                # masquerade as a second host's cold measurement.
                task = dict(host=host, kind=kind, input=entry, labels="invented-only")
                task_id = digest(task)
                cap = watchdogs.get(f"{host}/{kind}/{entry.get('route', entry.get('task_id'))}")
                if type(cap) is not int or cap <= 0:
                    missing.append(task_id)
                    cap = None
                tasks.append(task | dict(task_id=task_id, namespace=f"preflight/{host}/{task_id}",
                                         watchdog_seconds=cap, count=1))
    for stage, count in (("maps", 672), ("statistics", 1), ("three-seed-reports", 1)):
        cap = watchdogs.get(f"{supporting_host}/{stage}")
        if type(cap) is not int or cap <= 0:
            missing.append(stage)
            cap = None
        tasks.append(dict(task_id=digest([supporting_host, stage]), host=supporting_host,
                          kind=stage, count=count, watchdog_seconds=cap, labels="invented-only"))
    reserved = None if missing else sum(t["count"]*t["watchdog_seconds"] for t in tasks)
    status = "DECLARED" if (reserved is not None and type(preparation_cap) is int
                             and 0 < reserved <= preparation_cap) else "UNREADY"
    return dict(schema="budget-preflight-v1", status=status, executed=False,
                source_inventory_sha256=digest(sources), shapes_sha256=digest(shapes),
                generation_per_host=len(gen["tasks"]), writes_per_host=33*len(gen["tasks"]),
                fits_per_host=len(fit), unattainable=gen["unattainable"], tasks=tasks,
                preparation_cap_seconds=preparation_cap, reserved_slot_seconds=reserved,
                missing_watchdogs=missing, production_budget="UNBOUND", measured=False,
                acceptance_harness="later-separate-preread")


def measured_watchdog(elapsed_seconds):
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("positive actual route measurement required")
    return math.ceil(3*elapsed_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    result = manifest(**inputs)
    with args.out.open("x") as stream:
        stream.write(canonical(result)+"\n")


if __name__ == "__main__":
    main()
