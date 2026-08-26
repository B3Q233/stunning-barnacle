#!/usr/bin/env python3
"""Generate learning-path.json from all.json.

Usage:
    python plan_path.py all.json --target p_sgld -o learning-path.json \
        [--state user-state.json] [--max-depth N]

Computes the dependency closure (components + prerequisites) of the target node,
orders it topologically, drops hs nodes and mastery=known nodes, and writes the
ordered learning path.
"""
import argparse
import json
import sys


def fail(msg):
    sys.exit("ERROR: " + msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("all", help="path to all.json")
    ap.add_argument("--target", required=True, help="target node id (usually a contribution)")
    ap.add_argument("-o", "--output", default="learning-path.json", help="output path")
    ap.add_argument("--state", default=None, help="optional user-state.json")
    ap.add_argument("--max-depth", type=int, default=None, help="cap path by dependency_depth")
    ap.add_argument("--keep-hs", action="store_true", help="include hs nodes in the path")
    args = ap.parse_args()

    try:
        with open(args.all, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        fail(f"cannot read {args.all}: {e}")
    except json.JSONDecodeError as e:
        fail(f"invalid JSON in {args.all}: {e}")

    by = {n["id"]: n for n in data.get("nodes", [])}
    if args.target not in by:
        fail(f"target {args.target!r} not found in all.json")

    user_mastery = {}
    user_importance = {}
    if args.state:
        try:
            with open(args.state, encoding="utf-8") as f:
                state = json.load(f)
            user_mastery = state.get("mastery", {}) or {}
            user_importance = state.get("importance", {}) or {}
        except OSError as e:
            fail(f"cannot read {args.state}: {e}")
        except json.JSONDecodeError as e:
            fail(f"invalid JSON in {args.state}: {e}")

    def deps(n):
        seen = []
        for d in (n.get("components") or []) + (n.get("prerequisites") or []):
            if d not in seen:
                seen.append(d)
        return seen

    closure = {}
    stack = [args.target]
    while stack:
        nid = stack.pop()
        if nid in closure:
            continue
        closure[nid] = True
        for d in deps(by[nid]):
            if d in by:
                stack.append(d)
            else:
                fail(f"node {nid} references missing id {d!r}")

    topo_index = {nid: i for i, nid in enumerate(data.get("topo", []))}
    ordered = sorted(closure, key=lambda nid: topo_index.get(nid, 10**9))

    path = []
    excluded = {"hs": 0, "mastery_known": 0, "max_depth": 0}
    for nid in ordered:
        n = by[nid]
        mastery = user_mastery.get(nid, n.get("mastery"))
        if not args.keep_hs and n["kind"] == "hs":
            excluded["hs"] += 1
            continue
        if mastery == "known":
            excluded["mastery_known"] += 1
            continue
        if args.max_depth is not None and (n.get("dependency_depth") or 0) > args.max_depth:
            excluded["max_depth"] += 1
            continue
        importance = user_importance.get(nid, n.get("importance") or 0)
        path.append({
            "id": nid,
            "name": n.get("name") or nid,
            "kind": n["kind"],
            "role": "target" if nid == args.target else "prerequisite-chain",
            "dependency_depth": n.get("dependency_depth") or 0,
            "effective_cost": n.get("effective_cost") or 0,
            "importance": importance,
            "mastery": mastery,
        })

    result = {
        "target": args.target,
        "query": by[args.target].get("name") or args.target,
        "path": path,
        "excluded": excluded,
        "stats": {
            "path_length": len(path),
            "closure_size": len(closure),
            "max_depth": max((n.get("dependency_depth") or 0 for n in path), default=0),
        },
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except OSError as e:
        fail(f"cannot write {args.output}: {e}")

    print(f"target={args.target} closure={len(closure)} path={len(path)} -> {args.output}")


if __name__ == "__main__":
    main()
