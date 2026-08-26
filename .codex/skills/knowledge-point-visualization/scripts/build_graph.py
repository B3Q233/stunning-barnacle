#!/usr/bin/env python3
"""Build all.json (knowledge-point graph) from expanded meta.json.

Usage:
    python build_graph.py meta.json [-o all.json] [--alpha 0.5] [--beta 0.5]

- Two edge types: components (composition) and prerequisites (dependency).
- DAG constraint applies to the UNION graph only.
- Five kinds: hs / atom / concept / method / contribution (knowledge role).
- dependency_depth = 1 + max(parent depth) over the union (hs = 0).
- effective_cost normalized to [0,1]: alpha*depth/maxDepth + beta*H/10.
- proof domain: proof_status + proof_type + proof_deps.
"""
import argparse
import json
import sys

KINDS = {"hs", "atom", "concept", "method", "contribution"}
IMPORTANCE_DEFAULT = {"hs": 0, "atom": 3, "concept": 4, "method": 6, "contribution": 8}
PROOF_TYPES = {"definition", "derivation", "empirical"}


def fail(msg):
    sys.exit("ERROR: " + msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meta", help="path to meta.json")
    ap.add_argument("-o", "--output", default="all.json", help="output path (default: all.json)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of normalized depth (default 0.5)")
    ap.add_argument("--beta", type=float, default=0.5, help="weight of H/10 (default 0.5)")
    args = ap.parse_args()
    if args.alpha + args.beta <= 0:
        fail("alpha + beta must be > 0")

    try:
        with open(args.meta, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        fail(f"cannot read {args.meta}: {e}")
    except json.JSONDecodeError as e:
        fail(f"invalid JSON in {args.meta}: {e}")

    source = data.get("source", "") if isinstance(data, dict) else ""
    phase = data.get("phase", "expanded") if isinstance(data, dict) else "expanded"
    expansion = data.get("expansion", {}) if isinstance(data, dict) else {}
    nodes = data.get("nodes", data) if isinstance(data, dict) else data
    if not isinstance(nodes, list):
        fail("meta.json must be a list or an object with a 'nodes' list")

    by = {}
    for raw in nodes:
        nid = raw.get("id")
        if not nid:
            fail("every node needs a non-empty 'id'")
        if nid in by:
            fail(f"duplicate id: {nid}")
        kind = raw.get("kind")
        if kind not in KINDS:
            fail(f"node {nid}: bad kind {kind!r} (expected one of {sorted(KINDS)})")
        components = list(raw.get("components") or [])
        if "prerequisites" in raw:
            prerequisites = list(raw.get("prerequisites") or [])
        else:
            prerequisites = list(raw.get("parents") or [])  # deprecated fallback
        if not isinstance(components, list) or not isinstance(prerequisites, list):
            fail(f"node {nid}: components/prerequisites must be lists")
        if kind == "hs":
            components = []
            prerequisites = []
        importance = raw.get("importance")
        if importance is None:
            importance = IMPORTANCE_DEFAULT.get(kind, 3)
        proof_type = raw.get("proof_type", "derivation")
        if proof_type not in PROOF_TYPES:
            fail(f"node {nid}: bad proof_type {proof_type!r}")
        by[nid] = {
            "id": nid,
            "name": raw.get("name") or nid,
            "short": raw.get("short") or raw.get("name") or nid,
            "kind": kind,
            "section": raw.get("section"),
            "proofAnchor": raw.get("proofAnchor"),
            "description": raw.get("description", ""),
            "provenance": raw.get("provenance", "explicit"),
            "proof_type": proof_type,
            "components": components,
            "prerequisites": prerequisites,
            "importance": importance,
            "learning_cost": raw.get("learning_cost"),
            "mastery": raw.get("mastery"),
        }

    for nid, n in by.items():
        for edge_type in ("components", "prerequisites"):
            for t in n[edge_type]:
                if t not in by:
                    fail(f"node {nid}: unknown {edge_type[:-1]} {t!r}")
        if n["kind"] == "hs" and (n["components"] or n["prerequisites"]):
            print(f"warning: hs node {nid} has dependencies; they are ignored", file=sys.stderr)

    for nid, n in by.items():
        n["children_components"] = []
        n["children_prerequisites"] = []
        for cid, c in by.items():
            if nid in c["components"]:
                n["children_components"].append(cid)
            if nid in c["prerequisites"]:
                n["children_prerequisites"].append(cid)
        n["children_components"].sort()
        n["children_prerequisites"].sort()

    def deps(n):
        seen = []
        for d in n["components"] + n["prerequisites"]:
            if d not in seen:
                seen.append(d)
        return seen

    # Union-graph topological sort (the only DAG constraint).
    indeg = {nid: len(deps(n)) for nid, n in by.items()}
    queue = sorted(nid for nid, d in indeg.items() if d == 0)
    topo = []
    while queue:
        nid = queue.pop(0)
        topo.append(nid)
        for child in by[nid]["children_components"] + by[nid]["children_prerequisites"]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
                queue.sort()
    if len(topo) != len(by):
        cyclic = [nid for nid, d in indeg.items() if d > 0]
        fail(f"cycle detected in union graph among nodes: {cyclic}")

    depth = {}
    for nid in topo:
        n = by[nid]
        depth[nid] = 0 if n["kind"] == "hs" else 1 + max((depth[d] for d in deps(n)), default=0)

    max_depth = max(depth.values()) if depth else 0
    max_expand = expansion.get("max_expand_depth")
    if max_expand is not None:
        for nid, n in by.items():
            if n["provenance"] == "expanded" and depth[nid] > int(max_expand):
                print(f"warning: expanded node {nid} depth {depth[nid]} > max_expand_depth {max_expand}",
                      file=sys.stderr)

    def effective(nid):
        lc = by[nid]["learning_cost"]
        if max_depth <= 0:
            return 0.0
        if isinstance(lc, (int, float)):
            e = args.alpha * (depth[nid] / max_depth) + args.beta * (lc / 10.0)
        else:
            e = depth[nid] / max_depth
        return round(max(0.0, min(1.0, e)), 3)

    initial = sorted(nid for nid, n in by.items() if n["kind"] == "hs")
    initial_set = set(initial)
    states = {}
    for nid in topo:
        n = by[nid]
        if n["kind"] == "hs":
            states[nid] = "proven"
        elif all(d in initial_set for d in deps(n)):
            states[nid] = "provable"
        else:
            states[nid] = "unproven"

    out_nodes = []
    for nid, n in by.items():
        proof_deps = deps(n)
        out_nodes.append({
            "id": nid,
            "name": n["name"],
            "short": n["short"],
            "kind": n["kind"],
            "section": n["section"],
            "proofAnchor": n["proofAnchor"],
            "description": n["description"],
            "provenance": n["provenance"],
            "proof_type": n["proof_type"],
            "components": list(n["components"]),
            "prerequisites": list(n["prerequisites"]),
            "children_components": list(n["children_components"]),
            "children_prerequisites": list(n["children_prerequisites"]),
            "dependency_depth": depth[nid],
            "importance": n["importance"],
            "learning_cost": n["learning_cost"],
            "mastery": n["mastery"],
            "effective_cost": effective(nid),
            "proof_status": states[nid],
            "proof_deps": proof_deps,
        })

    edges = ([{"from": d, "to": nid, "type": "component"}
              for nid, n in by.items() for d in n["components"]] +
             [{"from": d, "to": nid, "type": "prerequisite"}
              for nid, n in by.items() for d in n["prerequisites"]])
    stats = {k: sum(1 for n in by.values() if n["kind"] == k) for k in sorted(KINDS)}
    stats["total"] = len(by)

    result = {
        "meta": {
            "source": source,
            "phase": phase,
            "expansion": expansion,
            "stats": stats,
            "maxDepth": max_depth,
        },
        "nodes": out_nodes,
        "edges": edges,
        "topo": topo,
        "proof": {
            "initial": initial,
            "rule": ("derivation/empirical nodes are provable iff every proof_deps id is proven; "
                     "definition nodes are introduced once their referenced deps exist"),
            "states": states,
        },
        "defaults": {
            "thresholdK": 0.5,
            "importanceThreshold": 6,
            "alpha": args.alpha,
            "beta": args.beta,
            "learningCostMax": 10,
        },
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except OSError as e:
        fail(f"cannot write {args.output}: {e}")

    print(
        f"nodes={len(by)} edges={len(edges)} topo={len(topo)} "
        f"maxDepth={max_depth} -> {args.output}"
    )


if __name__ == "__main__":
    main()
