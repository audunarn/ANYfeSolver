#!/usr/bin/env python3
"""Deterministic collector for the Q4 / V6 source-binding successor record.

SOURCE BINDING ONLY.  This script measures bytes, Git object ids and AST
symbol hashes.  It runs no numerics, edits no production source, grants no CI,
scientific or physical acceptance, and cannot claim reviewer approval.

Every digest in the output is either (a) computed here from live files or Git
objects, or (b) a cited constant from the independent comparison evidence
(q4-source-comparison.json) that is *asserted* against the computed value.  Any
disagreement is reported as a problem and ``--write`` refuses to write.

Usage (from the repository root, with the historical commits present):

    python scripts/build_q4_source_binding_20261111.py            # dry run
    python scripts/build_q4_source_binding_20261111.py --write    # write record
    python scripts/build_q4_source_binding_20261111.py --check    # freshness
    python scripts/build_q4_source_binding_20261111.py --verify-evidence-files

Git is used read-only (cat-file / rev-parse) for historical objects only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reference_cases" / "e4_pl_source_binding_20261111.json"
SCHEMA = "anyfesolver.q4-source-binding-successor-v2"
BOUNDARY = "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"

RECORDED_PREFIX = "src/anysolver/"
LIVE_PREFIX = "src/anyfesolver/"
TEXT_LIVE_TO_RECORDED = (("ANYfeSolver", "ANYsolver"), ("anyfesolver", "anysolver"))
ENTRY = "e4_pl_element.py"

REVISIONS = {
    "c31a223f": "c31a223fe63f886a1a9b0b47c926bbe748ad4554",
    "5c34dc2e4": "5c34dc2e47f7f0114c2a5832d9515fc9ebc2dc13",
    "a0781c0bc": "a0781c0bc8863a9190dc5647c32d263bdf4f581d",
    "e0c5f32ab": "e0c5f32ab40e9548ba0ea8981cf5e288ed375a1c",
}
CARRIERS = (
    "e4_pl_element.py",
    "e4_pl_s3_element.py",
    "_q4_assumed_strain_plasticity.py",
    "vectorized_nonlinear.py",
    "elements.py",
    "_compensated_contractions.py",
)
# Cited constants: independently measured Git ids (identical at 5c34dc2e4,
# a0781c0bc and e0c5f32ab) and the c31a223f terminal blobs.
UPSTREAM_BLOBS = {
    "e4_pl_element.py": "47efceacb07a9be0ab92542164e5dd3ca1ea0c63",
    "e4_pl_s3_element.py": "8047c66f677be5214fb7457a0312492481fb7e93",
    "_q4_assumed_strain_plasticity.py": "b78cf67beb8f11d1e0a239c05ee098417ecdfad7",
    "vectorized_nonlinear.py": "b2ed521ff2b16710d50b7ee7b407447eb455ec4e",
    "elements.py": "0654d419641bf591ee21ddbe9ddf0c98bc8890c8",
    "_compensated_contractions.py": "e82e10b1f74aeee44ccc9e3009444b668f1ed53d",
}
TERMINAL_BLOBS = {
    "e4_pl_element.py": "bd6c0dc2cb3661eb73942fc0fea56e527aa8d0b4",
    "e4_pl_s3_element.py": "adc6562cbd3fb768d7f0d0ab69011f21ff7c9392",
}
# Cited constants for the live candidate: (raw blob, LF blob, LF sha256,
# blob of LF bytes after the reverse spelling map).
EXPECTED_LIVE = {
    "e4_pl_element.py": (
        "3ded2fff1472b8532f88223792232bf4aea2250e",
        "3ded2fff1472b8532f88223792232bf4aea2250e",
        "d597c84dde8e3b280134c9ddf57ca5ff4f39fb748038c521ae50f2aef5051e3e",
        "47efceacb07a9be0ab92542164e5dd3ca1ea0c63",
    ),
    "e4_pl_s3_element.py": (
        "336cbf9673242908f4bc4b9250a58e55072181ed",
        "be978bd2beff2111166c754a35e911aa740196b8",
        "f56a3101a74333f35ba0a7162d221b491230e249899e5ab5a7d485ece7d0ed1c",
        "8047c66f677be5214fb7457a0312492481fb7e93",
    ),
    "_q4_assumed_strain_plasticity.py": (
        "68622a264aa806b5228d6c17839f1327cb170e03",
        "b78cf67beb8f11d1e0a239c05ee098417ecdfad7",
        "cdca874f3572914c13c0ac710632714035322cda0531a06b56b701fc0e240e0c",
        "b78cf67beb8f11d1e0a239c05ee098417ecdfad7",
    ),
    "vectorized_nonlinear.py": (
        "b2ed521ff2b16710d50b7ee7b407447eb455ec4e",
        "656c7baa056660fced221c572be4760b9807e526",
        "13d0683f5c91cbfcb39dc36cca0b9f978a59f09af8e85270fbee6df574729efc",
        "656c7baa056660fced221c572be4760b9807e526",
    ),
    "elements.py": (
        "cd24e94b4fbb8134d6e019b4cf7c45c96f72fc39",
        "beae44a0c8beb310072c3e12556e249cbd739fc8",
        "47f3d2d8474accb4fb5de75384a914fa643ebb0ec4ca0c15f89e15f038e98215",
        "a90a8da82c4f5603161eeb76f754babc3f4c243e",
    ),
    "_compensated_contractions.py": (
        "e82e10b1f74aeee44ccc9e3009444b668f1ed53d",
        "6f0bce4a39de7a8ac74c7558978e346743719984",
        "43088d8059fe6481fad64e63d7e20ba2db32c223110f92bcc913bdd8c280b131",
        "6f0bce4a39de7a8ac74c7558978e346743719984",
    ),
}
# Two authentic states of the October 19 record (exact Git genealogy in the
# cited lineage evidence).  AS_REVIEWED is the original reader-pinned blob,
# restored byte-for-byte; OCT19_RECORD is the live in-place file whose only
# post-pin edits are six rebinds of external-dependency sha256 leaves.
OCT19_AS_REVIEWED = "docs/reference_cases/e4_pl_q4_source_binding_20261019_as_reviewed.json"
OCT19_AS_REVIEWED_SHA256 = "64B07F1A535E6D6EFCEAE0B2EC01BA539DF69BE7956C03DBDA65C2ACF7C7E74F"
OCT19_AS_REVIEWED_BLOB = "99b59279e9702b509a84f1932874e2b24680adcc"
OCT19_AS_REVIEWED_BYTES = 71924
OCT19_RECORD = "docs/reference_cases/e4_pl_q4_source_binding_20261019.json"
OCT19_RECORD_SHA256 = "06679901687621BD6696DB9C8777A80131EE08631BFCCA706FD03FA15042FB38"
OCT19_RECORD_BLOB = "3052505a185198142ffeee60fbee8ad46b3d968c"
OCT19_RECORD_BYTES = 70591
LEAF_PREFIX = "external_arithmetic_dependencies/"
# Auto-text sources (Git LF, Windows CRLF without an eol attribute).  Identity is
# the canonical LF form; exactly two physical forms are permitted (uniform LF or
# uniform CRLF).  Everything else, including -text carriers, keeps exact raw bytes.
PORTABLE_AUTO_TEXT = frozenset({
    "e4_pl_s3_element.py", "_q4_assumed_strain_plasticity.py", "_qualified_authority_epoch.py",
    "shell_sections.py", "e4_pl_s3_state.py", "plasticity.py",
})
# Independently audited inputs (whole-file SHA-256 of the supplied evidence).
LINEAGE_JSON_SHA256 = "1cef66aedcd866ed1feaa1ee28497f30a96966a78d01e1cb472ea5d58dbbd7ac"
LINEAGE_README_SHA256 = "0605298d0ec64b1f005e0139513da7a588cab256e58b36ab3140d04295bef523"
LINEAGE_DIR = Path("C:/ANYworkspaceWork/anyfesolver-rollout-20261111/evidence/q4-lineage")
HISTORICAL_FILES = (
    OCT19_AS_REVIEWED,
    OCT19_RECORD,
    "docs/reference_cases/qualified_shell_runtime_guard_performance_successor.json",
    "docs/reference_cases/e4_pl_q4_0_4_1_replay_hotfix.json",
    "docs/reference_cases/e4_pl_q4_0_4_2_replay_canonicalization.json",
    "docs/reference_cases/e4_pl_s3_v2d_default_activation_candidate.json",
    "docs/reference_cases/e4_pl_s3_v2d_default_integration_repair.json",
    "docs/reference_cases/e4_pl_s3_mixed_mesh_qualification_contract.json",
    "docs/STATIC_RUNTIME_PERFORMANCE.md",
)
CONTRACT = "docs/reference_cases/e4_pl_s3_mixed_mesh_qualification_contract.json"
STATIC_REPORT_SHA256 = "CBB92A59FDDDA847AA5953EDAA7A2ED6BE1EA64ED1995D415A59B72831BB31D1"
EXPECTED_CLOSURE = {
    "oct19": (159, "996C94581594E958545D5D64F25D1864942941A5B766BE88A79A4991BD4DE95C"),
    "upstream": (161, "CF0026B4621EA9180B0DFCE1A483AFB8A762A218CB5CC3F538517C1A8D0843AA"),
    "candidate": (161, "8F7841580C5A835C18BE351161DBF714F2D048A4F00203FE656EEEBD81E9C480"),
}
EVIDENCE_REPORTS = {
    "C:/ANYworkspaceWork/codex-q4-qual-notes/03_FINAL_REPORT.md": "3b0bfcf5fbf1499c9027d05b20af3404e21db73866e862fa085df61ed8f84f9e",
    "C:/ANYworkspaceWork/codex-q4-qual-notes/04_POSTMERGE_REPORT.md": "4af246083c78245db61e00ca42314ccfa6551cb3d3efd2091e4ea6d80aac9d98",
    "C:/ANYworkspaceWork/codex-q4-qual-notes/99_COORDINATOR_ACK.md": "fde1e77d52e92a850fac5f13bf8d34a57c58597f0d9c59d7c7f5a18d6197c504",
    "C:/ANYworkspaceWork/qualification-plan/DECISIONS_LOG.md": "f88475e28cc9802d8e898184652fa91fd73886a2fe00fdb8f0714cf8697ffead",
    "C:/ANYworkspaceWork/codex-s3-notes/03_FINAL_REPORT.md": "3de04929016b7ebc764924dd99cc6043f911059256b0a2d9e8a815dce7b1730e",
    "C:/ANYworkspaceWork/codex-s3-notes/99_COORDINATOR_ACK.md": "cac0ed2eacf957ee76a90621e60e9423109917d30cf93ad4bbfac875d2d1cd60",
}


# ---------------------------------------------------------------- primitives
def sha256_upper(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def identity(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "git_blob": git_blob_id(data), "sha256": sha256_upper(data)}


def reverse_spelling(data: bytes) -> bytes:
    for live, recorded in TEXT_LIVE_TO_RECORDED:
        data = data.replace(live.encode("ascii"), recorded.encode("ascii"))
    return data


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    return json.loads(raw, object_pairs_hook=pairs)


def git_blob_at(rev: str, logical_path: str) -> bytes | None:
    done = subprocess.run(
        ["git", "cat-file", "blob", f"{rev}:{logical_path}"], cwd=ROOT, capture_output=True, check=False
    )
    return done.stdout if done.returncode == 0 else None


# ------------------------------------------------------ AST closure (pure)
def canonical_ast(value: object) -> object:
    if isinstance(value, ast.AST):
        fields: list[list[object]] = []
        for name, item in ast.iter_fields(value):
            if name == "type_comment" and item is None:
                continue
            if name == "type_params" and item == []:
                continue
            if (
                name == "body"
                and isinstance(item, list)
                and item
                and isinstance(item[0], ast.Expr)
                and isinstance(item[0].value, ast.Constant)
                and isinstance(item[0].value.value, str)
            ):
                item = item[1:]
            fields.append([name, canonical_ast(item)])
        return [type(value).__name__, fields]
    if isinstance(value, (list, tuple)):
        return [canonical_ast(item) for item in value]
    if value is Ellipsis:
        return ["Ellipsis"]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    raise TypeError(f"unsupported AST field value {type(value).__name__}")


def symbol_nodes(source: str) -> dict[str, list[ast.AST]]:
    nodes: dict[str, list[ast.AST]] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{node.name}"
                nodes.setdefault(name, []).append(node)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{name}.")
                continue
            targets: list[str] = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            for target in targets:
                nodes.setdefault(f"{prefix}{target}", []).append(node)

    visit(ast.parse(source).body)
    return nodes


def _digest(value: object) -> str:
    return sha256_upper(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8"))


def closure_payload(source: str, roots: set[str], required: set[str], schema: str, path: str) -> dict[str, Any]:
    inventory = symbol_nodes(source)
    need = roots | required
    bad = sorted(n for n in need if len(inventory.get(n, ())) != 1)
    if bad:
        raise ValueError(f"missing or duplicate closure symbols: {bad}")
    module_symbols = {n for n in inventory if "." not in n}
    owners = {"QualifiedE4PLShellElement", "__class__", "cls", "self"}

    def deps(symbol: str) -> list[str]:
        found: set[str] = set()
        for item in ast.walk(inventory[symbol][0]):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in module_symbols:
                found.add(item.id)
            elif (
                isinstance(item, ast.Attribute)
                and isinstance(item.ctx, ast.Load)
                and isinstance(item.value, ast.Name)
                and item.value.id in owners
                and f"QualifiedE4PLShellElement.{item.attr}" in inventory
            ):
                found.add(f"QualifiedE4PLShellElement.{item.attr}")
        found.discard(symbol)
        return sorted(found)

    closure, pending, edges = set(roots), sorted(roots), {}
    while pending:
        symbol = pending.pop(0)
        edges[symbol] = deps(symbol)
        for dep in edges[symbol]:
            if len(inventory.get(dep, ())) != 1:
                raise ValueError(f"unresolved closure dependency {symbol} -> {dep}")
            if dep not in closure:
                closure.add(dep)
                pending.append(dep)
        pending.sort()
    edges = {k: edges[k] for k in sorted(edges)}
    symbols = {n: _digest(canonical_ast(inventory[n][0])) for n in sorted(edges)}
    return {"edges": edges, "path": path, "roots": sorted(roots), "schema": schema, "symbols": symbols}


def aggregate(payload: dict[str, Any]) -> str:
    return _digest(payload)


def module_symbol_hashes(source: str) -> dict[str, str]:
    return {n: _digest([canonical_ast(x) for x in v]) for n, v in sorted(symbol_nodes(source).items())}


def internal_imports(source: str, closure_symbols: list[str]) -> list[dict[str, Any]]:
    """Every sibling-package import of the entry module, with closure usage."""
    tree = ast.parse(source)
    inventory = symbol_nodes(source)
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        absolute = node.level == 0 and node.module and node.module.split(".")[0] in ("anyfesolver", "anysolver")
        if node.level == 0 and not absolute:
            continue
        parts = (node.module.split(".")[1:] if absolute else (node.module.split(".") if node.module else []))
        for alias in node.names:
            file = ("/".join(parts) if parts else alias.name) + ".py"
            rows.append(
                {
                    "bound_name": alias.asname or alias.name,
                    "file": file if node.level <= 1 else None,
                    "imported": alias.name,
                    "kind": "name" if parts else "module",
                    "relative_level": node.level,
                }
            )
    for row in rows:
        used: set[str] = set()
        loaded = False
        for symbol in closure_symbols:
            for item in ast.walk(inventory[symbol][0]):
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == row["bound_name"]:
                    loaded = True
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == row["bound_name"]
                ):
                    used.add(item.attr)
        row["used_attributes"] = sorted(used) if row["kind"] == "module" else []
        row["used_in_closure"] = loaded or bool(used)
    return sorted(rows, key=lambda r: (r["bound_name"], str(r["file"]), r["imported"]))


# ---------------------------------------------------- historical lineage
def leaves(value: object, path: tuple[str, ...] = ()) -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key in value:
            out.update(leaves(value[key], path + (key,)))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(leaves(item, path + (str(index),)))
        return out
    return {"/".join(path): value}


def physical(problems: list[str], name: str, raw: bytes) -> dict[str, Any]:
    """Canonical LF identity plus the exactly permitted physical representations."""
    canonical = lf(raw)
    out: dict[str, Any] = {"live_lf": identity(canonical), "portable_text": name in PORTABLE_AUTO_TEXT}
    if out["portable_text"]:
        crlf = canonical.replace(b"\n", b"\r\n")
        if b"\r" in canonical or raw not in (canonical, crlf):
            problems.append(f"{name}: raw is mixed-ending, double-CR or not uniform LF/CRLF")
        out["permitted_raw"] = {"crlf": identity(crlf), "lf": identity(canonical)}
    else:
        out["live_raw"] = identity(raw)
    return out


def git_text(*args: str) -> str | None:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def verify_lineage_against_git(problems: list[str], nodes: list[dict[str, Any]]) -> None:
    """Maintainer-only: every node, blob, digest and parent list must exist in Git."""
    for node in nodes:
        commit = node["commit"]
        data = git_blob_at(commit, OCT19_RECORD)
        if data is None or (git_blob_id(data), sha256_upper(data), len(data)) != (node["blob"], node["sha256"], node["length"]):
            problems.append(f"lineage node {commit}: record object differs from Git (or commit absent)")
        listed = git_text("rev-list", "--parents", "-n", "1", commit)
        if listed is None or listed.split()[1:] != [p["commit"] for p in node["parents"]]:
            problems.append(f"lineage node {commit}: parent list differs from Git")
        for parent in node["parents"]:
            if git_text("rev-parse", f"{parent['commit']}:{OCT19_RECORD}") != parent["blob"]:
                problems.append(f"lineage node {commit}: parent {parent['commit']} blob differs from Git")


def lineage_section(expect, problems: list[str], reviewed: Any, current: Any, lineage_dir: Path) -> dict[str, Any]:
    """Static, Git-free copy of the measured six-edit lineage (for later tests)."""
    lineage_file, readme_file = lineage_dir / "lineage.json", lineage_dir / "README.md"
    if not (lineage_file.is_file() and readme_file.is_file()):
        problems.append(f"lineage evidence missing under {lineage_dir}")
        return {}
    raw = lineage_file.read_bytes()
    expect("lineage.json whole-file sha256 (vetted input)", hashlib.sha256(raw).hexdigest(), LINEAGE_JSON_SHA256)
    expect("lineage README whole-file sha256 (vetted input)", hashlib.sha256(readme_file.read_bytes()).hexdigest(), LINEAGE_README_SHA256)
    lineage = strict_json(raw)
    nodes_all = [
        {key: node[key] for key in ("blob", "commit", "length", "parents", "sha256")} for node in lineage["nodes"]
    ]
    expect("lineage node count", len(nodes_all), 9)
    verify_lineage_against_git(problems, nodes_all)
    expect("lineage reader pin", lineage["original_reader_sha256"], OCT19_AS_REVIEWED_SHA256)
    expect("lineage anchor", lineage["anchor"], REVISIONS["e0c5f32ab"])
    expect("lineage anchor blob", lineage["anchor_blob"], OCT19_RECORD_BLOB)
    nodes = lineage["nodes"]
    pin_index = next(i for i, n in enumerate(nodes) if n["blob"] == OCT19_AS_REVIEWED_BLOB)
    pin = nodes[pin_index]
    expect("lineage pin node", (pin["sha256"], pin["length"]), (OCT19_AS_REVIEWED_SHA256, OCT19_AS_REVIEWED_BYTES))
    state = leaves(reviewed)
    previous_blob = OCT19_AS_REVIEWED_BLOB
    transitions: list[dict[str, Any]] = []
    merge: dict[str, Any] | None = None
    for node in nodes[pin_index + 1:]:
        if len(node["parents"]) != 1:
            merge = {"blob": node["blob"], "commit": node["commit"], "parents": node["parents"], "sha256": node["sha256"]}
            expect("lineage merge blob", node["blob"], OCT19_RECORD_BLOB)
            continue
        parent = node["parents"][0]
        expect(f"lineage continuity {node['commit'][:9]}", parent["blob"], previous_blob)
        for leaf, change in parent["changes"].items():
            if not (leaf.startswith(LEAF_PREFIX) and leaf.endswith("/sha256")):
                problems.append(f"unexpected changed leaf outside external sha256 leaves: {leaf}")
            expect(f"lineage leaf chain {leaf}", state.get(leaf), change["before"])
            state[leaf] = change["after"]
        transitions.append({
            "blob": node["blob"], "changes": parent["changes"], "commit": node["commit"],
            "parent_blob": parent["blob"], "parent_commit": parent["commit"],
            "sha256": node["sha256"], "length": node["length"],
        })
        previous_blob = node["blob"]
    expect("lineage transition count", len(transitions), 6)
    expect("lineage ends at current record blob", previous_blob, OCT19_RECORD_BLOB)
    expect("lineage replay reproduces current leaves", state == leaves(current), True)
    if merge is not None:
        for parent in merge["parents"]:
            for leaf in parent["changes"]:
                if not (leaf.startswith(LEAF_PREFIX) and leaf.endswith("/sha256")):
                    problems.append(f"unexpected merge-parent leaf: {leaf}")
    return {
        "anchor": lineage["anchor"],
        "lineage_json_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "chain_sha256": _digest(nodes_all),
        "nodes": nodes_all,
        "pin_node": {"blob": pin["blob"], "commit": pin["commit"], "length": pin["length"], "sha256": pin["sha256"]},
        "readme_sha256": sha256_upper(readme_file.read_bytes()),
        "scientific_transfer_endorsed": False,
    }


def historical_gaps(current: Any) -> list[dict[str, Any]]:
    """Whole-file comparisons of the current record that do not describe live bytes."""
    review = current["source_binding_integration_review"]
    external = current["external_arithmetic_dependencies"]["_elements_module"]
    checks = [
        ("source", current["source"]["path"], current["source"]["sha256"], "source/sha256"),
        ("lifecycle_delta", review["lifecycle_delta"]["path"], review["lifecycle_delta"]["sha256"],
         "source_binding_integration_review/lifecycle_delta/sha256"),
        ("external__elements_module", external["path"], external["sha256"],
         "external_arithmetic_dependencies/_elements_module/sha256"),
    ]
    for relative, row in sorted(current["observed_focused_pre_reader_integration"]["all_source_symbol_delta"]["files"].items()):
        if relative.startswith("tests/test_general_contact_c5_"):
            checks.append((
                f"observed_{relative}", relative, row["after_source_sha256"],
                f"observed_focused_pre_reader_integration/all_source_symbol_delta/files/{relative}/after_source_sha256",
            ))
    gaps = []
    for check, recorded, digest, leaf in checks:
        live = recorded.replace(RECORDED_PREFIX, LIVE_PREFIX, 1)
        live_bytes = (ROOT / live).read_bytes()  # protected (-text) carriers: raw is pinned too
        live_sha = sha256_upper(lf(live_bytes))
        gaps.append({
            "historical_leaf": leaf, "live_raw": identity(live_bytes), "live_lf": identity(lf(live_bytes)),
            "check": check, "historical_gap": True, "live_lf_sha256": live_sha,
            "live_lf_matches_recorded": live_sha == digest.upper(), "live_path": live,
            "recorded_path": recorded, "recorded_sha256": digest.upper(),
        })
    return gaps


# ------------------------------------------------------------------ builder
def build(lineage_dir: Path = LINEAGE_DIR) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []

    def expect(label: str, actual: object, expected: object) -> None:
        if actual != expected:
            problems.append(f"{label}: computed {actual!r} != cited {expected!r}")

    for key, full in REVISIONS.items():
        done = subprocess.run(["git", "cat-file", "-e", f"{full}^{{commit}}"], cwd=ROOT, capture_output=True, check=False)
        if done.returncode != 0:
            problems.append(f"revision {key} ({full}) is not present in this repository")

    contract = strict_json((ROOT / CONTRACT).read_bytes())
    authority = contract["candidate"]["qualified_q4_mechanics_authority"]
    roots, required = set(authority["roots"]), set(authority["symbols"])
    schema, logical_entry = authority["schema"], authority["path"]
    expect("contract entry path", logical_entry, RECORDED_PREFIX + ENTRY)

    historical_files = {}
    for relative in HISTORICAL_FILES:
        historical_files[relative] = sha256_upper((ROOT / relative).read_bytes())
    reviewed_raw = (ROOT / OCT19_AS_REVIEWED).read_bytes()
    current_raw = (ROOT / OCT19_RECORD).read_bytes()
    expect("as-reviewed record identity", (sha256_upper(reviewed_raw), git_blob_id(reviewed_raw), len(reviewed_raw)),
           (OCT19_AS_REVIEWED_SHA256, OCT19_AS_REVIEWED_BLOB, OCT19_AS_REVIEWED_BYTES))
    expect("current historical record identity", (sha256_upper(current_raw), git_blob_id(current_raw), len(current_raw)),
           (OCT19_RECORD_SHA256, OCT19_RECORD_BLOB, OCT19_RECORD_BYTES))
    expect("static performance report sha256", historical_files["docs/STATIC_RUNTIME_PERFORMANCE.md"], STATIC_REPORT_SHA256)
    oct19 = strict_json(reviewed_raw)  # the original reader semantic
    current_record = strict_json(current_raw)
    for key in ("mechanics", "mechanics_aggregate_sha256", "source_binding_integration_review", "observed_focused_pre_reader_integration"):
        expect(f"current historical record keeps {key}", current_record[key], oct19[key])
    expect("Oct19 aggregate", oct19["mechanics_aggregate_sha256"], EXPECTED_CLOSURE["oct19"][1])
    expect("Oct19 symbol count", len(oct19["mechanics"]["symbols"]), EXPECTED_CLOSURE["oct19"][0])
    oct19_flags = {
        key: oct19["source_binding_integration_review"][key]
        for key in ("physical_acceptance_transfer", "C5_exception_transfer", "CI_credit")
    }
    expect("Oct19 nonacceptance flags", oct19_flags, {k: False for k in oct19_flags})
    observed = oct19["observed_focused_pre_reader_integration"]
    expect("Oct19 observation", (observed["status"], observed["case_count"], observed["phase_count"]), ("PASS", 41, 123))

    carriers: dict[str, Any] = {}
    live_lf_sources: dict[str, str] = {}
    for name in CARRIERS:
        live_path = LIVE_PREFIX + name
        recorded_path = RECORDED_PREFIX + name
        raw = (ROOT / live_path).read_bytes()
        live_lf = lf(raw)
        reverse = reverse_spelling(live_lf)
        raw_blob, lf_blob, lf_sha, reverse_blob = EXPECTED_LIVE[name]
        if name not in PORTABLE_AUTO_TEXT:  # portable files: canonical LF identity only
            expect(f"{name} live raw blob", git_blob_id(raw), raw_blob)
        expect(f"{name} live LF blob", git_blob_id(live_lf), lf_blob)
        expect(f"{name} live LF sha256", sha256_upper(live_lf).lower(), lf_sha)
        expect(f"{name} reverse-spelling blob", git_blob_id(reverse), reverse_blob)
        upstream_raw = git_blob_at(REVISIONS["e0c5f32ab"], recorded_path)
        entry: dict[str, Any] = {
            "live_path": live_path,
            "recorded_path": recorded_path,
            **physical(problems, name, raw),
            "live_lf_after_reverse_spelling_map": identity(reverse),
        }
        if upstream_raw is None:
            problems.append(f"{name}: not readable at e0c5f32ab as {recorded_path}")
        else:
            expect(f"{name} upstream raw blob", git_blob_id(upstream_raw), UPSTREAM_BLOBS[name])
            for rev in ("5c34dc2e4", "a0781c0bc"):
                other = git_blob_at(REVISIONS[rev], recorded_path)
                expect(f"{name} blob at {rev}", None if other is None else git_blob_id(other), UPSTREAM_BLOBS[name])
            upstream_lf = lf(upstream_raw)
            entry["upstream_raw"] = identity(upstream_raw)
            entry["upstream_lf"] = identity(upstream_lf)
            entry["reverse_spelling_map_reproduces_upstream_lf"] = reverse == upstream_lf
            expect(f"{name} reverse map reproduces upstream LF", reverse == upstream_lf, True)
            up_syms = module_symbol_hashes(upstream_lf.decode("utf-8"))
            lv_syms = module_symbol_hashes(live_lf.decode("utf-8"))
            entry["module_symbols"] = {
                "added": sorted(set(lv_syms) - set(up_syms)),
                "candidate_aggregate_sha256": _digest(lv_syms),
                "changed": sorted(n for n in up_syms if n in lv_syms and up_syms[n] != lv_syms[n]),
                "removed": sorted(set(up_syms) - set(lv_syms)),
                "upstream_aggregate_sha256": _digest(up_syms),
            }
        if name in TERMINAL_BLOBS:
            entry["v6_terminal_git_blob"] = TERMINAL_BLOBS[name]
            entry["live_lf_equals_v6_terminal"] = git_blob_id(live_lf) == TERMINAL_BLOBS[name]
        carriers[name] = entry
        live_lf_sources[name] = live_lf.decode("utf-8")

    # Closure measurements, always with the recorded logical path.
    candidate = closure_payload(live_lf_sources[ENTRY], roots, required, schema, logical_entry)
    candidate_agg = aggregate(candidate)
    expect("candidate closure", (len(candidate["symbols"]), candidate_agg), EXPECTED_CLOSURE["candidate"])
    upstream_source = git_blob_at(REVISIONS["e0c5f32ab"], RECORDED_PREFIX + ENTRY)
    upstream_agg = ""
    upstream_symbols: dict[str, str] = {}
    if upstream_source is not None:
        upstream = closure_payload(lf(upstream_source).decode("utf-8"), roots, required, schema, logical_entry)
        upstream_agg = aggregate(upstream)
        upstream_symbols = upstream["symbols"]
        expect("upstream closure", (len(upstream_symbols), upstream_agg), EXPECTED_CLOSURE["upstream"])
    oct19_symbols = oct19["mechanics"]["symbols"]

    def delta(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
        return {
            "added": sorted(set(new) - set(old)),
            "changed": sorted(n for n in old if n in new and old[n] != new[n]),
            "removed": sorted(set(old) - set(new)),
        }

    imports = internal_imports(live_lf_sources[ENTRY], sorted(candidate["symbols"]))
    for row in imports:
        if row["file"] is None or row["relative_level"] > 1:
            problems.append(f"unsupported internal import form: {row}")
            continue
        data = (ROOT / LIVE_PREFIX / row["file"]).read_bytes()
        row["live_path"] = LIVE_PREFIX + row["file"]
        row["recorded_path"] = RECORDED_PREFIX + row["file"]
        row.update(physical(problems, row["file"], data))
    oct19_external = {
        alias: {"path": v["path"], "used_attributes": v["used_attributes"], "sha256": v["sha256"].upper()}
        for alias, v in oct19["external_arithmetic_dependencies"].items()
    }
    lineage = lineage_section(expect, problems, oct19, current_record, lineage_dir)
    gaps = historical_gaps(current_record)
    expect("historical gap count", len(gaps), 5)
    for gap in gaps:
        expect(f"gap {gap['check']} is a nonmatch", gap["live_lf_matches_recorded"], False)

    record: dict[str, Any] = {
        "adoption": {
            "independent_review_required_outside_generating_invocation": True,
            "owner_adoption_recorded": False,
            "reviewer_approval_claimed": False,
            "status": "UNREVIEWED_CANDIDATE_RECORD",
        },
        "boundary_flags": {
            "C5_exception_transfer": False,
            "CI_credit": False,
            "historical_observation_rerun": False,
            "historical_records_modified": False,
            "new_physical_acceptance": False,
            "new_scientific_acceptance": False,
            "numerical_tolerances_changed": False,
            "physical_acceptance_transfer": False,
            "production_mechanics_changed_by_this_record": False,
        },
        "carriers": carriers,
        "historical_evidence_cited": {
            "oct19_nonacceptance_flags_retained": oct19_flags,
            "oct19_observation_retained_historical_only": {"case_count": 41, "phase_count": 123, "rerun": False},
            "owner_decision": "Option A: public stress accepted with follow-up; elastic bit-identity waiver confined to qualified Q4; owner approved qualified S3 von Karman default subject to its gates",
            "postmerge_confirmation": {
                "commit": REVISIONS["a0781c0bc"],
                "product_digest": "9db4a9abac478313a3000d67a286174e1999d9259df74b030fdbe5878e2feb7f",
                "required_tests": 26,
                "test_outcomes_matching_target": 2679,
            },
            "q4_r4_grade": {
                "commit": REVISIONS["5c34dc2e4"],
                "exceptions": ["W1 2.5% at n8", "C4 1% same-mesh legacy and S8R-weighted band", "A8 one route-count exception", "H1 open HIGH untagged historical plastic state"],
                "grade": "READY-WITH-EXCEPTIONS",
                "matrix_cells": "68/68",
                "product_digest": "f2acc4da71e052fc4c8bee250963d012c80a5a6d95111d68ab2183e900d0780d",
                "product_files": 741,
            },
            "report_sha256_cited": {k: v for k, v in sorted(EVIDENCE_REPORTS.items())},
            "s3_r1_grade": "READY with coordinator-granted exceptions and OPEN-LOW limitations",
            "scope_statement": "Cited history only. Gives no credit to the current release source or any installed build.",
        },
        "historical_files_sha256": historical_files,
        "links": {
            "L1_OCT19_AND_V6_TERMINAL_TO_QUALIFIED_Q4_R4": {
                "external_kernel_added": "_q4_assumed_strain_plasticity.py",
                "mechanics_changed": True,
                "oct19_external_dependencies": oct19_external,
                "oct19_closure": {"aggregate_sha256": oct19["mechanics_aggregate_sha256"], "symbol_count": len(oct19_symbols)},
                "qualified_target_closure": {"aggregate_sha256": upstream_agg, "symbol_count": len(upstream_symbols)},
                "symbol_delta_oct19_to_qualified_target": delta(oct19_symbols, upstream_symbols),
                "v6_terminal_blobs": TERMINAL_BLOBS,
                "v6_terminal_mechanics_unchanged_claim_extended": False,
            },
            "L2_UPSTREAM_E0C5F32AB_TO_RELEASE_CANDIDATE": {
                "claim": "SIX_CARRIER_SPELLING_ONLY_AFTER_REVERSE_MAP_NOT_EXTENDED_TO_OTHER_PATHS",
                "symbol_delta_upstream_to_candidate": delta(upstream_symbols, candidate["symbols"]),
                "unbound_scope": "Other runtime and packaging paths that differ from 5c34dc2e4 to e0c5f32ab (37 in the comparison) and transitive imports of the carriers are not bound by this record.",
            },
        },
        "mechanics": {
            "aggregate_sha256": candidate_agg,
            "logical_path": logical_entry,
            "roots": candidate["roots"],
            "schema": schema,
            "symbol_count": len(candidate["symbols"]),
            "symbols": candidate["symbols"],
        },
        "predecessor": {
            "oct19_as_reviewed": {
                "bytes": OCT19_AS_REVIEWED_BYTES, "git_blob": OCT19_AS_REVIEWED_BLOB,
                "path": OCT19_AS_REVIEWED, "sha256": OCT19_AS_REVIEWED_SHA256,
            },
            "oct19_current_historical": {
                "bytes": OCT19_RECORD_BYTES, "git_blob": OCT19_RECORD_BLOB,
                "path": OCT19_RECORD, "sha256": OCT19_RECORD_SHA256,
            },
            "current_historical_equals_as_reviewed": False,
            "current_historical_describes_current_execution": False,
            "v6_runtime_guard_successor_commit": REVISIONS["c31a223f"],
        },
        "historical_gaps_of_current_historical_record": gaps,
        "historical_record_lineage": lineage,
        "qualification_boundary": BOUNDARY,
        "recorded_to_live_spelling_map": {
            "applies": "source resolution only; recorded bytes and recorded logical paths are never rewritten",
            "path_map": {RECORDED_PREFIX + n: LIVE_PREFIX + n for n in CARRIERS},
            "text_live_to_recorded": [list(pair) for pair in TEXT_LIVE_TO_RECORDED],
        },
        "revisions": REVISIONS,
        "schema": SCHEMA,
        "sibling_imports_of_entry_module": imports,
        "static_runtime_performance_report": {
            "bytes": 4561,
            "path": "docs/STATIC_RUNTIME_PERFORMANCE.md",
            "sha256": STATIC_REPORT_SHA256,
        },
    }
    return record, problems


def verify_evidence_files() -> list[str]:
    out = []
    for path, expected in EVIDENCE_REPORTS.items():
        file = Path(path)
        if not file.is_file():
            out.append(f"missing evidence file {path}")
        elif hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            out.append(f"evidence file digest differs: {path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the record when there are no problems")
    parser.add_argument("--check", action="store_true", help="fail unless the on-disk record is fresh")
    parser.add_argument("--verify-evidence-files", action="store_true")
    parser.add_argument("--lineage-dir", type=Path, default=LINEAGE_DIR, help="directory with lineage.json and README.md")
    args = parser.parse_args(argv)
    record, problems = build(args.lineage_dir)
    if args.verify_evidence_files:
        problems += verify_evidence_files()
    data = canonical_bytes(record)
    print(f"record sha256 (would-be bytes): {sha256_upper(data)} ({len(data)} bytes)")
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    if problems:
        return 1
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != data:
            print("record missing or stale", file=sys.stderr)
            return 1
    elif args.write:
        OUTPUT.write_bytes(data)
        print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
