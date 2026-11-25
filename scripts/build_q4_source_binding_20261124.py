#!/usr/bin/env python3
"""Deterministic collector for the 2026-11-24 Q4 source-binding delta record.

SOURCE BINDING ONLY.  The 2026-11-11 successor record binds the live sources
to the immutable Q4 history and to the upstream ``anysolver`` bytes.  Its
schema requires live == upstream (after the spelling map), so it cannot
describe a later, deliberate source change.  This record binds such a change
as a delta on top of that predecessor, without rewriting it:

* the predecessor record is identified by bytes, Git blob and SHA-256;
* every file the predecessor binds by live identity is either unchanged, or
  listed here with its predecessor and live identities and the exact reverse
  line hunks that rebuild the predecessor bytes from the live bytes (so the
  predecessor reader keeps running, on authenticated predecessor bytes);
* per-module AST symbol deltas and the live qualified-Q4 mechanics closure
  are recorded and recomputable.

The change bound by this record is the owner-decided warped-Q4 drilling floor
(spot-check finding 1, option C).  The record grants no CI, scientific or
physical acceptance, claims no reviewer approval and must be independently
reviewed outside the generating invocation before its SHA-256 is pinned.

Usage (repository root, base revision present):

    python scripts/build_q4_source_binding_20261124.py            # dry run
    python scripts/build_q4_source_binding_20261124.py --write    # write record
    python scripts/build_q4_source_binding_20261124.py --check    # freshness
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reference_cases" / "e4_pl_source_binding_20261124.json"
SCHEMA = "anyfesolver.q4-source-binding-delta-v1"
BOUNDARY = "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"
PREDECESSOR = "docs/reference_cases/e4_pl_source_binding_20261111.json"
PREDECESSOR_SHA256 = "2716B80F2792574F4720CC282C4368E3E936E5C1353FFF8B63439E72B30F42B1"
# origin/main when the change was branched; holds the predecessor-bound bytes.
BASE_REVISION = "b9b4a6e808d664ae08a3387b372c6cf7ecbfdf45"
ENTRY = "src/anyfesolver/e4_pl_element.py"
CONTRACT = "docs/reference_cases/e4_pl_s3_mixed_mesh_qualification_contract.json"
CHANGE = {
    "id": "WARPED_Q4_DRILLING_FLOOR_20261124",
    "summary": (
        "Warped qualified Q4 facets (equation7_frame warpage above planar_tolerance, "
        "varying-frame kernels) apply max(drilling_stabilization, 0.1) in the linear, "
        "nonlinear and batched kernels; planar facets and every other element apply "
        "the stored value unchanged; the stored attribute, to_dict and cache keys keep "
        "the raw value."
    ),
    "owner_decision_cited": (
        "Spot-check finding 1 (MacNeal-Harder twisted beam +31 %/+24 % at 24x4 with "
        "alpha=1e-3, non-convergent); owner selected option C (floor alpha at 0.1 only "
        "for warped qualified Q4) and approved re-sealing the qualified-Q4 source "
        "binding for this change."
    ),
    "mechanics_changed": True,
    "warped_qualified_q4_results_changed": True,
    "planar_and_other_element_results_changed": False,
    "regression_test": "tests/test_warped_q4_drilling_floor.py",
}

_PREVIOUS = ROOT / "scripts" / "build_q4_source_binding_20261111.py"
_spec = importlib.util.spec_from_file_location("_q4_source_binding_20261111", _PREVIOUS)
assert _spec is not None and _spec.loader is not None
_prev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prev)

sha256_upper = _prev.sha256_upper
git_blob_id = _prev.git_blob_id
lf = _prev.lf
identity = _prev.identity
canonical_bytes = _prev.canonical_bytes
strict_json = _prev.strict_json
closure_payload = _prev.closure_payload
aggregate = _prev.aggregate
module_symbol_hashes = _prev.module_symbol_hashes
internal_imports = _prev.internal_imports

_LINE = re.compile(rb"[^\n]*\n|[^\n]+\Z")


def lines(data: bytes) -> list[bytes]:
    return _LINE.findall(data)


def git_blob_at(rev: str, path: str) -> bytes | None:
    done = subprocess.run(
        ["git", "cat-file", "blob", f"{rev}:{path}"], cwd=ROOT, capture_output=True, check=False
    )
    return done.stdout if done.returncode == 0 else None


def reverse_hunks(predecessor: bytes, live: bytes) -> list[dict[str, Any]]:
    """Line hunks that turn ``live`` back into ``predecessor`` (applied bottom-up)."""

    old, new = lines(predecessor), lines(live)
    hunks = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunks.append({
            "live_line_start": j1,
            "live_line_end": j2,
            "live_lines_sha256": sha256_upper(b"".join(new[j1:j2])),
            "predecessor_text": b"".join(old[i1:i2]).decode("utf-8"),
        })
    return hunks


def apply_reverse(live: bytes, hunks: list[dict[str, Any]]) -> bytes:
    current = lines(live)
    for hunk in sorted(hunks, key=lambda h: h["live_line_start"], reverse=True):
        start, end = hunk["live_line_start"], hunk["live_line_end"]
        current[start:end] = [hunk["predecessor_text"].encode("utf-8")]
    return b"".join(current)


def predecessor_bound_files(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every live path the predecessor binds by exact live identity."""

    bound: dict[str, dict[str, Any]] = {}
    for row in record["carriers"].values():
        if "live_raw" in row:
            bound[row["live_path"]] = row["live_raw"]
    for row in record["sibling_imports_of_entry_module"]:
        if "live_raw" in row:
            bound.setdefault(row["live_path"], row["live_raw"])
    for row in record["historical_gaps_of_current_historical_record"]:
        bound.setdefault(row["live_path"], row["live_raw"])
    return bound


def symbol_delta(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(after) - set(before)),
        "changed": sorted(n for n in set(after) & set(before) if after[n] != before[n]),
        "removed": sorted(set(before) - set(after)),
    }


def build() -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    predecessor_raw = (ROOT / PREDECESSOR).read_bytes()
    if sha256_upper(predecessor_raw) != PREDECESSOR_SHA256:
        problems.append("predecessor record bytes differ from the cited pin")
    predecessor = strict_json(predecessor_raw)

    files: dict[str, Any] = {}
    for path, bound in sorted(predecessor_bound_files(predecessor).items()):
        base = git_blob_at(BASE_REVISION, path)
        live = (ROOT / path).read_bytes()
        if base is None:
            problems.append(f"{path}: absent at base revision {BASE_REVISION}")
            continue
        if identity(base) != bound:
            problems.append(f"{path}: base-revision bytes are not the predecessor-bound bytes")
            continue
        if live == base:
            continue
        hunks = reverse_hunks(base, live)
        if apply_reverse(live, hunks) != base:
            problems.append(f"{path}: reverse hunks do not rebuild the predecessor bytes")
        before = module_symbol_hashes(lf(base).decode("utf-8"))
        after = module_symbol_hashes(lf(live).decode("utf-8"))
        files[path] = {
            "live_lf": identity(lf(live)),
            "live_raw": identity(live),
            "module_symbols": {
                **symbol_delta(before, after),
                "live_aggregate_sha256": aggregate(after),
                "predecessor_aggregate_sha256": aggregate(before),
            },
            "predecessor_raw": bound,
            "reverse_hunks": hunks,
        }

    contract = strict_json((ROOT / CONTRACT).read_bytes())
    authority = contract["candidate"]["qualified_q4_mechanics_authority"]
    roots, required = set(authority["roots"]), set(authority["symbols"])
    live_entry = lf((ROOT / ENTRY).read_bytes()).decode("utf-8")
    live_closure = closure_payload(live_entry, roots, required, authority["schema"], authority["path"])
    old_symbols = predecessor["mechanics"]["symbols"]
    live_imports = internal_imports(live_entry, sorted(live_closure["symbols"]))
    live_keys = {"live_path", "recorded_path", "live_raw", "live_lf", "portable_text", "permitted_raw"}
    previous_imports = [
        {k: v for k, v in row.items() if k not in live_keys}
        for row in predecessor["sibling_imports_of_entry_module"]
    ]
    if live_imports != previous_imports:
        problems.append("sibling imports of the entry module changed; extend this collector")

    record = {
        "adoption": {
            "independent_review_required_outside_generating_invocation": True,
            "owner_adoption_recorded": False,
            "reviewer_approval_claimed": False,
            "status": "UNREVIEWED_CANDIDATE_RECORD",
        },
        "base_revision": BASE_REVISION,
        "boundary_flags": {key: False for key in predecessor["boundary_flags"]},
        "change": CHANGE,
        "files": files,
        "mechanics": {
            "aggregate_sha256": aggregate(live_closure),
            "delta_from_predecessor": symbol_delta(old_symbols, live_closure["symbols"]),
            "logical_path": live_closure["path"],
            "predecessor_aggregate_sha256": predecessor["mechanics"]["aggregate_sha256"],
            "roots": live_closure["roots"],
            "schema": live_closure["schema"],
            "symbol_count": len(live_closure["symbols"]),
            "symbols": live_closure["symbols"],
        },
        "predecessor": {"path": PREDECESSOR, **identity(predecessor_raw)},
        "qualification_boundary": BOUNDARY,
        "schema": SCHEMA,
        "sibling_imports_of_entry_module_unchanged": live_imports == previous_imports,
    }
    if not files:
        problems.append("no predecessor-bound file changed; no delta record is needed")
    return record, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the record when there are no problems")
    parser.add_argument("--check", action="store_true", help="fail unless the on-disk record is fresh")
    args = parser.parse_args(argv)
    record, problems = build()
    data = canonical_bytes(record)
    print(f"record sha256 (would-be bytes): {sha256_upper(data)} ({len(data)} bytes)")
    for path, row in record["files"].items():
        print(f"  {path}: {len(row['reverse_hunks'])} hunks, symbols {row['module_symbols']['changed'] + row['module_symbols']['added']}")
    print(f"  mechanics delta: {record['mechanics']['delta_from_predecessor']}")
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
