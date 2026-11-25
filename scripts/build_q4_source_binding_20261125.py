#!/usr/bin/env python3
"""Deterministic collector for the 2026-11-25 Q4 source-binding delta record.

SOURCE BINDING ONLY.  This record is chained from the 2026-11-24 delta record
(warped-Q4 drilling floor), which is itself a delta on the 2026-11-11
successor record.  It binds a later, deliberate change to a file the chain
binds, without rewriting either predecessor:

* the predecessor (2026-11-24) record is identified by bytes, Git blob and
  SHA-256, and its own predecessor (2026-11-11) by the 2026-11-24 pin;
* every file the chain binds by exact live identity (the 2026-11-24 live
  identity for files that record changed, the 2026-11-11 identity for every
  other file) is either unchanged, or listed here with its predecessor and
  live identities and the exact reverse line hunks that rebuild the
  predecessor bytes from the live bytes (so both predecessor readers keep
  running, on authenticated predecessor bytes);
* per-module AST symbol deltas and the live qualified-Q4 mechanics closure
  are recorded and recomputable.

The change bound by this record is S2, the bit-identical beam frame-math trim
in ``src/anyfesolver/elements.py``: ``BeamElement._beam_frame_and_transform``,
``BeamElement._linear_rigid_arm_map`` and ``_beam_rotation_matrix`` change,
``_contiguous_vector_norm`` and ``_frame_columns`` are added, no qualified-Q4
mechanics symbol changes and no result changes.  The 2026-11-11 claim that
live ``elements.py`` reproduces the upstream ``anysolver`` bytes (after the
spelling map) is replaced by this successor lineage: the live bytes bind
through this record and the 2026-11-24 record to the bytes the 2026-11-11
record bound.

The record grants no CI, scientific or physical acceptance, claims no
reviewer approval and must be independently reviewed outside the generating
invocation before its SHA-256 is pinned.

Usage (repository root, base revision present):

    python scripts/build_q4_source_binding_20261125.py            # dry run
    python scripts/build_q4_source_binding_20261125.py --write    # write record
    python scripts/build_q4_source_binding_20261125.py --check    # freshness
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reference_cases" / "e4_pl_source_binding_20261125.json"
SCHEMA = "anyfesolver.q4-source-binding-delta-v1"
BOUNDARY = "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"
PREDECESSOR = "docs/reference_cases/e4_pl_source_binding_20261124.json"
PREDECESSOR_SHA256 = "340CF7EB6B15B01799FA5838CC558BE74C1D6033C267120C5A142F6C257E50A7"
# origin/main after PR #79 merged; holds the 2026-11-24-bound bytes.
BASE_REVISION = "cb74b21599cf12d4ded3b82dd06c2161e01d1c5a"
ENTRY = "src/anyfesolver/e4_pl_element.py"
CONTRACT = "docs/reference_cases/e4_pl_s3_mixed_mesh_qualification_contract.json"
ELEMENTS = "src/anyfesolver/elements.py"
EXPECTED_MODULE_SYMBOLS = {
    "added": ["_contiguous_vector_norm", "_frame_columns"],
    "changed": [
        "BeamElement",
        "BeamElement._beam_frame_and_transform",
        "BeamElement._linear_rigid_arm_map",
        "_beam_rotation_matrix",
    ],
    "removed": [],
}
CHANGE = {
    "id": "BEAM_FRAME_EXACT_HELPERS_S2_20261125",
    "summary": (
        "S2 beam frame-math trim in src/anyfesolver/elements.py: "
        "BeamElement._beam_frame_and_transform takes its length norm through "
        "_contiguous_vector_norm; _beam_rotation_matrix takes its norms through "
        "_contiguous_vector_norm and builds the frame through _frame_columns; "
        "BeamElement._linear_rigid_arm_map builds the default (no stored offsets) "
        "rigid arm directly, selected by except AttributeError exactly as getattr "
        "with a default.  The two new module helpers evaluate exactly the original "
        "np.linalg.norm and np.column_stack expressions for C-contiguous float64 "
        "input and call them unchanged otherwise.  No qualified-Q4 mechanics symbol "
        "and no other elements.py symbol changes; results are bit-identical."
    ),
    "owner_decision_cited": (
        "Owner decision 2026-11-24/25: S2 may land once PR #79 has put its "
        "successor source-binding record on main, with a new successor record "
        "chained from it that replaces the 'reproduces upstream' claim by the "
        "successor lineage."
    ),
    "changed_module_symbols": {ELEMENTS: EXPECTED_MODULE_SYMBOLS},
    "bit_identical_results_claimed": True,
    "upstream_reproduction_claim_replaced_by_successor_lineage": True,
    "mechanics_changed": False,
    "warped_qualified_q4_results_changed": False,
    "planar_and_other_element_results_changed": False,
    "regression_test": "tests/test_beam_frame_exact_helpers.py",
}

_PREVIOUS = ROOT / "scripts" / "build_q4_source_binding_20261124.py"
_spec = importlib.util.spec_from_file_location("_q4_source_binding_20261124", _PREVIOUS)
assert _spec is not None and _spec.loader is not None
_prev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prev)

sha256_upper = _prev.sha256_upper
lf = _prev.lf
identity = _prev.identity
canonical_bytes = _prev.canonical_bytes
strict_json = _prev.strict_json
closure_payload = _prev.closure_payload
aggregate = _prev.aggregate
module_symbol_hashes = _prev.module_symbol_hashes
internal_imports = _prev.internal_imports
git_blob_at = _prev.git_blob_at
reverse_hunks = _prev.reverse_hunks
apply_reverse = _prev.apply_reverse
symbol_delta = _prev.symbol_delta


def chain_bound_files(
    record: dict[str, Any], origin: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Every live path the 2026-11-24 chain binds by exact live identity."""

    bound = dict(_prev.predecessor_bound_files(origin))
    for path, row in record["files"].items():
        bound[path] = row["live_raw"]
    return bound


def build() -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    predecessor_raw = (ROOT / PREDECESSOR).read_bytes()
    if sha256_upper(predecessor_raw) != PREDECESSOR_SHA256:
        problems.append("predecessor record bytes differ from the cited pin")
    predecessor = strict_json(predecessor_raw)
    origin_path = predecessor["predecessor"]["path"]
    origin_raw = (ROOT / origin_path).read_bytes()
    if origin_path != _prev.PREDECESSOR or sha256_upper(origin_raw) != _prev.PREDECESSOR_SHA256:
        problems.append("the 2026-11-24 record's predecessor is not its pinned record")
    if predecessor["predecessor"] != {"path": origin_path, **identity(origin_raw)}:
        problems.append("the 2026-11-24 record does not cite its predecessor's bytes")
    origin = strict_json(origin_raw)

    files: dict[str, Any] = {}
    for path, bound in sorted(chain_bound_files(predecessor, origin).items()):
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

    if sorted(files) != [ELEMENTS]:
        problems.append(f"changed bound files {sorted(files)} are not exactly [{ELEMENTS}]")
    elif {
        k: files[ELEMENTS]["module_symbols"][k] for k in ("added", "changed", "removed")
    } != EXPECTED_MODULE_SYMBOLS:
        problems.append("elements.py symbol delta differs from the declared S2 change")

    contract = strict_json((ROOT / CONTRACT).read_bytes())
    authority = contract["candidate"]["qualified_q4_mechanics_authority"]
    roots, required = set(authority["roots"]), set(authority["symbols"])
    live_entry = lf((ROOT / ENTRY).read_bytes()).decode("utf-8")
    live_closure = closure_payload(live_entry, roots, required, authority["schema"], authority["path"])
    old_symbols = predecessor["mechanics"]["symbols"]
    mechanics_delta = symbol_delta(old_symbols, live_closure["symbols"])
    if any(mechanics_delta.values()) or aggregate(live_closure) != predecessor["mechanics"]["aggregate_sha256"]:
        problems.append("the qualified-Q4 mechanics closure changed; S2 must not change it")
    live_imports = internal_imports(live_entry, sorted(live_closure["symbols"]))
    live_keys = {"live_path", "recorded_path", "live_raw", "live_lf", "portable_text", "permitted_raw"}
    previous_imports = [
        {k: v for k, v in row.items() if k not in live_keys}
        for row in origin["sibling_imports_of_entry_module"]
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
            "delta_from_predecessor": mechanics_delta,
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
