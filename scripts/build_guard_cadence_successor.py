#!/usr/bin/env python3
"""Deterministic builder for the nonlinear guard cadence successor record.

SOURCE BINDING ONLY.  Owner decisions G2, L1 and G1 (2026-11-24) plus the
canonical-state-JSON fast paths (V1) change files whose exact bytes are bound
by ``docs/reference_cases/e4_pl_source_binding_20261111.json``.  This builder
records, for every such bound file, the identity of its bytes at the base
commit, the identity of the live bytes and an exact reverse line patch from the
live text to the base text, so the source-binding record keeps being verified
in full on the reproduced base bytes while the live bytes are pinned exactly.

Byte identities are checkout-independent: ``e4_pl_element.py`` is ``-text``
(raw bytes are the same on every platform) and every other hashed file that is
not pinned by ``.gitattributes`` is identified by its LF-normalised text.  The
change commit is validated (ancestor of HEAD whose diff touches the bound
entry carrier) both when writing and by ``--check``.

Git is used read-only (``git show``, ``merge-base --is-ancestor``,
``diff --name-only``).  No numerics run here.

Usage (repository root):
    python scripts/build_guard_cadence_successor.py EVIDENCE.json   # write
    python scripts/build_guard_cadence_successor.py --check          # freshness
"""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reference_cases" / "qualified_shell_nonlinear_guard_cadence_successor.json"
BASE_REF = "071910d60"  # origin/main before V1; resolved to the full id below
# Bound files of the source-binding record changed by this successor:
# (file, portable auto-text?)  Portable files are identified by their LF form.
BOUND = (("e4_pl_element.py", False), ("e4_pl_s3_state.py", True))
OTHER = (
    "e4_pl_s3_v2d_state.py",
    "matrix_assembly.py",
    "nonlinear_performance_batch_c.py",
    "nonlinear_performance_bootstrap.py",
    "nonlinear_static.py",
)
# The commit that introduced the G2/L1/G1 change to the bound entry carrier.
CHANGE_COMMIT = "7a5bad38fef15d29db90008a1ebb08d0d6a442c1"
CHANGE_COMMIT_TOUCHES = "src/anyfesolver/e4_pl_element.py"
CHANGE_COMMITS = (
    "d81e40216 Canonical state JSON: exact-type fast paths with identical bytes",
    "8d8fbd3ed Canonical fast paths: only under the internal exact-str path; trace tests",
    "7a5bad38f Nonlinear guard cadence: Q4 class authority per evaluation (G2), lifecycle per step (L1), exact sweeps (G1)",
    "ac2bfd828 Guard cadence: validate every window close, no publication before validation",
)


def ident(data: bytes) -> dict:
    return {
        "bytes": len(data),
        "git_blob": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def validated_change_commit(commit: object) -> str:
    """Return *commit* if it is a full id, an ancestor of HEAD and touches the entry carrier."""

    if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit(f"change.commit is not a full commit id: {commit!r}")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise SystemExit(f"change.commit {commit} is not an ancestor of HEAD")
    touched = git("diff", "--name-only", f"{commit}^", commit).decode().split()
    if CHANGE_COMMIT_TOUCHES not in touched:
        raise SystemExit(f"change.commit {commit} does not touch {CHANGE_COMMIT_TOUCHES}")
    return commit


def reverse_patch(current: bytes, prior: bytes) -> list:
    cur = current.decode("utf-8").splitlines(keepends=True)
    pri = prior.decode("utf-8").splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, cur, pri, autojunk=False)
    return [[i1, i2, pri[j1:j2]] for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def apply_patch(current: bytes, hunks: list) -> bytes:
    lines = current.decode("utf-8").splitlines(keepends=True)
    for start, end, replacement in reversed(hunks):
        lines[start:end] = replacement
    return "".join(lines).encode("utf-8")


def build(evidence: dict) -> dict:
    base_commit = git("rev-parse", BASE_REF).decode().strip()
    base_tree = git("rev-parse", f"{BASE_REF}^{{tree}}").decode().strip()
    bound = {}
    for name, portable in BOUND:
        rel = f"src/anyfesolver/{name}"
        live = (ROOT / rel).read_bytes()
        prior = git("show", f"{base_commit}:{rel}")
        current_form, prior_form = (lf(live), lf(prior)) if portable else (live, prior)
        hunks = reverse_patch(current_form, prior_form)
        assert apply_patch(current_form, hunks) == prior_form
        row = {
            "live_path": rel,
            "portable_text": portable,
            "prior_lf": ident(lf(prior)),
            "current_lf": ident(lf(live)),
            "reverse_line_patch": hunks,
        }
        if not portable:
            row["prior_raw"] = ident(prior)
            row["current_raw"] = ident(live)
        bound[name] = row
    other = {}
    for name in OTHER:
        rel = f"src/anyfesolver/{name}"
        other[name] = {
            "live_path": rel,
            "prior_lf_git_blob": ident(lf(git("show", f"{base_commit}:{rel}")))["git_blob"],
            "current_lf_git_blob": ident(lf((ROOT / rel).read_bytes()))["git_blob"],
        }
    binding = ROOT / "docs" / "reference_cases" / "e4_pl_source_binding_20261111.json"
    note = ROOT / "docs" / "NONLINEAR_STATIC_GUARD_SPEEDUP.md"
    return {
        "schema": "anysolver.qualified-shell-nonlinear-guard-cadence-successor-v1",
        "base": {"commit": base_commit, "tree": base_tree},
        "change": {
            "commit": validated_change_commit(CHANGE_COMMIT),
            "commits": list(CHANGE_COMMITS),
            "sealed_carriers": bound,
            "other_changed_sources": other,
            "scope": [
                "G2: Q4 class/namespace authority validated once per Newton evaluation or lease scan window (first use and window close)",
                "G2: per-element instance, quadrature and epoch checks unchanged on every guarded call",
                "L1: solver-internal complete lifecycle scans of a force-control step attempt at step entry and before commit",
                "L1: caller-controlled observations keep the immediate complete lifecycle scan",
                "G1: exact C-speed identity sweeps with unchanged order, messages and outcomes",
                "V1: canonical state JSON exact-type fast paths with identical bytes",
            ],
        },
        "owner_decisions": {
            "date": "2026-11-24",
            "approved": ["G2", "L1", "G1"],
            "not_approved_not_implemented": [
                "S2 final seal replay reduction",
                "V2 frozen V2D states",
                "sampling",
            ],
            "trust_boundary": (
                "NumPy/SciPy/Python builtins/ANYmaterial internals and ANYfeSolver classes and "
                "guard machinery are not replaced at run time; hostile data, instance attributes "
                "and subclasses stay in scope; concurrent mid-request mutation is out of contract"
            ),
        },
        "qualification_boundary": {
            "default_policy_changed": False,
            "formulation_ids_changed": False,
            "mass_policy_changed": False,
            "mechanics_coefficients_or_operators_changed": False,
            "mechanics_symbols_changed": False,
            "numerical_tolerances_changed": False,
            "prior_evidence_modified": False,
            "qualified_recovery_changed": False,
        },
        "preserved_evidence": {"path": binding.relative_to(ROOT).as_posix(), **ident(binding.read_bytes())},
        "validation": {
            "status": "PENDING",
            "bit_identical_result_hashes": evidence["hashes"],
            "regression_tests": "tests/test_nonlinear_guard_cadence.py",
            # Auto-text file: identified by its LF form (checkout-independent).
            "performance_note": {
                "path": note.relative_to(ROOT).as_posix(),
                "text_identity": "lf",
                **ident(lf(note.read_bytes())),
            },
        },
        "review": {"outcome": "PENDING", "finding_count": None, "reviewer": "PENDING", "scope": "PENDING"},
        "terminal": "PENDING_INDEPENDENT_REVIEW",
    }


def canonical(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        current = json.loads(OUTPUT.read_bytes())
        # Validated, not copied: the recorded commit must be the builder's
        # validated change commit (the canonical comparison below binds it).
        validated_change_commit(current["change"]["commit"])
        rebuilt = build({"hashes": current["validation"]["bit_identical_result_hashes"]})
        rebuilt["validation"]["status"] = current["validation"]["status"]
        rebuilt["review"] = current["review"]
        rebuilt["terminal"] = current["terminal"]
        fresh = canonical(rebuilt) == OUTPUT.read_bytes()
        print("fresh" if fresh else "STALE")
        return 0 if fresh else 1
    evidence = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    OUTPUT.write_bytes(canonical(build(evidence)))
    print(OUTPUT, OUTPUT.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
