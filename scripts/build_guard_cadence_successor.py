#!/usr/bin/env python3
"""Deterministic builder for the nonlinear guard cadence successor record.

SOURCE BINDING ONLY.  Owner decisions G2, L1 and G1 (2026-11-24) plus the
canonical-state-JSON fast paths (V1) change files whose exact bytes are bound
by the Q4 source-binding chain 2026-11-11 -> 2026-11-24 -> 2026-11-25.  This
record is the next and newest link of that chain:

* its predecessor is the pinned 2026-11-25 delta record, identified by bytes,
  Git blob and SHA-256 (the way that record binds the 2026-11-24 one);
* its base is origin/main holding the 2026-11-25-bound bytes; every file the
  chain binds is either unchanged from the base, or listed here (``sealed
  carriers``) with the identity of its base bytes, of its live bytes and an
  exact reverse line patch from the live text to the base text, so every chain
  reader keeps running on authenticated predecessor bytes;
* the change commits are validated (ancestors of HEAD whose diff touches a
  sealed carrier) both when writing and by ``--check``.

Byte identities are checkout-independent: ``e4_pl_element.py`` is ``-text``
(raw bytes are the same on every platform) and every other hashed file that is
not pinned by ``.gitattributes`` is identified by its LF-normalised text.

Git is used read-only (``git show``, ``rev-parse``, ``merge-base
--is-ancestor``, ``diff --name-only``, ``log``).  No numerics run here.

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
BASE_REF = "72faf02b1"  # origin/main after S2 (#90); resolved to the full id below
# The chain this record extends (pinned identities, newest last).
CHAIN_ORIGIN = "docs/reference_cases/e4_pl_source_binding_20261111.json"
CHAIN_ORIGIN_SHA256 = "2716B80F2792574F4720CC282C4368E3E936E5C1353FFF8B63439E72B30F42B1"
CHAIN_DELTA = "docs/reference_cases/e4_pl_source_binding_20261124.json"
CHAIN_DELTA_SHA256 = "340CF7EB6B15B01799FA5838CC558BE74C1D6033C267120C5A142F6C257E50A7"
PREDECESSOR = "docs/reference_cases/e4_pl_source_binding_20261125.json"
PREDECESSOR_SHA256 = "CFBA512F71CF589FC8C2E31E7EE12FA7A58E90F6ADDBAA28EA7E52FD7B3B3032"
# Chain-bound files changed by this successor:
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
CHANGE_COMMIT = "1b74926691245c2dd4c4f117b9ef87acc2d0e46c"
CHANGE_COMMIT_TOUCHES = "src/anyfesolver/e4_pl_element.py"
# Every commit of this change whose diff touches a sealed carrier.
CHANGE_COMMITS = (
    "c569bd617 Canonical state JSON: exact-type fast paths with identical bytes",
    "11eb4531c Canonical fast paths: only under the internal exact-str path; trace tests",
    "1b7492669 Nonlinear guard cadence: Q4 class authority per evaluation (G2), lifecycle per step (L1), exact sweeps (G1)",
    "5c4c8f8ba Guard cadence: validate every window close, no publication before validation",
    "2a5ddf5a3 Guard window: re-check swallowed nested refusals and hand nested elements to the parent",
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


def strict_json(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise SystemExit(f"duplicate key {key!r}")
            out[key] = value
        return out

    return json.loads(raw, object_pairs_hook=pairs)


def is_ancestor(commit: str) -> bool:
    done = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"], capture_output=True
    )
    return done.returncode == 0


def touched(commit: str) -> list[str]:
    return git("diff", "--name-only", f"{commit}^", commit).decode().split()


def validated_change_commit(commit: object) -> str:
    """Return *commit* if it is a full id, an ancestor of HEAD and touches the entry carrier."""

    if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit(f"change.commit is not a full commit id: {commit!r}")
    if not is_ancestor(commit):
        raise SystemExit(f"change.commit {commit} is not an ancestor of HEAD")
    if CHANGE_COMMIT_TOUCHES not in touched(commit):
        raise SystemExit(f"change.commit {commit} does not touch {CHANGE_COMMIT_TOUCHES}")
    return commit


def validated_change_commits(commits: object) -> list[str]:
    """Each entry is "<abbrev id> <subject>" of an ancestor of HEAD whose diff
    touches a sealed carrier; the subject is the commit's own subject."""

    if type(commits) is not list or not commits:
        raise SystemExit("change.commits is not a non-empty list")
    carriers = {f"src/anyfesolver/{name}" for name, _portable in BOUND}
    for entry in commits:
        if type(entry) is not str or " " not in entry:
            raise SystemExit(f"change.commits entry is malformed: {entry!r}")
        short, subject = entry.split(" ", 1)
        if len(short) < 9 or any(c not in "0123456789abcdef" for c in short):
            raise SystemExit(f"change.commits id is not an abbreviated commit id: {short!r}")
        resolved = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", f"{short}^{{commit}}"],
            capture_output=True,
        )
        if resolved.returncode != 0:
            raise SystemExit(f"change.commits {short} is not a commit")
        full = resolved.stdout.decode().strip()
        if not is_ancestor(full):
            raise SystemExit(f"change.commits {short} is not an ancestor of HEAD")
        if not carriers & set(touched(full)):
            raise SystemExit(f"change.commits {short} does not touch a sealed carrier")
        if git("log", "-1", "--format=%s", full).decode().strip() != subject:
            raise SystemExit(f"change.commits {short}: subject differs from the commit's subject")
    if CHANGE_COMMIT[:9] not in {entry.split(" ", 1)[0] for entry in commits}:
        raise SystemExit("change.commits does not list change.commit")
    return list(commits)


def chain_bound() -> dict[str, tuple[bool, dict]]:
    """Every path the chain 2026-11-11 -> 2026-11-24 -> 2026-11-25 binds:
    path -> (portable, identity).  Exact-raw files carry the newest link's live
    raw identity; portable auto-text files carry the 2026-11-11 LF identity
    (the delta links bind exact-raw files only)."""

    origin_raw = (ROOT / CHAIN_ORIGIN).read_bytes()
    delta_raw = (ROOT / CHAIN_DELTA).read_bytes()
    predecessor_raw = (ROOT / PREDECESSOR).read_bytes()
    for raw, pin, path in (
        (origin_raw, CHAIN_ORIGIN_SHA256, CHAIN_ORIGIN),
        (delta_raw, CHAIN_DELTA_SHA256, CHAIN_DELTA),
        (predecessor_raw, PREDECESSOR_SHA256, PREDECESSOR),
    ):
        if ident(raw)["sha256"] != pin:
            raise SystemExit(f"{path}: bytes differ from the pinned chain identity")
    origin, delta, predecessor = (strict_json(r) for r in (origin_raw, delta_raw, predecessor_raw))
    if delta["predecessor"] != {"path": CHAIN_ORIGIN, **ident(origin_raw)}:
        raise SystemExit("the 2026-11-24 delta does not chain from the 2026-11-11 record")
    if predecessor["predecessor"] != {"path": CHAIN_DELTA, **ident(delta_raw)}:
        raise SystemExit("the 2026-11-25 delta does not chain from the 2026-11-24 delta")
    bound: dict[str, tuple[bool, dict]] = {}
    rows = list(origin["carriers"].values()) + list(origin["sibling_imports_of_entry_module"])
    rows += list(origin["historical_gaps_of_current_historical_record"])
    for row in rows:
        if "live_raw" in row:
            entry = (False, row["live_raw"])
        elif row.get("portable_text") is True:
            entry = (True, row["live_lf"])
        else:
            raise SystemExit(f"{row['live_path']}: chain row has no bound identity")
        if bound.setdefault(row["live_path"], entry) != entry:
            raise SystemExit(f"{row['live_path']}: chain rows bind different identities")
    for record in (delta, predecessor):
        for path, row in record["files"].items():
            if bound.get(path, (False, None))[0]:
                raise SystemExit(f"{path}: a delta link changes a portable file")
            bound[path] = (False, row["live_raw"])
    return bound


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
    base_commit = git("rev-parse", f"{BASE_REF}^{{commit}}").decode().strip()
    base_tree = git("rev-parse", f"{BASE_REF}^{{tree}}").decode().strip()
    chain = chain_bound()
    # The base holds exactly the chain-bound bytes; the files this successor
    # changes among them are exactly BOUND.
    changed = {}
    for path, (portable, bound) in sorted(chain.items()):
        prior = git("show", f"{base_commit}:{path}")
        if ident(lf(prior) if portable else prior) != bound:
            raise SystemExit(f"{path}: base bytes are not the chain-bound bytes")
        live = (ROOT / path).read_bytes()
        if (lf(live) if portable else live) != (lf(prior) if portable else prior):
            changed[path] = portable
    expected = {f"src/anyfesolver/{name}": portable for name, portable in BOUND}
    if changed != expected:
        raise SystemExit(f"changed chain-bound files {changed} are not exactly {expected}")
    bound = {}
    for name, portable in BOUND:
        rel = f"src/anyfesolver/{name}"
        live = (ROOT / rel).read_bytes()
        prior = git("show", f"{base_commit}:{rel}")
        current_form, prior_form = (lf(live), lf(prior)) if portable else (live, prior)
        hunks = reverse_patch(current_form, prior_form)
        if apply_patch(current_form, hunks) != prior_form:
            raise SystemExit(f"{rel}: reverse line patch does not rebuild the base bytes")
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
        if rel in chain:
            raise SystemExit(f"{rel} is chain-bound; it must be a sealed carrier")
        other[name] = {
            "live_path": rel,
            "prior_lf_git_blob": ident(lf(git("show", f"{base_commit}:{rel}")))["git_blob"],
            "current_lf_git_blob": ident(lf((ROOT / rel).read_bytes()))["git_blob"],
        }
    predecessor = ROOT / PREDECESSOR
    note = ROOT / "docs" / "NONLINEAR_STATIC_GUARD_SPEEDUP.md"
    return {
        "schema": "anysolver.qualified-shell-nonlinear-guard-cadence-successor-v1",
        "base": {"commit": base_commit, "tree": base_tree},
        "predecessor": {"path": PREDECESSOR, **ident(predecessor.read_bytes())},
        "change": {
            "commit": validated_change_commit(CHANGE_COMMIT),
            "commits": validated_change_commits(list(CHANGE_COMMITS)),
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
        # Validated, not copied: the recorded commits must be the builder's
        # validated change commits (the canonical comparison below binds them).
        validated_change_commit(current["change"]["commit"])
        validated_change_commits(current["change"]["commits"])
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
