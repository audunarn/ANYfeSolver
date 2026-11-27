#!/usr/bin/env python3
"""Deterministic builder for the qualified Q4 planar batch successor record.

SOURCE BINDING ONLY.  The owner-approved L4 change (batched, bit-identical
formation and default recovery numerics for the exact qualified Q4 planar
route, run inside the existing authority lease) changes files whose exact
bytes the source-binding chain binds
(2026-11-11 -> 2026-11-24 -> 2026-11-25 -> guard cadence successor).  This
record is the next and newest link of that chain:

* its predecessor is the pinned guard cadence successor
  (``qualified_shell_nonlinear_guard_cadence_successor.json``), identified by
  bytes, Git blob and SHA-256; this record is that link's single recorded
  successor;
* its base is origin/main holding the guard-cadence-bound bytes; every file
  the chain or the guard cadence successor identifies is either unchanged from
  the base, or listed here (``sealed carriers``) with the identity of its base
  bytes, of its live bytes and an exact reverse line patch from the live text
  to the base text, so every earlier chain reader keeps running on
  authenticated predecessor bytes (``planar_batch_prior``);
* the change commits are validated (ancestors of HEAD whose diff touches a
  sealed carrier) both when writing and by ``--check``.

Byte identities are checkout-independent: ``e4_pl_element.py`` is ``-text``
(raw bytes are the same on every platform) and every other hashed file that
is not pinned by ``.gitattributes`` is identified by its LF-normalised text.
Git is used read-only.  No numerics run here.

Usage (repository root):
    python scripts/build_q4_planar_batch_successor.py EVIDENCE.json   # write
    python scripts/build_q4_planar_batch_successor.py --check          # freshness
"""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reference_cases" / "qualified_q4_planar_batch_successor.json"
SCHEMA = "anysolver.qualified-q4-planar-batch-successor-v1"
BASE_REF = "3b5f23d5d"  # origin/main after #93 (G2 #92 is d4cf5011f); resolved to the full id below
PREDECESSOR = "docs/reference_cases/qualified_shell_nonlinear_guard_cadence_successor.json"
PREDECESSOR_SHA256 = "96EB8040773D4E05F33938C333F34BF63372428D0A7EEE5D099867690A88936D"
PERFORMANCE_NOTE = "docs/Q4_PLANAR_BATCH_SPEEDUP.md"
# Files identified by an earlier link and changed here: (file, portable auto-text?).
BOUND = (("e4_pl_element.py", False), ("matrix_assembly.py", True))
# Changed or new sources no earlier link identifies (identity only).
OTHER = ("_q4_planar_batch.py", "assembly.py")
# The commit that introduced the L4 wiring to the bound entry carrier.
CHANGE_COMMIT = "704536ab0c590f09af99ed53a09d8b74eefda119"
CHANGE_COMMIT_TOUCHES = "src/anyfesolver/e4_pl_element.py"
# Every commit of this change whose diff touches a sealed carrier.
CHANGE_COMMITS = (
    "704536ab0 Wire the batched Q4 planar numerics into assembly and recovery (L4)",
    "84810a7dc Publish batch rows in assembly order and match the per-element backward error",
)
QUALIFICATION_BOUNDARY = {
    "default_policy_changed": False,
    "formulation_ids_changed": False,
    "mass_policy_changed": False,
    "mechanics_coefficients_or_operators_changed": False,
    "mechanics_symbols_changed": False,
    "numerical_tolerances_changed": False,
    "prior_evidence_modified": False,
    "qualified_recovery_changed": False,
}


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


def canonical(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def apply_patch(current: bytes, hunks: list) -> bytes:
    lines = current.decode("utf-8").splitlines(keepends=True)
    for start, end, replacement in reversed(hunks):
        lines[start:end] = replacement
    return "".join(lines).encode("utf-8")


def reverse_patch(current: bytes, prior: bytes) -> list:
    cur = current.decode("utf-8").splitlines(keepends=True)
    pri = prior.decode("utf-8").splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, cur, pri, autojunk=False)
    return [[i1, i2, pri[j1:j2]] for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def planar_batch_prior(problems: list[str], relative: str, raw: bytes) -> bytes:
    """The bytes the guard cadence successor binds as live for *relative*:
    *raw*, or for a file this record changes, the prior bytes rebuilt through
    this record.

    This record is the single recorded successor of the guard cadence
    successor: its ``predecessor`` must be exactly the pinned guard cadence
    record bytes.  It binds the exact live bytes of each changed file
    (``current_raw`` / ``current_lf``) to an exact reverse line patch that
    reproduces the bytes the earlier chain binds (``prior_raw`` /
    ``prior_lf``).  Any mismatch is a problem, never a silent fallback.  Files
    it does not bind are returned as given; without the record the bytes are
    returned unchanged.
    """

    if not OUTPUT.is_file():
        return raw
    record_raw = OUTPUT.read_bytes()
    try:
        record = strict_json(record_raw)
    except (SystemExit, ValueError) as exc:
        problems.append(f"planar batch successor is not strict JSON: {exc}")
        return raw
    if record_raw != canonical(record):
        problems.append("planar batch successor is not canonical JSON")
        return raw
    if record.get("schema") != SCHEMA:
        problems.append("planar batch successor schema differs")
        return raw
    if record.get("qualification_boundary") != QUALIFICATION_BOUNDARY:
        problems.append("planar batch successor claims a qualification change")
        return raw
    predecessor_raw = (ROOT / PREDECESSOR).read_bytes()
    if ident(predecessor_raw)["sha256"] != PREDECESSOR_SHA256 or record.get("predecessor") != {
        "path": PREDECESSOR, **ident(predecessor_raw),
    }:
        problems.append("planar batch successor does not chain from the pinned guard cadence record")
        return raw
    for name, row in record["change"]["sealed_carriers"].items():
        if row["live_path"] != relative:
            continue
        if row["live_path"] != f"src/anyfesolver/{name}":
            problems.append(f"planar batch successor {name}: row binds another file")
            return raw
        portable = dict(BOUND).get(name)
        if portable is None or row["portable_text"] is not portable:
            problems.append(f"planar batch successor {name}: portable_text differs")
            return raw
        canonical_text = lf(raw)
        if ident(canonical_text) != row["current_lf"]:
            problems.append(f"planar batch successor {name}: live LF bytes are not the recorded current bytes")
            return raw
        if portable:
            if raw not in (canonical_text, canonical_text.replace(b"\n", b"\r\n")):
                problems.append(f"planar batch successor {name}: live raw is not uniform LF/CRLF")
                return raw
            source = canonical_text
        else:
            if ident(raw) != row["current_raw"]:
                problems.append(f"planar batch successor {name}: live raw bytes are not the recorded current bytes")
                return raw
            source = raw
        prior = apply_patch(source, row["reverse_line_patch"])
        if ident(lf(prior)) != row["prior_lf"] or (not portable and ident(prior) != row["prior_raw"]):
            problems.append(f"planar batch successor {name}: reverse patch does not reproduce the prior bytes")
            return raw
        if portable and raw != canonical_text:
            return prior.replace(b"\n", b"\r\n")
        return prior
    return raw


def read_prior(relative: str) -> bytes:
    """Working-tree bytes of *relative* rebuilt through this record; a problem raises."""

    problems: list[str] = []
    data = planar_batch_prior(problems, relative, (ROOT / relative).read_bytes())
    if problems:
        raise SystemExit("; ".join(problems))
    return data


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


def _guard_cadence_builder():
    spec = importlib.util.spec_from_file_location(
        "_planar_batch_guard_cadence_builder", ROOT / "scripts" / "build_guard_cadence_successor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predecessor_bound() -> tuple[dict[str, tuple[bool, dict]], dict[str, str]]:
    """What the chain up to and including the guard cadence successor binds.

    Returns ``(bound, other)``: ``bound`` maps every chain-bound path to
    ``(portable, identity)`` with the guard cadence successor's current
    identities applied; ``other`` maps each of that successor's other changed
    sources to its current LF Git blob."""

    raw = (ROOT / PREDECESSOR).read_bytes()
    if ident(raw)["sha256"] != PREDECESSOR_SHA256:
        raise SystemExit(f"{PREDECESSOR}: bytes differ from the pinned chain identity")
    predecessor = strict_json(raw)
    if raw != canonical(predecessor):
        raise SystemExit(f"{PREDECESSOR}: not canonical JSON")
    bound = dict(_guard_cadence_builder().chain_bound())
    for name, row in predecessor["change"]["sealed_carriers"].items():
        path = row["live_path"]
        if path != f"src/anyfesolver/{name}" or path not in bound:
            raise SystemExit(f"{PREDECESSOR}: {name} is not a chain-bound carrier")
        portable = row["portable_text"]
        bound[path] = (portable, row["current_lf"] if portable else row["current_raw"])
    other = {}
    for name, row in predecessor["change"]["other_changed_sources"].items():
        if row["live_path"] != f"src/anyfesolver/{name}" or row["live_path"] in bound:
            raise SystemExit(f"{PREDECESSOR}: other source {name} is malformed")
        other[row["live_path"]] = row["current_lf_git_blob"]
    return bound, other


def build(evidence: dict) -> dict:
    base_commit = git("rev-parse", f"{BASE_REF}^{{commit}}").decode().strip()
    base_tree = git("rev-parse", f"{BASE_REF}^{{tree}}").decode().strip()
    bound, other_bound = predecessor_bound()
    # The base holds exactly the bytes the predecessor binds as live; the
    # files this successor changes among them are exactly BOUND.
    changed = {}
    for path, (portable, identity) in sorted(bound.items()):
        prior = git("show", f"{base_commit}:{path}")
        if ident(lf(prior) if portable else prior) != identity:
            raise SystemExit(f"{path}: base bytes are not the chain-bound bytes")
        live = (ROOT / path).read_bytes()
        if (lf(live) if portable else live) != (lf(prior) if portable else prior):
            changed[path] = portable
    for path, blob in sorted(other_bound.items()):
        prior = lf(git("show", f"{base_commit}:{path}"))
        if ident(prior)["git_blob"] != blob:
            raise SystemExit(f"{path}: base bytes are not the guard-cadence-bound bytes")
        if lf((ROOT / path).read_bytes()) != prior:
            changed[path] = True
    expected = {f"src/anyfesolver/{name}": portable for name, portable in BOUND}
    if changed != expected:
        raise SystemExit(f"changed bound files {changed} are not exactly {expected}")
    carriers = {}
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
        carriers[name] = row
    other = {}
    for name in OTHER:
        rel = f"src/anyfesolver/{name}"
        if rel in bound or rel in other_bound:
            raise SystemExit(f"{rel} is bound by an earlier link; it must be a sealed carrier")
        exists = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{base_commit}:{rel}"], capture_output=True
        ).returncode == 0
        prior_blob = ident(lf(git("show", f"{base_commit}:{rel}")))["git_blob"] if exists else None
        other[name] = {
            "live_path": rel,
            "prior_lf_git_blob": prior_blob,
            "current_lf_git_blob": ident(lf((ROOT / rel).read_bytes()))["git_blob"],
        }
    note = ROOT / PERFORMANCE_NOTE
    return {
        "schema": SCHEMA,
        "base": {"commit": base_commit, "tree": base_tree},
        "predecessor": {"path": PREDECESSOR, **ident((ROOT / PREDECESSOR).read_bytes())},
        "change": {
            "commit": validated_change_commit(CHANGE_COMMIT),
            "commits": validated_change_commits(list(CHANGE_COMMITS)),
            "sealed_carriers": carriers,
            "other_changed_sources": other,
            "scope": [
                "L4: formation cache misses of exact qualified Q4 planar elements run through one batched, "
                "bit-identical kernel (src/anyfesolver/_q4_planar_batch.py) inside the existing assembly lease",
                "L4: default four-Gauss-point recovery of the same elements runs through the batched kernel "
                "inside the existing recovery lease",
                "every per-element instance, quadrature, cache-key, admission and binding check stays per element",
                "a runtime known-answer self-check of the emulated summation order gates the batch; any refusal, "
                "self-check failure or input failure reruns the whole batch through the per-element path",
                "minimum batch sizes: 96 formation rows, 128 recovery rows; below them the per-element loops run "
                "unchanged",
                "batch-formed rows are bound only when the ordered per-element assembly loop reaches them",
            ],
        },
        "owner_decisions": {
            "date": "2026-11-26",
            "approved": ["L4"],
            "not_approved_not_implemented": ["L1", "L2", "L3", "L5"],
            "decisions": {
                "D1": "keep the runtime-authority generation bracket of each replaced guarded call, through a "
                      "private hook, with regression tests",
                "D2": "accept: the batched formation observes the mesh revision_signature once per element "
                      "(the caller's cache key) and checks that the coordinates it reads reproduce the key's "
                      "relative-coordinate entry, instead of recomputing the key inside the formation",
            },
            "trust_boundary": (
                "NumPy/Numba/SciPy/Python builtins, the compensated contraction modules and ANYfeSolver "
                "classes and guard machinery are not rebound at run time; hostile data, instance attributes "
                "and subclasses stay in scope and are refused or sent to the per-element path; mid-request "
                "mutation is out of contract"
            ),
            "contract_notes": [
                "D2: a foreign mutate-and-restore of the mesh during the single revision_signature observation "
                "is outside the contract; persistent mutations are still refused at the lease boundaries",
                "a runtime-authority generation move during the shared batch kernel, which belongs to no single "
                "element, discards the batch and reruns every element per element under the new generation",
                "in error cases, cache-hit elements that precede a failing element may already be bound when "
                "the error is raised",
                "both differences are visible only outside a lease; the caller sees the same error and state",
                "owner decision A (2026-11-26): in a failing assembly, elements after the failing one may "
                "already have had their cache-hit steps applied (stale caches cleared, guards rebound); the "
                "error and every result are identical, and a following successful solve ends in the same state; "
                "further members of this class are documented, not repaired",
            ],
        },
        "qualification_boundary": dict(QUALIFICATION_BOUNDARY),
        "validation": {
            "status": "PENDING",
            "bit_identical_result_hashes": evidence["hashes"],
            "regression_tests": [
                "tests/test_q4_planar_batch.py",
                "tests/test_q4_batch_wiring.py",
                "tests/test_q4_batch_paths.py",
                "tests/test_q4_planar_batch_successor.py",
            ],
            "performance_note": {
                "path": PERFORMANCE_NOTE,
                "text_identity": "lf",
                **ident(lf(note.read_bytes())),
            },
        },
        "review": {"outcome": "PENDING", "finding_count": None, "reviewer": "PENDING", "scope": "PENDING"},
        "terminal": "PENDING_INDEPENDENT_REVIEW",
    }


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        current = strict_json(OUTPUT.read_bytes())
        validated_change_commit(current["change"]["commit"])
        validated_change_commits(current["change"]["commits"])
        rebuilt = build({"hashes": current["validation"]["bit_identical_result_hashes"]})
        rebuilt["validation"]["status"] = current["validation"]["status"]
        rebuilt["review"] = current["review"]
        rebuilt["terminal"] = current["terminal"]
        problems: list[str] = []
        for _name, row in current["change"]["sealed_carriers"].items():
            planar_batch_prior(problems, row["live_path"], (ROOT / row["live_path"]).read_bytes())
        for problem in problems:
            print("PROBLEM:", problem)
        fresh = canonical(rebuilt) == OUTPUT.read_bytes() and not problems
        print("fresh" if fresh else "STALE")
        return 0 if fresh else 1
    evidence = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    OUTPUT.write_bytes(canonical(build(evidence)))
    print(OUTPUT, OUTPUT.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
