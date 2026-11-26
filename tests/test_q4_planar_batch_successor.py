"""Source binding of the batched qualified Q4 planar change (L4).

The Q4 planar batch successor is the single recorded successor of the guard
cadence successor.  These tests check that it binds the exact live bytes of
every file it changes to an exact reverse patch reproducing the bytes the
guard cadence successor binds, that its reader refuses anything else, and
that its builder is fresh.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "reference_cases" / "qualified_q4_planar_batch_successor.json"
BUILDER = ROOT / "scripts" / "build_q4_planar_batch_successor.py"
PREDECESSOR = ROOT / "docs" / "reference_cases" / "qualified_shell_nonlinear_guard_cadence_successor.json"
PREDECESSOR_SHA256 = "96EB8040773D4E05F33938C333F34BF63372428D0A7EEE5D099867690A88936D"
RECORDED_BASE_COMMIT = "d4cf5011fea7539b315a37b5de4359591c801ae0"
# Pinned without history (Codex L4 round 1 P2): a shallow checkout that skips
# the ancestry checks still refuses a substituted change identity.
RECORDED_BASE_TREE = "736b9aa3c5c35429d339c0c836b91c474c0bea61"
RECORDED_CHANGE_COMMIT = "704536ab0c590f09af99ed53a09d8b74eefda119"
RECORDED_CHANGE_COMMIT_IDS = ("704536ab0", "84810a7dc")
SEALED_CARRIERS = ("e4_pl_element.py", "matrix_assembly.py")
OTHER_CHANGED_SOURCES = ("_q4_planar_batch.py", "assembly.py")
SHALLOW_BOUNDARY_SKIP = (
    "commit history is absent at the explicit GitHub shallow boundary; every byte binding was "
    "checked, ancestry is checked in full clones and by the builder's --check"
)


def _identity(data: bytes) -> dict:
    return {
        "bytes": len(data),
        "git_blob": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def _canonical(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _load_builder(name: str = "_q4_planar_batch_successor_builder"):
    spec = importlib.util.spec_from_file_location(name, BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record() -> dict:
    raw = RECORD.read_bytes()
    record = json.loads(raw)
    assert raw == _canonical(record)
    return record


def _git_environment() -> dict:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    environment.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    return environment


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False,
                          env=_git_environment())


def _require_history(*revisions: str) -> None:
    """Skip only at the explicit GitHub shallow boundary; refuse elsewhere."""

    if all(_git("cat-file", "-e", f"{r}^{{commit}}").returncode == 0 for r in revisions):
        return
    shallow = _git("rev-parse", "--is-shallow-repository")
    assert os.environ.get("GITHUB_ACTIONS") == "true" and shallow.stdout.strip() == b"true", (
        f"recorded commits are not in this repository: {revisions}"
    )
    pytest.skip(SHALLOW_BOUNDARY_SKIP)


def _guard_cadence_bound() -> dict[str, tuple[bool, dict | str]]:
    """What the guard cadence successor binds as live for the files it names."""

    predecessor = json.loads(PREDECESSOR.read_bytes())
    bound: dict[str, tuple[bool, dict | str]] = {}
    for row in predecessor["change"]["sealed_carriers"].values():
        bound[row["live_path"]] = (row["portable_text"],
                                   row["current_lf"] if row["portable_text"] else row["current_raw"])
    for row in predecessor["change"]["other_changed_sources"].values():
        bound[row["live_path"]] = (True, row["current_lf_git_blob"])
    return bound


def _assert_change_identity(record: dict) -> None:
    """History-free checks of the recorded change identity."""

    assert record["change"]["commit"] == RECORDED_CHANGE_COMMIT
    assert [entry.split(" ", 1)[0] for entry in record["change"]["commits"]] == list(
        RECORDED_CHANGE_COMMIT_IDS
    )
    assert all(type(entry) is str and len(entry.split(" ", 1)) == 2 for entry in record["change"]["commits"])


@pytest.mark.parametrize("path,value", [
    (("change", "commit"), "0" * 40),
    (("change", "commit"), "071910d60031beb455ef0fa2328a027d8ba77c3d"),
    (("change", "commits"), []),
    (("change", "commits"), ["704536ab0 Wire the batched Q4 planar numerics into assembly and recovery (L4)"]),
])
def test_change_identity_is_refused_without_history(path, value) -> None:
    """Codex L4 round 1 P2: a substituted change identity is refused even
    where the ancestry checks are deferred (GitHub shallow checkout)."""

    _assert_change_identity(_record())
    with pytest.raises(AssertionError):
        _assert_change_identity(_set(_record(), path, value))


def test_record_chains_from_the_pinned_guard_cadence_successor() -> None:
    record = _record()
    raw = PREDECESSOR.read_bytes()
    assert _identity(raw)["sha256"] == PREDECESSOR_SHA256
    assert record["predecessor"] == {"path": PREDECESSOR.relative_to(ROOT).as_posix(), **_identity(raw)}
    assert record["schema"] == "anysolver.qualified-q4-planar-batch-successor-v1"
    assert record["base"]["commit"] == RECORDED_BASE_COMMIT
    assert record["base"]["tree"] == RECORDED_BASE_TREE  # Codex L4 round 2 P3
    _assert_change_identity(record)
    assert all(value is False for value in record["qualification_boundary"].values())
    assert len(record["qualification_boundary"]) == 8
    assert record["owner_decisions"]["approved"] == ["L4"]
    assert set(record["owner_decisions"]["decisions"]) == {"D1", "D2"}
    assert len(record["owner_decisions"]["contract_notes"]) == 5
    assert record["review"]["outcome"] in {"PENDING", "ACCEPT"}


def test_sealed_carriers_rebuild_the_guard_cadence_bound_bytes() -> None:
    record = _record()
    bound = _guard_cadence_bound()
    assert tuple(sorted(record["change"]["sealed_carriers"])) == SEALED_CARRIERS
    builder = _load_builder()
    for name, row in record["change"]["sealed_carriers"].items():
        relative = f"src/anyfesolver/{name}"
        assert row["live_path"] == relative
        live = (ROOT / relative).read_bytes()
        problems: list[str] = []
        prior = builder.planar_batch_prior(problems, relative, live)
        assert problems == []
        assert prior != live
        portable, identity = bound[relative]
        assert row["portable_text"] is portable
        if relative == "src/anyfesolver/e4_pl_element.py":
            assert portable is False and _identity(prior) == identity == row["prior_raw"]
        else:
            # A guard-cadence "other changed source": bound by its LF Git blob.
            assert _identity(prior.replace(b"\r\n", b"\n"))["git_blob"] == identity
        assert _identity(prior.replace(b"\r\n", b"\n")) == row["prior_lf"]


def test_other_changed_sources_bind_the_files_their_keys_name() -> None:
    record = _record()
    assert tuple(sorted(record["change"]["other_changed_sources"])) == OTHER_CHANGED_SOURCES
    bound = _guard_cadence_bound()
    for name, row in record["change"]["other_changed_sources"].items():
        assert row["live_path"] == f"src/anyfesolver/{name}"
        assert row["live_path"] not in bound
        live = (ROOT / row["live_path"]).read_bytes().replace(b"\r\n", b"\n")
        assert _identity(live)["git_blob"] == row["current_lf_git_blob"]
        assert row["prior_lf_git_blob"] != row["current_lf_git_blob"]
    assert record["change"]["other_changed_sources"]["_q4_planar_batch.py"]["prior_lf_git_blob"] is None


def test_portable_carrier_is_checkout_independent() -> None:
    builder = _load_builder()
    relative = "src/anyfesolver/matrix_assembly.py"
    canonical = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    priors = []
    for data in (canonical, canonical.replace(b"\n", b"\r\n")):
        problems: list[str] = []
        prior = builder.planar_batch_prior(problems, relative, data)
        assert problems == []
        assert (b"\r\n" in prior) is (b"\r\n" in data)
        priors.append(prior.replace(b"\r\n", b"\n"))
    assert priors[0] == priors[1]


@pytest.mark.parametrize("relative,mutate", [
    ("src/anyfesolver/e4_pl_element.py", lambda data: data + b"\n# unrecorded\n"),
    ("src/anyfesolver/e4_pl_element.py", lambda data: data.replace(b"\n", b"\r\n")),
    ("src/anyfesolver/matrix_assembly.py", lambda data: data + b"# unrecorded\n"),
    ("src/anyfesolver/matrix_assembly.py", lambda data: data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n", 1)),
])
def test_reader_refuses_unrecorded_live_bytes(relative, mutate) -> None:
    builder = _load_builder()
    live = (ROOT / relative).read_bytes()
    problems: list[str] = []
    assert builder.planar_batch_prior(problems, relative, mutate(live)) == mutate(live)
    assert problems


def _set(record: dict, path: tuple, value) -> dict:
    mutated = copy.deepcopy(record)
    node = mutated
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return mutated


@pytest.mark.parametrize("path,value", [
    (("predecessor", "sha256"), "0" * 64),
    (("schema",), "anysolver.qualified-q4-planar-batch-successor-v0"),
    (("qualification_boundary", "mechanics_symbols_changed"), True),
    (("change", "sealed_carriers", "e4_pl_element.py", "current_raw", "sha256"), "0"),
    (("change", "sealed_carriers", "e4_pl_element.py", "prior_raw", "sha256"), "0"),
    (("change", "sealed_carriers", "e4_pl_element.py", "portable_text"), True),
    (("change", "sealed_carriers", "matrix_assembly.py", "current_lf", "sha256"), "0"),
    (("change", "sealed_carriers", "matrix_assembly.py", "reverse_line_patch"), []),
])
def test_reader_refuses_a_mutated_record(tmp_path, path, value) -> None:
    builder = _load_builder()
    mutated = tmp_path / "record.json"
    mutated.write_bytes(_canonical(_set(_record(), path, value)))
    builder.OUTPUT = mutated
    affected = (
        [f"src/anyfesolver/{path[2]}"] if path[0] == "change"
        else ["src/anyfesolver/e4_pl_element.py", "src/anyfesolver/matrix_assembly.py"]
    )
    for relative in affected:
        problems: list[str] = []
        live = (ROOT / relative).read_bytes()
        assert builder.planar_batch_prior(problems, relative, live) == live
        assert problems, (path, relative)


def test_reader_refuses_a_non_canonical_record(tmp_path) -> None:
    builder = _load_builder()
    mutated = tmp_path / "record.json"
    mutated.write_bytes(json.dumps(_record(), indent=1).encode("utf-8"))
    builder.OUTPUT = mutated
    problems: list[str] = []
    relative = "src/anyfesolver/e4_pl_element.py"
    builder.planar_batch_prior(problems, relative, (ROOT / relative).read_bytes())
    assert problems == ["planar batch successor is not canonical JSON"]


def test_builder_check_is_fresh() -> None:
    record = _record()
    _require_history(record["base"]["commit"], record["change"]["commit"])
    done = subprocess.run([sys.executable, "-B", str(BUILDER), "--check"], cwd=ROOT,
                          capture_output=True, check=False, env=_git_environment())
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.decode().strip() == "fresh"


def test_builder_validates_the_change_commits() -> None:
    record = _record()
    _require_history(record["base"]["commit"], record["change"]["commit"])
    builder = _load_builder()
    saved = dict(os.environ)
    sanitized = _git_environment()
    os.environ.clear()
    os.environ.update(sanitized)
    try:
        assert builder.validated_change_commits(list(record["change"]["commits"])) == record["change"]["commits"]
        short, subject = record["change"]["commits"][0].split(" ", 1)
        for bogus in ([short + " another subject"], ["0" * 9 + " " + subject], []):
            with pytest.raises(SystemExit):
                builder.validated_change_commits(bogus)
        with pytest.raises(SystemExit):
            builder.validated_change_commit("0" * 40)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_performance_note_is_bound_by_its_lf_text() -> None:
    note = _record()["validation"]["performance_note"]
    data = (ROOT / note["path"]).read_bytes().replace(b"\r\n", b"\n")
    assert note["text_identity"] == "lf"
    assert {k: note[k] for k in ("bytes", "git_blob", "sha256")} == _identity(data)
