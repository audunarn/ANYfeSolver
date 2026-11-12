"""Confirmation infrastructure tests, separate from the G2 mechanics inventory."""
from copy import deepcopy
from hashlib import sha256
import json
from unittest.mock import patch
import pytest
from scripts import confirm_ge_beam3_g2 as c


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1}'])
def test_strict_rejects_malformed(raw):
    with pytest.raises(ValueError): c.strict(raw)


def good_results():
    return dict(collected=["one"], exitcode=0,
                reports=[dict(node="one", phase=p, outcome="passed") for p in ("setup", "call", "teardown")])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "skipped", "failed", "inventory", "exit", "extra"])
def test_inventory_phase_mutations(mutation):
    body = good_results()
    if mutation == "missing": body["reports"].pop()
    elif mutation == "duplicate": body["reports"].append(body["reports"][0])
    elif mutation in ("skipped", "failed"): body["reports"][1]["outcome"] = mutation
    elif mutation == "inventory": body["collected"] = []
    elif mutation == "exit": body["exitcode"] = 1
    else: body["extra"] = True
    with pytest.raises(ValueError): c.validate_results(body, ["one"])


def test_complete_results_and_canonical_bytes():
    body = good_results(); c.validate_results(body, ["one"])
    assert c.strict(c.canonical(body)) == body


@pytest.mark.parametrize("mutation", ["hash", "commit", "tree", "findings", "decision", "independence", "schema"])
def test_review_rejection(tmp_path, mutation):
    body = dict(decision=c.ACCEPTED, findings=[], subject_commit="candidate",
                scope=dict(subject_tree="tree"), reviewer=dict(independent=True))
    if mutation == "commit": body["subject_commit"] = "other"
    elif mutation == "tree": body["scope"]["subject_tree"] = "other"
    elif mutation == "findings": body["findings"] = ["state-safety"]
    elif mutation == "decision": body["decision"] = "BLOCKED"
    elif mutation == "independence": body["reviewer"]["independent"] = False
    elif mutation == "schema": body["extra"] = True
    path = tmp_path/"review.json"; c.write(path, body)
    digest = "0"*64 if mutation == "hash" else sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): c.review_input(path, digest, "candidate", "tree")


def test_publication_failure_and_exclusivity(tmp_path):
    path = tmp_path/"aggregate.json"; body = dict(result="scoped")
    with patch.object(c.os, "link", side_effect=OSError("injected")):
        with pytest.raises(OSError): c.publish(path, body)
    assert not path.exists() and path.with_name("aggregate.json.pending").exists()
    other = tmp_path/"accepted.json"; c.publish(other, body); before = other.read_bytes()
    with pytest.raises(FileExistsError): c.publish(other, dict(result="overwrite"))
    assert other.read_bytes() == before


def test_identity_rejects_dirty_candidate():
    with patch.object(c, "git", side_effect=["candidate", " M dirty.py"]):
        with pytest.raises(ValueError, match="dirty"): c.identity("candidate")


def test_registered_inventory_is_complete_and_unique():
    body = json.loads(c.INVENTORY.read_bytes())
    assert len(body) == len(set(body)) == 54
    assert all(n.split("::")[0] in c.TESTS for n in body)


@pytest.mark.parametrize("kind", ["timeout", "memory", "inactivity", "crash"])
def test_resource_or_crash_releases_tree_without_acceptance(tmp_path, kind):
    from types import SimpleNamespace
    class Job:
        terminated = False
        closed = False
        def launch(self, *args, **kwargs):
            assert all(kwargs["env"][name] == "1" for name in c.THREADS)
            return SimpleNamespace(poll=lambda: 1 if kind == "crash" else None,
                                   wait=lambda timeout: None, returncode=1)
        def accounting(self):
            return (0, 0 if self.terminated or kind == "crash" else 1,
                    24*1024**3+1 if kind == "memory" else 1000)
        def terminate(self): self.terminated = True
        def close(self): self.closed = True
    job = Job(); module = SimpleNamespace(_ProcessJob=lambda limit: job)
    spec = SimpleNamespace(name="fake_g2_job", loader=SimpleNamespace(exec_module=lambda module: None))
    times = iter([0, 1, 122, 123] if kind == "inactivity" else [0, 601 if kind == "timeout" else 1, 602])
    with patch.object(c.importlib.util, "spec_from_file_location", return_value=spec), \
         patch.object(c.importlib.util, "module_from_spec", return_value=module), \
         patch.object(c.time, "monotonic", side_effect=lambda: next(times)), \
         patch.object(c.time, "sleep"):
        with pytest.raises(ValueError, match="bounded child failure"): c.run_child(tmp_path)
    assert job.closed and (job.terminated or kind == "crash")
    assert c.strict((tmp_path/"process.json").read_bytes())["active_processes"] == 0
    assert not (tmp_path/"aggregate.json").exists()


@pytest.mark.parametrize("mutation", [None, "packet", "correction", "missing", "process", "identity"])
def test_coordinator_determinism_and_failure_no_aggregate(tmp_path, monkeypatch, mutation):
    frozen = dict(commit="candidate", tree="tree", files={})
    review = dict(decision=c.ACCEPTED, findings=[], reviewer=dict(independent=True),
                  scope=dict(subject_tree="tree"), subject_commit="candidate")
    path = tmp_path/"review.json"; c.write(path, review)
    output = tmp_path/"external"
    expected = json.loads(c.INVENTORY.read_bytes())
    records = json.loads((c.ROOT/"docs/reference_cases/ge_beam3_g2_correction_records.json").read_bytes())
    calls = []
    def identity(*args, **kwargs):
        return dict(frozen, tree="mutated") if mutation == "identity" and len(calls) == 2 else frozen
    def child(directory):
        calls.append(directory.name)
        if mutation == "process": raise ValueError("injected process failure")
        c.write(directory/"tests.json", dict(collected=expected, exitcode=0,
            reports=[dict(node=n, phase=p, outcome="passed") for n in expected for p in ("setup", "call", "teardown")]))
        c.write(directory/"checkpoints.json", sorted(["capture", "trial_assembly", "local_solve", "acceptance", "restart", "output"]))
        packet = dict(schema="GE_BEAM3_G2_DEVELOPMENT_PACKET_V1", qualification=False,
                      first={}, second={}, recovery={}, checkpoint_sha256="0"*64)
        if mutation == "packet" and len(calls) == 2: packet["first"] = dict(changed=True)
        (directory/"packet.json").write_bytes((json.dumps(packet, sort_keys=True, separators=(",", ":"))+"\n").encode())
        (directory/"corrections").mkdir()
        for i, name in enumerate(records):
            if mutation == "missing" and i == 0: continue
            value = b'{"value":2}\n' if mutation == "correction" and len(calls) == 2 else b'{"value":1}\n'
            (directory/"corrections"/name).write_bytes(value)
    monkeypatch.setattr(c, "identity", identity); monkeypatch.setattr(c, "run_child", child)
    monkeypatch.setattr(c.sys, "argv", ["confirm", "--candidate", "candidate", "--review", str(path),
        "--review-sha256", sha256(path.read_bytes()).hexdigest(), "--output", str(output)])
    if mutation:
        with pytest.raises(ValueError): c.main()
        assert not (output/"aggregate.json").exists() and (output/"failure.json").exists()
    else:
        assert c.main() == 0
        aggregate = c.strict((output/"aggregate.json").read_bytes())
        assert aggregate["cycles"] == 2 and aggregate["general_static_parity"] is False
        assert len(aggregate["correction_records"]) == 18
    assert len(calls) <= 2  # Never retry a consumed cycle.


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_compact_scientific_duplicate_nonfinite_rejection(raw):
    with pytest.raises(ValueError): c.strict(raw, compact=True)


def test_compact_native_serializer_compatibility():
    from anyfesolver._ge_beam3_g1_elastic import canonical
    body = dict(schema="example", value=[1., -0.0], qualification=False)
    raw = canonical(body)
    assert c.strict(raw, compact=True) == body
    assert raw.endswith(b"\n")
    with pytest.raises(ValueError): c.strict(raw[:-1], compact=True)
