"""G1 S01-S08 development checks on the actual assembler/state seam."""
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from anyfesolver._ge_beam3_g1_elastic import canonical
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_analysis import ElasticAnalysis
from anyfesolver.nonlinear_state import StaleStateTokenError
from test_ge_beam3_g1_elastic import reference, section


def problem():
    return ElasticAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),
                            ElasticElement(2, (3, 4, 5), reference(1., .37), section(True))), tuple(range(6)))


def loads(owner):
    f = np.zeros(owner.size); f[-6:] = [.01, -.02, .015, .002, .003, -.001]
    return f


def evaluate(owner, total):
    return owner._evaluate(total, np.zeros(owner.size), np.zeros((2, 3)), np.zeros((2, 3)))


def discard(owner):
    if owner.store.has_active_trial:
        owner.store.discard_trial(owner.store.active_trial_token())


def test_S01_shared_rotation_and_different_rolls():
    owner = problem(); u = np.zeros(owner.size); u[15:18] = [.01, -.02, .015]
    before = owner.checkpoint()
    try:
        evaluate(owner, u); token = owner.store.active_trial_token()
        views = [owner.store.native_element_rotation_view(token, e.element_id, e.node_ids,
                 e.native_reference_directors(owner.model.mesh)) for e in owner.elements]
        np.testing.assert_array_equal(views[0].trial_rotation_matrices[2], views[1].trial_rotation_matrices[0])
        assert not np.array_equal(owner.elements[0].operator.reference.nodal_triads[2], owner.elements[1].operator.reference.nodal_triads[0])
    finally: discard(owner)
    assert owner.checkpoint() == before


def test_S02_rejected_noncommuting_trials():
    owner = problem(); before = owner.checkpoint(); a = np.zeros(owner.size); b = a.copy()
    a[15:18] = [.01, .02, 0.]; b[15:18] = [0., -.015, .01]
    try:
        evaluate(owner, a); discard(owner)
        first = evaluate(owner, b)[0]; discard(owner)
        evaluate(owner, a); discard(owner)
        second = evaluate(owner, b)[0]
        np.testing.assert_array_equal(first, second)
    finally: discard(owner)
    assert before == owner.checkpoint()


def test_S03_second_prepare_failure_is_atomic():
    owner = problem(); before = owner.checkpoint(); u = np.zeros(owner.size)
    evaluate(owner, u); token = owner.store.active_trial_token()
    victim = owner.elements[1]
    try:
        with patch.object(victim, "_validate", side_effect=ValueError("injected preparation failure")):
            with pytest.raises(ValueError, match="injected"):
                owner.store.commit(token, accepted_full_displacement=u,
                                   accepted_full_coordinates=np.array([n.coords() for n in owner.model.mesh.nodes.values()]))
        assert owner.store.generation == 0
        assert owner.store.native_rotation_store.generation == 0
    finally: discard(owner)
    assert owner.checkpoint() == before


def test_S04_stale_foreign_tokens_and_concurrent_owner():
    owner = problem(); other = problem(); u = np.zeros(owner.size)
    evaluate(owner, u); token = owner.store.active_trial_token(); discard(owner)
    with pytest.raises(StaleStateTokenError): owner.store.discard_trial(token)
    evaluate(other, u); foreign = other.store.active_trial_token()
    try:
        with pytest.raises(StaleStateTokenError): owner.store.commit(foreign)
    finally: discard(other)
    owner._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already in use"): owner.solve(loads(owner))
    finally: owner._lock.release()


def test_S07_actual_assembly_recovery_and_directional_tangent():
    owner = problem(); u = np.linspace(0., .001, owner.size); u[:6] = 0
    direction = np.linspace(.0005, -.0007, owner.size); direction[:6] = 0
    try:
        r, K, _ = evaluate(owner, u)
        for h in (1e-4, 1e-5, 1e-6):
            plus = evaluate(owner, u+h*direction)[0]
            minus = evaluate(owner, u-h*direction)[0]
            assert np.linalg.norm((plus-minus)/(2*h)-K @ direction) / max(1., np.linalg.norm(K @ direction)) <= 1e-7
    finally: discard(owner)
    result = owner.solve(loads(owner))
    assert np.linalg.norm(result["reactions"][owner.free]) <= 1e-11
    assert owner.store.generation == 1
    before = owner.checkpoint(); recovered = owner.recover()
    assert sum(map(len, recovered.values())) == 16 and owner.checkpoint() == before


def test_S08_restart_and_atomic_publication(tmp_path):
    owner = problem(); owner.solve(loads(owner))
    data = owner.checkpoint(); digest = sha256(data).hexdigest()
    resumed = ElasticAnalysis.resume(data, digest)
    assert resumed.checkpoint() == data
    a = owner.solve(loads(owner)*.5); b = resumed.solve(loads(resumed)*.5)
    assert canonical(a) == canonical(b)
    target = tmp_path / "checkpoint.json"
    resumed.publish(target); old = target.read_bytes()
    with pytest.raises(ValueError): resumed.publish(target)
    assert target.read_bytes() == old
    failed = tmp_path / "failed.json"
    with patch("anyfesolver._ge_beam3_g1_analysis.os.link", side_effect=OSError("injected publication fault")):
        with pytest.raises(OSError): resumed.publish(failed)
    assert not failed.exists() and target.read_bytes() == old


def test_restart_tamper_and_cancellation():
    owner = problem(); data = owner.checkpoint()
    with pytest.raises(ValueError): ElasticAnalysis.resume(data, "0"*64)
    body = json.loads(data); body["schema"] = "legacy"
    changed = canonical(body)
    with pytest.raises(ValueError): ElasticAnalysis.resume(changed, sha256(changed).hexdigest())
    with pytest.raises(InterruptedError): owner.solve(loads(owner), cancel=lambda: True)
    assert owner.checkpoint() == data


def test_graph_mutation_and_unsupported_partial_rotations():
    owner = problem(); owner.model.add_node(6, 3., 0., 0.)
    with pytest.raises(ValueError): owner.solve(np.zeros(owner.size))
    with pytest.raises(ValueError, match="complete rotational"):
        ElasticAnalysis((ElasticElement(1, (1, 2, 3), reference(), section()),), (0, 1, 2, 3))


def test_distributed_force_and_couple_recovery_and_restart():
    owner = problem()
    line = np.tile([.01, -.02, .005], (2, 1))
    couple = np.tile([.001, .002, -.001], (2, 1))
    result = owner.solve(np.zeros(owner.size), lines=line, couples=couple)
    assert np.linalg.norm(result["reactions"][owner.free]) < 1e-11
    # Two unit reference lengths; internal couples contribute no net force.
    np.testing.assert_allclose(result["reactions"][:3]+2*line[0], 0., atol=1e-11)
    data = owner.checkpoint()
    restored = ElasticAnalysis.resume(data, sha256(data).hexdigest())
    assert canonical(owner.recover()) == canonical(restored.recover())


@pytest.mark.parametrize("field", ["state", "history", "definition", "initial", "runtime"])
def test_resealed_checkpoint_tamper_rejected(field):
    owner = problem(); owner.solve(loads(owner)); data = json.loads(owner.checkpoint())
    if field == "state": data["state"]["1"]["response"]["resultants"][0] += .1
    elif field == "history": data["history"][0]["nodal"][-1] += .01
    elif field == "definition": data["definition"]["elements"][0]["section"]["stiffness"][0][0] += 1.
    else: data[field] = "0"*64
    changed = canonical(data)
    with pytest.raises(ValueError): ElasticAnalysis.resume(changed, sha256(changed).hexdigest())


def test_candidate_tamper_and_formulation_mutation_rejected():
    owner = problem(); before = owner.checkpoint()
    evaluate(owner, np.zeros(owner.size)); token = owner.store.active_trial_token()
    try:
        altered = owner.store.materialize(trial_token=token)[1]
        altered["response"]["resultants"][0] += .1
        with pytest.raises(ValueError): owner.store.set_trial_state(token, 1, altered)
    finally: discard(owner)
    assert owner.checkpoint() == before
    owner.elements[0].production_qualified = True
    with pytest.raises(ValueError, match="authority changed"): owner.solve(loads(owner))


def test_deterministic_development_packet(tmp_path):
    owner = problem(); first = owner.solve(loads(owner))
    first_checkpoint = owner.checkpoint()
    restored = ElasticAnalysis.resume(first_checkpoint, sha256(first_checkpoint).hexdigest())
    second = owner.solve(loads(owner)*.5)
    replayed = restored.solve(loads(restored)*.5)
    assert canonical(second) == canonical(replayed)
    assert owner.checkpoint() == restored.checkpoint()
    payload = canonical(dict(schema="GE_BEAM3_G1_DEVELOPMENT_PACKET_V1",
                             qualification=False, first=first, second=second,
                             recovery=owner.recover(), checkpoint_sha256=sha256(owner.checkpoint()).hexdigest()))
    target = Path(os.environ.get("G1_DIAGNOSTIC_PAYLOAD", str(tmp_path / "scientific-development.json")))
    with target.open("xb") as stream:
        stream.write(payload)


def test_accepted_journal_cannot_be_mutated_or_detached():
    owner = problem(); owner.solve(loads(owner))
    owner._accepted[0]["nodal"][-1] += .1
    with pytest.raises(ValueError, match="journal"): owner.checkpoint()


def accepted_snapshot(owner):
    rotations = owner.store.native_rotation_store
    return canonical(dict(state=owner.store.materialize(), u=rotations.committed_full_displacement,
                          Q=rotations.committed_rotation_matrices, generation=owner.store.generation,
                          rotation_generation=rotations.generation, history=owner._accepted,
                          journal=owner._journal_digest))


@pytest.mark.parametrize("accepted", [False, True])
def test_free_dof_public_alias_is_immutable(accepted):
    owner = problem()
    if accepted: owner.solve(loads(owner))
    before = accepted_snapshot(owner)
    with pytest.raises(AttributeError): owner.free = owner.free[:-1]
    with pytest.raises(ValueError): owner.free[-1] = 0
    with pytest.raises(ValueError): owner.free.setflags(write=True)
    with pytest.raises(ValueError): owner.free.view().setflags(write=True)
    assert accepted_snapshot(owner) == before
    data = owner.checkpoint()
    assert ElasticAnalysis.resume(data, sha256(data).hexdigest()).checkpoint() == data


@pytest.mark.parametrize("mutation", ["omitted", "writable", "size"])
def test_free_dof_private_replacement_rejected_before_evaluation(mutation):
    owner = problem(); before = accepted_snapshot(owner)
    if mutation == "omitted": owner._free = owner.free[:-1]
    elif mutation == "writable": owner._free = owner.free.copy()
    else: owner.size -= 1
    force = np.zeros(30); force[-1] = .001
    with patch.object(owner, "_evaluate", side_effect=AssertionError("must reject before evaluation")):
        with pytest.raises(ValueError, match="free-DOF"): owner.solve(force)
    assert accepted_snapshot(owner) == before and not owner.store.has_active_trial


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("mutation", ["constraint", "free", "fixed", "node", "section", "manager_support", "model_load"])
def test_final_callback_mutation_preserves_accepted_prefix(accepted, mutation):
    owner = problem(); force = np.zeros(owner.size)
    if accepted: owner.solve(force)
    before = accepted_snapshot(owner); calls = []
    def callback():
        calls.append(True)
        if len(calls) == 2:
            if mutation == "constraint": owner.model.constraint_equations.append("unsupported")
            elif mutation == "free": owner._free = owner.free[:-1]
            elif mutation == "fixed": owner.fixed = tuple(range(12))
            elif mutation == "node": owner.model.add_node(6, 3., 0., 0.)
            elif mutation == "manager_support": owner.model.mesh.dof_manager.constrain_dof(owner.size-1)
            elif mutation == "model_load":
                from anyfesolver.boundary import LoadCase
                owner.model.add_load_case(LoadCase("unsupported model load"))
            else: owner.model.materials[owner.elements[0].material_name] = section(True)
        return False
    with pytest.raises(ValueError): owner.solve(force, cancel=callback)
    assert len(calls) == 2
    assert accepted_snapshot(owner) == before and not owner.store.has_active_trial


def test_prepublication_preparation_mutation_is_rejected():
    owner = problem(); before = accepted_snapshot(owner)
    original = owner.store.native_element_rotation_view
    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        owner.model.constraint_equations.append("unsupported")
        return result
    with patch.object(owner.store, "native_element_rotation_view", side_effect=prepare):
        with pytest.raises(ValueError): owner.solve(np.zeros(owner.size))
    assert accepted_snapshot(owner) == before and not owner.store.has_active_trial


@pytest.mark.parametrize("accepted", [False, True])
def test_store_commit_material_preparation_mutation_is_atomic(accepted):
    owner = problem(); force = np.zeros(owner.size)
    if accepted: owner.solve(force)
    before = accepted_snapshot(owner); victim = owner.elements[-1]
    original_validate = victim._validate; original_commit = owner.store.commit
    preparing = []; triggered = []
    def commit(*args, **kwargs):
        preparing.append(True)
        try: return original_commit(*args, **kwargs)
        finally: preparing.pop()
    def validate(*args, **kwargs):
        original_validate(*args, **kwargs)
        if preparing:
            triggered.append(True)
            owner.model.constraint_equations.append("unsupported during commit")
    with patch.object(owner.store, "commit", side_effect=commit), patch.object(victim, "_validate", side_effect=validate):
        with pytest.raises(ValueError, match="unsupported graph"): owner.solve(force)
    assert triggered and accepted_snapshot(owner) == before and not owner.store.has_active_trial
