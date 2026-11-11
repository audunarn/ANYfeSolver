"""Bounded G4 material/state completion and authenticated replay tests."""
from hashlib import sha256
import json

import numpy as np
import pytest

from anyfesolver._ge_beam3_g4_completion import (
    AcceptedHistoryEnvelope, RestartAuthority)
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_retained_arc import NodalDeadForces, Program as ArcProgram, solve as arc_solve
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_retained_fibre_control import (
    TranslationProgram, solve_translation_program)
from anyfesolver._ge_beam3_retained_generalized_program import solve as generalized_solve
from anyfesolver._ge_beam3_retained_generalized_state import Program as GeneralizedProgram
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver.control import CancellationToken
from test_ge_beam3_g4a_station_transaction import make_owner, strains
from test_ge_beam3_native_generalized_modal import make as arc_model
from test_ge_beam3_native_generalized_restart import make as generalized_model, pattern as generalized_pattern
from test_ge_beam3_retained_fibre import make_model as fibre_model
from test_ge_beam3_shell_joint_state import owner as joint_owner


SCIENTIFIC_RECORDS = []


def authority(**changes):
    values = dict(graph_identity="G4_THREE_FAMILY_GRAPH_V1",
        formulation_identity="GE_BEAM3_G4_PRIVATE_FAMILY_SET_V1",
        section_identity="ELASTIC_ELLIPSOID_FIBRE_SET_V1",
        constraint_identity="G4_FROZEN_CONSTRAINT_SET_V1",
        load_identity="G4_FROZEN_LOAD_SET_V1",
        control_identity="FORCE_DISPLACEMENT_ARC_CONTROL_SET_V1",
        provenance_identity="G4_BOUNDED_QUALIFICATION_V1")
    values.update(changes)
    return RestartAuthority(**values)


def complete_envelope():
    made = AcceptedHistoryEnvelope(make_owner(), authority())
    for multiplier in (0.0, 1.0, 0.4, 0.0, -0.8, 0.0):
        made.accept(strains(multiplier))
    return made


def rehash(value):
    value["checkpoint_sha256"] = sha({key: item for key, item in value.items()
                                      if key != "checkpoint_sha256"})
    return canonical(value)


def test_complete_material_state_control_and_recovery(tmp_path):
    envelope = complete_envelope(); raw = envelope.checkpoint()
    restored = AcceptedHistoryEnvelope.restore(make_owner(), authority(), raw,
        expected_sha256=sha256(raw).hexdigest())
    assert restored.checkpoint() == raw
    assert restored.owner.snapshot() == envelope.owner.snapshot()
    mixed_recovery = restored.owner.recover()
    assert "fibres" not in mixed_recovery[0] and "fibres" not in mixed_recovery[1]
    assert len(mixed_recovery[2]["fibres"]) == 4

    gm = generalized_model("connected-plastic")
    gp = GeneralizedProgram((.25, .5, 1., .5, 0., -.5, 0.), generalized_pattern(gm))
    generalized = generalized_solve(gm, gp)
    assert generalized.status == "completed", generalized.failure
    assert any(sum(row.accumulated) > 0 for cell in generalized.state.histories
               for row in cell.stations)

    fp = ForceProgram((.25, .5, 1., .5, 0., -.5, 0.),
                      ((5, .4, -.005, 0.),), max_iterations=24)
    fibre = solve_force_program(fibre_model(), fp)
    assert fibre.status == "completed", fibre.failure
    assert any(row[2] > 0 for cell in fibre.state.histories
               for station in cell.stations for row in station.rows)
    force_prefix = solve_force_program(fibre_model(), fp, stop_after=3)
    force_rows = json.loads(force_prefix.checkpoint)["records"]
    reference_x = fibre_model().mesh.nodes[5].coords()[0]
    targets = tuple(float(row["mechanical"]["positions"][-1][0]
        + row["mechanical"]["position_low"][-1][0] - reference_x)
        for row in force_rows)
    translation = solve_translation_program(fibre_model(),
        TranslationProgram(targets, 5, (1., 0., 0.), fp.nodal_forces))
    assert translation.status == "completed", translation.failure
    for row, expected in zip(json.loads(translation.checkpoint)["records"], fp.targets[:3]):
        assert abs(row["parameter"] - expected) <= 1e-9

    am, _, _ = arc_model(False, 1, clamped=True, coupled=False)
    ap = ArcProgram((.01, .02, .015), 2., NodalDeadForces(((3, 1., 0., 0.),)))
    arc = arc_solve(am, ap)
    assert arc.status == "completed", arc.failure

    joint = joint_owner("S3-V2D", False).solve()
    assert joint.status == "completed", joint.failure
    recovered_joint = joint_owner("S3-V2D", False)
    prefix = recovered_joint.solve(stop_after=2)
    resumed_owner = joint_owner("S3-V2D", False)
    resumed = resumed_owner.solve(checkpoint=prefix.checkpoint,
        expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert resumed.status == "completed" and resumed.checkpoint == joint.checkpoint

    record = dict(test="G4_COMPLETE_MATERIAL_STATE_CONTROL_AND_RESTART",
        accepted_epochs=6, station_families=["EXACT_ELASTIC", "GENERALIZED_ELLIPSOID", "PHYSICAL_FIBRE"],
        mixed_atomic_publication=True, generalized_connected_history=True,
        fibre_connected_history=True, force_control=True, displacement_control=True,
        arc_control=True, load_unload_reversal=True, shell_joint_history=True,
        accepted_origin_replay=True, authenticated_restart=True,
        physical_recovery=True, resultant_only_no_invented_fibres=True,
        checkpoint_sha256=sha256(raw).hexdigest(), production_qualified=False)
    SCIENTIFIC_RECORDS.append(record)
    (tmp_path / "g4-science.json").write_bytes(canonical(record))


@pytest.mark.parametrize("field", (
    "schema", "owner", "graph", "formulation", "section", "constraint",
    "load", "control", "provenance", "epoch", "previous", "state",
    "recovery", "record_count", "production", "extra"))
def test_resealed_checkpoint_mutations_fail_before_publication(field):
    original = complete_envelope().checkpoint(); value = json.loads(original)
    if field == "schema": value["schema"] = "LEGACY_B3"
    elif field == "owner": value["owner_identity"] = "0" * 64
    elif field in {"graph", "formulation", "section", "constraint", "load", "control", "provenance"}:
        key = field + "_identity"; value["authority"][key] = "changed"
    elif field == "epoch": value["records"][-1]["epoch"] = 5
    elif field == "previous": value["records"][-1]["previous_sha256"] = "0" * 64
    elif field == "state": value["records"][-1]["state_sha256"] = "0" * 64
    elif field == "recovery": value["records"][-1]["recovery_sha256"] = "0" * 64
    elif field == "record_count": value["completed_epochs"] = 5
    elif field == "production": value["production_qualified"] = True
    else: value["extra"] = 1
    if value.get("records"):
        for row in value["records"]:
            row["record_sha256"] = sha({key: item for key, item in row.items()
                                        if key != "record_sha256"})
    changed = rehash(value)
    fresh = make_owner()
    with pytest.raises(ValueError):
        AcceptedHistoryEnvelope.restore(fresh, authority(), changed,
            expected_sha256=sha256(changed).hexdigest())
    assert fresh.accepted.epoch == 0


@pytest.mark.parametrize("kind", ("duplicate", "nonfinite", "external_hash", "cross_family"))
def test_untrusted_restart_bytes_and_foreign_authority_fail_closed(kind):
    raw = complete_envelope().checkpoint(); expected = sha256(raw).hexdigest()
    auth = authority()
    if kind == "duplicate": raw = raw.replace(b"{", b'{"schema":"duplicate",', 1); expected = sha256(raw).hexdigest()
    elif kind == "nonfinite": raw = raw.replace(b'"epoch":1', b'"epoch":NaN', 1); expected = sha256(raw).hexdigest()
    elif kind == "external_hash": expected = "0" * 64
    else: auth = authority(section_identity="FOREIGN_SECTION_FAMILY")
    fresh = make_owner()
    with pytest.raises(ValueError):
        AcceptedHistoryEnvelope.restore(fresh, auth, raw, expected_sha256=expected)
    assert fresh.accepted.epoch == 0


@pytest.mark.parametrize("stage", ("trial", "prepare", "before_publication"))
def test_atomic_failure_keeps_last_accepted_epoch(stage):
    envelope = AcceptedHistoryEnvelope(make_owner(), authority())
    before = envelope.checkpoint()
    def fail(actual, index):
        if actual == stage and (index in (None, 1)):
            raise RuntimeError("injected G4 failure")
    with pytest.raises(RuntimeError): envelope.accept(strains(1.0), check=fail)
    assert envelope.checkpoint() == before and envelope.owner.accepted.epoch == 0


def test_actual_cancellation_keeps_fibre_accepted_prefix():
    program = ForceProgram((.25, .5), ((5, .4, -.005, 0.),), max_iterations=24)
    prefix = solve_force_program(fibre_model(), program, stop_after=1)
    token = CancellationToken()
    def stop(row):
        if row["stage"] == "retained-fibre.before_commit": token.cancel()
    result = solve_force_program(fibre_model(), program, checkpoint=prefix.checkpoint,
        expected_checkpoint_sha256=sha256(prefix.checkpoint).hexdigest(),
        cancellation_token=token, progress=stop)
    assert result.status == "cancelled" and result.checkpoint == prefix.checkpoint


def test_missing_record_and_noncanonical_bytes_are_rejected():
    raw = complete_envelope().checkpoint(); value = json.loads(raw)
    value["records"].pop(); changed = rehash(value)
    with pytest.raises(ValueError):
        AcceptedHistoryEnvelope.restore(make_owner(), authority(), changed,
            expected_sha256=sha256(changed).hexdigest())
    with pytest.raises(ValueError):
        AcceptedHistoryEnvelope.restore(make_owner(), authority(), raw + b" ",
            expected_sha256=sha256(raw + b" ").hexdigest())
