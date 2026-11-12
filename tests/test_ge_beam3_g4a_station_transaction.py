"""Bounded tests for the private heterogeneous G4a station transaction."""
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_section import Fibre, FlowCurve, PhysicalFibreSection, canonical
from anyfesolver._ge_beam3_g1_elastic import ElasticSection
from anyfesolver._ge_beam3_g4a_station_transaction import (
    MixedStationTransaction,
    PHYSICAL_FIBRES,
    RESULTANT_ONLY,
    StationDefinition,
)
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "docs/reference_cases/ge_beam3_g4a_station_transaction_contract_v1.json").read_text())
CHECKER_PATH = ROOT / "docs/reference_cases/ge_beam3_g4a_station_transaction_checker.py"
SPEC = importlib.util.spec_from_file_location("g4a_checker", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
SCIENTIFIC_RECORDS = []


def make_owner():
    rows = CONTRACT["fixture"]["stations"]
    elastic = ElasticSection(rows[0]["law"]["stiffness"], rows[0]["law"]["name"])
    data = rows[1]["law"]
    generalized = EllipsoidalGeneralizedSection(
        data["elastic"], data["metric"], data["yield_force"], data["hardening"])
    data = rows[2]["law"]
    fibres = tuple(Fibre(
        row["fibre_id"], row["y"], row["z"], row["area"], row["young"],
        FlowCurve.linear(row["flow"]["yield_stress"], row["flow"]["hardening"]))
        for row in data["fibres"])
    fibre = PhysicalFibreSection(fibres, data["background_factor"])
    return MixedStationTransaction(tuple(StationDefinition(row["station_id"], law)
        for row, law in zip(rows, (elastic, generalized, fibre))))


def strains(multiplier):
    return tuple(np.array(row["direction"]) * multiplier
                 for row in CONTRACT["fixture"]["stations"])


def accept(owner, multiplier, check=None):
    proposal = owner.trial_all(strains(multiplier))
    prepared = owner.prepare_all(proposal)
    return owner.commit_all(prepared, check=check)


def test_six_stage_history_independent_replay_and_capabilities(tmp_path):
    owner = make_owner(); snapshots = []
    for epoch, multiplier in enumerate(CONTRACT["fixture"]["history_multipliers"], 1):
        state = accept(owner, multiplier)
        assert state.epoch == epoch
        snapshots.append(owner.snapshot())
        reference = CHECKER.evaluate(owner.definitions, state.strains,
                                     tuple(row.origin for row in state.responses))
        for actual, expected in zip(state.responses, reference):
            assert actual.station_id == expected["station_id"]
            assert actual.law_identity == expected["law_identity"]
            np.testing.assert_array_equal(actual.resultants, expected["resultants"])
            np.testing.assert_array_equal(actual.tangent, expected["tangent"])
            assert list(actual.incremental_potential) == expected["incremental_potential"]
    assert len(set(snapshots)) == 6
    assert sum(owner.accepted.responses[1].history.accumulated) > 0
    assert any(row[2] > 0 for row in owner.accepted.responses[2].history.rows)
    recovery = owner.recover()
    assert [row["capability"] for row in recovery] == [RESULTANT_ONLY, RESULTANT_ONLY, PHYSICAL_FIBRES]
    assert "fibres" not in recovery[0] and "fibres" not in recovery[1]
    assert [row.fibre_id for row in recovery[2]["fibres"]] == ["0", "1", "2", "3"]
    (tmp_path / "accepted.json").write_bytes(owner.snapshot())
    (tmp_path / "recovery.json").write_bytes(canonical(recovery))
    SCIENTIFIC_RECORDS.append(dict(
        test="G4A_HETEROGENEOUS_SIX_STAGE_TRANSACTION",
        station_families=["EXACT_ELASTIC", "GENERALIZED_ELLIPSOID", "PHYSICAL_FIBRE"],
        accepted_epochs=6,
        independent_replay=True,
        atomic_publication=True,
        physical_fibre_recovery=True,
        snapshot_sha256=sha256(owner.snapshot()).hexdigest(),
        production_qualified=False))


@pytest.mark.parametrize("index", range(3))
def test_trial_failure_is_nonpublishing(index):
    owner = make_owner(); before = owner.snapshot()
    def fail(stage, station):
        if stage == "trial" and station == index:
            raise RuntimeError("injected trial failure")
    with pytest.raises(RuntimeError, match="trial failure"):
        owner.trial_all(strains(1.0), check=fail)
    assert owner.snapshot() == before and owner.accepted.epoch == 0
    proposal = owner.trial_all(strains(0.4)); owner.discard(proposal)


@pytest.mark.parametrize("index", range(3))
def test_prepare_failure_discards_whole_proposal(index):
    owner = make_owner(); before = owner.snapshot(); proposal = owner.trial_all(strains(1.0))
    def fail(stage, station):
        if stage == "prepare" and station == index:
            raise RuntimeError("injected prepare failure")
    with pytest.raises(RuntimeError, match="prepare failure"):
        owner.prepare_all(proposal, check=fail)
    assert owner.snapshot() == before
    with pytest.raises(ValueError): owner.prepare_all(proposal)
    proposal = owner.trial_all(strains(0.4)); owner.discard(proposal)


def test_before_and_after_publication_boundaries():
    owner = make_owner(); before = owner.snapshot()
    proposal = owner.trial_all(strains(1.0)); prepared = owner.prepare_all(proposal)
    def before_failure(stage, _):
        if stage == "before_publication": raise RuntimeError("before")
    with pytest.raises(RuntimeError, match="before"):
        owner.commit_all(prepared, check=before_failure)
    assert owner.snapshot() == before and owner.accepted.epoch == 0
    proposal = owner.trial_all(strains(1.0)); prepared = owner.prepare_all(proposal)
    def after_failure(stage, _):
        if stage == "after_publication": raise RuntimeError("after")
    with pytest.raises(RuntimeError, match="after"):
        owner.commit_all(prepared, check=after_failure)
    assert owner.snapshot() != before and owner.accepted.epoch == 1


def test_foreign_stale_replayed_and_reordered_inputs():
    owner = make_owner(); other = make_owner(); proposal = owner.trial_all(strains(1.0))
    with pytest.raises(ValueError): other.prepare_all(proposal)
    owner.discard(proposal)
    with pytest.raises(ValueError): owner.prepare_all(proposal)
    proposal = owner.trial_all(strains(1.0))
    forged = replace(proposal, responses=tuple(reversed(proposal.responses)))
    with pytest.raises(ValueError): owner.prepare_all(forged)
    proposal = owner.trial_all(strains(1.0))
    prepared = owner.prepare_all(proposal)
    owner.commit_all(prepared)
    with pytest.raises(ValueError): owner.commit_all(prepared)
    with pytest.raises(ValueError): owner.discard(proposal)


def test_mutated_response_history_and_law_fail_closed():
    owner = make_owner(); proposal = owner.trial_all(strains(1.0))
    object.__setattr__(proposal.responses[1], "derivative_kind", "MUTATED")
    with pytest.raises(ValueError): owner.prepare_all(proposal)
    owner = make_owner(); proposal = owner.trial_all(strains(1.0))
    object.__setattr__(proposal, "origins", ((), (), ()))
    with pytest.raises(ValueError): owner.prepare_all(proposal)
    owner = make_owner()
    object.__setattr__(owner.definitions[0].law, "name", "changed")
    with pytest.raises(ValueError): owner.trial_all(strains(1.0))


def test_concurrent_writer_and_immutable_outputs():
    owner = make_owner(); owner._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            owner.trial_all(strains(1.0))
    finally:
        owner._lock.release()
    state = accept(owner, 1.0)
    for response in state.responses:
        for array in (response.strain, response.resultants, response.tangent):
            with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(ValueError): owner.recover(replace(state))


def test_tangent_and_incremental_work_directional_agreement():
    owner = make_owner(); accept(owner, 1.0); state = owner.accepted
    direction = np.array([0.7, -0.3, 0.2, -0.4, 0.5, -0.1])
    step = 1e-6
    for definition, response in zip(owner.definitions, state.responses):
        law = definition.law; origin = response.origin; x = response.strain
        if type(law) is ElasticSection:
            plus = law.response(x + step * direction, origin)
            minus = law.response(x - step * direction, origin)
            numerical = (plus["resultants"] - minus["resultants"]) / (2 * step)
            energy_derivative = (plus["potential"] - minus["potential"]) / (2 * step)
        elif type(law) is EllipsoidalGeneralizedSection:
            plus = law.response(x + step * direction, origin=origin)
            minus = law.response(x - step * direction, origin=origin)
            numerical = ((plus.resultants + plus.resultants_low)
                         - (minus.resultants + minus.resultants_low)) / (2 * step)
            energy_derivative = (sum(plus.incremental_potential)
                                 - sum(minus.incremental_potential)) / (2 * step)
        else:
            plus = law.response(x + step * direction, origin=origin)
            minus = law.response(x - step * direction, origin=origin)
            numerical = ((plus.resultants + plus.resultants_low)
                         - (minus.resultants + minus.resultants_low)) / (2 * step)
            energy_derivative = (sum(plus.incremental_potential)
                                 - sum(minus.incremental_potential)) / (2 * step)
        tangent = response.tangent + response.tangent_low
        error = np.linalg.norm(numerical - tangent @ direction) / max(1.0, np.linalg.norm(tangent @ direction))
        assert error <= 1e-7
        work = float((response.resultants + response.resultants_low) @ direction)
        assert abs(energy_derivative - work) / max(1.0, abs(work)) <= 1e-7


def test_checker_does_not_import_transaction_owner():
    raw = CHECKER_PATH.read_text(encoding="utf-8")
    assert "_ge_beam3_g4a_station_transaction" not in raw
