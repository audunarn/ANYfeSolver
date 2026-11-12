"""Prepared consumer contract regression tests; execution needs fresh authority."""

from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell
from anyfesolver import _stiffened_shell_consumer_capabilities as consumer
from anyfesolver._stiffened_shell_b2_experimental import ExperimentalB2Capabilities
from anyfesolver._stiffened_shell_b2_q4_panel_experimental import (
    ExperimentalB2MixedReferenceModalCapabilities,
    ExperimentalB2Q4FootprintCapabilities, B2PhysicalRigidFootprintCapabilities,
)
from anyfesolver._stiffened_shell_b2_nodal_footprint import (
    B2NodalRigidFootprint, LAW_ID as NODAL_LAW,
)
from anyfesolver._stiffened_shell_b2_physical_rigid_footprint import (
    B2PhysicalRigidFootprint, LAW_ID as PHYSICAL_LAW,
)


def _owner(monkeypatch, report, owner_type=consumer.ExperimentalGEAnalysis):
    # Exercise the exact existing owner type without compiling a numerical model.
    monkeypatch.setattr(owner_type, "capabilities", property(lambda self: report))
    return object.__new__(owner_type)


def _ge_report(**extra):
    return dict(profile=consumer.GE_PROFILE, status="EXPERIMENTAL_NOT_QUALIFIED",
                analysis=("reference_linear",), excluded=("qualified response",),
                **extra)


def test_ge_record_roundtrip_preserves_unknown_fields_and_owns_values(monkeypatch):
    array = np.array([[1., 2.], [3., 4.]])
    report = _ge_report(
        unknown_future_policy={"flag": np.bool_(True), "count": np.int64(7),
                               "value": np.float32(.5), "label": np.str_("policy"),
                               "matrix": array, "scalar_array": np.array(6)},
        source_mapping=MappingProxyType({"nested": ["first"]}))
    owner = _owner(monkeypatch, report)
    record = stiffened_shell.capabilities_record(owner)
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["schema"] == consumer.SCHEMA
    assert record["schema_version"] == 1
    assert record["raw_report"]["analysis"] == ["reference_linear"]
    assert record["raw_report"]["excluded"] == ["qualified response"]
    assert record["raw_report"]["unknown_future_policy"] == {
        "flag": True, "count": 7, "value": .5, "label": "policy",
        "matrix": [[1., 2.], [3., 4.]], "scalar_array": 6}
    record["raw_report"]["unknown_future_policy"]["matrix"][0][0] = 99.
    record["raw_report"]["source_mapping"]["nested"].append("record mutation")
    record["qualification"]["accepted_evidence"]["ids"].append("not accepted")
    assert array[0, 0] == 1.
    assert report["source_mapping"]["nested"] == ["first"]
    array[1, 1] = 88.
    assert record["raw_report"]["unknown_future_policy"]["matrix"][1][1] == 4.
    fresh = stiffened_shell.capabilities_record(owner)
    assert fresh["qualification"]["accepted_evidence"]["ids"] == []
    assert fresh["raw_report"]["unknown_future_policy"]["matrix"][0][0] == 1.


@pytest.mark.parametrize("owner_type", consumer._B2_OWNERS)
def test_all_existing_b2_owner_types_preserve_dataclass_status(monkeypatch, owner_type):
    report = ExperimentalB2Capabilities()
    record = stiffened_shell.capabilities_record(_owner(monkeypatch, report, owner_type))
    assert record["profile"] == consumer.B2_PROFILE
    assert record["raw_status"] == report.status == "experimental"
    assert record["raw_report"]["operations"] == list(report.operations)
    assert record["raw_report"]["shell_attachment_supported"] is False
    assert record["status"] == "EXPERIMENTAL_NOT_QUALIFIED"
    assert record["qualification"] == {
        "production_qualified": False,
        "accepted_evidence": {"status": "UNAVAILABLE", "ids": []}}
    assert record["platform_qualification"] == {"status": "NOT_BOUND"}
    assert record["resource_qualification"] == {"status": "NOT_BOUND"}
    assert record["resources"]["resources"] is None
    assert record["resources"]["scope"] == "owner_resource_argument_admission_only"
    assert record["exports"]["status"] == "NO_EXPORT_OPERATION_SUPPLIED"
    assert "operations" not in record


def test_mixed_modal_law_mass_and_unknown_dataclass_fields_preserved(monkeypatch):
    @dataclass(frozen=True)
    class ExtendedCapabilities(ExperimentalB2MixedReferenceModalCapabilities):
        unknown_policy: tuple = ("future bounded policy",)

    report = ExtendedCapabilities(
        shell_formulations=(("shell", "S3-V2D"),),
        point_laws=(("joint", "exact-point-law"),),
        shell_mass_policies=(("shell", "exact-mass-policy"),),
        boundary_policy="explicit-modal-only")
    record = stiffened_shell.capabilities_record(
        _owner(monkeypatch, report, consumer.ExperimentalB2MixedPanelAnalysis))
    raw = record["raw_report"]
    assert raw["point_laws"] == [["joint", "exact-point-law"]]
    assert raw["shell_mass_policies"] == [["shell", "exact-mass-policy"]]
    assert raw["inertia_policy"] == report.inertia_policy
    assert raw["boundary_policy"] == report.boundary_policy
    assert raw["unknown_policy"] == ["future bounded policy"]


@pytest.mark.parametrize("physical", (False, True))
def test_footprint_existing_descriptor_preserves_law_semantics(monkeypatch, physical):
    if physical:
        footprint = B2PhysicalRigidFootprint(
            "region", (1, 2, 3, 4), 10, (0., 0., 1.),
            ((1., 0.), (0., 1.), (0., 0.)), ((1, 2, 3, 4),))
        report = B2PhysicalRigidFootprintCapabilities(footprints=(footprint,))
        law = PHYSICAL_LAW
    else:
        footprint = B2NodalRigidFootprint("region", (1, 2, 3, 4), 10,
                                         ((1, 2, 3, 4),))
        report = ExperimentalB2Q4FootprintCapabilities(footprints=(footprint,))
        law = NODAL_LAW
    record = stiffened_shell.capabilities_record(
        _owner(monkeypatch, report, consumer.ExperimentalB2Q4PanelAnalysis))
    descriptor = record["raw_report"]["footprints"][0]
    assert descriptor["law"] == law
    assert descriptor["semantics"] == "complete_selected_cell_patch"
    assert descriptor["production_qualified"] is False
    assert descriptor["shell_nodes"] == [1, 2, 3, 4]
    if physical:
        assert descriptor["director"] == [0., 0., 1.]
        assert descriptor["normal_spin"] == "independent"
        descriptor["tangent_basis"][0][0] = 99.
        assert footprint.tangent_basis[0, 0] == 1.
    json.dumps(record, allow_nan=False)


def test_reads_no_owner_attribute_except_capabilities(monkeypatch):
    report = _ge_report()
    reads = []

    def guarded(owner, name):
        reads.append(name)
        if name == "capabilities":
            return report
        raise AssertionError("unexpected owner access: " + name)

    monkeypatch.setattr(consumer.ExperimentalGEAnalysis, "__getattribute__", guarded)
    owner = object.__new__(consumer.ExperimentalGEAnalysis)
    stiffened_shell.capabilities_record(owner)
    assert reads == ["capabilities"]


def test_foreign_and_subclass_owners_refused_before_capability_access():
    class Foreign:
        @property
        def capabilities(self):
            raise AssertionError("foreign owner must not be read")

    class Subclass(consumer.ExperimentalGEAnalysis):
        @property
        def capabilities(self):
            raise AssertionError("subclass owner must not be read")

    for owner in (Foreign(), object.__new__(Subclass)):
        with pytest.raises(TypeError, match="existing stiffened-shell"):
            stiffened_shell.capabilities_record(owner)


@pytest.mark.parametrize("value", (
    float("nan"), float("inf"), np.float64("-inf"),
    np.array([1., float("nan")]), complex(1, 2), np.complex64(1),
    b"bytes", {"set"}, object(), {1: "non-string key"},
))
def test_unsupported_or_nonfinite_values_refused_without_dropping_fields(monkeypatch, value):
    report = _ge_report(unknown_future_field=value)
    with pytest.raises((TypeError, ValueError)):
        stiffened_shell.capabilities_record(_owner(monkeypatch, report))


@pytest.mark.parametrize("kind", ("mapping", "list", "array"))
def test_cycles_refused(monkeypatch, kind):
    report = _ge_report()
    if kind == "mapping":
        report["cycle"] = report
    elif kind == "list":
        cyclic = []
        cyclic.append(cyclic)
        report["cycle"] = cyclic
    else:
        cyclic = np.empty(1, dtype=object)
        cyclic[0] = cyclic
        report["cycle"] = cyclic
    with pytest.raises(ValueError, match="cyclic"):
        stiffened_shell.capabilities_record(_owner(monkeypatch, report))


@pytest.mark.parametrize("conflict", (
    {"profile": consumer.B2_PROFILE}, {"profile_id": consumer.B2_PROFILE},
    {"production_qualified": True}, {"production_qualified": 0},
    {"qualified": True}, {"status": "QUALIFIED"},
    {"status": "accepted"}, {"status": "unknown"},
    {"future": {"production_qualified": True}},
))
def test_conflicting_profile_or_qualification_claims_refused(monkeypatch, conflict):
    report = _ge_report()
    report.update(conflict)
    with pytest.raises(ValueError):
        stiffened_shell.capabilities_record(_owner(monkeypatch, report))


@pytest.mark.parametrize("missing", ("profile", "status"))
def test_required_raw_claims_not_inferred(monkeypatch, missing):
    report = _ge_report()
    del report[missing]
    with pytest.raises(ValueError):
        stiffened_shell.capabilities_record(_owner(monkeypatch, report))


def test_excessive_record_depth_refused(monkeypatch):
    nested = None
    for _ in range(70):
        nested = [nested]
    with pytest.raises(ValueError, match="depth"):
        stiffened_shell.capabilities_record(_owner(monkeypatch, _ge_report(nested=nested)))


def test_shared_acyclic_values_are_copied_independently(monkeypatch):
    shared = [1, 2]
    record = stiffened_shell.capabilities_record(
        _owner(monkeypatch, _ge_report(first=shared, second=shared)))
    record["raw_report"]["first"].append(3)
    assert record["raw_report"]["second"] == [1, 2]
    assert shared == [1, 2]


@pytest.mark.parametrize("report", (None, ["capabilities"], "capabilities"))
def test_non_record_capabilities_refused(monkeypatch, report):
    with pytest.raises(TypeError, match="mapping or dataclass"):
        stiffened_shell.capabilities_record(_owner(monkeypatch, report))


def test_real_ge_facade_report_is_preserved_without_request_execution():
    from test_stiffened_shell_ge_experimental import _panel

    owner = stiffened_shell.create_analysis(definition=_panel())
    try:
        raw = owner.capabilities
        record = stiffened_shell.capabilities_record(owner)
        assert record["profile"] == stiffened_shell.B3_GE_E4_PL_STIFFENED_SHELL_V1
        assert record["raw_report"] == json.loads(json.dumps(raw, allow_nan=False))
    finally:
        owner.close()


def test_real_b2_panel_facade_report_is_preserved_without_request_execution():
    # Keep this public-facade smoke independent of legacy diagnostic test imports.
    nodes = {
        1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
        4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.),
    }
    shells = (
        stiffened_shell.Q4PanelShell("left", (1, 2, 5, 4), 1000., .3, .15),
        stiffened_shell.Q4PanelShell("right", (2, 3, 6, 5), 1000., .3, .15),
    )
    section = stiffened_shell.LinearSection.isotropic(
        E=2000., nu=.25, area=.04, Iy=.00015, Iz=.0003,
        J=.0004, ky=.8, kz=.7)
    members = (
        stiffened_shell.PhysicalB2Member(
            "beam-left", "beam-source-left", (10, 11),
            ((.2, .5, .2), (1.2, .5, .2)),
            (0., 1., 0.), "beam-section", section),
        stiffened_shell.PhysicalB2Member(
            "beam-right", "beam-source-right", (11, 12),
            ((1.2, .5, .2), (2.2, .5, .2)),
            (0., 1., 0.), "beam-section", section),
    )
    points = (
        stiffened_shell.Q4B2Point("joint-left", "left", "beam-left", .5,
                                 (0., .08, -.03)),
        stiffened_shell.Q4B2Point("joint-right", "right", "beam-right", .5,
                                 (0., .08, -.03)),
    )
    shell_index = {node_id: i for i, node_id in enumerate(sorted(nodes))}
    beam_index = {node_id: i+len(nodes)
                  for i, node_id in enumerate((10, 11, 12))}
    fixed = set()
    for node_id in (1, 4):
        fixed.update(6*shell_index[node_id]+axis for axis in range(6))
    fixed.update(6*beam_index[10]+axis for axis in range(6))
    for node_id in (2, 3, 5, 6):
        fixed.add(6*shell_index[node_id]+5)
    free = np.array([i for i in range(54) if i not in fixed])
    owner = stiffened_shell.create_analysis(definition=stiffened_shell.B2Q4PanelDefinition(
        stiffened_shell.B2_E4_PL_FAST_LINEAR_V1, nodes, shells, members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54)))
    try:
        raw = owner.capabilities
        record = stiffened_shell.capabilities_record(owner)
        assert record["profile"] == stiffened_shell.B2_E4_PL_FAST_LINEAR_V1
        assert record["raw_report"]["point_law_id"] == raw.point_law_id
        assert record["raw_report"]["operations"] == list(raw.operations)
        assert record["raw_report"]["point_trace_policy"] == raw.point_trace_policy
        json.dumps(record, allow_nan=False)
    finally:
        owner.close()
