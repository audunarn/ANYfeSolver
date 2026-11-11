"""Regression boundaries for qualified-S3 descriptor provenance and failures."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    FEModel,
    FixedSupport,
    QualifiedE4PLS3ShellElement,
    shell_element_from_dict,
    solve_free_vibration,
)
from anyfesolver.e4_pl_s3_element import (
    ALGEBRAIC_COORDINATE_POLICY_ID,
    FORMULATION_ID,
)
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import BeamElement


_PROVENANCE = [
    {
        "element_id": 1,
        "formulation_id": FORMULATION_ID,
        "algebraic_coordinate_policy": ALGEBRAIC_COORDINATE_POLICY_ID,
    }
]


def _s3_model(*, density: float = 7850.0) -> FEModel:
    model = FEModel("qualified-s3-modal-provenance")
    model.add_material("steel", 210.0e9, 0.3, density=density)
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        dtype=float,
    )
    for node_id, coordinate in enumerate(nodes, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.02,
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed-edge", [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
    )
    return model


def _beam_model() -> FEModel:
    model = FEModel("non-descriptor-modal-regression")
    model.add_material("steel", 100.0, 0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(
        1,
        BeamElement(
            1,
            [1, 2],
            "steel",
            {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6},
        ),
    )
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "axial-only",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _canonical(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _stable_non_descriptor_payload(result: Any) -> dict[str, Any]:
    """Remove wall-clock observations from the deterministic science payload."""

    payload = result.to_dict()
    for lane in ("stiffness", "mass"):
        info = payload["assembly_info"][lane]
        info.pop("assembly_time", None)
        info.pop("element_times", None)
    payload["diagnostics"].pop("phase_timings_seconds", None)
    return payload


def test_descriptor_success_binds_formulation_and_coordinate_policy() -> None:
    result = solve_free_vibration(_s3_model(), num_modes=2)

    assert result.solver_status == "ok"
    assert result.diagnostics["descriptor_modal"] is True
    assert result.diagnostics["declared_algebraic_formulations"] == _PROVENANCE
    assert result.result_case["metadata"]["descriptor_modal_provenance"] == {
        "policy_id": "SWAPPED_MASSLESS_ALGEBRAIC_PENCIL_V1",
        "elements": _PROVENANCE,
    }


def test_descriptor_failure_provenance_and_diagnostics_are_stable() -> None:
    results = [solve_free_vibration(_s3_model(density=0.0), num_modes=1) for _ in range(2)]

    for result in results:
        assert result.solver_status == "failed"
        assert result.diagnostics["error_code"] == "ALGEBRAIC_DESCRIPTOR_INVALID"
        assert result.diagnostics["error_type"] == "AlgebraicDynamicsError"
        assert result.diagnostics["declared_algebraic_element_ids"] == [1]
        assert result.diagnostics["declared_algebraic_formulations"] == _PROVENANCE
        assert result.result_case["metadata"]["descriptor_modal_provenance"] == {
            "policy_id": "SWAPPED_MASSLESS_ALGEBRAIC_PENCIL_V1",
            "elements": _PROVENANCE,
        }
    assert _canonical(results[0].diagnostics) == _canonical(results[1].diagnostics)


def test_descriptor_backend_failure_is_wrapped_without_backend_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anyfesolver.algebraic_dynamics as algebraic_dynamics

    calls = 0

    def volatile_backend(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"volatile backend detail {calls}")

    # This bounded supported S3 intentionally takes the coordinate-invariant
    # static-condensation branch.  Patch its spectral backend, not an earlier
    # certificate backend.
    monkeypatch.setattr(algebraic_dynamics.linalg, "eigh", volatile_backend)
    errors = []
    for _ in range(2):
        with pytest.raises(
            ElementCapabilityError,
            match=r"NUMERICAL_AUTHORITY_MISMATCH=scipy\.linalg\.eigh",
        ) as captured:
            solve_free_vibration(_s3_model(), num_modes=1, dense_size_limit=1)
        errors.append(str(captured.value))

    assert calls == 0
    assert errors[0] == errors[1]


@pytest.mark.parametrize("declaration", [None, True, -1, 1.5, "three"])
def test_malformed_algebraic_declaration_has_typed_stable_failure(
    declaration: Any,
) -> None:
    model = _s3_model()
    model.mesh.elements[1].dynamic_algebraic_nullity = declaration

    with pytest.raises(
        ElementCapabilityError,
        match=(
            r"CLASS_NAMESPACE_INSTANCE_SHADOW="
            r"dynamic_algebraic_nullity"
        ),
    ):
        solve_free_vibration(model, num_modes=1)


def test_missing_serialized_algebraic_policy_fails_closed() -> None:
    element = _s3_model().mesh.elements[1]
    payload = element.to_dict()
    assert payload.pop("algebraic_coordinate_policy") == ALGEBRAIC_COORDINATE_POLICY_ID

    with pytest.raises(
        ValueError,
        match="serialized qualified S3 algebraic coordinate policy is incompatible",
    ):
        shell_element_from_dict(payload)


def test_non_descriptor_modal_payload_has_no_descriptor_provenance() -> None:
    first = solve_free_vibration(_beam_model(), num_modes=1)
    second = solve_free_vibration(_beam_model(), num_modes=1)

    assert first.solver_status == second.solver_status == "ok"
    assert first.modes[0].eigenvalue == pytest.approx(100.0, rel=1.0e-14)
    assert first.frequencies_hz[0] == pytest.approx(10.0 / (2.0 * np.pi), rel=1.0e-14)
    forbidden = {
        "descriptor_modal",
        "policy_id",
        "declared_algebraic_element_ids",
        "declared_algebraic_formulations",
        "declared_algebraic_mass_certificate",
    }
    assert forbidden.isdisjoint(first.diagnostics)
    assert set(first.diagnostics["phase_timings_seconds"]) == {
        "validation",
        "stiffness",
        "mass",
        "geometric_stiffness",
        "constraint_reduction",
        "factorization",
        "eigen_iterations",
        "residual_batch",
        "residual_checks_and_mode_recovery",
        "total",
    }
    assert all(
        value >= 0.0 for value in first.diagnostics["phase_timings_seconds"].values()
    )
    assert _canonical(_stable_non_descriptor_payload(first)) == _canonical(
        _stable_non_descriptor_payload(second)
    )
