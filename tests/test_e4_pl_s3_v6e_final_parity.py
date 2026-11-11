from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.dynamics import TransientConfig, solve_transient_newmark
from anyfesolver.e4_pl_s3_v2d_element import (
    DYNAMIC_ALGEBRAIC_MASS_WITNESS,
    DYNAMIC_ALGEBRAIC_POLICY_ID,
    FORMULATION_ID,
    GEOMETRIC_STIFFNESS_POLICY_ID,
    IMPLEMENTATION_ID,
    MASS_POLICY_ID,
    NativeParityCapabilityError,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.elements import (
    DEFAULT_Q4_FORMULATION,
    DEFAULT_S3_FORMULATION,
    create_shell_element,
    shell_element_from_dict,
)
from anyfesolver.fe_core import FEModel, FEMesh, Material
from anyfesolver.modal import solve_free_vibration
from anyfesolver.recovery import recover_stress_result


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/reference_cases/e4_pl_s3_v6e_v2d_final_parity_contract.json"
COORDINATES = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.3, 1.1, 0.0))


def _element() -> NativeParityE4PLS3V2DShellElement:
    made = create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=0.08,
        reference_normal=(0.0, 0.0, 1.0),
    )
    assert type(made) is NativeParityE4PLS3V2DShellElement
    return made


def _model() -> FEModel:
    model = FEModel("s3-v6e")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    model.add_element(1, _element())
    model.add_boundary_condition(FixedSupport("fixed-edge", [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
    )
    return model


def _mesh() -> FEMesh:
    mesh = FEMesh()
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        mesh.add_node(node_id, *coordinate)
    return mesh


def _material() -> Material:
    return Material("steel", 210.0e9, 0.3, density=7850.0)


def test_v6e_descriptor_mass_witness_is_exact() -> None:
    element = _element()
    components = element.compute_mass_components(_mesh(), _material())
    directions = element.dynamic_algebraic_directions(_mesh(), _material())
    assert element.dynamic_algebraic_nullity == 9
    assert element.dynamic_algebraic_policy == DYNAMIC_ALGEBRAIC_POLICY_ID
    assert element.dynamic_algebraic_mass_witness == DYNAMIC_ALGEBRAIC_MASS_WITNESS
    assert components["mass_policy_id"] == MASS_POLICY_ID
    assert components["condensed_rank"] == 9
    np.testing.assert_array_equal(components["global"] @ directions, np.zeros((18, 9)))
    np.testing.assert_array_equal(directions.T @ directions, np.eye(9))


def test_v6e_descriptor_modal_route_is_finite_and_provenanced() -> None:
    result = solve_free_vibration(_model(), num_modes=1)
    assert result.solver_status == "ok"
    assert result.frequencies_hz.shape == (1,)
    assert float(result.frequencies_hz[0]) > 0.0
    assert result.diagnostics["descriptor_modal"] is True
    assert result.diagnostics["declared_algebraic_element_ids"] == [1]
    declared = result.diagnostics["declared_algebraic_formulations"]
    assert declared == [
        {
            "element_id": 1,
            "formulation_id": FORMULATION_ID,
            "algebraic_coordinate_policy": DYNAMIC_ALGEBRAIC_POLICY_ID,
        }
    ]


def test_v6e_rayleigh_transient_uses_certified_descriptor_reduction() -> None:
    initial = np.zeros(18, dtype=np.float64)
    initial[14] = 1.0e-5
    result = solve_transient_newmark(
        _model(),
        TransientConfig(
            dt=1.0e-4,
            t_end=2.0e-4,
            initial_displacement=initial,
            rayleigh_alpha=0.02,
            rayleigh_beta=1.0e-5,
        ),
    )
    assert result.diagnostics["descriptor_transient"] is True
    assert result.diagnostics["declared_algebraic_element_ids"] == [1]
    assert result.diagnostics["rayleigh_alpha"] == 0.02
    assert result.diagnostics["rayleigh_beta"] == 1.0e-5
    assert np.all(np.isfinite(result.displacements))


def test_v6e_reference_buckling_route_is_finite_and_source_bounded() -> None:
    state = {1: {"membrane_compression": [1.0, 1.0, 0.0]}}
    result = solve_eigenvalue_buckling(_model(), state, num_modes=1)
    assert result.solver_status == "ok"
    assert result.critical_load_factor is not None
    assert float(result.critical_load_factor) > 0.0
    element = _element()
    matrix = element.compute_geometric_stiffness_matrix(
        _mesh(), _material(), state[1]
    )
    np.testing.assert_array_equal(matrix, matrix.T)
    assert np.linalg.matrix_rank(matrix) == 6
    changed = dict(state[1])
    changed["bending_compression"] = [1.0, 0.0, 0.0]
    with pytest.raises(NativeParityCapabilityError, match="bending_compression"):
        element.compute_geometric_stiffness_matrix(_mesh(), _material(), changed)
    assert GEOMETRIC_STIFFNESS_POLICY_ID.startswith("S3_V2D_CST_")


def test_v6e_local_recovery_and_public_recovery_preserve_provenance() -> None:
    model = _model()
    displacements = np.arange(18, dtype=np.float64) / 8192.0
    local = model.mesh.elements[1].compute_stresses(
        model.mesh,
        displacements,
        model.get_material("steel"),
        return_global=False,
    )
    assert local["formulation_id"] == FORMULATION_ID
    assert local["implementation_id"] == IMPLEMENTATION_ID
    assert local["recovery_scope"] == "PHYSICAL_LOCAL_SURFACE_STRESSES"
    assert local["physical_stress_available"] is True
    assert np.asarray(local["von_mises"]).shape == (3,)
    assert local["numerical_fields_excluded"] is True
    public = recover_stress_result(model, displacements, return_global=False)
    assert public.element_stresses[1]["formulation_id"] == FORMULATION_ID
    assert public.element_stresses[1]["recovery_scope"] == "PHYSICAL_LOCAL_SURFACE_STRESSES"
    np.testing.assert_array_equal(
        public.element_stresses[1]["von_mises"], local["von_mises"]
    )


def test_v6e_serialization_round_trip_is_exact_and_pickle_stays_blocked() -> None:
    element = _element()
    payload = element.to_dict()
    restored = shell_element_from_dict(payload)
    assert type(restored) is NativeParityE4PLS3V2DShellElement
    assert restored.to_dict() == payload
    assert payload["dynamic_algebraic_policy_id"] == DYNAMIC_ALGEBRAIC_POLICY_ID
    assert payload["geometric_stiffness_policy_id"] == GEOMETRIC_STIFFNESS_POLICY_ID
    changed = dict(payload)
    changed["mass_policy_id"] = "MUTATED"
    with pytest.raises(ValueError, match="fingerprint"):
        shell_element_from_dict(changed)
    with pytest.raises(NativeParityCapabilityError, match="python_pickle_restart"):
        element.__getstate__()


def test_v6e_contract_and_production_boundary_are_exact() -> None:
    raw = CONTRACT.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["candidate"]["implementation_id"] == (
        "E4_PL_S3_V2D_FINAL_PARITY_GATE_V1"
    )
    assert contract["candidate"]["formulation_id"] == FORMULATION_ID
    assert contract["runtime_policy"] == {
        "automatic_retry": False,
        "focused_cycle_limit_seconds": 600,
        "long_running_scientific_cycle_authorized": False,
        "required_focused_cycles": 2,
    }
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"
    assert contract["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
        "stage4a_scientific_rerun_authorized": False,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "formulation_id",
        "implementation_id",
        "dynamic_algebraic_policy_id",
        "geometric_stiffness_policy_id",
        "mass_policy_id",
    ),
)
def test_v6e_serialized_identity_mutations_fail_closed(mutation: str) -> None:
    payload = _element().to_dict()
    payload[mutation] = "MUTATED"
    with pytest.raises((ValueError, TypeError)):
        shell_element_from_dict(payload)
