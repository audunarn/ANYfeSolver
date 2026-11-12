from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.buckling import solve_eigenvalue_buckling
from anyfesolver.corotational import corotational_element_response
from anyfesolver.current_state_tangent import (
    S3_V2D_CURRENT_STATE_PROJECTION_POLICY_ID,
    S3_V2D_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID,
    assemble_committed_current_tangent_components,
)
from anyfesolver.e4_pl_s3_v2d_element import (
    FORMULATION_ID,
    IMPLEMENTATION_ID,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import DEFAULT_Q4_FORMULATION, DEFAULT_S3_FORMULATION, create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.modal import solve_free_vibration
from anyfesolver.recovery import PatchRecoveryConfig, recover_stress_result


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/reference_cases/e4_pl_s3_v6g_recovery_current_eigen_contract.json"
COORDINATES = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.3, 1.1, 0.0))
SURFACE_KEYS = tuple(
    f"global_{component}_{surface}"
    for surface in ("top", "bot")
    for component in ("xx", "yy", "zz", "xy", "yz", "xz")
)


def _model() -> FEModel:
    model = FEModel("s3-v6g")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(COORDINATES, start=1):
        model.add_node(node_id, *coordinate)
    element = create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=0.08,
        reference_normal=(0.0, 0.0, 1.0),
    )
    assert type(element) is NativeParityE4PLS3V2DShellElement
    model.add_element(1, element)
    model.add_boundary_condition(FixedSupport("fixed-edge", [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition("node-3-in-plane", [3], {"ux": 0.0, "uy": 0.0})
    )
    return model


def _committed(model: FEModel, displacement: np.ndarray) -> dict[str, object]:
    element = model.mesh.elements[1]
    material = model.get_material("steel")
    initial = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    _force, _tangent, trial = corotational_element_response(
        model,
        1,
        element,
        displacement,
        tangent=True,
        committed_state=initial,
        num_layers=5,
        tangent_mode="consistent",
    )
    return element.seal_solver_integrated_nonlinear_state(
        model.mesh,
        material,
        trial,
        5,
        displacement,
        kinematics="corotational",
    )


def _compression() -> np.ndarray:
    displacement = np.zeros(18, dtype=np.float64)
    displacement[12] = -2.0e-4
    displacement[14] = 8.0e-5
    displacement[16] = 1.0e-4
    return displacement


def test_v6g_global_and_patch_recovery_are_physical_and_finite() -> None:
    model = _model()
    displacement = np.arange(18, dtype=np.float64) / 65536.0
    element = model.mesh.elements[1]
    direct = element.compute_stresses(
        model.mesh,
        displacement,
        model.get_material("steel"),
        return_global=True,
    )
    assert direct["formulation_id"] == FORMULATION_ID
    assert direct["implementation_id"] == IMPLEMENTATION_ID
    assert direct["physical_stress_available"] is True
    assert direct["numerical_fields_excluded"] is True
    for key in SURFACE_KEYS:
        assert np.asarray(direct[key]).shape == (3,)
        assert np.all(np.isfinite(direct[key]))

    public = recover_stress_result(
        model,
        displacement,
        return_global=True,
        patch_config=PatchRecoveryConfig(),
    )
    recovered = public.element_stresses[1]
    for key in SURFACE_KEYS:
        np.testing.assert_array_equal(recovered[key], direct[key])
    assert public.nodal_stresses is not None
    assert public.provenance.per_element_source[1] == "elastic_displacement_reconstruction"


def test_v6g_committed_recovery_uses_stored_layers_and_objective_frame() -> None:
    model = _model()
    displacement = _compression()
    state = _committed(model, displacement)
    before = canonical_json_bytes(state)
    public = recover_stress_result(
        model,
        displacement,
        element_states={1: state},
        kinematics="corotational",
        return_global=True,
    )
    recovered = public.element_stresses[1]
    assert recovered["recovery_history_source"] == "committed_s3_v2d_hammer3_state"
    assert recovered["recovery_scope"] == "PHYSICAL_COMMITTED_GLOBAL_SURFACE_TENSORS"
    assert public.history_aware is True
    assert public.provenance.per_element_source[1] == "committed_s3_v2d_hammer3_state"
    assert public.provenance.per_element_component_sources[1]["stress_frame"] == (
        "element_independent_corotational_physical_frame"
    )
    for key in SURFACE_KEYS:
        assert np.asarray(recovered[key]).shape == (3,)
        assert np.all(np.isfinite(recovered[key]))
    assert canonical_json_bytes(state) == before


def test_v6g_committed_tangent_and_current_eigen_routes_are_registered() -> None:
    model = _model()
    displacement = _compression()
    state = _committed(model, displacement)
    before = canonical_json_bytes(state)
    material, geometric, total, info = assemble_committed_current_tangent_components(
        model,
        displacement,
        {1: state},
        5,
    )
    material = material.toarray()
    geometric = geometric.toarray()
    total = total.toarray()
    np.testing.assert_allclose(material + geometric, total, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(total, total.T, rtol=0.0, atol=1.0e-8)
    element_info = info["element_components"]["1"]
    assert element_info["decomposition_policy_id"] == (
        S3_V2D_CURRENT_STATE_TANGENT_DECOMPOSITION_POLICY_ID
    )
    assert element_info["projection_policy_id"] == (
        S3_V2D_CURRENT_STATE_PROJECTION_POLICY_ID
    )
    assert canonical_json_bytes(state) == before

    modal = solve_free_vibration(
        model,
        num_modes=1,
        current_state_displacements=displacement,
        current_state_element_states={1: state},
    )
    assert modal.solver_status == "ok"
    assert float(modal.frequencies_hz[0]) > 0.0
    assert modal.assembly_info["current_state_route"]["route"] == "qualified_s3_v2d"

    buckling = solve_eigenvalue_buckling(
        model,
        num_modes=1,
        current_state_displacements=displacement,
        current_state_element_states={1: state},
    )
    assert buckling.solver_status == "ok"
    assert buckling.critical_load_factor is not None
    assert float(buckling.critical_load_factor) > 0.0
    assert buckling.assembly_info["current_state_route"]["route"] == "qualified_s3_v2d"
    assert canonical_json_bytes(state) == before


def test_v6g_state_mutation_and_api_mutation_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    displacement = _compression()
    state = _committed(model, displacement)
    mutated = json.loads(canonical_json_bytes(state))
    mutated["committed_total_u"][12] += 1.0
    with pytest.raises((ValueError, ElementCapabilityError)):
        assemble_committed_current_tangent_components(
            model, displacement, {1: mutated}, 5
        )

    monkeypatch.setattr(
        NativeParityE4PLS3V2DShellElement,
        "compute_mass_matrix",
        lambda *args, **kwargs: np.zeros((18, 18)),
    )
    with pytest.raises(ElementCapabilityError, match="compute_mass_matrix"):
        solve_free_vibration(
            model,
            num_modes=1,
            current_state_displacements=displacement,
            current_state_element_states={1: state},
        )


def test_v6g_contract_is_canonical_and_defaults_remain_frozen() -> None:
    raw = CONTRACT.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    contract = json.loads(raw)
    assert raw == (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert contract["production_boundary"]["activation_authorized"] is False
    assert contract["production_boundary"]["stage4a_scientific_rerun_authorized"] is False
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"


def test_v6g_closes_the_frozen_v6f_readiness_inventory() -> None:
    audit_path = ROOT / "docs/reference_cases/e4_pl_s3_v6f_final_parity_audit.py"
    spec = importlib.util.spec_from_file_location("v6f_readiness", audit_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(ROOT)
    assert result["audit"]["open_routes"] == []
    assert result["audit"]["open_route_count"] == 0
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V6F_STAGE4A_RERUN_REVIEW"
    assert result["stage4a_scientific_rerun_authorized"] is False
