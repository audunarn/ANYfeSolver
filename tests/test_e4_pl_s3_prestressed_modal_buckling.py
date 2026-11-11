"""End-to-end prestressed modal and buckling parity for qualified S3."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest
from scipy import linalg

import anyfesolver.modal as modal_module
import anyfesolver.e4_pl_element as q4_element_module
import anyfesolver.e4_pl_s3_element as s3_element_module
import anyfesolver.e4_pl_s3_state as s3_state_module
import anyfesolver.matrix_assembly as matrix_assembly_module
from anyfesolver import (
    AnalysisSession,
    BoundaryCondition,
    FEModel,
    FixedSupport,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    solve_static_nonlinear,
    solve_eigenvalue_buckling,
    solve_free_vibration,
)
from anyfesolver.boundary import LoadCase
from anyfesolver.control import CancellationToken
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.algebraic_dynamics import (
    DESCRIPTOR_RAYLEIGH_REFINEMENT_POLICY_ID,
)
from anyfesolver.e4_pl_s3_element import (
    REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID,
)
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import BeamElement, Element, ShellElement
from anyfesolver.linalg import FactorizationCache
from anyfesolver.matrix_assembly import (
    AssemblyError,
    assemble_geometric_stiffness_matrix,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
)
from anyfesolver.modal import (
    CURRENT_STATE_MODAL_POLICY_ID,
    PRESTRESSED_MODAL_POLICY_ID,
    PRESTRESS_INPUT_SCHEMA_ID,
    QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID,
)
from anyfesolver.nonlinear_state import (
    NonlinearStateStore,
    create_model_native_rotation_store,
    discard_active_state_candidate,
)
from anyfesolver.shell_sections import GeneralizedShellSection


def _model() -> FEModel:
    model = FEModel("qualified-s3-prestressed-eigen")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
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


def _qualified_q4_reference_model(*, warped: bool = False) -> FEModel:
    model = FEModel("qualified-q4-reference-prestress-lifecycle")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.08 if warped else 0.0),
            (0.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    return model


def _qualified_q4_beam_reference_model() -> FEModel:
    """Small mixed model exercising qualified-shell input authority."""

    return _qualified_q4_beam_chain_reference_model(1)


def _qualified_q4_beam_chain_reference_model(beam_count: int) -> FEModel:
    """Mixed qualified-Q4 model with an exact built-in beam chain."""

    model = _qualified_q4_reference_model()
    previous_node_id = 2
    for offset in range(beam_count):
        node_id = 5 + offset
        element_id = 2 + offset
        model.add_node(node_id, 2.0 + offset, 0.0, 0.0)
        model.add_element(
            element_id,
            BeamElement(
                element_id,
                [previous_node_id, node_id],
                "steel",
                {
                    "area": 0.02,
                    "Iy": 1.0e-6,
                    "Iz": 1.0e-6,
                    "J": 1.0e-6,
                },
            ),
        )
        previous_node_id = node_id
    return model


@pytest.mark.parametrize("route", ("modal", "buckling"))
def test_q4_formulation_descriptor_is_never_invoked_by_reference_eigen_routes(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    model = _qualified_q4_reference_model()
    expected = vars(QualifiedE4PLShellElement)["formulation_id"]
    reached: list[str] = []

    class SplitFormulationDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> str:
            self.reads += 1
            return expected if instance is None else str(expected)

    descriptor = SplitFormulationDescriptor()
    monkeypatch.setattr(
        QualifiedE4PLShellElement, "formulation_id", descriptor
    )
    monkeypatch.setattr(
        model,
        "apply_boundary_conditions",
        lambda: reached.append("boundary"),
    )

    with pytest.raises(ElementCapabilityError, match="FORMULATION_ID_CLASS_MISMATCH"):
        if route == "modal":
            solve_free_vibration(model, num_modes=1)
        else:
            solve_eigenvalue_buckling(
                model,
                {1: {"membrane_compression": [1.0e4, 0.0, 0.0]}},
                num_modes=1,
                reference_elastic_only=True,
            )
    assert descriptor.reads == 0
    assert reached == []


@pytest.mark.parametrize("route", ("modal", "buckling"))
def test_warped_q4_shape_spoof_rejects_before_reference_eigen_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    model = _qualified_q4_reference_model(warped=True)
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("shape")
        raise AssertionError("spoofed warped-Q4 shape mechanics reached")

    monkeypatch.setattr(
        ShellElement, "_compute_4node_shape_functions", forbidden
    )
    monkeypatch.setattr(
        model,
        "apply_boundary_conditions",
        lambda: reached.append("boundary"),
    )

    with pytest.raises(ElementCapabilityError, match="_compute_4node_shape_functions"):
        if route == "modal":
            solve_free_vibration(model, num_modes=1)
        else:
            solve_eigenvalue_buckling(
                model,
                {1: {"membrane_compression": [1.0e4, 0.0, 0.0]}},
                num_modes=1,
                reference_elastic_only=True,
            )
    assert reached == []


@pytest.mark.parametrize(
    ("family", "route"),
    (
        ("q4", "modal"),
        ("q4", "buckling"),
        ("s3", "modal"),
        ("s3", "buckling"),
    ),
)
def test_reference_eigen_quadrature_descriptor_rejects_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    route: str,
) -> None:
    model = _qualified_q4_reference_model() if family == "q4" else _model()
    owner = ShellElement if family == "q4" else QualifiedE4PLS3ShellElement
    expected = vars(owner)["gauss_points"]
    reached: list[str] = []

    class StatefulQuadratureDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> object:
            self.reads += 1
            if instance is None:
                return expected
            return np.zeros((1, 2), dtype=np.float64)

    descriptor = StatefulQuadratureDescriptor()
    monkeypatch.setattr(owner, "gauss_points", descriptor)
    monkeypatch.setattr(
        model,
        "apply_boundary_conditions",
        lambda: reached.append("boundary"),
    )
    state = (
        {"membrane_compression": [1.0e4, 0.0, 0.0]}
        if family == "q4"
        else _compression()
    )

    with pytest.raises(ElementCapabilityError, match="gauss_points"):
        if route == "modal":
            solve_free_vibration(model, num_modes=1, prestress_states={1: state})
        else:
            solve_eigenvalue_buckling(
                model,
                {1: state},
                num_modes=1,
                reference_elastic_only=True,
            )
    assert descriptor.reads == 0
    assert reached == []


def test_s3_serialization_quadrature_descriptor_rejects_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = _model().mesh.elements[1]
    expected = vars(QualifiedE4PLS3ShellElement)["gauss_points"]

    class SplitQuadratureDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> object:
            self.reads += 1
            return expected if instance is None else np.zeros((7, 2))

    descriptor = SplitQuadratureDescriptor()
    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement, "gauss_points", descriptor
    )

    with pytest.raises(
        ValueError,
        match=r"qualified S3 .*class.* changed",
    ):
        element.to_dict()
    assert descriptor.reads == 0


@pytest.mark.parametrize("family", ("q4", "s3"))
@pytest.mark.parametrize("mutation", ("blatant", "split_descriptor"))
def test_deserialization_revalidates_static_quadrature_authority(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    mutation: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    payload = element.to_dict()
    owner = ShellElement if family == "q4" else QualifiedE4PLS3ShellElement
    element_type = (
        QualifiedE4PLShellElement
        if family == "q4"
        else QualifiedE4PLS3ShellElement
    )
    expected = vars(owner)["gauss_points"]

    class SplitQuadratureDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> object:
            self.reads += 1
            return expected if instance is None else np.zeros((1, 2))

    descriptor = SplitQuadratureDescriptor()
    replacement: object = (
        property(lambda _self: np.zeros((1, 2), dtype=np.float64))
        if mutation == "blatant"
        else descriptor
    )
    monkeypatch.setattr(owner, "gauss_points", replacement)

    with pytest.raises(
        ValueError,
        match=rf"qualified {family.upper()} .*class.* changed",
    ):
        element_type.from_dict(payload)
    if mutation == "split_descriptor":
        assert descriptor.reads == 0


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_serialization_guards_survive_global_authority_handle_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    element_type = type(element)
    payload = element.to_dict()
    module = q4_element_module if family == "q4" else s3_element_module
    identity_name = (
        "_Q4_SERIALIZATION_GLOBAL_IDENTITY"
        if family == "q4"
        else "_S3_SERIALIZATION_GLOBAL_IDENTITY"
    )
    validator_name = (
        "_validate_q4_serialization_authority"
        if family == "q4"
        else "_validate_s3_serialization_authority"
    )
    quadrature_name = (
        "_validate_q4_quadrature_authority"
        if family == "q4"
        else "_validate_s3_quadrature_values"
    )
    monkeypatch.setattr(module, identity_name, {})
    monkeypatch.setattr(module, validator_name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, quadrature_name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "FORMULATION_ID", f"FOREIGN_{family.upper()}")

    assert not hasattr(element_type.to_dict, "__wrapped__")
    assert not hasattr(element_type.from_dict.__func__, "__wrapped__")
    with pytest.raises(ValueError, match="global FORMULATION_ID authority"):
        element.to_dict()

    forged = dict(payload)
    forged["formulation_id"] = getattr(module, "FORMULATION_ID")
    with pytest.raises(ValueError, match="global FORMULATION_ID authority"):
        element_type.from_dict(forged)


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_serialized_configuration_descriptor_is_never_invoked(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    owner = type(element)

    class ForbiddenThicknessDescriptor:
        reads = 0

        def __get__(self, _instance: object, _owner: object) -> float:
            self.reads += 1
            raise AssertionError("serialization invoked thickness descriptor")

        def __set__(self, _instance: object, _value: object) -> None:
            raise AssertionError("serialization wrote thickness descriptor")

    descriptor = ForbiddenThicknessDescriptor()
    monkeypatch.setattr(owner, "thickness", descriptor, raising=False)

    with pytest.raises(
        ValueError,
        match=rf"qualified {family.upper()} .*class.* changed",
    ):
        element.to_dict()
    assert descriptor.reads == 0


@pytest.mark.parametrize(
    "name", ("capability_restrictions", "capability_gaps")
)
def test_broad_buckling_capability_descriptor_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    model = _model()
    reached: list[str] = []

    class ForbiddenCapabilityDescriptor:
        reads = 0

        def __get__(self, _instance: object, _owner: object) -> object:
            self.reads += 1
            raise AssertionError("buckling invoked capability descriptor")

    descriptor = ForbiddenCapabilityDescriptor()
    monkeypatch.setattr(QualifiedE4PLS3ShellElement, name, descriptor)
    monkeypatch.setattr(
        model, "apply_boundary_conditions", lambda: reached.append("boundary")
    )

    with pytest.raises(ElementCapabilityError, match=name):
        solve_eigenvalue_buckling(
            model,
            {1: _compression()},
            num_modes=1,
        )
    assert descriptor.reads == 0
    assert reached == []


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_direct_serialization_rejects_spoofed_dynamic_station_table(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    module = q4_element_module if family == "q4" else s3_element_module
    name = "_GAUSS" if family == "q4" else "TRIANGLE_QUADRATURE"

    class EqualitySpoof:
        def __ne__(self, _other: object) -> bool:
            return False

        def __iter__(self):
            return iter(((0.0, 0.0, 1.0),))

    monkeypatch.setattr(module, name, EqualitySpoof())
    with pytest.raises(ValueError, match="station-table authority"):
        element.to_dict()


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_direct_serialization_binds_complete_super_serializer_chain(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    monkeypatch.setattr(
        Element,
        "to_dict",
        lambda _self: {"element_id": -1, "node_ids": [], "material_name": "x"},
    )
    with pytest.raises(ValueError, match="root serialization authority"):
        element.to_dict()


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_direct_serialization_rejects_attacker_config_without_conversion(
    family: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    reached: list[str] = []

    class ForbiddenFloat:
        def __float__(self) -> float:
            reached.append("float")
            raise AssertionError("attacker configuration was evaluated")

    object.__setattr__(element, "thickness", ForbiddenFloat())
    with pytest.raises(ValueError, match="instance-data authority"):
        element.to_dict()
    assert reached == []


@pytest.mark.parametrize("family", ("q4", "s3"))
def test_direct_serialization_binds_generalized_section_method_namespace(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    element = (
        _qualified_q4_reference_model().mesh.elements[1]
        if family == "q4"
        else _model().mesh.elements[1]
    )
    reached: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("rotated")
        raise AssertionError("mutated generalized-section API was evaluated")

    monkeypatch.setattr(GeneralizedShellSection, "rotated", forbidden)
    with pytest.raises(ValueError, match="generalized-section serialization"):
        element.to_dict()
    assert reached == []


def _exact_noncurrent_reference_state(
    model: FEModel,
    status: str,
) -> dict[str, object]:
    element = model.mesh.elements[1]
    material = model.get_material(element.material_name)
    node_count = len(element.node_ids)
    local_u = np.zeros(6 * node_count, dtype=np.float64)
    if isinstance(element, QualifiedE4PLS3ShellElement):
        active = element.init_model_bound_nonlinear_state(
            model.mesh,
            material,
            3,
        )
    else:
        _force, tangent, candidate = element.compute_nonlinear_response(
            model.mesh,
            material,
            local_u,
            None,
            3,
            True,
        )
        assert tangent is not None
        active = element.seal_committed_current_tangent_state(
            model.mesh,
            material,
            local_u,
            candidate,
            3,
        )
    if status == "FAILED_NONAUTHORITATIVE":
        state = element.mark_noncurrent_failed_state(
            model.mesh,
            material,
            local_u,
            active,
            3,
            failure_reason="reference-prestress-lifecycle-regression",
        )
        element.validate_noncurrent_failed_state(
            model.mesh,
            material,
            state,
            3,
        )
        return state
    assert status == "DELETED_FROZEN_NONCURRENT"
    state = element.seal_noncurrent_deleted_state(
        model.mesh,
        material,
        local_u,
        active,
        3,
        deletion_step_index=1,
        deletion_load_factor=0.25,
        residual_stiffness_fraction=0.2,
        trigger_name="reference-prestress-lifecycle-regression",
    )
    element.validate_noncurrent_deleted_state(
        model.mesh,
        material,
        state,
        3,
        expected_deletion_step_index=1,
        expected_deletion_load_factor=0.25,
        expected_residual_stiffness_fraction=0.2,
        expected_trigger_name="reference-prestress-lifecycle-regression",
    )
    return state


def _compression(scale: float = 1.0) -> dict[str, object]:
    pressure = 1.0e5 * float(scale)
    thickness = 0.02
    return {
        "bubble_linearization_policy": (
            REFERENCE_ELASTIC_BUBBLE_LINEARIZATION_ID
        ),
        "membrane_compression": [pressure, pressure, 0.0],
        "bending_compression": [0.0, 0.0, 0.0],
        "stress_second_moment": [
            pressure * thickness**2 / 12.0,
            pressure * thickness**2 / 12.0,
            0.0,
        ],
    }


def _finite_descriptor_values(
    stiffness: np.ndarray, mass: np.ndarray
) -> np.ndarray:
    alpha_beta, _vectors = linalg.eig(
        stiffness,
        mass,
        homogeneous_eigvals=True,
        right=True,
    )
    alpha, beta = alpha_beta
    scale = max(float(np.max(np.abs(beta))), np.finfo(float).tiny)
    finite = np.abs(beta) > np.finfo(float).eps * len(beta) * scale
    values = np.real_if_close(alpha[finite] / beta[finite], tol=1000)
    values = np.asarray(np.real(values), dtype=float)
    return np.sort(values[np.isfinite(values)])


def _reduced_operators(
    model: FEModel, state: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.apply_boundary_conditions()
    stiffness, _stiffness_info = assemble_stiffness_matrix(model)
    geometric, _geometric_info = assemble_geometric_stiffness_matrix(
        model, {1: state}
    )
    mass, _mass_info = assemble_mass_matrix(model)
    reduced_stiffness, _, transform, _, _, _ = build_constraint_transformation(
        stiffness,
        np.zeros(stiffness.shape[0], dtype=float),
        model,
    )
    return (
        reduced_stiffness.toarray(),
        (transform.T @ geometric @ transform).toarray(),
        (transform.T @ mass @ transform).toarray(),
    )


def _axial_nonlinear_model() -> tuple[FEModel, LoadCase]:
    model = FEModel("qualified-s3-current-state-modal")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.01,
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    model.add_boundary_condition(FixedSupport("node-1", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-2-axial-only",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("node-3", [3]))
    load = LoadCase("pull")
    load.add_nodal_load(2, [1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _committed_trial_force(
    model: FEModel,
    committed_displacements: np.ndarray,
    trial_displacements: np.ndarray,
    states: dict[int, object],
) -> np.ndarray:
    store = NonlinearStateStore.from_shell_layouts((), states)
    native = create_model_native_rotation_store(
        model, store, committed_displacements
    )
    assert native is not None
    store.attach_native_rotation_store(native)
    try:
        from anyfesolver.nonlinear_static import _assemble_nonlinear_system

        force, _tangent, _candidate = _assemble_nonlinear_system(
            model,
            trial_displacements,
            store,
            3,
            tangent=False,
            require_full_coordinates=True,
        )
        return np.asarray(force, dtype=float)
    finally:
        discard_active_state_candidate(store)


def test_prestressed_modal_matches_independent_descriptor_pencil() -> None:
    model = _model()
    state = _compression()
    material, geometric, mass = _reduced_operators(model, state)
    oracle = _finite_descriptor_values(material - geometric, mass)

    baseline = solve_free_vibration(model, num_modes=3)
    result = solve_free_vibration(
        model,
        num_modes=3,
        prestress_states={1: state},
    )

    assert result.solver_status == "ok"
    assert result.assembly_info["prestressed_modal_policy_id"] == (
        PRESTRESSED_MODAL_POLICY_ID
    )
    assert result.assembly_info["prestress_operator_authority_policy_id"] == (
        QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
    )
    assert result.assembly_info["reference_operator_authority_policy_id"] == (
        QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
    )
    assert baseline.assembly_info["reference_operator_authority_policy_id"] == (
        QUALIFIED_PRESTRESS_OPERATOR_AUTHORITY_POLICY_ID
    )
    assert result.assembly_info["geometric_stiffness"]["matrix_type"] == (
        "geometric_stiffness"
    )
    np.testing.assert_allclose(
        np.asarray([mode.eigenvalue for mode in result.modes]),
        oracle[:3],
        rtol=3.0e-10,
        atol=2.0e-7,
    )
    assert result.frequencies_hz[0] < baseline.frequencies_hz[0]
    assert result.diagnostics["max_residual_norm"] < 3.0e-10


def test_prestressed_modal_session_matches_uncached_path() -> None:
    model = _model()
    states = {1: _compression(0.4)}
    expected = solve_free_vibration(
        model,
        num_modes=3,
        prestress_states=states,
    )

    with AnalysisSession(model) as session:
        first = solve_free_vibration(
            model,
            num_modes=3,
            session=session,
            prestress_states=states,
        )
        repeated = solve_free_vibration(
            model,
            num_modes=3,
            session=session,
            prestress_states=states,
        )

    np.testing.assert_allclose(
        first.frequencies_hz, expected.frequencies_hz, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        repeated.frequencies_hz, first.frequencies_hz, rtol=2.0e-12
    )
    assert repeated.diagnostics["analysis_session"]["plan_reused"] is True


def test_prestress_input_is_closed_world_and_prevalidated_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def forbidden() -> None:
        raise AssertionError("prestress guard reached stiffness mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="not in the model"):
        solve_free_vibration(model, prestress_states={999: _compression()})
    with pytest.raises(ValueError, match="duplicate or ambiguous"):
        solve_free_vibration(
            model, prestress_states={1: _compression(), "1": _compression()}
        )
    with pytest.raises(ElementCapabilityError, match="bubble_policy"):
        solve_free_vibration(
            model,
            prestress_states={1: {"membrane_compression": [1.0, 1.0, 0.0]}},
        )
    with pytest.raises(ValueError, match="strict canonical"):
        solve_free_vibration(
            model,
            prestress_states={
                1: {
                    **_compression(),
                    "membrane_compression": [float("nan"), 0.0, 0.0],
                }
            },
        )


@pytest.mark.parametrize("family", ("qualified_q4", "qualified_s3"))
@pytest.mark.parametrize(
    "status", ("FAILED_NONAUTHORITATIVE", "DELETED_FROZEN_NONCURRENT")
)
@pytest.mark.parametrize("analysis", ("modal", "buckling"))
def test_reference_prestress_rejects_exact_noncurrent_lifecycle_before_mechanics(
    family: str,
    status: str,
    analysis: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_reference_model() if family == "qualified_q4" else _model()
    state = _exact_noncurrent_reference_state(model, status)
    marker_key = (
        "qualified_q4_activity_disposition"
        if family == "qualified_q4"
        else "qualified_s3_activity_disposition"
    )
    assert state[marker_key]["status"] == status
    calls = 0

    def forbidden() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("noncurrent reference prestress reached mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="noncurrent activity dispositions"):
        if analysis == "modal":
            solve_free_vibration(
                model,
                num_modes=1,
                prestress_states={1: state},
            )
        else:
            solve_eigenvalue_buckling(
                model,
                {1: state},
                num_modes=1,
                reference_elastic_only=True,
            )
    assert calls == 0


@pytest.mark.parametrize("analysis", ("modal", "buckling"))
@pytest.mark.parametrize(
    "marker_key",
    (
        "qualified_q4_activity_disposition",
        "qualified_s3_activity_disposition",
    ),
)
def test_reference_prestress_rejects_malformed_noncurrent_marker_before_mechanics(
    analysis: str,
    marker_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_reference_model()
    state = {
        "membrane_compression": [1.0e4, 0.0, 0.0],
        marker_key: {"status": "ACTIVE", "disposition_sha256": "tampered"},
    }
    calls = 0

    def forbidden() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("malformed reference prestress reached mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="noncurrent activity dispositions"):
        if analysis == "modal":
            solve_free_vibration(
                model,
                num_modes=1,
                prestress_states={1: state},
            )
        else:
            solve_eigenvalue_buckling(
                model,
                {1: state},
                num_modes=1,
                reference_elastic_only=True,
            )
    assert calls == 0


def test_prestress_callable_is_evaluated_once_and_bound_to_provenance() -> None:
    model = _model()
    calls: list[tuple[int, int]] = []

    def provider(element_id: int, element: object) -> dict[str, object]:
        calls.append((element_id, int(getattr(element, "element_id"))))
        return _compression(0.35)

    result = solve_free_vibration(model, num_modes=2, prestress_states=provider)

    assert result.solver_status == "ok"
    assert calls == [(1, 1)]
    provenance = result.assembly_info["prestress_input"]
    assert provenance["schema_id"] == PRESTRESS_INPUT_SCHEMA_ID
    assert provenance["source_kind"] == "callable_evaluated_once"
    assert provenance["element_ids"] == [1]
    assert provenance["supplied_element_ids"] == [1]
    assert provenance["explicitly_unstressed_element_ids"] == []
    assert len(provenance["state_sha256"]["1"]) == 64


@pytest.mark.parametrize("source_kind", ("callable", "mapping"))
def test_prestress_observation_rechecks_authority_before_canonicalization(
    source_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> np.ndarray:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    def change_runtime() -> None:
        reached.append(source_kind)
        monkeypatch.setattr(np, "asarray", forbidden_numeric)

    if source_kind == "callable":
        def source(_element_id: int, _element: object) -> dict[str, object]:
            change_runtime()
            return _compression(0.35)
    else:
        class ObservedStates(dict[int, dict[str, object]]):
            def items(self):  # type: ignore[override]
                change_runtime()
                return super().items()

        source = ObservedStates({1: _compression(0.35)})

    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        solve_free_vibration(model, num_modes=1, prestress_states=source)
    assert reached == [source_kind]


@pytest.mark.parametrize("analysis", ("modal", "buckling"))
def test_eigen_cancellation_callback_rechecks_authority_before_recovery(
    analysis: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> np.ndarray:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    class ObservedToken(CancellationToken):
        def raise_if_cancelled(self, stage: str = "") -> None:
            target = "modal.recovery:" if analysis == "modal" else "buckling.root:"
            if stage.startswith(target):
                reached.append("cancellation")
                monkeypatch.setattr(np, "asarray", forbidden_numeric)

    with pytest.raises(ElementCapabilityError, match="NUMERICAL_AUTHORITY_MISMATCH"):
        if analysis == "modal":
            solve_free_vibration(
                model,
                num_modes=1,
                cancellation_token=ObservedToken(),
            )
        else:
            solve_eigenvalue_buckling(
                model,
                {1: _compression(0.35)},
                num_modes=1,
                cancellation_token=ObservedToken(),
                reference_elastic_only=True,
            )
    assert reached == ["cancellation"]


@pytest.mark.parametrize("owner_kind", ("session", "cache"))
def test_qualified_eigen_routes_require_exact_runtime_owners(
    owner_kind: str,
) -> None:
    model = _model()
    reached: list[str] = []

    class ObservedSession(AnalysisSession):
        def stiffness_plan(self, owned_model: FEModel):  # type: ignore[override]
            reached.append("session")
            return super().stiffness_plan(owned_model)

    class ObservedCache(FactorizationCache):
        def linear_operator(self, *args: object, **kwargs: object):  # type: ignore[override]
            reached.append("cache")
            return super().linear_operator(*args, **kwargs)

    with pytest.raises(ElementCapabilityError, match="exact AnalysisSession|exact FactorizationCache"):
        if owner_kind == "session":
            solve_free_vibration(
                model,
                num_modes=1,
                session=ObservedSession(model),
            )
        else:
            solve_free_vibration(
                model,
                num_modes=1,
                shift=0.0,
                factorization_cache=ObservedCache(),
            )
    assert reached == []


@pytest.mark.parametrize("source_kind", ("callable", "mapping"))
def test_prestress_states_are_owned_when_each_value_is_observed(
    source_kind: str,
) -> None:
    model = _model()
    for node_id, coordinates in enumerate(
        ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.2, 0.9, 0.0)),
        start=4,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        2,
        QualifiedE4PLS3ShellElement(
            2,
            [4, 5, 6],
            "steel",
            thickness=0.02,
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    first = _compression(0.25)
    second = _compression(0.5)

    if source_kind == "callable":
        def source(element_id: int, _element: object) -> dict[str, object]:
            if element_id == 1:
                return first
            first["membrane_compression"][0] = 9.0e99  # type: ignore[index]
            return second
    else:
        class MutatingItems(dict[int, dict[str, object]]):
            def items(self):  # type: ignore[override]
                yield 1, first
                first["membrane_compression"][0] = 9.0e99  # type: ignore[index]
                yield 2, second

        source = MutatingItems({1: first, 2: second})

    normalized, provenance = modal_module._normalize_prestress_states(
        model, source
    )

    assert normalized[1]["membrane_compression"][0] == 2.5e4
    assert normalized[2]["membrane_compression"][0] == 5.0e4
    assert provenance["source_kind"] == (
        "callable_evaluated_once" if source_kind == "callable" else "mapping"
    )


def test_builtin_prestress_map_amortizes_only_internal_observations() -> None:
    model = _model()
    full_contexts: list[str] = []
    trusted_contexts: list[str] = []

    def full_guard(_model: FEModel, *, context: str) -> dict[str, object]:
        full_contexts.append(context)
        return {}

    def lease(_model: FEModel, *, context: str) -> None:
        raise AssertionError(f"outer lease callback unexpectedly used at {context}")

    def trusted_guard(_model: FEModel, *, context: str) -> None:
        trusted_contexts.append(context)

    vars(lease)["_qualified_trusted_require"] = trusted_guard
    state = {
        1: {
            "membrane_compression": [2.5e4, 0.0, 0.0],
            "nested": {"values": (1.0, 2.0, 3.0)},
        }
    }
    normalized, provenance = modal_module._normalize_prestress_states(
        model,
        state,
        _exact_guard=full_guard,
        _qualified_runtime_guard=lease,
    )

    assert normalized == {
        1: {
            "membrane_compression": [2.5e4, 0.0, 0.0],
            "nested": {"values": [1.0, 2.0, 3.0]},
        }
    }
    assert provenance["supplied_element_ids"] == [1]
    assert full_contexts == [
        "qualified reference-prestress normalization preflight",
        "qualified reference-prestress normalization output",
    ]
    assert len(trusted_contexts) >= 20


def test_numpy_and_custom_prestress_inputs_keep_full_observation_guards() -> None:
    model = _model()
    events: list[tuple[str, str]] = []

    def full_guard(_model: FEModel, *, context: str) -> dict[str, object]:
        events.append(("full", context))
        return {}

    def lease(_model: FEModel, *, context: str) -> None:
        del context

    def trusted_guard(_model: FEModel, *, context: str) -> None:
        events.append(("trusted", context))

    vars(lease)["_qualified_trusted_require"] = trusted_guard

    class ObservedMapping(dict[int, object]):
        def items(self):  # type: ignore[override]
            events.append(("callback", "mapping.items"))
            return super().items()

    source = ObservedMapping(
        {1: {"membrane_compression": np.asarray((2.5e4, 0.0, 0.0))}}
    )
    normalized, _provenance = modal_module._normalize_prestress_states(
        model,
        source,
        _exact_guard=full_guard,
        _qualified_runtime_guard=lease,
    )

    assert normalized[1]["membrane_compression"] == [2.5e4, 0.0, 0.0]
    callback_index = events.index(("callback", "mapping.items"))
    assert events[callback_index + 1][0] == "full"
    numpy_full_observations = [
        context
        for kind, context in events
        if kind == "full" and "prestress_states[1].membrane_compression" in context
    ]
    assert len(numpy_full_observations) == 2


def test_builtin_prestress_traversal_ignores_hostile_module_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    callbacks: list[str] = []

    def trap(*_args: object, **_kwargs: object) -> object:
        callbacks.append("primitive")
        raise AssertionError("modal module primitive shadow must not run")

    class TrapObject:
        __getattribute__ = staticmethod(trap)

    class TrapDict:
        get = staticmethod(trap)
        items = staticmethod(trap)

    for name, replacement in {
        "callable": trap,
        "dict": TrapDict,
        "int": trap,
        "isinstance": trap,
        "iter": trap,
        "list": trap,
        "next": trap,
        "object": TrapObject,
        "set": trap,
        "sorted": trap,
        "str": trap,
        "tuple": trap,
        "type": trap,
    }.items():
        monkeypatch.setattr(modal_module, name, replacement, raising=False)

    full_contexts: list[str] = []
    trusted_contexts: list[str] = []

    def full_guard(_model: FEModel, *, context: str) -> dict[str, object]:
        full_contexts.append(context)
        return {}

    def lease(_model: FEModel, *, context: str) -> None:
        del context

    def trusted_guard(_model: FEModel, *, context: str) -> None:
        trusted_contexts.append(context)

    vars(lease)["_qualified_trusted_require"] = trusted_guard
    normalized, _provenance = modal_module._normalize_prestress_states(
        model,
        {1: {"membrane_compression": [2.5e4, 0.0, 0.0]}},
        _exact_guard=full_guard,
        _qualified_runtime_guard=lease,
    )

    assert normalized[1]["membrane_compression"] == [2.5e4, 0.0, 0.0]
    assert len(full_contexts) == 2
    assert trusted_contexts
    assert callbacks == []


@pytest.mark.parametrize("source_kind", ("callable", "mapping"))
def test_prestress_callback_cannot_replace_selected_trusted_guard(
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    model = _model()
    original_calls: list[str] = []
    replacement_calls: list[str] = []
    helper_calls: list[str] = []

    def full_guard(_model: FEModel, *, context: str) -> dict[str, object]:
        del context
        return {}

    def lease(_model: FEModel, *, context: str) -> None:
        del context

    def original_guard(_model: FEModel, *, context: str) -> None:
        original_calls.append(context)

    def replacement_guard(_model: FEModel, *, context: str) -> None:
        replacement_calls.append(context)
        raise AssertionError("callback-selected replacement guard must not run")

    vars(lease)["_qualified_trusted_require"] = original_guard
    snapshot_function = modal_module._guarded_prestress_snapshot
    canonical_id_function = modal_module._canonical_prestress_element_id
    normalize_function = modal_module._normalize_prestress_states

    def forbidden_helper(*_args: object, **_kwargs: object) -> object:
        helper_calls.append("helper")
        raise AssertionError("callback-replaced prestress helper must not run")

    def replace_guard() -> None:
        vars(lease)["_qualified_trusted_require"] = replacement_guard
        snapshot_defaults = dict(snapshot_function.__kwdefaults__ or {})
        snapshot_defaults.update(
            {
                "_dict_items": forbidden_helper,
                "_exact_iter": forbidden_helper,
                "_exact_next": forbidden_helper,
                "_exact_type": forbidden_helper,
                "_snapshotter": forbidden_helper,
            }
        )
        monkeypatch.setattr(
            snapshot_function,
            "__kwdefaults__",
            snapshot_defaults,
        )
        monkeypatch.setattr(
            canonical_id_function,
            "__defaults__",
            (
                (),
                forbidden_helper,
                forbidden_helper,
                (),
                forbidden_helper,
            ),
        )
        normalize_defaults = dict(normalize_function.__kwdefaults__ or {})
        normalize_defaults.update(
            {
                "_canonical_prestress_id": forbidden_helper,
                "_exact_type": forbidden_helper,
                "_snapshotter": forbidden_helper,
            }
        )
        monkeypatch.setattr(
            normalize_function,
            "__kwdefaults__",
            normalize_defaults,
        )
        for name in (
            "_canonical_prestress_element_id",
            "_evaluate_prestress_provider",
            "_guarded_prestress_snapshot",
        ):
            monkeypatch.setattr(modal_module, name, forbidden_helper)
        for name in ("canonical_json_bytes", "canonical_plain_data"):
            monkeypatch.setattr(s3_state_module, name, forbidden_helper)

    if source_kind == "callable":
        def source(_element_id: int, _element: object) -> dict[str, object]:
            replace_guard()
            return {"membrane_compression": [2.5e4, 0.0, 0.0]}
    else:
        class SwitchingMapping(dict[int, object]):
            def items(self):  # type: ignore[override]
                replace_guard()
                return super().items()

        source = SwitchingMapping(
            {1: {"membrane_compression": [2.5e4, 0.0, 0.0]}}
        )

    normalized, _provenance = modal_module._normalize_prestress_states(
        model,
        source,
        _exact_guard=full_guard,
        _qualified_runtime_guard=lease,
    )

    assert normalized[1]["membrane_compression"] == [2.5e4, 0.0, 0.0]
    assert original_calls
    assert replacement_calls == []
    assert helper_calls == []


@pytest.mark.parametrize("attack", ("code", "globals"))
def test_prestress_provider_cannot_mutate_private_helper_authority(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    model = _model()
    expected_state = {"membrane_compression": [2.5e4, 0.0, 0.0]}
    expected_schema = modal_module.PRESTRESS_INPUT_SCHEMA_ID
    expected_hash = hashlib.sha256(
        s3_state_module.canonical_json_bytes(expected_state)
    ).hexdigest().upper()
    normalize_function = modal_module._normalize_prestress_states
    shared_helpers = (
        normalize_function,
        modal_module._guarded_prestress_snapshot,
        modal_module._canonical_prestress_element_id,
        modal_module._evaluate_prestress_provider,
        s3_state_module.canonical_plain_data,
        s3_state_module.canonical_json_bytes,
    )

    def redirected(*_args: object, **_kwargs: object) -> str:
        return "CALLBACK_REDIRECTED_SHARED_HELPER"

    def full_guard(_model: FEModel, *, context: str) -> dict[str, object]:
        del context
        return {}

    def lease(_model: FEModel, *, context: str) -> None:
        del context

    def trusted_guard(_model: FEModel, *, context: str) -> None:
        del context

    vars(lease)["_qualified_trusted_require"] = trusted_guard

    def source(_element_id: int, _element: object) -> dict[str, object]:
        if attack == "code":
            for helper in shared_helpers:
                monkeypatch.setattr(helper, "__code__", redirected.__code__)
        else:
            modal_globals = normalize_function.__globals__
            state_globals = s3_state_module.canonical_plain_data.__globals__
            for name in (
                "Mapping",
                "PRESTRESS_INPUT_SCHEMA_ID",
                "Sequence",
                "TypeError",
                "ValueError",
                "_JSON_ENCODE_BASESTRING",
                "_canonical_prestress_element_id",
                "_evaluate_prestress_provider",
                "_guarded_prestress_snapshot",
                "dict",
                "hashlib",
                "inspect",
                "int",
                "isinstance",
                "iter",
                "list",
                "math",
                "next",
                "np",
                "repr",
                "set",
                "sorted",
                "str",
                "tuple",
                "type",
            ):
                monkeypatch.setitem(modal_globals, name, redirected)
            for name in ("_canonical_value", "json", "math", "np"):
                monkeypatch.setitem(state_globals, name, redirected)
        return expected_state

    normalized, provenance = normalize_function(
        model,
        source,
        _exact_guard=full_guard,
        _qualified_runtime_guard=lease,
    )

    assert normalized[1] == expected_state
    assert provenance["schema_id"] == expected_schema
    assert provenance["state_sha256"] == {"1": expected_hash}


def test_prestress_provider_matrix_assembly_primitive_shadow_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_reference_model()
    lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="prestress provider primitive-shadow test",
    )
    callbacks: list[str] = []

    def trap(*_args: object, **_kwargs: object) -> object:
        callbacks.append("type")
        raise AssertionError("shadowed matrix_assembly.type must not run")

    def source(_element_id: int, _element: object) -> dict[str, object]:
        monkeypatch.setattr(
            matrix_assembly_module,
            "type",
            trap,
            raising=False,
        )
        return {"membrane_compression": [2.5e4, 0.0, 0.0]}

    def exact_guard(
        observed_model: FEModel,
        *,
        context: str,
    ) -> dict[str, object]:
        result = modal_module._EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD(
            observed_model,
            context=context,
        )
        lease(observed_model, context=context)
        return result

    with pytest.raises(AssemblyError, match="incompatible qualified shell authority"):
        modal_module._normalize_prestress_states(
            model,
            source,
            _exact_guard=exact_guard,
            _qualified_runtime_guard=lease,
        )
    assert callbacks == []


def test_mixed_cached_q4_beam_route_exposes_narrow_input_capability() -> None:
    model = _qualified_q4_beam_reference_model()
    # Populate the real warm-Q4 plan before capturing the same route used by
    # the stiffened mixed-model solver.
    assemble_stiffness_matrix(model)
    lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="mixed cached prestress input capability test",
        allow_q4_cached_stiffness=True,
    )
    namespace = vars(lease)

    assert namespace["_qualified_q4_raw_mesh"] is model.mesh
    assert namespace["_qualified_q4_only"] is False
    assert callable(namespace.get("_qualified_trusted_input_require"))
    assert "_qualified_trusted_require" not in namespace


def test_mixed_q4_beam_builtin_prestress_amortization_preserves_evidence() -> None:
    model = _qualified_q4_beam_reference_model()
    lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="mixed prestress normalization equality test",
    )
    assert callable(vars(lease).get("_qualified_trusted_input_require"))
    assert "_qualified_trusted_require" not in vars(lease)
    state = {
        1: {
            "membrane_compression": [2.5e4, 0.0, 0.0],
            "nested": {"values": (1.0, 2.0, 3.0)},
        },
        2: {"axial_compression": 125.0},
    }
    full_contexts: list[str] = []

    def full_guard(observed_model: FEModel, *, context: str) -> dict[str, object]:
        full_contexts.append(context)
        modal_module._EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD(
            observed_model,
            context=context,
        )
        lease(observed_model, context=context)
        return {}

    fast = modal_module._normalize_prestress_states(
        model,
        state,
        _exact_guard=full_guard,
        _qualified_runtime_guard=lease,
    )
    reference = modal_module._normalize_prestress_states(
        model,
        state,
        _exact_guard=lambda _model, *, context: {},
    )

    assert fast == reference
    assert full_contexts == [
        "qualified reference-prestress normalization preflight",
        "qualified reference-prestress normalization output",
    ]


def test_mixed_q4_beam_exact_geometric_state_amortization_is_identical() -> None:
    fast_model = _qualified_q4_beam_reference_model()
    reference_model = _qualified_q4_beam_reference_model()
    states = {
        1: {"membrane_compression": [2.5e4, 0.0, 0.0]},
        2: {"axial_compression": 125.0},
    }
    fast_lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        fast_model,
        context="mixed geometric-state amortization test",
    )
    trusted = vars(fast_lease).get("_qualified_trusted_input_require")
    assert callable(trusted)
    trusted_contexts: list[str] = []

    def counted_trusted(observed_model: FEModel, *, context: str) -> None:
        trusted_contexts.append(context)
        trusted(observed_model, context=context)

    vars(fast_lease)["_qualified_trusted_input_require"] = counted_trusted
    fast_matrix, fast_info = (
        matrix_assembly_module._assemble_geometric_stiffness_matrix_under_lease(
            fast_model,
            states,
            qualified_runtime_guard=fast_lease,
        )
    )

    reference_lease = (
        matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
            reference_model,
            context="mixed geometric-state full-guard reference test",
        )
    )

    def reference_guard(observed_model: FEModel, *, context: str) -> None:
        reference_lease(observed_model, context=context)

    reference_matrix, reference_info = (
        matrix_assembly_module._assemble_geometric_stiffness_matrix_under_lease(
            reference_model,
            states,
            qualified_runtime_guard=reference_guard,
        )
    )

    np.testing.assert_array_equal(
        fast_matrix.toarray(),
        reference_matrix.toarray(),
    )
    assert trusted_contexts
    for key in ("matrix_type", "num_elements", "state_source", "sparsity_signature"):
        assert fast_info[key] == reference_info[key]


@pytest.mark.parametrize("attack", ("getter_lookup", "getter_call"))
def test_exact_dict_state_never_crosses_a_custom_geometric_getter_guard(
    attack: str,
) -> None:
    model = _qualified_q4_reference_model()
    shell = model.mesh.elements[1]

    class HostileGenericElement(Element):
        def __init__(self) -> None:
            super().__init__(2, [2, 5], "steel")
            self.armed = False

        @property
        def num_nodes(self) -> int:
            return 2

        @property
        def dofs_per_node(self) -> int:
            return 6

        def get_node_coordinates(self, mesh: object) -> np.ndarray:
            return np.asarray(
                [mesh.nodes[node_id].coords() for node_id in self.node_ids],
                dtype=float,
            )

        def compute_stiffness_matrix(
            self,
            _mesh: object,
            _material: object,
        ) -> np.ndarray:
            return np.zeros((12, 12), dtype=float)

        def __getattribute__(self, name: str) -> object:
            if (
                name == "compute_geometric_stiffness_matrix"
                and object.__getattribute__(self, "armed")
                and attack == "getter_lookup"
            ):
                thickness = shell.thickness
                shell.thickness = thickness * 2.0
                shell.thickness = thickness
            return super().__getattribute__(name)

        def compute_geometric_stiffness_matrix(
            self,
            _mesh: object,
            _material: object,
            _state: object = None,
        ) -> np.ndarray:
            if attack == "getter_call":
                thickness = shell.thickness
                shell.thickness = thickness * 2.0
                shell.thickness = thickness
            return np.zeros((12, 12), dtype=float)

    model.add_node(5, 2.0, 0.0, 0.0)
    hostile = HostileGenericElement()
    model.add_element(2, hostile)
    hostile.armed = True
    exact_builtin_states = {
        1: {"membrane_compression": [2.5e4, 0.0, 0.0]},
        2: {"axial_compression": 125.0},
    }

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        assemble_geometric_stiffness_matrix(model, exact_builtin_states)


@pytest.mark.parametrize("shadow_scope", ("instance", "class"))
def test_nested_beam_helper_shadow_uses_the_fully_guarded_fallback(
    monkeypatch: pytest.MonkeyPatch,
    shadow_scope: str,
) -> None:
    model = _qualified_q4_beam_reference_model()
    shell = model.mesh.elements[1]
    beam = model.mesh.elements[2]
    callbacks: list[str] = []

    def hostile_axial_state(*_args: object, **_kwargs: object) -> float:
        callbacks.append("axial-state")
        thickness = shell.thickness
        shell.thickness = thickness * 2.0
        shell.thickness = thickness
        return 125.0

    if shadow_scope == "instance":
        monkeypatch.setattr(
            beam,
            "_axial_compression_from_state",
            hostile_axial_state,
        )
    else:
        monkeypatch.setattr(
            BeamElement,
            "_axial_compression_from_state",
            staticmethod(hostile_axial_state),
        )

    with pytest.raises((AssemblyError, ElementCapabilityError)):
        assemble_geometric_stiffness_matrix(
            model,
            {
                1: {"membrane_compression": [2.5e4, 0.0, 0.0]},
                2: {"axial_compression": 125.0},
            },
        )
    assert callbacks == (["axial-state"] if shadow_scope == "instance" else [])


def test_exact_builtin_beam_geometric_guards_scale_linearly() -> None:
    def guard_counts(beam_count: int) -> tuple[int, int]:
        model = _qualified_q4_beam_chain_reference_model(beam_count)
        lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
            model,
            context=f"mixed geometric scaling probe with {beam_count} beams",
        )
        trusted = vars(lease).get("_qualified_trusted_input_require")
        assert callable(trusted)
        full_contexts: list[str] = []
        narrow_contexts: list[str] = []

        def counted_trusted(observed_model: FEModel, *, context: str) -> None:
            narrow_contexts.append(context)
            trusted(observed_model, context=context)

        def counted_full(observed_model: FEModel, *, context: str) -> None:
            full_contexts.append(context)
            lease(observed_model, context=context)

        vars(counted_full)["_qualified_trusted_input_require"] = counted_trusted
        states = {
            1: {"membrane_compression": [2.5e4, 0.0, 0.0]},
            **{
                element_id: {"axial_compression": 125.0}
                for element_id in range(2, beam_count + 2)
            },
        }
        matrix_assembly_module._assemble_geometric_stiffness_matrix_under_lease(
            model,
            states,
            qualified_runtime_guard=counted_full,
        )
        return len(full_contexts), len(narrow_contexts)

    one_full, one_narrow = guard_counts(1)
    sixteen_full, sixteen_narrow = guard_counts(16)

    # Full lifecycle/lease scans are cycle-boundary work, not one scan per
    # built-in beam.  The narrow constant-time checks grow by a fixed amount
    # per beam, proving that the mixed built-in route remains O(N).
    assert one_full == sixteen_full
    assert 2 * 15 <= sixteen_narrow - one_narrow <= 12 * 15


@pytest.mark.parametrize("attack", ("qualified_aba", "builtin_shadow"))
def test_mixed_geometric_custom_mapping_observation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    model = _qualified_q4_beam_reference_model()
    shell = model.mesh.elements[1]
    callbacks: list[str] = []

    def trap(*_args: object, **_kwargs: object) -> object:
        callbacks.append("type")
        raise AssertionError("shadowed matrix_assembly.type must not run")

    class HostileStates(dict[int, object]):
        def __contains__(self, key: object) -> bool:
            if attack == "qualified_aba":
                thickness = shell.thickness
                shell.thickness = thickness * 2.0
                shell.thickness = thickness
            else:
                monkeypatch.setattr(
                    matrix_assembly_module,
                    "type",
                    trap,
                    raising=False,
                )
            return super().__contains__(key)

    states = HostileStates(
        {
            1: {"membrane_compression": [2.5e4, 0.0, 0.0]},
            2: {"axial_compression": 125.0},
        }
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        assemble_geometric_stiffness_matrix(model, states)
    assert callbacks == []


@pytest.mark.parametrize("attack", ("element", "assembly_module"))
def test_builtin_prestress_guard_rejects_operation_local_aba(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    model = _qualified_q4_reference_model()
    element = model.mesh.elements[1]
    lease = matrix_assembly_module._CAPTURE_QUALIFIED_ASSEMBLY_RUNTIME_LEASE(
        model,
        context="prestress normalization ABA test",
    )
    trusted_name = (
        "_qualified_trusted_input_require"
        if "_qualified_trusted_input_require" in vars(lease)
        else "_qualified_trusted_require"
    )
    trusted = vars(lease).get(trusted_name)
    assert callable(trusted)
    original_assembly = matrix_assembly_module._assemble_element_matrix_under_lease
    calls = 0

    def injected_guard(observed_model: FEModel, *, context: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            if attack == "element":
                original_thickness = element.thickness
                element.thickness = original_thickness * 2.0
                element.thickness = original_thickness
            else:
                def replacement(*_args: object, **_kwargs: object) -> object:
                    raise AssertionError("restored assembly replacement must not run")

                monkeypatch.setattr(
                    matrix_assembly_module,
                    "_assemble_element_matrix_under_lease",
                    replacement,
                )
                monkeypatch.setattr(
                    matrix_assembly_module,
                    "_assemble_element_matrix_under_lease",
                    original_assembly,
                )
        trusted(observed_model, context=context)

    vars(lease)[trusted_name] = injected_guard

    def exact_guard(
        observed_model: FEModel,
        *,
        context: str,
    ) -> dict[str, object]:
        result = modal_module._EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD(
            observed_model,
            context=context,
        )
        lease(observed_model, context=context)
        return result

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        modal_module._normalize_prestress_states(
            model,
            {1: {"membrane_compression": [2.5e4, 0.0, 0.0]}},
            _exact_guard=exact_guard,
            _qualified_runtime_guard=lease,
        )


def test_omitted_prestress_ids_are_explicitly_unstressed() -> None:
    model = _model()
    baseline = solve_free_vibration(model, num_modes=2)
    explicit = solve_free_vibration(model, num_modes=2, prestress_states={})

    np.testing.assert_array_equal(explicit.frequencies_hz, baseline.frequencies_hz)
    provenance = explicit.assembly_info["prestress_input"]
    assert provenance["supplied_element_ids"] == []
    assert provenance["explicitly_unstressed_element_ids"] == [1]


def test_prestressed_modal_uses_transient_factorization_not_session_cache() -> None:
    model = FEModel("qualified-q4-prestress-transient-factor")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02
        ),
    )
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    state = {1: {"membrane_compression": [1.0e4, 0.0, 0.0]}}

    with AnalysisSession(model) as session:
        before = session.factorization_cache.diagnostics()
        result = solve_free_vibration(
            model,
            num_modes=2,
            dense_size_limit=1,
            shift=1.0,
            session=session,
            prestress_states=state,
        )
        after = session.factorization_cache.diagnostics()

    assert result.solver_status == "ok"
    assert before["entries"] == after["entries"] == 0
    assert result.diagnostics["factorization_cache"]["name"] == (
        "modal_state_dependent_transient"
    )


def test_prestressed_modal_rejects_persistent_cache_and_api_spoof_pre_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    with pytest.raises(ValueError, match="persistent factorization_cache"):
        solve_free_vibration(
            model,
            prestress_states={1: _compression()},
            factorization_cache=FactorizationCache("forbidden-prestress-cache"),
        )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed S3 stiffness mechanics reached")

    monkeypatch.setattr(
        QualifiedE4PLS3ShellElement,
        "_compute_stiffness_components",
        forbidden,
    )
    monkeypatch.setattr(modal_module, "assemble_stiffness_matrix", forbidden)
    with pytest.raises(ElementCapabilityError, match="_compute_stiffness_components"):
        solve_free_vibration(model, prestress_states={1: _compression()})


def test_reference_buckling_prestress_map_is_closed_world_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def forbidden() -> None:
        raise AssertionError("buckling prestress guard reached mechanics")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="not in the model"):
        solve_eigenvalue_buckling(
            model,
            {999: _compression()},
            reference_elastic_only=True,
        )
    with pytest.raises(ValueError, match="duplicate or ambiguous"):
        solve_eigenvalue_buckling(
            model,
            {1: _compression(), "1": _compression()},
            reference_elastic_only=True,
        )


def test_qualified_q4_reference_buckling_binds_vectorized_samples_and_cache() -> None:
    model = FEModel("qualified-q4-reference-buckling-authority")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02
        ),
    )
    model.add_boundary_condition(FixedSupport("left", [1, 4]))
    states = {1: {"membrane_compression": [1.0e4, 1.0e4, 0.0]}}

    with pytest.raises(ValueError, match="persistent factorization_cache"):
        solve_eigenvalue_buckling(
            model,
            states,
            reference_elastic_only=True,
            factorization_cache=FactorizationCache("forbidden-qualified-buckling"),
        )

    with AnalysisSession(model) as session:
        before = session.factorization_cache.diagnostics()
        result = solve_eigenvalue_buckling(
            model,
            states,
            num_modes=2,
            dense_size_limit=1,
            shift_load_factor=1000.0,
            allow_dense_fallback=True,
            allow_free_mechanisms=True,
            reference_elastic_only=True,
            session=session,
        )
        after = session.factorization_cache.diagnostics()

    assert result.solver_status == "ok"
    assert before["entries"] == after["entries"] == 0
    assert result.diagnostics["factorization_cache"]["name"] == (
        "buckling_qualified_prestress_transient"
    )
    provenance = result.assembly_info["reference_prestress_input"]
    assert provenance["schema_id"] == PRESTRESS_INPUT_SCHEMA_ID


def test_q4_vectorized_prestress_sample_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FEModel("qualified-q4-reference-buckling-spoof")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02
        ),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed Q4 vectorized prestress sampling reached")

    monkeypatch.setattr(
        QualifiedE4PLShellElement,
        "_membrane_compression_samples",
        classmethod(forbidden),
    )
    with pytest.raises(ElementCapabilityError, match="_membrane_compression_samples"):
        solve_eigenvalue_buckling(
            model,
            {1: {"membrane_compression": [1.0e4, 0.0, 0.0]}},
            reference_elastic_only=True,
        )


@pytest.mark.parametrize(
    "helper_name",
    (
        "_membrane_compression_samples",
        "_bending_compression_samples",
        "_stress_second_moment_samples",
        "_membrane_compression_from_state",
        "_resultant_samples",
        "_local_dof_transform",
    ),
)
@pytest.mark.parametrize("route", ("modal", "buckling"))
def test_s3_inherited_prestress_helper_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    route: str,
) -> None:
    model = _model()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed inherited S3 helper reached mechanics")

    monkeypatch.setattr(ShellElement, helper_name, forbidden)
    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ElementCapabilityError, match=helper_name):
        if route == "modal":
            solve_free_vibration(
                model,
                num_modes=1,
                prestress_states={1: _compression()},
            )
        else:
            solve_eigenvalue_buckling(
                model,
                {1: _compression()},
                num_modes=1,
                reference_elastic_only=True,
            )


@pytest.mark.parametrize(
    "helper_name",
    (
        "compute_stiffness_components",
        "compute_geometric_stiffness_components",
        "_constitutive",
        "_director_generalized_transform",
    ),
)
@pytest.mark.parametrize("route", ("modal", "buckling"))
def test_s3_transitive_prestress_api_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    route: str,
) -> None:
    model = _model()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed transitive S3 API reached mechanics")

    monkeypatch.setattr(QualifiedE4PLS3ShellElement, helper_name, forbidden)
    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ElementCapabilityError, match=helper_name):
        if route == "modal":
            solve_free_vibration(
                model,
                num_modes=1,
                prestress_states={1: _compression()},
            )
        else:
            solve_eigenvalue_buckling(
                model,
                {1: _compression()},
                num_modes=1,
                reference_elastic_only=True,
            )


@pytest.mark.parametrize(
    ("owner", "helper_name"),
    (
        (QualifiedE4PLShellElement, "_constitutive_and_drill_stiffness"),
        (QualifiedE4PLShellElement, "_bind_qualified_component_guard"),
        (QualifiedE4PLShellElement, "_warped_generalized_drilling_correction"),
        (QualifiedE4PLShellElement, "_generalized_section_in_frame"),
        (QualifiedE4PLShellElement, "_physical_director_context"),
        (ShellElement, "_material_angle"),
        (ShellElement, "_build_drilling_b_matrix"),
        (ShellElement, "compute_jacobian"),
        (ShellElement, "_fallback_edge_direction"),
        (ShellElement, "_normalize"),
    ),
)
def test_q4_transitive_prestress_api_spoof_rejects_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    owner: type[object],
    helper_name: str,
) -> None:
    model = FEModel("qualified-q4-transitive-prestress-spoof")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1,
            [1, 2, 3, 4],
            "steel",
            thickness=0.02,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spoofed transitive Q4 API reached mechanics")

    monkeypatch.setattr(owner, helper_name, forbidden)
    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ElementCapabilityError, match=helper_name):
        solve_eigenvalue_buckling(
            model,
            {1: {"membrane_compression": [1.0e4, 0.0, 0.0]}},
            num_modes=1,
            reference_elastic_only=True,
        )


def test_committed_current_state_modal_matches_force_difference_tangent() -> None:
    model, load = _axial_nonlinear_model()
    static = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        num_layers=3,
    )
    assert static.status == "completed"
    frozen_state = canonical_json_bytes(static.element_states[1])

    result = solve_free_vibration(
        model,
        num_modes=1,
        current_state_displacements=static.displacements,
        current_state_element_states=static.element_states,
        current_state_num_layers=3,
    )

    model.apply_boundary_conditions()
    mass, _mass_info = assemble_mass_matrix(model)
    material, _stiffness_info = assemble_stiffness_matrix(model)
    _reduced, _, transform, _, _, _ = build_constraint_transformation(
        material,
        np.zeros(material.shape[0], dtype=float),
        model,
    )
    assert transform.shape[1] == 1
    direction = np.asarray(transform[:, 0].toarray(), dtype=float).reshape(-1)
    step = 1.0e-7
    plus = _committed_trial_force(
        model,
        np.asarray(static.displacements),
        np.asarray(static.displacements) + step * direction,
        static.element_states,
    )
    minus = _committed_trial_force(
        model,
        np.asarray(static.displacements),
        np.asarray(static.displacements) - step * direction,
        static.element_states,
    )
    tangent_action = (plus - minus) / (2.0 * step)
    stiffness_scalar = float(direction @ tangent_action)
    mass_scalar = float(direction @ (mass @ direction))
    oracle = stiffness_scalar / mass_scalar

    assert result.solver_status == "ok"
    assert canonical_json_bytes(static.element_states[1]) == frozen_state
    assert result.assembly_info["current_state_modal_policy_id"] == (
        CURRENT_STATE_MODAL_POLICY_ID
    )
    assert result.assembly_info["current_state_tangent"][
        "relative_symmetry_error"
    ] <= 512.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        result.modes[0].eigenvalue,
        oracle,
        rtol=2.0e-7,
        atol=0.0,
    )


def test_mixed_q4_s3_current_state_modal_matches_virgin_reference() -> None:
    model = FEModel("mixed-qualified-current-state-modal")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        QualifiedE4PLShellElement(
            1, [1, 2, 3, 4], "steel", thickness=0.02
        ),
    )
    s3 = QualifiedE4PLS3ShellElement(
        2,
        [2, 5, 3],
        "steel",
        thickness=0.02,
        reference_normal=np.asarray((0.0, 0.0, 1.0)),
    )
    model.add_element(2, s3)
    model.add_boundary_condition(FixedSupport("left-edge", [1, 4]))
    zero = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    q4 = model.mesh.elements[1]
    states = {
        1: q4.seal_committed_current_tangent_state(
            model.mesh,
            model.get_material("steel"),
            zero[q4.get_dof_mapping(model.mesh)],
            q4.init_nonlinear_state(3),
            3,
        ),
        2: s3.init_model_bound_nonlinear_state(
            model.mesh, model.get_material("steel"), 3
        )
    }

    reference = solve_free_vibration(model, num_modes=4)
    current = solve_free_vibration(
        model,
        num_modes=4,
        current_state_displacements=zero,
        current_state_element_states=states,
        current_state_num_layers=3,
    )

    assert reference.solver_status == current.solver_status == "ok"
    for result in (reference, current):
        assert result.diagnostics["candidate_eigenvalue_policy_id"] == (
            DESCRIPTOR_RAYLEIGH_REFINEMENT_POLICY_ID
        )
        assert result.diagnostics["candidate_eigenvalue_disposition"] == (
            "FULL_SYSTEM_RAYLEIGH_REFINED"
        )
        assert result.diagnostics["candidate_rayleigh_refined_count"] > 0
        assert result.diagnostics["candidate_condensed_preserved_count"] == 0
    np.testing.assert_allclose(
        current.frequencies_hz,
        reference.frequencies_hz,
        rtol=8.0e-10,
        atol=1.0e-8,
    )
    assert current.diagnostics["declared_algebraic_element_ids"] == [2]


def test_generalized_section_current_state_modal_uses_native_tangent() -> None:
    model = FEModel("generalized-s3-current-state-modal")
    model.add_material("carrier", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    section = GeneralizedShellSection(
        A=np.asarray(((2.3e9, 0.7e9, 0.0), (0.7e9, 2.1e9, 0.0), (0.0, 0.0, 0.8e9))),
        B=np.asarray(((1.2e5, -0.3e5, 0.0), (-0.3e5, 0.9e5, 0.0), (0.0, 0.0, 0.2e5))),
        D=np.asarray(((1.9e5, 0.5e5, 0.0), (0.5e5, 1.7e5, 0.0), (0.0, 0.0, 0.6e5))),
        As=np.asarray(((7.0e8, 0.0), (0.0, 6.0e8))),
    )
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "carrier",
            thickness=0.01,
            shell_section=section,
            material_direction=np.asarray((1.0, 0.0, 0.0)),
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    model.add_boundary_condition(FixedSupport("node-1", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-2-axial-only",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("node-3", [3]))
    load = LoadCase("pull")
    load.add_nodal_load(2, [25.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    static = solve_static_nonlinear(
        model, load, num_steps=2, num_layers=3
    )
    assert static.status == "completed"

    result = solve_free_vibration(
        model,
        num_modes=1,
        current_state_displacements=static.displacements,
        current_state_element_states=static.element_states,
        current_state_num_layers=3,
    )

    assert result.solver_status == "ok"
    assert result.modes[0].eigenvalue > 0.0
    assert result.assembly_info["current_state_tangent"]["state_digests"] == {
        "1": static.element_states[1]["state_integrity_sha256"]
    }


def test_buckling_matches_independent_pencil_and_inverse_load_scaling() -> None:
    model = _model()
    state = _compression()
    material, geometric, _mass = _reduced_operators(model, state)
    oracle = _finite_descriptor_values(material, geometric)
    oracle = oracle[oracle > 0.0]

    result = solve_eigenvalue_buckling(
        model,
        {1: state},
        num_modes=4,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        reference_elastic_only=True,
    )
    doubled = solve_eigenvalue_buckling(
        model,
        {1: _compression(2.0)},
        num_modes=4,
        dense_size_limit=1000,
        allow_free_mechanisms=True,
        reference_elastic_only=True,
    )

    factors = np.asarray([mode.load_factor for mode in result.modes])
    doubled_factors = np.asarray([mode.load_factor for mode in doubled.modes])
    assert result.solver_status == "ok"
    np.testing.assert_allclose(factors, oracle[: len(factors)], rtol=3.0e-12)
    np.testing.assert_allclose(doubled_factors, 0.5 * factors, rtol=3.0e-12)
    assert result.diagnostics["max_residual_norm"] < 1.0e-10


def test_capability_matrix_records_both_eigen_workflows_as_native() -> None:
    matrix = _model().mesh.elements[1].capability_matrix()

    assert matrix["reference_elastic_prestressed_modal"] == "PARITY_REPLACED"
    assert matrix["reference_elastic_buckling"] == "PARITY_REPLACED"
    assert matrix["current_state_modal"] == "PARITY_REPLACED"
    assert matrix["mixed_current_state_modal"] == "PARITY_REPLACED"
    assert matrix["current_state_buckling_s3"] == "PARITY_REPLACED"
    assert matrix["mixed_current_state_buckling"] == "PARITY_REPLACED"
    assert matrix["buckling"] == (
        "EXPLICIT_REFERENCE_ELASTIC_OR_CURRENT_STATE_AUTHORITY_REQUIRED"
    )


def test_eigen_authority_profiles_reject_ambiguous_inputs_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def forbidden() -> None:
        raise AssertionError("eigen input validation reached model evaluation")

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden)
    with pytest.raises(ValueError, match="requires both committed"):
        solve_free_vibration(
            model,
            current_state_displacements=np.zeros(18),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_free_vibration(
            model,
            prestress_states={1: _compression()},
            current_state_displacements=np.zeros(18),
            current_state_element_states={},
        )
    malformed = _compression()
    malformed.pop("bubble_linearization_policy")
    with pytest.raises(ValueError, match="bubble_linearization_policy"):
        solve_eigenvalue_buckling(
            model,
            {1: malformed},
            reference_elastic_only=True,
        )
