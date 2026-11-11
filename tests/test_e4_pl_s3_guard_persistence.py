from __future__ import annotations

import copy
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import anyfesolver.arc_length as arc_length_module
import anyfesolver.imperfections as imperfections_module
import anyfesolver.nonlinear_static as nonlinear_static_module
import anyfesolver.recovery as recovery_module
import anyfesolver.results as results_module
from anyfesolver import (
    ElementCapabilityError,
    FEModel,
    GeneralizedShellSection,
    ImperfectionField,
    LegacyS3MigrationWarning,
    LegacyShellElement,
    PatchRecoveryConfig,
    QualifiedE4PLS3ShellElement,
    QualifiedE4PLShellElement,
    RecoveryConfig,
    create_shell_element,
    shell_element_from_dict,
)
from anyfesolver.assembly import compute_stresses as assembly_compute_stresses
from anyfesolver.element_capabilities import require_model_element_capabilities


OWNER_NORMAL = np.asarray((0.0, 0.0, 1.0))
S3_ID = 7
Q4_ID = 20


class _DeclaredGuardGapS3(QualifiedE4PLS3ShellElement):
    """Negative-profile element proving the workflow guard stays fail closed."""

    @property
    def capability_gaps(self) -> frozenset[str]:
        # The exact qualified S3 profile has no required gaps.  Do not call
        # its now-protected property on this deliberately invalid descendant;
        # this fixture exists only to expose the declared negative profile to
        # consumer-side capability guards.
        return frozenset({
            "initial_fields",
            "static_restart_history",
            "arc_length_restart_history",
        })

    @property
    def capability_restrictions(self) -> dict[str, str]:
        return {}


def _mixed_model(
    *,
    generalized_s3: bool = False,
    declared_guard_gaps: bool = False,
) -> FEModel:
    model = FEModel("qualified-s3-guard-boundary")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    coordinates = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.2, 0.9, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    )
    for node_id, coordinate in enumerate(coordinates, start=1):
        model.add_node(node_id, *coordinate)
    section = (
        GeneralizedShellSection(
            A=np.asarray(
                ((120.0, 12.0, 0.0), (12.0, 100.0, 0.0), (0.0, 0.0, 44.0))
            ),
            B=np.zeros((3, 3)),
            D=np.asarray(
                ((14.0, 1.0, 0.0), (1.0, 11.0, 0.0), (0.0, 0.0, 5.0))
            ),
            As=np.asarray(((28.0, 0.0), (0.0, 24.0))),
        )
        if generalized_s3
        else None
    )
    s3 = QualifiedE4PLS3ShellElement(
        S3_ID,
        [1, 2, 3],
        "steel",
        thickness=0.1,
        reference_normal=OWNER_NORMAL,
        shell_section=section,
        material_direction=(1.0, 0.0, 0.0) if generalized_s3 else None,
    )
    if declared_guard_gaps:
        # Production construction correctly rejects descendants of the exact
        # qualified class.  Inject the deliberately invalid runtime object
        # only after exact construction so this negative workflow fixture
        # continues to test the consumer-side fail-closed guard.
        object.__setattr__(s3, "__class__", _DeclaredGuardGapS3)
    model.add_element(
        S3_ID,
        s3,
    )
    q4 = create_shell_element(
        Q4_ID,
        [4, 5, 6, 7],
        "steel",
        thickness=0.1,
    )
    assert type(q4) is QualifiedE4PLShellElement
    model.add_element(Q4_ID, q4)
    return model


def _zero_displacements(model: FEModel) -> np.ndarray:
    return np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)


def test_capability_guard_scopes_exact_element_ids_and_orders_failures() -> None:
    model = SimpleNamespace(
        mesh=SimpleNamespace(
            elements={
                11: SimpleNamespace(capability_gaps=("zeta", "alpha")),
                3: SimpleNamespace(capability_gaps=("beta", "alpha")),
            }
        )
    )

    require_model_element_capabilities(
        model,
        ("alpha", "beta", "zeta"),
        context="empty-selection",
        element_ids=(),
    )
    require_model_element_capabilities(
        model,
        "alpha",
        context="unknown-selection",
        element_ids=(99,),
    )

    selected = (value for value in (11, 3, 11, 99))
    with pytest.raises(ElementCapabilityError) as caught:
        require_model_element_capabilities(
            model,
            ("zeta", "beta", "alpha"),
            context="ordered-selection",
            element_ids=selected,
        )
    message = str(caught.value)
    assert message.startswith("ordered-selection is unavailable")
    assert "3 (alpha, beta)" in message
    assert "11 (alpha, zeta)" in message
    assert message.index("3 (alpha, beta)") < message.index("11 (alpha, zeta)")

    with pytest.raises(ElementCapabilityError) as caught_single:
        require_model_element_capabilities(
            model,
            "zeta",
            context="single-selection",
            element_ids=(11,),
        )
    assert "11 (zeta)" in str(caught_single.value)
    assert "3 (" not in str(caught_single.value)


def test_static_nonlinear_restart_gap_rejects_before_any_model_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(generalized_s3=True, declared_guard_gaps=True)
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"nonlinear guard evaluated {name}")

        return fail

    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden("boundary"))
    monkeypatch.setattr(
        nonlinear_static_module,
        "_has_follower_pressure",
        forbidden("follower"),
    )
    monkeypatch.setattr(
        nonlinear_static_module,
        "_ensure_nonlinear_acceleration",
        forbidden("acceleration"),
    )
    monkeypatch.setattr(
        nonlinear_static_module,
        "assemble_stiffness_matrix",
        forbidden("stiffness"),
    )
    monkeypatch.setattr(
        nonlinear_static_module,
        "_prepare_initial_states",
        forbidden("initial-state"),
    )

    with pytest.raises(
        ElementCapabilityError,
        match="exact qualified component/lifecycle APIs.*FORMULATION_ID_CLASS_MISMATCH",
    ):
        nonlinear_static_module.solve_static_nonlinear(
            model,
            num_steps=1,
            initial_element_states={S3_ID: {}},
        )
    assert calls == []


def test_static_nonlinear_state_and_field_guards_are_id_scoped() -> None:
    model = _mixed_model()
    generalized_model = _mixed_model(generalized_s3=True)
    guarded_generalized_model = _mixed_model(
        generalized_s3=True,
        declared_guard_gaps=True,
    )

    with pytest.raises(
        ElementCapabilityError,
        match="exact qualified component/lifecycle APIs.*FORMULATION_ID_CLASS_MISMATCH",
    ):
        nonlinear_static_module.solve_static_nonlinear(
            guarded_generalized_model,
            num_steps=1,
            initial_fields={S3_ID: object()},
        )
    require_model_element_capabilities(
        generalized_model,
        "initial_fields",
        context="qualified-generalized-s3-field-selection",
        element_ids=(S3_ID,),
    )
    require_model_element_capabilities(
        model,
        ("static_restart_history", "arc_length_restart_history"),
        context="qualified-s3-checkpoint-selection",
        element_ids=(S3_ID,),
    )
    with pytest.raises(
        ElementCapabilityError,
        match=(
            "restart_history=STATIC_AND_ARC_LENGTH_CHECKPOINTS_ONLY"
        ),
    ):
        require_model_element_capabilities(
            model,
            "restart_history",
            context="generic-restart-selection",
            element_ids=(S3_ID,),
        )
    assert model.mesh.elements[S3_ID].capability_matrix()[
        "restart_history"
    ] == "STATIC_AND_ARC_LENGTH_CHECKPOINTS_ONLY"

    require_model_element_capabilities(
        generalized_model,
        "initial_fields",
        context="q4-field-selection",
        element_ids=(Q4_ID,),
    )
    require_model_element_capabilities(
        generalized_model,
        "restart_history",
        context="q4-state-selection",
        element_ids=(Q4_ID,),
    )


def test_arc_length_restart_gap_rejects_before_load_copy_boundary_or_stiffness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(generalized_s3=True, declared_guard_gaps=True)
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"arc-length guard evaluated {name}")

        return fail

    monkeypatch.setattr(
        arc_length_module,
        "_has_follower_pressure",
        forbidden("follower"),
    )
    monkeypatch.setattr(
        arc_length_module,
        "_copy_model_with_imperfection",
        forbidden("copy"),
    )
    monkeypatch.setattr(model, "apply_boundary_conditions", forbidden("boundary"))
    monkeypatch.setattr(
        arc_length_module,
        "assemble_stiffness_matrix",
        forbidden("stiffness"),
    )

    with pytest.raises(
        ElementCapabilityError,
        match="exact qualified component/lifecycle APIs.*FORMULATION_ID_CLASS_MISMATCH",
    ):
        arc_length_module.solve_static_arc_length(
            model,
            None,
            initial_element_states={S3_ID: {}},
        )
    assert calls == []
    with pytest.raises(
        ElementCapabilityError,
        match="exact qualified component/lifecycle APIs.*FORMULATION_ID_CLASS_MISMATCH",
    ):
        arc_length_module.solve_static_arc_length(
            model,
            None,
            imperfection=ImperfectionField({3: (0.0, 0.0, 0.1)}),
        )
    assert calls == []


def test_apply_imperfection_rejects_before_copy_conversion_or_geometry_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(generalized_s3=True, declared_guard_gaps=True)
    coordinates = {
        node_id: node.coords().copy() for node_id, node in model.mesh.nodes.items()
    }
    revisions = model.revision_signature()
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"imperfection guard evaluated {name}")

        return fail

    monkeypatch.setattr(
        imperfections_module.copy,
        "deepcopy",
        forbidden("deepcopy"),
    )
    monkeypatch.setattr(
        imperfections_module,
        "to_imperfection_field",
        forbidden("conversion"),
    )

    with pytest.raises(
        ElementCapabilityError,
        match="apply_imperfection.*initial_fields",
    ):
        imperfections_module.apply_imperfection(
            model,
            ImperfectionField({3: (0.0, 0.0, 0.1)}),
            copy_model=True,
        )
    assert calls == []
    assert model.revision_signature() == revisions
    for node_id, expected in coordinates.items():
        np.testing.assert_array_equal(model.mesh.nodes[node_id].coords(), expected)


def test_patch_recovery_rejects_before_displacement_or_state_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Layered S3 patch recovery is now native.  A pre-integrated generalized
    # section still has no physical top/bottom stress field and therefore
    # retains the patch capability guard.
    model = _mixed_model(generalized_s3=True)
    selected = RecoveryConfig(element_ids=[S3_ID])
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"recovery guard evaluated {name}")

        return fail

    monkeypatch.setattr(
        recovery_module,
        "_recovery_displacements",
        forbidden("displacements"),
    )
    monkeypatch.setattr(
        recovery_module,
        "_recover_generalized_committed_state",
        forbidden("generalized-state"),
    )
    with pytest.raises(
        ElementCapabilityError,
        match="recover_stress_result.*patch_recovery",
    ):
        recovery_module.recover_stress_result(
            model,
            object(),
            selected,
            patch_config=PatchRecoveryConfig(),
        )
    assert calls == []


def test_committed_state_recovery_is_selection_scoped_and_model_bound() -> None:
    model = _mixed_model()
    element = model.mesh.elements[S3_ID]
    state = element.init_model_bound_nonlinear_state(
        model.mesh,
        model.get_material(element.material_name),
        3,
    )

    native = recovery_module.recover_stress_result(
        model,
        _zero_displacements(model),
        RecoveryConfig(element_ids=[S3_ID]),
        element_states={S3_ID: state},
        return_global=True,
    )
    assert tuple(native.element_stresses) == (S3_ID,)
    assert native.provenance.per_element_source[S3_ID] == (
        "committed_qualified_s3_native_state"
    )

    recovered = recovery_module.recover_stress_result(
        model,
        _zero_displacements(model),
        RecoveryConfig(element_ids=[Q4_ID]),
        element_states={S3_ID: {}},
        return_global=True,
    )
    assert tuple(recovered.element_stresses) == (Q4_ID,)
    assert recovered.committed_element_states == {}


@pytest.mark.parametrize(
    "mode",
    ("explicit_von_karman", "explicit_corotational", "inferred_von_karman", "inferred_corotational"),
)
def test_nonlinear_context_recovery_reaches_displacement_after_native_closure(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    model = _mixed_model()
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("displacements")
        raise RuntimeError("native-recovery-displacement-sentinel")

    monkeypatch.setattr(recovery_module, "_recovery_displacements", forbidden)
    resolved = "corotational" if mode.endswith("corotational") else "von_karman"
    kwargs = {
        "nonlinear_result": SimpleNamespace(
            info={"kinematics": resolved},
            element_states={},
            displacements=object(),
        )
    } if mode.startswith("inferred") else {"kinematics": resolved}
    with pytest.raises(RuntimeError, match="native-recovery-displacement-sentinel"):
        recovery_module.recover_stress_result(
            model,
            object(),
            RecoveryConfig(element_ids=[S3_ID]),
            return_global=True,
            **kwargs,
        )
    assert calls == ["displacements"]


def test_nonlinear_recovery_accepts_closed_native_geometry_capability() -> None:
    model = _mixed_model()
    nonlinear_result = SimpleNamespace(
        info={"kinematics": "von_karman"},
        element_states={},
        displacements=_zero_displacements(model),
    )
    recovered = recovery_module.recover_stress_result(
        model,
        None,
        RecoveryConfig(element_ids=[S3_ID]),
        nonlinear_result=nonlinear_result,
        return_global=True,
    )
    assert tuple(recovered.element_stresses) == (S3_ID,)
    assert recovered.provenance.analysis_context["kinematics"] == "von_karman"


def test_disabled_global_recovery_preserves_noop_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled recovery evaluated mechanics")

    monkeypatch.setattr(recovery_module, "_compute_one_element_stress", forbidden)
    monkeypatch.setattr(recovery_module, "_element_recovery_context", forbidden)
    monkeypatch.setattr(
        recovery_module,
        "_recover_generalized_committed_state",
        forbidden,
    )
    monkeypatch.setattr(recovery_module, "_recovery_displacements", forbidden)
    monkeypatch.setattr(
        model.mesh.elements[S3_ID],
        "compute_stresses",
        forbidden,
    )
    stresses, report = recovery_module.recover_element_stresses_with_report(
        model,
        object(),
        RecoveryConfig(element_ids=[S3_ID], include_stresses=False),
        return_global=True,
    )
    assert stresses == {}
    assert report.backend == "disabled"
    assert report.item_count == 0

    for kwargs in (
        {"kinematics": "corotational"},
        {
            "nonlinear_result": SimpleNamespace(
                info={"kinematics": "corotational"},
                element_states={},
                displacements=object(),
            )
        },
    ):
        result = recovery_module.recover_stress_result(
            model,
            object(),
            RecoveryConfig(element_ids=[S3_ID], include_stresses=False),
            return_global=True,
            **kwargs,
        )
        assert result.element_stresses == {}
        assert result.provenance.mode == "disabled"
        assert result.provenance.analysis_context["kinematics"] == "corotational"
        assert result.execution_report is not None
        assert result.execution_report.backend == "disabled"


def test_patch_and_nodal_recovery_reject_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _mixed_model(generalized_s3=True)
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"patch guard evaluated {name}")

        return fail

    monkeypatch.setattr(
        recovery_module,
        "_prepare_shell_patch_elements",
        forbidden("patch-preparation"),
    )
    with pytest.raises(
        ElementCapabilityError,
        match="recover_shell_patch_stresses.*patch_recovery",
    ):
        recovery_module.recover_shell_patch_stresses(
            model,
            {S3_ID: {}},
            element_ids=[S3_ID],
        )

    operator = results_module._gauss_to_node_extrapolation(
        model.mesh.elements[S3_ID]
    )
    assert operator is not None
    assert operator.shape == (3, 7)
    monkeypatch.setattr(
        results_module,
        "_gauss_to_node_extrapolation",
        forbidden("extrapolation"),
    )
    with pytest.raises(
        ElementCapabilityError,
        match="recover_nodal_stresses.*patch_recovery",
    ):
        results_module.recover_nodal_stresses(
            model,
            object(),
            element_ids=[S3_ID],
        )
    assert calls == []


def test_assembly_global_s3_recovery_executes() -> None:
    model = _mixed_model()
    recovered = assembly_compute_stresses(
        model,
        _zero_displacements(model),
        return_global=True,
        element_ids=[S3_ID],
    )
    assert tuple(recovered) == (S3_ID,)
    assert recovered[S3_ID]["recovery_scope"] == (
        "qualified_s3_local_and_global_physical"
    )


def test_nonfinite_s3_recovery_rejects_before_mechanics(
) -> None:
    model = _mixed_model()
    element = model.mesh.elements[S3_ID]
    derived_names = (
        "_qualified_components",
        "_qualified_cache_key",
        "_qualified_component_guard",
        "_stiffness_matrix",
        "_internal_forces",
    )
    assert all(getattr(element, name) is None for name in derived_names)
    plan_revision = element._qualified_plan_state_revision
    direct_revision = int(element._qualified_direct_state_token[0])
    displacement = np.zeros(18, dtype=float)
    displacement[0] = np.nan
    with pytest.raises(ValueError, match="recovery requires finite displacements"):
        element.compute_stresses(
            model.mesh,
            displacement,
            model.get_material(element.material_name),
            return_global=True,
        )
    assert all(getattr(element, name) is None for name in derived_names)
    assert element._qualified_plan_state_revision == plan_revision
    assert int(element._qualified_direct_state_token[0]) == direct_revision


def test_public_recovery_propagates_qualified_s3_fail_closed_errors() -> None:
    model = _mixed_model()
    displacement = _zero_displacements(model)
    s3_node = model.mesh.nodes[model.mesh.elements[S3_ID].node_ids[0]]
    displacement[s3_node.dofs[0]] = np.nan

    with pytest.raises(ValueError, match="recovery requires finite displacements"):
        assembly_compute_stresses(
            model,
            displacement,
            return_global=True,
            element_ids=[S3_ID],
        )
    with pytest.raises(ValueError, match="recovery requires finite displacements"):
        recovery_module.recover_element_stresses_with_report(
            model,
            displacement,
            RecoveryConfig(element_ids=[S3_ID]),
            return_global=True,
        )

    with pytest.raises(ValueError, match="complete in-range DOF mapping"):
        assembly_compute_stresses(
            model,
            np.zeros(1),
            return_global=True,
            element_ids=[S3_ID],
        )
    with pytest.raises(ValueError, match="complete in-range DOF mapping"):
        recovery_module.recover_element_stresses_with_report(
            model,
            np.zeros(1),
            RecoveryConfig(element_ids=[S3_ID]),
            return_global=True,
        )


def test_recovery_selection_does_not_block_unselected_s3_and_local_s3_works() -> None:
    model = _mixed_model()
    displacements = _zero_displacements(model)
    q4_only = RecoveryConfig(element_ids=[Q4_ID])

    element_stresses, _report = recovery_module.recover_element_stresses_with_report(
        model,
        displacements,
        q4_only,
        return_global=True,
    )
    assert tuple(element_stresses) == (Q4_ID,)

    stress_result = recovery_module.recover_stress_result(
        model,
        displacements,
        q4_only,
        return_global=True,
    )
    assert tuple(stress_result.element_stresses) == (Q4_ID,)

    assembly_stresses = assembly_compute_stresses(
        model,
        displacements,
        return_global=True,
        element_ids=[Q4_ID],
    )
    assert tuple(assembly_stresses) == (Q4_ID,)

    nodal = results_module.recover_nodal_stresses(
        model,
        displacements,
        element_ids=[Q4_ID],
    )
    assert tuple(nodal["element_nodal"]) == (Q4_ID,)

    local_s3, _local_report = recovery_module.recover_element_stresses_with_report(
        model,
        displacements,
        RecoveryConfig(element_ids=[S3_ID]),
        return_global=False,
    )
    assert tuple(local_s3) == (S3_ID,)
    assert local_s3[S3_ID]["recovery_scope"] == "qualified_s3_local_physical_only"

    global_s3, _global_report = recovery_module.recover_element_stresses_with_report(
        model,
        displacements,
        RecoveryConfig(element_ids=[S3_ID]),
        return_global=True,
    )
    assert tuple(global_s3) == (S3_ID,)
    assert global_s3[S3_ID]["recovery_scope"] == (
        "qualified_s3_local_and_global_physical"
    )


def test_historical_tri3_warns_once_stays_legacy_and_preserves_input() -> None:
    historical = {
        "type": "ShellElement",
        "element_id": 91,
        "node_ids": [1, 2, 3],
        "material_name": "steel",
        "thickness": 0.07,
        "drilling_stabilization": 1.0e-3,
        "hourglass_stabilization": 1.0e-8,
        "reduced_integration": False,
    }
    before = copy.deepcopy(historical)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rebuilt = shell_element_from_dict(historical)

    migration = [
        item for item in caught if issubclass(item.category, LegacyS3MigrationWarning)
    ]
    assert len(migration) == 1
    message = str(migration[0].message)
    assert "LEGACY_S3_MISSING_FORMULATION_ID" in message
    assert "loaded as legacy-s3" in message
    assert "qualified-s3 was not inferred" in message
    assert "qualified-s3 hot restart" in message
    assert type(rebuilt) is LegacyShellElement
    assert historical == before


def test_explicit_legacy_and_qualified_records_do_not_emit_migration_warning() -> None:
    qualified = QualifiedE4PLS3ShellElement(
        92,
        [1, 2, 3],
        "steel",
        thickness=0.07,
        reference_normal=OWNER_NORMAL,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explicit_legacy = create_shell_element(
            93,
            [1, 2, 3],
            "steel",
            formulation="legacy-s3",
        )
        rebuilt = shell_element_from_dict(qualified.to_dict())
    assert type(explicit_legacy) is LegacyShellElement
    assert type(rebuilt) is QualifiedE4PLS3ShellElement
    assert not any(
        issubclass(item.category, LegacyS3MigrationWarning) for item in caught
    )

    historical_q4 = {
        "type": "ShellElement",
        "element_id": 94,
        "node_ids": [1, 2, 3, 4],
        "material_name": "steel",
        "thickness": 0.07,
    }
    missing_identity = qualified.to_dict()
    missing_identity.pop("formulation_id")
    with warnings.catch_warnings(record=True) as deserialization_warnings:
        warnings.simplefilter("always")
        historical_q4_element = shell_element_from_dict(historical_q4)
        with pytest.raises(ValueError, match="missing formulation_id"):
            shell_element_from_dict(missing_identity)
    assert type(historical_q4_element) is LegacyShellElement
    assert not any(
        issubclass(item.category, LegacyS3MigrationWarning)
        for item in deserialization_warnings
    )


@pytest.mark.parametrize(
    "retained_marker",
    (
        "director_polarity",
        "director_polarity_policy_id",
        "director_reversal_transform_id",
        "quadrature_authority_id",
        "reference_normal",
    ),
)
def test_new_qualified_s3_fingerprint_markers_cannot_downgrade_to_legacy(
    retained_marker: str,
) -> None:
    qualified = QualifiedE4PLS3ShellElement(
        95,
        [1, 2, 3],
        "steel",
        reference_normal=OWNER_NORMAL,
        director_polarity=-1,
    ).to_dict()
    stripped = {
        "type": "ShellElement",
        "element_id": 95,
        "node_ids": [1, 2, 3],
        "material_name": "steel",
        retained_marker: copy.deepcopy(qualified[retained_marker]),
    }
    before = copy.deepcopy(stripped)
    with pytest.raises(ValueError, match="retains qualified S3 fingerprint"):
        shell_element_from_dict(stripped)
    assert stripped == before
