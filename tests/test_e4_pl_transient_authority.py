from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.dynamics as dynamics
from anyfesolver import (
    AnalysisSession,
    BoundaryCondition,
    ElementActivity,
    FEModel,
    FixedSupport,
    PressurePatch,
    TransientConfig,
)
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.element_capabilities import ElementCapabilityError
from anyfesolver.elements import ShellElement
from anyfesolver.control import CancellationToken


def _qualified_q4_model(
    *, constrained: bool = False, warped: bool = False
) -> FEModel:
    model = FEModel("qualified-q4-transient-authority")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
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
        ),
    )
    if constrained:
        model.add_boundary_condition(FixedSupport("fixed", [1, 2, 4]))
        model.add_boundary_condition(
            BoundaryCondition(
                "single-uz",
                [3],
                {
                    "ux": 0.0,
                    "uy": 0.0,
                    "rx": 0.0,
                    "ry": 0.0,
                    "rz": 0.0,
                },
            )
        )
    return model


def test_transient_formulation_descriptor_is_rejected_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_model()
    expected = vars(QualifiedE4PLShellElement)["formulation_id"]

    class SplitFormulationDescriptor:
        reads = 0

        def __get__(self, instance: object, _owner: object) -> str:
            self.reads += 1
            return expected if instance is None else str(expected)

    descriptor = SplitFormulationDescriptor()
    monkeypatch.setattr(
        QualifiedE4PLShellElement, "formulation_id", descriptor
    )
    reached = _install_premechanics_tripwires(monkeypatch, model)

    with pytest.raises(ElementCapabilityError, match="FORMULATION_ID_CLASS_MISMATCH"):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
        )
    assert descriptor.reads == 0
    assert reached == []


def test_warped_q4_shape_spoof_rejects_before_transient_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_model(warped=True)
    reached = _install_premechanics_tripwires(monkeypatch, model)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        reached.append("shape")
        raise AssertionError("spoofed warped-Q4 shape mechanics reached")

    monkeypatch.setattr(
        ShellElement, "_compute_4node_shape_functions", forbidden
    )

    with pytest.raises(ElementCapabilityError, match="_compute_4node_shape_functions"):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
        )
    assert reached == []


def _qualified_s3_model() -> FEModel:
    model = FEModel("qualified-s3-transient-authority")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
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
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    return model


def _install_premechanics_tripwires(
    monkeypatch: pytest.MonkeyPatch,
    model: FEModel,
) -> list[str]:
    reached: list[str] = []

    def tripwire(name: str):
        def fail(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"transient reached {name} before authority rejection")

        return fail

    monkeypatch.setattr(model, "apply_boundary_conditions", tripwire("boundary"))
    monkeypatch.setattr(dynamics, "assemble_load_vector", tripwire("load"))
    monkeypatch.setattr(
        dynamics, "assemble_stiffness_matrix", tripwire("stiffness")
    )
    monkeypatch.setattr(dynamics, "assemble_mass_matrix", tripwire("mass"))
    return reached


@pytest.mark.parametrize("disposition", ("foreign", "closed"))
def test_transient_rejects_foreign_or_closed_session_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    model = _qualified_q4_model()
    owner = _qualified_q4_model()
    session = AnalysisSession(model if disposition == "closed" else owner)
    if disposition == "closed":
        session.close()
    reached = _install_premechanics_tripwires(monkeypatch, model)

    expected = RuntimeError if disposition == "closed" else ValueError
    with pytest.raises(expected):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
            session=session,
        )
    assert reached == []


@pytest.mark.parametrize(
    "mutation",
    (
        "callable_mass_spoof",
        "class_mass_identity",
        "q4_implementation_identity",
        "q4_descriptor_spoof",
        "s3_descriptor_spoof",
    ),
)
def test_transient_rejects_qualified_identity_mutation_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    model = (
        _qualified_s3_model()
        if mutation == "s3_descriptor_spoof"
        else _qualified_q4_model()
    )
    element = model.mesh.elements[1]
    if mutation == "callable_mass_spoof":
        element.__dict__["compute_mass_matrix"] = lambda *_args: np.eye(24)
    elif mutation == "class_mass_identity":
        monkeypatch.setattr(
            QualifiedE4PLShellElement,
            "compute_mass_matrix",
            lambda *_args: np.eye(24),
        )
    elif mutation == "q4_implementation_identity":
        element.__dict__["implementation_id"] = "FOREIGN_Q4_IMPLEMENTATION"
    elif mutation == "q4_descriptor_spoof":
        element.__dict__["dynamic_algebraic_nullity"] = 3
    else:
        element.__dict__["dynamic_algebraic_policy"] = "FOREIGN_DESCRIPTOR"
    reached = _install_premechanics_tripwires(monkeypatch, model)

    with pytest.raises(ElementCapabilityError):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
        )
    assert reached == []


@pytest.mark.parametrize("activity", (0.5, 0.0))
def test_transient_rejects_softened_or_hard_deleted_model_activity_before_mechanics(
    monkeypatch: pytest.MonkeyPatch,
    activity: float,
) -> None:
    model = _qualified_q4_model()
    manager = ElementActivity([1])
    model.set_element_activity(manager)
    if activity == 0.0:
        manager.hard_delete([1], reason="transient-authority-test")
    else:
        manager.set_activity(
            [1], activity, reason="transient-authority-test"
        )
    reached = _install_premechanics_tripwires(monkeypatch, model)

    with pytest.raises(ElementCapabilityError, match="softened or hard-deleted"):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
        )
    assert reached == []


def test_warm_session_cannot_bypass_later_qualified_callable_spoof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_model(constrained=True)
    session = AnalysisSession(model)
    first = dynamics.solve_transient_newmark(
        model,
        TransientConfig(dt=1.0e-3, t_end=0.0),
        session=session,
    )
    assert first.status == "completed"

    model.mesh.elements[1].__dict__["compute_mass_matrix"] = (
        lambda *_args: np.eye(24)
    )
    reached = _install_premechanics_tripwires(monkeypatch, model)
    with pytest.raises(ElementCapabilityError):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
            session=session,
        )
    assert reached == []


def test_qualified_transient_preserves_nonzero_initial_conditions_and_authority() -> None:
    model = _qualified_q4_model(constrained=True)
    model.set_element_activity(ElementActivity([1]))
    initial = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    free_uz = model.mesh.get_node(3).dofs[2]
    initial[free_uz] = 1.0e-8

    result = dynamics.solve_transient_newmark(
        model,
        TransientConfig(
            dt=1.0e-3,
            t_end=0.0,
            initial_displacement=initial,
        ),
    )

    assert result.status == "completed"
    assert result.displacements[0, free_uz] == initial[free_uz]
    authority = result.diagnostics["qualified_reference_transient_authority"]
    assert authority["active"] is True
    assert authority["qualified_element_ids"] == [1]
    assert authority["activity_disposition"] == "ACTIVE_REFERENCE_ONLY"


def test_pressure_selector_rechecks_authority_before_load_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_model(constrained=True)
    reached: list[str] = []

    def forbidden_shape(*_args: object, **_kwargs: object) -> object:
        reached.append("shape")
        raise AssertionError("pressure mechanics reached changed shape operation")

    def selector(_element_id: int, _element: object, _centroid: np.ndarray) -> bool:
        monkeypatch.setattr(
            ShellElement,
            "compute_shape_functions",
            forbidden_shape,
        )
        return True

    patch = PressurePatch(
        "selector-authority",
        1.0,
        selector=selector,
    )
    with pytest.raises(ElementCapabilityError, match="compute_shape_functions"):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
            pressure_patches=(patch,),
        )
    assert reached == []


def test_pressure_callback_rechecks_before_numpy_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_model(constrained=True)
    original_all = np.all
    reached: list[str] = []

    def changed_all(*args: object, **kwargs: object) -> object:
        reached.append("np.all")
        return original_all(*args, **kwargs)

    def pressure_at(_time: float) -> float:
        monkeypatch.setattr(np, "all", changed_all)
        return 1.0

    patch = PressurePatch(
        "pressure-callback-authority",
        pressure_at,
        element_ids=(1,),
    )
    with pytest.raises(ElementCapabilityError, match="numpy.all"):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=0.0),
            pressure_patches=(patch,),
        )
    assert reached == []


def test_transient_step_cancellation_rechecks_authority_before_step_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _qualified_q4_model(constrained=True)
    reached: list[str] = []

    def forbidden_numeric(*_args: object, **_kwargs: object) -> np.ndarray:
        reached.append("numeric")
        raise AssertionError("changed numerical operation was invoked")

    class ObservedToken(CancellationToken):
        def raise_if_cancelled(self, stage: str = "") -> None:
            if stage == "transient.step:1":
                reached.append("cancellation")
                monkeypatch.setattr(np, "asarray", forbidden_numeric)

    with pytest.raises(ElementCapabilityError, match="numpy.asarray"):
        dynamics.solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=1.0e-3),
            cancellation_token=ObservedToken(),
        )
    assert reached == ["cancellation"]
