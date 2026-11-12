from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Callable, Sequence

import numpy as np
import pytest

import anyfesolver.dynamics as dynamics
import anyfesolver.e4_pl_element as q4_runtime
import anyfesolver.e4_pl_s3_element as s3_runtime
import anyfesolver.nonlinear_static as nonlinear_static_module
import test_e4_pl_s3_current_tangent_buckling as current_fixture
import test_e4_pl_s3_generalized_nonlinear as nonlinear_fixture
import test_e4_pl_workflow_parity as workflow_fixture
from anyfesolver import (
    LoadCase,
    FractureConfig,
    ImperfectionField,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    PressurePatch,
    RecoveryConfig,
    ResourceConfig,
    TransientConfig,
    solve_eigenvalue_buckling,
    solve_free_vibration,
    solve_transient_newmark,
)
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.matrix_assembly import AssemblyError
from anyfesolver.nonlinear_static import DisplacementControl, solve_static_nonlinear


class _MutatingSequence(Sequence[object]):
    def __init__(self, values: Sequence[object], module: ModuleType, name: str) -> None:
        self._values = tuple(values)
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.mutations = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> object:
        if index == 0 and self.armed:
            self.armed = False
            self.mutations += 1
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        return self._values[index]


class _MutatingMapping(Mapping[object, object]):
    def __init__(self, values: Mapping[object, object], module: ModuleType, name: str) -> None:
        self._values = dict(values)
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.mutations = 0

    def __iter__(self):
        if self.armed:
            self.armed = False
            self.mutations += 1
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: object) -> object:
        return self._values[key]


class _MutatingTruthMapping(Mapping[object, object]):
    def __init__(self, values: Mapping[object, object], module: ModuleType, name: str) -> None:
        self._values = dict(values)
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.mutations = 0

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: object) -> object:
        return self._values[key]

    def __bool__(self) -> bool:
        if self.armed:
            self.armed = False
            self.mutations += 1
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        return bool(self._values)


class _MutatingArrayProtocol:
    def __init__(self, values: Sequence[float], module: ModuleType, name: str) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.mutations = 0

    def __array__(
        self,
        dtype: object = None,
        copy: object = None,
    ) -> np.ndarray:
        if self.armed:
            self.armed = False
            self.mutations += 1
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        made = np.asarray(self._values, dtype=dtype)
        if copy is True:
            made = made.copy()
        return made


class _MutatingFloat:
    def __init__(self, value: float, module: ModuleType, name: str) -> None:
        self._value = float(value)
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.mutations = 0

    def __float__(self) -> float:
        if self.armed:
            self.armed = False
            self.mutations += 1
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        return self._value


class _MutatingAttributeProxy:
    def __init__(
        self,
        value: object,
        attribute: str,
        module: ModuleType,
        name: str,
    ) -> None:
        self._value = value
        self._attribute = attribute
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.mutations = 0

    def __getattr__(self, name: str) -> object:
        if name == self._attribute and self.armed:
            self.armed = False
            self.mutations += 1
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        return getattr(self._value, name)


class _MutatingStatusCallback:
    def __init__(self, module: ModuleType, name: str) -> None:
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.calls = 0
        self.truth_tests = 0

    def __bool__(self) -> bool:
        self.truth_tests += 1
        raise AssertionError("status callbacks must not be truth-tested")

    def __call__(self, _message: str) -> None:
        self.calls += 1
        if self.armed:
            self.armed = False
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)


class _MutatingImperfectionConverter:
    def __init__(self, module: ModuleType, name: str) -> None:
        self._module = module
        self._name = name
        self._original = vars(module)[name]
        self.armed = True
        self.calls = 0

    def to_field(self, _model: object) -> ImperfectionField:
        self.calls += 1
        if self.armed:
            self.armed = False
            setattr(self._module, self._name, lambda *_args, **_kwargs: None)
            setattr(self._module, self._name, self._original)
        return ImperfectionField({1: (0.0, 0.0, 0.0)})


def _two_q4_transient_model() -> object:
    model = workflow_fixture.generate_simple_panel_mesh(
        2.0,
        1.0,
        0.05,
        num_divisions_x=2,
        num_divisions_y=1,
    )
    for element_id, legacy in tuple(model.mesh.elements.items()):
        model.mesh.elements[element_id] = (
            workflow_fixture.QualifiedE4PLShellElement(
                element_id,
                list(legacy.node_ids),
                legacy.material_name,
                thickness=legacy.thickness,
            )
        )
    node_ids = tuple(sorted(model.mesh.nodes))
    model.add_boundary_condition(
        workflow_fixture.FixedSupport("lease-fixed", node_ids[:-1])
    )
    model.add_boundary_condition(
        workflow_fixture.BoundaryCondition(
            "lease-free-transverse",
            [node_ids[-1]],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _install_mutate_restore_boundary_callback(
    model: object,
    module: ModuleType,
    name: str,
) -> tuple[list[int], Callable[[], None]]:
    """Mutate and restore tracked authority inside one model callback."""

    original_apply = getattr(model, "apply_boundary_conditions")
    original_authority = vars(module)[name]
    calls: list[int] = []

    def callback() -> None:
        calls.append(1)
        setattr(module, name, lambda *_args, **_kwargs: None)
        try:
            pass
        finally:
            setattr(module, name, original_authority)
        original_apply()

    setattr(model, "apply_boundary_conditions", callback)

    def restore() -> None:
        delattr(model, "apply_boundary_conditions")

    return calls, restore


@pytest.mark.parametrize("route", ("modal", "buckling", "transient"))
def test_q4_linear_orchestration_rejects_transient_authority_mutation_and_recovers(
    route: str,
) -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    load = LoadCase("lease-load")
    load.add_nodal_load(3, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def run() -> object:
        if route == "modal":
            return solve_free_vibration(model, num_modes=1)
        if route == "buckling":
            return solve_eigenvalue_buckling(
                model,
                {1: {"membrane_compression_x": 1.0, "membrane_compression_y": 1.0}},
                num_modes=1,
            )
        return solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=1.0e-3),
            base_load_case=load,
        )

    calls, restore = _install_mutate_restore_boundary_callback(
        model,
        q4_runtime,
        "equation7_frame",
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        run()
    assert calls == [1]
    restore()

    result = run()
    if route == "modal":
        assert result.solver_status == "ok"
    elif route == "buckling":
        assert result.solver_status == "ok"
    else:
        assert result.status == "completed"


def test_modal_scalar_config_is_owned_before_model_observation() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    observed = _MutatingFloat(
        1.0e-9,
        q4_runtime,
        "equation7_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_free_vibration(
            model,
            num_modes=1,
            eigen_tolerance=observed,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_free_vibration(
        model,
        num_modes=1,
        eigen_tolerance=observed,
    )
    baseline_model, _element = workflow_fixture._candidate_model(constrained=True)
    baseline = solve_free_vibration(
        baseline_model,
        num_modes=1,
        eigen_tolerance=1.0e-9,
    )
    assert clean.solver_status == baseline.solver_status
    np.testing.assert_array_equal(clean.frequencies_hz, baseline.frequencies_hz)
    assert boundary_calls == [1]


def test_resource_config_is_owned_before_thread_policy_observation() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    base = ResourceConfig(solver_threads=1, metadata={"lane": "focused"})
    observed = _MutatingAttributeProxy(
        base,
        "solver_threads",
        q4_runtime,
        "equation7_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_free_vibration(
            model,
            num_modes=1,
            resource_config=observed,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_free_vibration(
        model,
        num_modes=1,
        resource_config=observed,
    )
    baseline_model, _element = workflow_fixture._candidate_model(constrained=True)
    baseline = solve_free_vibration(
        baseline_model,
        num_modes=1,
        resource_config=base,
    )
    assert clean.solver_status == baseline.solver_status
    np.testing.assert_array_equal(clean.frequencies_hz, baseline.frequencies_hz)
    assert boundary_calls == [1]


def test_buckling_range_config_is_owned_before_model_observation() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    observed = _MutatingSequence(
        (None, None),
        q4_runtime,
        "equation7_frame",
    )
    states = {
        1: {
            "membrane_compression_x": 1.0,
            "membrane_compression_y": 1.0,
        }
    }
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_eigenvalue_buckling(
            model,
            states,
            num_modes=1,
            load_factor_range=observed,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        load_factor_range=observed,
    )
    baseline_model, _element = workflow_fixture._candidate_model(constrained=True)
    baseline = solve_eigenvalue_buckling(
        baseline_model,
        states,
        num_modes=1,
        load_factor_range=(None, None),
    )
    assert clean.solver_status == baseline.solver_status
    assert [mode.load_factor for mode in clean.modes] == [
        mode.load_factor for mode in baseline.modes
    ]
    assert boundary_calls == [1]


def test_transient_config_is_owned_before_model_observation() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    base = TransientConfig(dt=1.0e-3, t_end=1.0e-3)
    observed = _MutatingAttributeProxy(
        base,
        "dt",
        q4_runtime,
        "equation7_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_transient_newmark(model, observed)
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_transient_newmark(model, observed)
    baseline_model, _element = workflow_fixture._candidate_model(constrained=True)
    baseline = solve_transient_newmark(baseline_model, base)
    assert clean.status == baseline.status == "completed"
    np.testing.assert_array_equal(clean.times, baseline.times)
    np.testing.assert_array_equal(clean.displacements, baseline.displacements)
    assert boundary_calls == [1]


def test_arc_control_is_owned_before_model_observation() -> None:
    model, load = nonlinear_fixture._constrained_model()
    base = ArcLengthControl(
        initial_load_increment=0.05,
        minimum_load_increment=0.001,
        maximum_load_increment=0.10,
        maximum_absolute_load_factor=0.05,
        max_steps=1,
    )
    observed = _MutatingAttributeProxy(
        base,
        "initial_load_increment",
        s3_runtime,
        "triangle_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_arc_length(
            model,
            load,
            control=observed,
            num_layers=5,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_arc_length(
        model,
        load,
        control=observed,
        num_layers=5,
    )
    baseline_model, baseline_load = nonlinear_fixture._constrained_model()
    baseline = solve_static_arc_length(
        baseline_model,
        baseline_load,
        control=base,
        num_layers=5,
    )
    assert clean.status == baseline.status
    assert clean.load_factor == baseline.load_factor
    np.testing.assert_array_equal(clean.displacements, baseline.displacements)
    assert boundary_calls == [1]


def test_pressure_selector_uses_the_active_generation_before_load_mechanics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    original_load_assembler = dynamics.assemble_load_vector
    original_authority = q4_runtime.equation7_frame
    selected: list[int] = []
    reached: list[str] = []

    def selector(element_id: int, _element: object, _centroid: object) -> bool:
        first_observation = not selected
        selected.append(int(element_id))
        if first_observation:
            setattr(q4_runtime, "equation7_frame", lambda *_args, **_kwargs: None)
            setattr(q4_runtime, "equation7_frame", original_authority)
        return True

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        reached.append("load")
        raise AssertionError("pressure load mechanics ran after a stale generation")

    patch = PressurePatch("lease-selector", 1.0, selector=selector)
    monkeypatch.setattr(dynamics, "assemble_load_vector", forbidden_load)
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        dynamics.assemble_pressure_patch_load_vector(model, patch)
    assert selected == [1]
    assert reached == []

    monkeypatch.setattr(dynamics, "assemble_load_vector", original_load_assembler)
    vector, info = dynamics.assemble_pressure_patch_load_vector(model, patch)
    assert vector.shape == (model.mesh.dof_manager.total_dofs,)
    assert info["selected_element_ids"] == [1]
    assert selected == [1, 1]


def test_pressure_selector_stops_before_the_next_element_after_mutation() -> None:
    model = _two_q4_transient_model()
    original_authority = q4_runtime.equation7_frame
    selected: list[int] = []
    armed = [True]

    def selector(element_id: int, _element: object, _centroid: object) -> bool:
        selected.append(int(element_id))
        if armed[0]:
            armed[0] = False
            setattr(q4_runtime, "equation7_frame", lambda *_args, **_kwargs: None)
            setattr(q4_runtime, "equation7_frame", original_authority)
        return True

    patch = PressurePatch("lease-two-element-selector", 1.0, selector=selector)
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        dynamics.assemble_pressure_patch_load_vector(model, patch)
    assert selected == [min(model.mesh.elements)]

    _vector, info = dynamics.assemble_pressure_patch_load_vector(model, patch)
    assert info["selected_element_ids"] == sorted(model.mesh.elements)
    assert selected == [
        min(model.mesh.elements),
        *sorted(model.mesh.elements),
    ]


def test_pressure_patch_collection_is_owned_before_selection_mechanics() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    selected: list[int] = []

    def selector(element_id: int, _element: object, _centroid: object) -> bool:
        selected.append(int(element_id))
        return True

    patch = PressurePatch("lease-patch-sequence", 1.0, selector=selector)
    patches = _MutatingSequence(
        (patch,),
        q4_runtime,
        "equation7_frame",
    )
    config = TransientConfig(dt=1.0e-3, t_end=1.0e-3)

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_transient_newmark(model, config, pressure_patches=patches)
    assert patches.mutations == 1
    assert selected == []

    clean = solve_transient_newmark(model, config, pressure_patches=patches)
    assert clean.status == "completed"
    assert selected == [1]


def test_pressure_time_callbacks_stop_at_the_first_stale_generation() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    original_authority = q4_runtime.equation7_frame
    observed_times: list[float] = []
    armed = [True]

    def pressure_at(time: float) -> float:
        observed_times.append(float(time))
        if armed[0]:
            armed[0] = False
            setattr(q4_runtime, "equation7_frame", lambda *_args, **_kwargs: None)
            setattr(q4_runtime, "equation7_frame", original_authority)
        return 1.0

    patch = PressurePatch(
        "lease-pressure-time",
        pressure_at,
        element_ids=(1,),
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=1.0e-3),
            pressure_patches=(patch,),
        )
    assert observed_times == [0.0]

    result = solve_transient_newmark(
        model,
        TransientConfig(dt=1.0e-3, t_end=1.0e-3),
        pressure_patches=(patch,),
    )
    assert result.status == "completed"
    assert observed_times == [0.0, 0.0, 1.0e-3]


def test_pressure_time_sequence_observation_keeps_the_original_generation() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    original_authority = q4_runtime.equation7_frame

    class MutatingPressureHistory(Sequence[tuple[float, float]]):
        def __init__(self) -> None:
            self.armed = True
            self.mutations = 0
            self.rows = ((0.0, 1.0), (1.0e-3, 1.0))

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[float, float]:
            if index == 0 and self.armed:
                self.armed = False
                self.mutations += 1
                setattr(
                    q4_runtime,
                    "equation7_frame",
                    lambda *_args, **_kwargs: None,
                )
                setattr(q4_runtime, "equation7_frame", original_authority)
            return self.rows[index]

    history = MutatingPressureHistory()
    patch = PressurePatch(
        "lease-pressure-table",
        history,
        element_ids=(1,),
    )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_transient_newmark(
            model,
            TransientConfig(dt=1.0e-3, t_end=1.0e-3),
            pressure_patches=(patch,),
        )
    assert history.mutations == 1

    clean = solve_transient_newmark(
        model,
        TransientConfig(dt=1.0e-3, t_end=1.0e-3),
        pressure_patches=(patch,),
    )
    assert clean.status == "completed"


@pytest.mark.parametrize(
    "route",
    (
        "output_nodes",
        "output_elements",
        "recovery_nodes",
        "recovery_elements",
        "recovery_components",
    ),
)
def test_transient_output_and_recovery_sequences_are_owned_under_the_lease(
    route: str,
) -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    values: tuple[object, ...] = (
        (3,)
        if route in {"output_nodes", "recovery_nodes"}
        else (("von_mises",) if route == "recovery_components" else (1,))
    )
    observed = _MutatingSequence(
        values,
        q4_runtime,
        "equation7_frame",
    )
    config_options: dict[str, object] = {}
    if route == "output_nodes":
        config_options["output_nodes"] = observed
    elif route == "output_elements":
        config_options["output_elements"] = observed
    else:
        recovery_options: dict[str, object] = {
            "history_mode": "selected",
            "include_stresses": False,
        }
        recovery_options[
            {
                "recovery_nodes": "node_ids",
                "recovery_elements": "element_ids",
                "recovery_components": "components",
            }[route]
        ] = observed
        config_options["recovery"] = RecoveryConfig(**recovery_options)
    config = TransientConfig(
        dt=1.0e-3,
        t_end=1.0e-3,
        **config_options,
    )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_transient_newmark(model, config)
    assert observed.mutations == 1

    clean = solve_transient_newmark(model, config)
    assert clean.status == "completed"


def test_callback_exception_cannot_publish_a_transiently_mutated_modal_result() -> None:
    model, _element = workflow_fixture._candidate_model(constrained=True)
    original_authority = q4_runtime.equation7_frame
    callbacks: list[str] = []

    def progress(_event: object) -> None:
        callbacks.append("modal")
        setattr(q4_runtime, "equation7_frame", lambda *_args, **_kwargs: None)
        setattr(q4_runtime, "equation7_frame", original_authority)
        raise RuntimeError("caller callback failed")

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_free_vibration(model, num_modes=1, progress_callback=progress)
    assert callbacks == ["modal"]

    clean = solve_free_vibration(model, num_modes=1)
    assert clean.solver_status == "ok"


@pytest.mark.parametrize("route", ("nonlinear", "arc_length"))
def test_s3_nonlinear_orchestration_rejects_transient_authority_mutation_and_recovers(
    route: str,
) -> None:
    model, load = nonlinear_fixture._constrained_model()

    def run() -> object:
        if route == "nonlinear":
            return solve_static_nonlinear(
                model,
                load,
                num_steps=1,
                num_layers=5,
            )
        return solve_static_arc_length(
            model,
            load,
            control=ArcLengthControl(
                initial_load_increment=0.05,
                minimum_load_increment=0.001,
                maximum_load_increment=0.10,
                maximum_absolute_load_factor=0.05,
                max_steps=1,
            ),
            num_layers=5,
        )

    calls, restore = _install_mutate_restore_boundary_callback(
        model,
        s3_runtime,
        "triangle_frame",
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        run()
    assert calls == [1]
    restore()

    result = run()
    if route == "nonlinear":
        assert result.status == "completed"
    else:
        assert result.status in {
            "load_factor_limit_reached",
            "maximum_steps_reached",
        }


def test_nonlinear_status_callback_is_not_truth_tested_and_keeps_lease() -> None:
    model, load = nonlinear_fixture._constrained_model()
    callback = _MutatingStatusCallback(s3_runtime, "triangle_frame")

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load,
            num_steps=1,
            num_layers=5,
            status_callback=callback,
        )
    assert callback.truth_tests == 0
    assert callback.calls == 1

    clean = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=5,
        status_callback=callback,
    )
    assert clean.status == "completed"
    assert callback.truth_tests == 0
    assert callback.calls > 1


def test_exact_qualified_newton_loop_uses_trusted_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load = nonlinear_fixture._constrained_model()
    original = nonlinear_static_module._EXACT_QUALIFIED_LIFECYCLE_GUARD
    contexts: list[str] = []

    def counted_guard(observed_model: object, *, context: str) -> object:
        contexts.append(context)
        return original(observed_model, context=context)

    monkeypatch.setattr(
        nonlinear_static_module,
        "_EXACT_QUALIFIED_LIFECYCLE_GUARD",
        counted_guard,
    )
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=2,
        num_layers=5,
    )
    assert result.status == "completed"
    assert "solve_static_nonlinear preflight" in contexts
    assert "nonlinear static constraint postcheck" in contexts
    assert "nonlinear static force tangent assembly" not in contexts
    assert "nonlinear static force external load" not in contexts
    assert "nonlinear static force iteration cancellation" not in contexts
    assert "nonlinear static step external load" not in contexts


def test_nonlinear_load_program_is_owned_before_model_observation() -> None:
    model, load = nonlinear_fixture._constrained_model()
    base = NonlinearLoadProgram(
        (NonlinearLoadStage("primary", load, 1.0),)
    )
    observed = _MutatingAttributeProxy(
        base,
        "stages",
        s3_runtime,
        "triangle_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load_program=observed,
            num_steps=1,
            num_layers=5,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_nonlinear(
        model,
        load_program=observed,
        num_steps=1,
        num_layers=5,
    )
    assert clean.status == "completed"
    assert boundary_calls == [1]


def test_displacement_control_weights_are_owned_before_model_observation() -> None:
    model, load = nonlinear_fixture._constrained_model()
    observed = _MutatingMapping(
        {(2, "ux"): 1.0},
        s3_runtime,
        "triangle_frame",
    )
    control = DisplacementControl(
        target_displacement=1.0e-4,
        weighted_dofs=observed,
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load,
            control="displacement",
            displacement_control=control,
            num_steps=1,
            num_layers=5,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_nonlinear(
        model,
        load,
        control="displacement",
        displacement_control=control,
        num_steps=1,
        num_layers=5,
    )
    assert clean.status == "completed"
    assert boundary_calls == [1]


def test_fracture_config_is_owned_before_model_observation() -> None:
    model, load = nonlinear_fixture._constrained_model()
    config = FractureConfig(threshold=1.0)
    observed = _MutatingFloat(
        1.0,
        s3_runtime,
        "triangle_frame",
    )
    object.__setattr__(config, "threshold", observed)
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load,
            fracture_config=config,
            num_steps=1,
            num_layers=5,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_nonlinear(
        model,
        load,
        fracture_config=config,
        num_steps=1,
        num_layers=5,
    )
    assert clean.status == "completed"
    assert boundary_calls == [1]


@pytest.mark.parametrize("route", ("nonlinear", "arc_length"))
def test_imperfection_offsets_are_owned_before_model_observation(
    route: str,
) -> None:
    model, load = nonlinear_fixture._constrained_model()
    observed = _MutatingMapping(
        {1: (0.0, 0.0, 0.0)},
        s3_runtime,
        "triangle_frame",
    )
    imperfection = ImperfectionField(observed)
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions

    def run() -> object:
        if route == "nonlinear":
            return solve_static_nonlinear(
                model,
                load,
                imperfection=imperfection,
                num_steps=1,
                num_layers=5,
            )
        return solve_static_arc_length(
            model,
            load,
            control=ArcLengthControl(
                initial_load_increment=0.05,
                minimum_load_increment=0.001,
                maximum_load_increment=0.10,
                maximum_absolute_load_factor=0.05,
                max_steps=1,
            ),
            imperfection=imperfection,
            num_layers=5,
        )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        run()
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = run()
    if route == "nonlinear":
        assert clean.status == "completed"
    else:
        assert clean.status in {
            "load_factor_limit_reached",
            "maximum_steps_reached",
        }
    assert boundary_calls == [1]


def test_imperfection_converter_callback_keeps_original_lease() -> None:
    model, load = nonlinear_fixture._constrained_model()
    imperfection = _MutatingImperfectionConverter(
        s3_runtime,
        "triangle_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load,
            imperfection=imperfection,
            num_steps=1,
            num_layers=5,
        )
    assert imperfection.calls == 1
    assert boundary_calls == []

    clean = solve_static_nonlinear(
        model,
        load,
        imperfection=imperfection,
        num_steps=1,
        num_layers=5,
    )
    assert clean.status == "completed"
    assert imperfection.calls == 2
    assert boundary_calls == [1]


def test_accelerated_displacement_control_retains_the_outer_generation_lease() -> None:
    model, load = nonlinear_fixture._constrained_model()
    original_authority = s3_runtime.triangle_frame
    callbacks: list[str] = []
    armed = [True]

    def progress(_event: object) -> None:
        callbacks.append("displacement")
        if armed[0]:
            armed[0] = False
            setattr(s3_runtime, "triangle_frame", lambda *_args, **_kwargs: None)
            setattr(s3_runtime, "triangle_frame", original_authority)

    def run(*, callback: Callable[[object], None] | None = None) -> object:
        return solve_static_nonlinear(
            model,
            load,
            control="displacement",
            displacement_control=DisplacementControl(
                node_id=2,
                dof="ux",
                target_displacement=1.0e-4,
            ),
            num_steps=1,
            num_layers=5,
            progress_callback=callback,
        )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        run(callback=progress)
    assert callbacks == ["displacement"]

    clean = run()
    assert clean.status == "completed"


@pytest.mark.parametrize(
    "route",
    ("convergence_settings", "initial_fields"),
)
def test_nonlinear_mapping_inputs_are_owned_before_model_observation(
    route: str,
) -> None:
    model, load = nonlinear_fixture._constrained_model()
    observed = _MutatingMapping(
        {"profile": "legacy"} if route == "convergence_settings" else {},
        s3_runtime,
        "triangle_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    options = {route: observed}

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load,
            num_steps=1,
            num_layers=5,
            **options,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=5,
        **options,
    )
    assert clean.status == "completed"
    assert boundary_calls == [1]


@pytest.mark.parametrize(
    "route",
    ("initial_element_states", "initial_displacements"),
)
def test_nonlinear_restart_inputs_are_owned_before_model_observation(
    route: str,
) -> None:
    model, load = nonlinear_fixture._constrained_model()
    if route == "initial_element_states":
        observed: object = _MutatingMapping(
            {},
            s3_runtime,
            "triangle_frame",
        )
    else:
        observed = _MutatingArrayProtocol(
            np.zeros(model.mesh.dof_manager.total_dofs, dtype=np.float64),
            s3_runtime,
            "triangle_frame",
        )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    options = {route: observed}

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_nonlinear(
            model,
            load,
            num_steps=1,
            num_layers=5,
            equilibrate_initial_state=False,
            **options,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        num_layers=5,
        equilibrate_initial_state=False,
        **options,
    )
    assert clean.status == "completed"
    assert boundary_calls == [1]


def test_arc_restart_states_are_owned_before_model_observation() -> None:
    model, load = nonlinear_fixture._constrained_model()
    observed = _MutatingMapping(
        {},
        s3_runtime,
        "triangle_frame",
    )
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions
    control = ArcLengthControl(
        initial_load_increment=0.05,
        minimum_load_increment=0.001,
        maximum_load_increment=0.10,
        maximum_absolute_load_factor=0.05,
        max_steps=1,
    )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        solve_static_arc_length(
            model,
            load,
            control=control,
            initial_element_states=observed,
            num_layers=5,
        )
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = solve_static_arc_length(
        model,
        load,
        control=control,
        initial_element_states=observed,
        num_layers=5,
    )
    assert clean.status in {
        "load_factor_limit_reached",
        "maximum_steps_reached",
    }
    assert boundary_calls == [1]


@pytest.mark.parametrize("route", ("nonlinear", "arc_length"))
def test_nonlinear_follower_policy_observation_keeps_original_lease(
    route: str,
) -> None:
    model, load = nonlinear_fixture._constrained_model()
    observed = _MutatingTruthMapping(
        {},
        s3_runtime,
        "triangle_frame",
    )
    load.pressure_loads = observed
    boundary_calls: list[int] = []
    original_apply = model.apply_boundary_conditions

    def apply_boundary_conditions() -> None:
        boundary_calls.append(1)
        original_apply()

    model.apply_boundary_conditions = apply_boundary_conditions

    def run() -> object:
        if route == "nonlinear":
            return solve_static_nonlinear(
                model,
                load,
                num_steps=1,
                num_layers=5,
            )
        return solve_static_arc_length(
            model,
            load,
            control=ArcLengthControl(
                initial_load_increment=0.05,
                minimum_load_increment=0.001,
                maximum_load_increment=0.10,
                maximum_absolute_load_factor=0.05,
                max_steps=1,
            ),
            num_layers=5,
        )

    with pytest.raises(AssemblyError, match="qualified shell authority"):
        run()
    assert observed.mutations == 1
    assert boundary_calls == []

    clean = run()
    if route == "nonlinear":
        assert clean.status == "completed"
    else:
        assert clean.status in {
            "load_factor_limit_reached",
            "maximum_steps_reached",
        }
    assert boundary_calls == [1]


@pytest.mark.parametrize("route", ("modal", "buckling"))
def test_current_state_orchestration_keeps_original_generation_lease(
    route: str,
) -> None:
    model, committed = current_fixture._compressed_committed_state()

    def run() -> object:
        common = {
            "current_state_displacements": committed.displacements,
            "current_state_element_states": committed.element_states,
            "current_state_num_layers": 3,
        }
        if route == "modal":
            return solve_free_vibration(model, num_modes=1, **common)
        return solve_eigenvalue_buckling(
            model,
            num_modes=1,
            dense_size_limit=1000,
            allow_free_mechanisms=True,
            **common,
        )

    calls, restore = _install_mutate_restore_boundary_callback(
        model,
        s3_runtime,
        "triangle_frame",
    )
    with pytest.raises(AssemblyError, match="qualified shell authority"):
        run()
    assert calls == [1]
    restore()

    result = run()
    assert result.solver_status in {"ok", "no_positive_modes"}
