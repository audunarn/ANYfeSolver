from __future__ import annotations

import math

import numpy as np
import pytest

from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import Element
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import (
    _support_reaction_resultants,
    _support_reaction_resultants_from_forces,
)


class SofteningSpringElement(Element):
    """One-DOF spring embedded in the solver's six-DOF node convention.

    The active equilibrium path under a unit reference load is

        lambda = k u - c u^3,

    which has a closed-form limit point when c > 0.
    """

    def __init__(self, element_id: int, node_id: int, k: float = 1.0, c: float = 1.0):
        super().__init__(element_id, [node_id], "default")
        self.k = float(k)
        self.c = float(c)

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray([mesh.get_node(self.node_ids[0]).coords()], dtype=float)

    def compute_stiffness_matrix(self, mesh, material):
        matrix = np.eye(6, dtype=float)
        matrix[0, 0] = self.k
        return matrix

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        u = float(np.asarray(u_elem, dtype=float)[0])
        force = np.asarray(u_elem, dtype=float).copy()
        force[0] = self.k * u - self.c * u**3
        stiffness = None
        if tangent:
            stiffness = np.eye(6, dtype=float)
            stiffness[0, 0] = self.k - 3.0 * self.c * u**2
        trial_state = {"spring_displacement": u}
        return force, stiffness, trial_state


class CoupledLinearElement(Element):
    """Two translational DOFs with one free and one prescribed component."""

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray(
            [mesh.get_node(node_id).coords() for node_id in self.node_ids],
            dtype=float,
        )

    @staticmethod
    def _matrix():
        matrix = np.eye(12, dtype=float)
        matrix[np.ix_((0, 6), (0, 6))] = ((2.0, -1.0), (-1.0, 1.0))
        return matrix

    def compute_stiffness_matrix(self, mesh, material):
        return self._matrix()

    def compute_nonlinear_response(
        self, mesh, material, u_elem, state=None, num_layers=5, tangent=True
    ):
        matrix = self._matrix()
        displacement = np.asarray(u_elem, dtype=float)
        return matrix @ displacement, matrix if tangent else None, {}


def _spring_model(*, k: float = 1.0, c: float = 1.0):
    model = FEModel("softening_spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, SofteningSpringElement(1, 1, k=k, c=c))
    model.add_boundary_condition(
        BoundaryCondition(
            "one_dof",
            [1],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("unit_reference")
    load.add_nodal_load(1, load_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _prescribed_path_model():
    model = FEModel("prescribed_path")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, CoupledLinearElement(1, [1, 2], "default"))
    model.add_boundary_condition(
        BoundaryCondition(
            "free-x-only",
            [1],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "prescribed-x",
            [2],
            {"ux": 0.1, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def test_arc_length_crosses_softening_limit_point_and_reports_peak():
    model, load = _spring_model(k=1.0, c=1.0)
    control = ArcLengthControl(
        initial_load_increment=0.02,
        minimum_load_increment=1.0e-5,
        maximum_load_increment=0.05,
        target_iterations=5,
        max_steps=120,
        stop_after_peak_steps=5,
        peak_drop_tolerance=1.0e-4,
    )

    result = solve_static_arc_length(
        model,
        load,
        control=control,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
    )

    exact_peak = 2.0 / (3.0 * math.sqrt(3.0))
    assert result.status == "peak_confirmed"
    assert result.peak_load_factor == pytest.approx(exact_peak, rel=0.03)
    assert result.peak_step_index is not None
    assert len(result.steps) > result.peak_step_index
    assert result.steps[-1].load_factor < result.peak_load_factor
    assert any(step.load_increment < 0.0 for step in result.steps[result.peak_step_index :])
    assert sum(step.is_peak for step in result.steps) == 1


def test_arc_length_linear_path_stops_at_load_factor_guard():
    model, load = _spring_model(k=2.0, c=0.0)
    control = ArcLengthControl(
        initial_load_increment=0.05,
        minimum_load_increment=1.0e-4,
        maximum_load_increment=0.10,
        max_steps=30,
        maximum_absolute_load_factor=0.50,
    )

    result = solve_static_arc_length(
        model,
        load,
        control=control,
        max_iterations=10,
        tolerance=1.0e-10,
        arc_tolerance=1.0e-10,
    )

    assert result.status == "load_factor_limit_reached"
    assert result.load_factor >= 0.50
    assert result.peak_load_factor == pytest.approx(result.load_factor)
    assert all(step.load_increment > 0.0 for step in result.steps)


def test_arc_length_rejects_zero_reference_load():
    model, _load = _spring_model()
    zero = LoadCase("zero")
    result = solve_static_arc_length(model, zero)
    assert result.status == "zero_reference_load"
    assert not result.steps


def test_arc_length_continues_a_prescribed_displacement_without_load_case():
    model = _prescribed_path_model()
    control = ArcLengthControl(
        initial_load_increment=0.05,
        minimum_load_increment=1.0e-4,
        maximum_load_increment=0.10,
        maximum_absolute_load_factor=0.5,
        max_steps=30,
    )

    progress = []
    result = solve_static_arc_length(
        model,
        None,
        control=control,
        tolerance=1.0e-10,
        arc_tolerance=1.0e-10,
        progress_callback=progress.append,
    )

    assert result.status == "load_factor_limit_reached"
    assert result.load_factor >= 0.5
    # The prescribed node follows lambda * 0.1 and the free node equilibrates
    # to half that displacement for this coupled linear system.
    assert result.displacements[6] == pytest.approx(0.1 * result.load_factor)
    assert result.displacements[0] == pytest.approx(0.05 * result.load_factor)
    assert result.info["prescribed_displacement_path"]["active"] is True
    assert result.info["constraint_postcheck"]["status"] == "passed"
    final_reactions = result.steps[-1].support_reactions
    assert abs(final_reactions["prescribed-x"][0]) > 0.0
    assert progress[-1]["support_reactions"] == {
        name: list(values) for name, values in final_reactions.items()
    }
    assert result.info["support_reaction_history"] == [
        {
            "step_index": step.step_index,
            "load_factor": step.load_factor,
            "support_reactions": {
                name: list(values)
                for name, values in step.support_reactions.items()
            },
        }
        for step in result.steps
    ]


@pytest.mark.parametrize("plan_size", [32, 33])
def test_force_based_support_reactions_match_residual_oracle(plan_size):
    force_size = plan_size - 1
    internal = np.linspace(-2.0, 3.0, force_size)
    constant = np.linspace(0.5, -0.25, force_size)
    proportional = np.linspace(-0.75, 1.25, force_size)
    load_factor = 0.375
    dofs = np.arange(plan_size, dtype=np.intp)
    dofs[-1] = force_size + 7
    components = np.arange(plan_size, dtype=np.intp) % 6
    plan = {"shared support": (dofs, components)}
    residual = internal - (constant + load_factor * proportional)
    model = FEModel("support reaction oracle")

    oracle = _support_reaction_resultants(
        model, residual, dof_plan=plan
    )
    recovered = _support_reaction_resultants_from_forces(
        model,
        internal,
        constant,
        proportional,
        load_factor,
        dof_plan=plan,
    )

    assert recovered.keys() == oracle.keys()
    np.testing.assert_allclose(
        recovered["shared support"],
        oracle["shared support"],
        rtol=0.0,
        atol=0.0,
    )


def test_arc_length_control_validates_increment_bounds():
    with pytest.raises(ValueError):
        ArcLengthControl(initial_load_increment=0.05, minimum_load_increment=0.10)
    with pytest.raises(ValueError):
        ArcLengthControl(initial_load_increment=0.05, maximum_load_increment=0.01)


def test_arc_length_post_buckling_traces_descending_branch_to_load_fraction():
    """Post-buckling continuation: the trace passes the limit point, follows the
    descending branch and stops automatically at the configured load fraction."""
    model, load = _spring_model(k=1.0, c=1.0)
    control = ArcLengthControl(
        initial_load_increment=0.02,
        minimum_load_increment=1.0e-5,
        maximum_load_increment=0.05,
        target_iterations=5,
        max_steps=400,
        stop_after_peak_steps=10_000,     # disabled: rely on the fraction stop
        peak_drop_tolerance=1.0e-4,
        post_peak_load_fraction=0.5,
    )

    progress: list[dict] = []
    result = solve_static_arc_length(
        model,
        load,
        control=control,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        progress_callback=progress.append,
    )

    exact_peak = 2.0 / (3.0 * math.sqrt(3.0))
    assert result.status == "post_buckling_traced"
    assert result.converged
    assert result.peak_load_factor == pytest.approx(exact_peak, rel=0.03)
    # The final point is ON the descending branch at ~half the peak load.
    assert result.load_factor <= 0.5 * result.peak_load_factor + 1.0e-9
    assert result.load_factor > 0.25 * result.peak_load_factor, "should stop near the fraction, not run away"
    # Progress stream: one structured dict per converged step, matching steps.
    assert len(progress) == len(result.steps)
    assert all(item["type"] == "nonlinear_static_step" for item in progress)
    assert progress[-1]["load_factor"] == pytest.approx(result.load_factor)
    assert progress[-1]["max_translation"] > progress[0]["max_translation"]


def test_arc_length_post_buckling_displacement_guard_stops_trace():
    model, load = _spring_model(k=1.0, c=1.0)
    guard = 0.45  # spring peak sits at u = 1/sqrt(3) ~ 0.577, guard trips before
    control = ArcLengthControl(
        initial_load_increment=0.02,
        minimum_load_increment=1.0e-5,
        maximum_load_increment=0.05,
        max_steps=400,
        stop_after_peak_steps=10_000,
        post_peak_load_fraction=0.1,
        max_translation=guard,
    )

    result = solve_static_arc_length(
        model,
        load,
        control=control,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
    )

    assert result.status == "displacement_limit_reached"
    assert result.converged
    assert result.info["final_max_translation"] > guard


def test_arc_length_control_validates_post_buckling_fields():
    with pytest.raises(ValueError):
        ArcLengthControl(post_peak_load_fraction=1.5)
    with pytest.raises(ValueError):
        ArcLengthControl(max_translation=-1.0)
    control = ArcLengthControl(post_peak_load_fraction=0.5, max_translation=0.1)
    payload = control.to_dict()
    assert payload["post_peak_load_fraction"] == 0.5
    assert payload["max_translation"] == 0.1


def test_force_control_nonlinear_static_streams_progress_dicts():
    from anyfesolver.nonlinear_static import solve_static_nonlinear

    model, load = _spring_model(k=1.0, c=1.0)
    progress: list[dict] = []
    result = solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.3,   # below the ~0.385 limit point
        num_steps=4,
        max_iterations=20,
        tolerance=1.0e-9,
        progress_callback=progress.append,
    )

    assert result.converged
    # Adaptive stepping may merge increments; at least a few points stream.
    assert len(progress) >= 2
    assert all(item["type"] == "nonlinear_static_step" for item in progress)
    load_factors = [item["load_factor"] for item in progress]
    assert load_factors == sorted(load_factors)
    assert load_factors[-1] == pytest.approx(0.3)
    assert all(item["max_translation"] >= 0.0 for item in progress)
