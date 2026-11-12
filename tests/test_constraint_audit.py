import numpy as np
import pytest

from anyfesolver import audit_constraints, constraint_residual_summary
from anyfesolver.assembly import build_constraint_transformation
import anyfesolver.assembly as assembly_module
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.fe_core import FEModel
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement


class MPCElement:
    def __init__(self, element_id, constraints):
        self.element_id = element_id
        self.node_ids = []
        self.material_name = "default"
        self._constraints = constraints

    def get_dof_mapping(self, mesh):
        return []

    def get_mpc_constraints(self, mesh):
        return self._constraints


def model_with_nodes(count=3):
    model = FEModel("constraint-audit")
    for node_id in range(1, count + 1):
        model.add_node(node_id, float(node_id - 1), 0.0, 0.0)
    return model


def codes(report):
    return {issue.code for issue in report.issues}


def test_conflicting_prescriptions_and_fixed_slave_collision_are_infeasible():
    model = model_with_nodes(2)
    model.add_boundary_condition(BoundaryCondition("first", [1], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("conflict", [1], {"ux": 1.0}))
    dofs = model.mesh.get_node(1).dofs
    model.add_element(1, MPCElement(1, [{"slave": dofs[0], "masters": {dofs[1]: 1.0}}]))

    report = audit_constraints(model)
    assert not report.feasible
    assert "CONSTRAINT002" in codes(report)
    with pytest.raises(ValueError, match="Invalid constraint system"):
        build_constraint_transformation(
            np.eye(12), np.zeros(12), model
        )


def test_invalid_constraints_fail_before_assembly(monkeypatch):
    model = model_with_nodes(1)
    model.add_boundary_condition(BoundaryCondition("first", [1], {"ux": 0.0}))
    model.add_boundary_condition(BoundaryCondition("conflict", [1], {"ux": 1.0}))
    assembled = False

    def forbidden(*args, **kwargs):
        nonlocal assembled
        assembled = True
        raise AssertionError("assembly must not run")

    monkeypatch.setattr(assembly_module, "assemble_system", forbidden)
    with pytest.raises(ValueError, match="Invalid constraint system"):
        assembly_module.solve_linear(model)
    assert not assembled


def test_duplicate_slaves_self_reference_cycle_and_invalid_master_are_stable_diagnostics():
    model = model_with_nodes(3)
    d1 = model.mesh.get_node(1).dofs
    d2 = model.mesh.get_node(2).dofs
    model.add_element(
        1,
        MPCElement(
            1,
            [
                {"slave": d1[0], "masters": {d2[0]: 1.0}, "label": "a"},
                {"slave": d1[0], "masters": {d2[1]: 1.0}, "label": "duplicate"},
                {"slave": d1[1], "masters": {d1[1]: 1.0}, "label": "self"},
                {"slave": d1[2], "masters": {d2[2]: 1.0}, "label": "cycle-a"},
                {"slave": d2[2], "masters": {d1[2]: 1.0}, "label": "cycle-b"},
                {"slave": d1[3], "masters": {9999: 1.0}, "label": "invalid"},
            ],
        ),
    )
    report = audit_constraints(model)
    assert {"CONSTRAINT001", "CONSTRAINT002", "CONSTRAINT003"} <= codes(report)
    assert not report.feasible


@pytest.mark.parametrize("bad", [np.nan, np.inf, True, "1.0"])
def test_nonfinite_or_non_numeric_coefficients_are_rejected(bad):
    model = model_with_nodes(2)
    d1 = model.mesh.get_node(1).dofs
    d2 = model.mesh.get_node(2).dofs
    model.add_element(1, MPCElement(1, [{"slave": d1[0], "masters": {d2[0]: bad}}]))
    assert "CONSTRAINT004" in codes(audit_constraints(model))


def test_cascading_weighted_affine_mpcs_report_depth_and_residual():
    model = model_with_nodes(3)
    d1 = model.mesh.get_node(1).dofs
    d2 = model.mesh.get_node(2).dofs
    d3 = model.mesh.get_node(3).dofs
    model.add_element(
        1,
        MPCElement(
            1,
            [
                {"slave": d2[0], "masters": {d1[0]: 2.0}, "value": 1.5, "label": "first"},
                {"slave": d3[0], "masters": {d2[0]: -0.5}, "value": 2.0, "label": "second"},
            ],
        ),
    )
    report = audit_constraints(model)
    assert report.feasible and report.structural_rank == 2
    assert report.max_dependency_depth == 2
    assert not report.homogeneous
    assert [equation.origin for equation in report.equations] == ["mpc:1:first", "mpc:1:second"]
    assert report.equations[0].coefficients == ((d2[0], 1.0), (d1[0], -2.0))
    values = np.zeros(18)
    values[d1[0]] = 3.0
    values[d2[0]] = 7.5
    values[d3[0]] = -1.75
    assert constraint_residual_summary(model, values)["status"] == "passed"
    values[d3[0]] += 1.0e-4
    failed = constraint_residual_summary(model, values)
    assert failed["status"] == "failed"
    assert failed["issue_code"] == "CONSTRAINT006"


def test_valid_eccentric_beam_shell_coupling_is_feasible():
    model = model_with_nodes(5)
    coupling = InterpolatedBeamShellMPCElement(
        1,
        beam_node_id=5,
        shell_node_ids=[1, 2, 3, 4],
        shape_weights=np.full(4, 0.25),
        eccentricity=np.array([0.0, 0.0, 0.2]),
    )
    model.add_element(1, coupling)
    report = audit_constraints(model)
    assert report.status == "ok"
    assert report.feasible
    assert report.mpc_slave_dofs == 6
    assert report.structural_rank == 6


def test_modal_variations_ignore_affine_offsets_but_enforce_coefficients():
    model = model_with_nodes(2)
    d1 = model.mesh.get_node(1).dofs
    d2 = model.mesh.get_node(2).dofs
    model.add_element(1, MPCElement(1, [{"slave": d2[0], "masters": {d1[0]: 2.0}, "value": 5.0}]))
    mode = np.zeros(12)
    mode[d1[0]] = 2.0
    mode[d2[0]] = 4.0
    report = constraint_residual_summary(model, mode, homogeneous_variation=True)
    assert report["status"] == "passed"
