from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition,
    ElementCapabilityError,
    FEModel,
    FixedSupport,
    QualifiedE4PLS3ShellElement,
    RigidSphereImpact,
    TransientConfig,
    solve_transient_sphere_impact,
)
from anyfesolver.element_capabilities import require_model_element_capabilities


def _model() -> FEModel:
    model = FEModel("guarded-qualified-s3")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)), start=1
    ):
        model.add_node(node_id, *coordinate)
    model.add_element(
        7,
        QualifiedE4PLS3ShellElement(
            7,
            [1, 2, 3],
            "steel",
            reference_normal=np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    model.add_boundary_condition(FixedSupport("edge-12", [1, 2]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-3-guided",
            [3],
            {"ux": 0.0, "uy": 0.0, "rz": 0.0},
        )
    )
    return model


def test_linear_transient_is_no_longer_a_qualified_s3_capability_gap() -> None:
    element = _model().mesh.elements[7]
    assert "transient_algebraic_dynamics" not in element.capability_gaps
    assert element.capability_matrix()["transient_algebraic_dynamics"] == (
        "PARITY_REPLACED"
    )


def test_public_contact_uses_the_native_qualified_s3_path() -> None:
    model = _model()
    sphere = RigidSphereImpact(
        "guard",
        radius=0.1,
        mass=1.0,
        start_point=(0.3, 0.3, 0.2),
        travel_direction=(0.0, 0.0, -1.0),
        speed=1.0,
    )

    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.01, t_end=0.01),
        sphere,
    )
    assert result.status in {"completed", "no_contact"}
    assert "contact_state" not in model.mesh.elements[7].capability_gaps


def test_guard_diagnostics_are_sorted_and_bounded() -> None:
    model = FEModel("guard-order")
    model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    for node_id, coordinate in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinate)
    for element_id, node_ids in ((11, [1, 2, 3]), (3, [4, 5, 6])):
        model.add_element(
            element_id,
            QualifiedE4PLS3ShellElement(
                element_id,
                node_ids,
                "steel",
                reference_normal=np.asarray((0.0, 0.0, 1.0)),
            ),
        )

    with pytest.raises(ElementCapabilityError) as caught:
        require_model_element_capabilities(
            model,
            "buckling",
            context="broad-current-state-buckling",
        )

    message = str(caught.value)
    assert message.index("3 (") < message.index("11 (")
