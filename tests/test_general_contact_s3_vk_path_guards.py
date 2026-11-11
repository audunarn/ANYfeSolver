"""Guard logic of the S3 lifted path under the von Karman law (review findings P1-1, P1-2, P2-3)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
import test_general_contact_event_subdivision as events  # noqa: E402
from anyfesolver.dynamics import TransientConfig  # noqa: E402
from anyfesolver.general_contact import transient as driver  # noqa: E402
from anyfesolver.general_contact.algebraic_path import affine_descriptor_path  # noqa: E402

CONTROLS = SimpleNamespace(s3_path_position_tolerance_m=1e-10, s3_path_rotation_tolerance_rad=1e-9,
                           equilibrium_tolerance=1e-8)


def _adapter(cls=driver._S3AffinePathAdapter, **fields):
    """Six-dof synthetic adapter: dof 0 physical, dof 3 algebraic, unit stiffness, inactive contact."""
    n = 6
    base = dict(
        structural=np.eye(n), null_basis=np.eye(n)[:, [3]], gradient=np.eye(n)[[2]],
        reference_gap=np.array([1.]), weights=np.array([1.]), force_offset=np.zeros(n),
        corner_reference=np.zeros(1), corner_gradient=np.zeros((1, n)),
        model=None, transform=sparse.identity(n, format="csr"), committed_states=None)
    base.update(fields)
    return cls(**base)


def _endpoint_case(delta):
    adapter = _adapter()
    adapter.memo["force"] = (np.zeros(6), np.zeros(6), np.zeros(6))
    adapter.memo["load"] = np.zeros(6)
    path = affine_descriptor_path(adapter.structural, adapter.null_basis, np.zeros((3, 6)), adapter.gradient,
                                  adapter.reference_gap, adapter.weights, force=np.zeros(6))
    actual_end = np.zeros(6)
    actual_end[3] = delta
    return adapter, path, actual_end


@pytest.mark.parametrize("delta", [1e-6, 5e-9, 2e-9])
def test_p1_1_endpoint_mismatch_above_budget_is_refused_even_below_the_newton_tolerance(delta):
    # Algebraic dof off by delta rad against the 1e-9 rad budget. delta = 5e-9 and 2e-9 have a residual
    # below the 1e-8 Newton tolerance, so no residual-derived allowance may excuse them.
    adapter, path, actual_end = _endpoint_case(delta)
    with pytest.raises(ValueError, match="endpoint"):
        adapter.check_endpoints(path, np.zeros(6), actual_end, sparse.identity(6, format="csr"), CONTROLS)


def test_p1_1_converged_endpoint_within_budget_is_accepted():
    adapter, path, actual_end = _endpoint_case(5e-10)
    adapter.check_endpoints(path, np.zeros(6), actual_end, sparse.identity(6, format="csr"), CONTROLS)


class _CubicExcess(driver._S3AffinePathAdapter):
    """Synthetic excess F_vk = c*x0**3 acting on the algebraic dof (the real law is cubic too)."""
    coefficient = 1.

    def nonlinear_part(self, position):
        out = np.zeros(6)
        out[3] = self.coefficient*position[0]**3
        return out


def test_p2_3_probes_below_budget_cannot_hide_an_interior_error_above_it():
    # x0(s) = s - 1.4 s^2 gives a sextic excess; its quadratic model through s = 0, 1/2, 1 leaves an
    # error whose maximum (0.033) is 1.5x its value at the probes 1/4, 3/4. Budget between the two.
    adapter = _adapter(_CubicExcess)
    dt = 1.
    start, end, velocity = np.zeros(6), np.zeros(6), np.zeros(6)
    end[0], velocity[0] = -.4, 1.
    adapter.evaluate(start, end, velocity, dt, np.zeros(6))
    force = adapter.memo["force"]
    grid = np.linspace(0., 1., 4001)
    error = np.abs(np.array([adapter.nonlinear_part(np.array([s-1.4*s*s, 0, 0, 0, 0, 0]))[3]
                             -(force[0][3]+s*(force[1][3]+s*force[2][3])) for s in grid]))
    probes = max(error[1000], error[3000])
    assert error.max() > 1.15*probes     # interior error hidden from the two probes
    budget = SimpleNamespace(s3_path_position_tolerance_m=0.5*(probes+error.max()))
    path, _ = adapter.evaluate(start, end, velocity, dt, np.zeros(6))
    with pytest.raises(ValueError, match="interior"):
        adapter.check_interior(path, budget)


def test_p1_2_vanishing_endpoint_excess_does_not_make_the_path_linear():
    # Identical endpoints (zero excess) but a transverse slope ~ s(1-s): the interior excess is nonzero.
    captured = []
    original = driver._S3AffinePathAdapter.prepare.__func__

    def capture(cls, *args, **kwargs):
        adapter = original(cls, *args, **kwargs)
        captured.append(adapter)
        return adapter

    model, definition, initial = events._guided_s3_descriptor_impact(tilt=.002, span=.1)
    driver._S3AffinePathAdapter.prepare = classmethod(capture)
    try:
        driver.solve_transient_contact(model, definition, TransientConfig(.0002, .0004, initial_velocity=initial),
                                       solver_controls=events.CONTROLS)
    finally:
        driver._S3AffinePathAdapter.prepare = classmethod(original)
    adapter = captured[0]
    n = adapter.structural.shape[0]
    free_uz = np.arange(n)[::4]            # uz of the three nodes (reduced = free ordering uz,rx,ry,rz)
    velocity = np.zeros(n)
    velocity[free_uz] = (-1., 0., 1.)      # nonuniform: transverse slope
    start = np.zeros(n)
    adapter.memo.clear()
    path, _ = adapter.evaluate(start, start.copy(), velocity, 1e-4, np.zeros(n))
    assert adapter.memo.get("force") is not None   # interior excess detected, validated by check_interior
