"""Local, non-equilibrated V19 production-adapter discriminator.

This pose is not a transient initial state. The test does not call a time
integrator or commit material history. Execution is distinct from transient initial admission.
"""
from fractions import Fraction as Q
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path

import numpy as np
import pytest


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return [x-y for x, y in zip(a, b)]


def _exact_float(x):
    value = float(x)
    assert Q(value) == x, "fixture chart is not exactly representable"
    return value


def _build_local_pose():
    from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
                           ContactInteraction, ContactSolverSettings,
                           create_shell_element)
    from anyfesolver.elements import QuadraticBeamElement
    from anyfesolver.fe_core import FEModel
    m, delta, height, radius = 256, Q(1, 16384), Q(1, 128), Q(1, 128)
    X, Y, rise = (m*m-1)*delta, Q(4), 2*m*delta
    stations = (Q(1, 4), Q(5, 16), Q(3, 8))
    beam_x = [-1+t*X-radius/m for t in stations]
    refs = {1: (beam_x[0], Q(0), Q(1, 32)),
            6: (beam_x[1], Q(0), Q(1, 32)),
            2: (beam_x[2], Q(0), Q(1, 32)),
            3: (Q(-1), Q(-1), Q(0)),
            4: (-1+X, Q(-1), Q(0)),
            5: (Q(-1), Q(3), Q(0))}
    model = FEModel("LOCAL_nonzero_directional_null_coupling")
    model.add_material("beam", 1.e3, .3, density=1.)
    model.add_material("skin", 1.e5, .3, density=1.)
    for node, point in refs.items():
        model.add_node(node, *map(_exact_float, point))
    model.add_element(1, QuadraticBeamElement(1, [1, 6, 2], "beam",
        dict(area=1.e-3, Iy=1.e-4, Iz=1.e-4, J=1.e-4)))
    model.add_element(2, create_shell_element(2, (3, 4, 5), "skin",
        thickness=float(2*height), formulation="e4-pl-s3-v2d",
        reference_normal=(0., 0., 1.), reference_surface_offset=0.))
    model.add_boundary_condition(BoundaryCondition("beam_guides", [1, 6, 2],
        dict.fromkeys(("ux", "uy", "rx", "rz"), 0.)))
    model.add_boundary_condition(BoundaryCondition("skin_guides", [3, 4, 5],
        dict.fromkeys(("ux", "uy", "rz"), 0.)))
    dofs = {node: model.mesh.dof_manager.get_node_dofs(node) for node in refs}
    for axis in (3, 4):
        for node in (4, 5):
            model.add_constraint_equation(
                terms=((dofs[node][axis], 1.), (dofs[3][axis], -1.)),
                dependent_dof=dofs[node][axis],
                source_id=f"uniform_skin_rotation_{axis}_{node}")
    definition = ContactDefinition(
        (ContactSurface("tube", (1,), beam_radius=float(radius)),
         ContactSurface("skin", (2,), side="top")), (),
        (ContactInteraction("line", "tube", "skin", None, 1.e-5,
                            line_stiffness=1.e5),))
    settings = ContactSolverSettings(pair_motion="general",
        beam_shell_rim_clearance=.02, line_integration_relative_tolerance=1.e-8,
        beam_shell_geometry="jet2_deformable_s3_dev",
        beam_shell_maximum_swept_intervals=128)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[dofs[4][2]] = float(rise)
    for node, t in zip((1, 6, 2), stations):
        u[dofs[node][2]] = _exact_float(height+radius+t*rise-Q(1, 32))
    return model, definition, settings, u, refs, dofs, height, radius, X, Y, rise


def _independent_gap_coefficients(refs, dofs, u, height):
    """Direct affine skin and moving-normal differential, no production map."""
    n = len(u)
    point_ids = (1, 6, 2, 3, 4, 5)
    maps = [[[Q(0) for _ in range(n)] for _ in range(3)] for _ in point_ids]
    points = []
    for k, node in enumerate(point_ids):
        for axis in range(3):
            maps[k][axis][dofs[node][axis]] = Q(1)
        if k >= 3:
            maps[k][0][dofs[node][4]] = height
            maps[k][1][dofs[node][3]] = -height
        base = list(refs[node])
        if k >= 3:
            base[2] += height
        points.append([base[a]+sum(maps[k][a][j]*Q(float(u[j]))
                                   for j in range(n)) for a in range(3)])
    beam, skin = points[:3], points[3:]
    edge1, edge2 = _sub(skin[1], skin[0]), _sub(skin[2], skin[0])
    C = _cross(edge1, edge2)
    C2 = _dot(C, C)
    # Pythagorean norm is exact for the declared dyadic construction.
    norm = Q(4, 16384)*(256*256+1)
    assert norm*norm == C2 and C[2] > 0
    E = [_sub(beam[0], skin[0]),
         [-3*beam[0][a]+4*beam[1][a]-beam[2][a] for a in range(3)],
         [2*beam[0][a]-4*beam[1][a]+2*beam[2][a] for a in range(3)]]
    B = np.empty((3, n), object)
    for j in range(n):
        variation = [[maps[k][a][j] for a in range(3)] for k in range(6)]
        db, ds = variation[:3], variation[3:]
        dC = [a+b for a, b in zip(_cross(_sub(ds[1], ds[0]), edge2),
                                  _cross(edge1, _sub(ds[2], ds[0])))]
        dE = [_sub(db[0], ds[0]),
              [-3*db[0][a]+4*db[1][a]-db[2][a] for a in range(3)],
              [2*db[0][a]-4*db[1][a]+2*db[2][a] for a in range(3)]]
        for i in range(3):
            B[i, j] = (_dot(dE[i], C)+_dot(E[i], dC)
                       -_dot(E[i], C)*_dot(C, dC)/C2)/norm
    return B, maps, points, E, C, norm


def _reference_directional(B, coefficient, velocity):
    """Independent quadratic roots and analytic degree-four integrals."""
    with localcontext() as ctx:
        ctx.prec = 70
        def dec(x):
            q = x if isinstance(x, Q) else Q(float(x))
            return Decimal(q.numerator)/Decimal(q.denominator)
        b = [[dec(x) for x in row] for row in B]
        v = [dec(x) for x in velocity]
        c = [sum(x*y for x, y in zip(row, v)) for row in b]
        roots = []
        if c[2]:
            discriminant = c[1]*c[1]-4*c[2]*c[0]
            if discriminant >= 0:
                root = discriminant.sqrt()
                roots = [(-c[1]-root)/(2*c[2]), (-c[1]+root)/(2*c[2])]
        elif c[1]:
            roots = [-c[0]/c[1]]
        cuts = sorted(set([Decimal(0), Decimal(1)]+
                          [x for x in roots if 0 < x < 1]))
        active = [(a, z) for a, z in zip(cuts, cuts[1:])
                  if sum(c[i]*((a+z)/2)**i for i in range(3)) < 0]
        moments = [[sum((z**(i+j+1)-a**(i+j+1))/Decimal(i+j+1)
                         for a, z in active) for j in range(3)] for i in range(3)]
        k = dec(coefficient)
        H = [[k*sum(b[i][p]*moments[i][j]*b[j][q]
                     for i in range(3) for j in range(3))
              for q in range(len(v))] for p in range(len(v))]
        D = [sum(h*x for h, x in zip(row, v)) for row in H]
        return np.array(D, float), np.array(H, float), [(float(a), float(z)) for a, z in active]


def test_production_nonzero_null_directional_coupling():
    """Future authorized selector; no transient-initial or C2 DAE claim."""
    from anyfesolver.general_contact.beam_s3_directional_rates import (
        DirectionalBeamS3Manager, directional_coefficients, directional_operator,
        exact_gap_polynomial, solve_directional_velocity, recover_directional_rates,
        PROFILE)
    from anyfesolver.assembly import assemble_mass_matrix, build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.general_contact.runtime import suspend_contact
    from anyfesolver.nonlinear_state import discard_active_state_candidate
    (model, definition, settings, u, refs, dofs, height, radius,
     X, Y, rise) = _build_local_pose()
    manager = DirectionalBeamS3Manager(model, definition, settings,
                                      initial_displacement=u)
    reference, maps, points, E, C, norm = _independent_gap_coefficients(
        refs, dofs, u, height)
    assert _dot(E[0], C) == radius*norm
    assert _dot(E[1], C) == _dot(E[2], C) == 0
    full, isolated, *_ = exact_gap_polynomial(manager, u)
    assert full and not isolated
    B, factor = directional_coefficients(manager, u)
    assert np.array_equal(B/norm, reference)
    assert np.array_equal(manager.work_map.toarray(), np.array(maps, float).reshape(18, -1))
    uniform_ry = np.zeros(len(u))
    uniform_ry[[dofs[node][4] for node in (3, 4, 5)]] = 1.
    expected_coupling = 2*256*height/Q(256*256+1)
    assert list(reference@np.array([Q(float(x)) for x in uniform_ry], object)) == [expected_coupling, Q(0), Q(0)]
    assert expected_coupling > 0
    model.apply_boundary_conditions()
    K0, _ = assemble_stiffness_matrix(model)
    M, _ = assemble_mass_matrix(model)
    _, _, T, _, independent, _ = build_constraint_transformation(K0, np.zeros(len(u)), model)
    Mred = (T.T@M@T).tocsr()
    basis = build_declared_algebraic_basis(model, M, Mred, T, independent, dense_size_limit=512)
    Z = basis.reduced_basis.toarray()
    assert Z.shape[1] == 2
    assert np.linalg.norm(Mred@Z) <= 1.e-13*np.linalg.norm(Mred.toarray())
    assert np.linalg.norm(np.array(reference, float)@T@Z) > 0.
    ry_red = np.linalg.lstsq(T.toarray(), uniform_ry, rcond=None)[0]
    assert np.linalg.norm(T@ry_red-uniform_ry) < 1.e-13
    assert np.linalg.norm(ry_red-Z@np.linalg.solve(Z.T@Z, Z.T@ry_red)) < 1.e-13
    # Independent physical mass: Q2 beam translation/rotary consistent mass,
    # and S3 lumped normal mass. No synthetic rotational inertia is introduced.
    shape = ((1, -3, 2), (0, 4, -4), (0, -1, 2))
    gram = np.array([[float(sum(Q(a*b, i+j+1) for i, a in enumerate(p)
                               for j, b in enumerate(q))) for q in shape] for p in shape])
    length = float(X/8)
    for local, density in ((2, 1.e-3), (4, 1.e-4)):
        ids = [dofs[node][local] for node in (1, 6, 2)]
        assert np.allclose(M[ids][:, ids].toarray(), density*length*gram, rtol=1.e-12, atol=1.e-15)
    shell_ids = [dofs[node][2] for node in (3, 4, 5)]
    assert np.allclose(M[shell_ids][:, shell_ids].toarray(),
                       np.eye(3)*float(2*height*X*Y/6), rtol=1.e-12, atol=1.e-15)
    committed = {}
    try:
        with suspend_contact():
            force, tangent, trial = _assemble_nonlinear_system(model, u, committed, 3,
                tangent=True, kinematics="von_karman")
    finally:
        discard_active_state_candidate(committed)
    assert committed == {}
    # Local pose is not claimed to satisfy algebraic equilibrium; retain the
    # actual residual rather than promoting it to a transient initial state.
    origin_null_residual = np.linalg.norm(Z.T@(T.T@force))
    K = (T.T@tangent@T).toarray()
    v = np.zeros(len(u))
    for node, value in zip((1, 6, 2), (.01, -.01, .01)):
        v[dofs[node][2]] = value
    for node, value in zip((3, 4, 5), (.001, .0008, .0012)):
        v[dofs[node][2]] = value
        v[dofs[node][4]] = .002
    reduced_v = np.linalg.lstsq(T.toarray(), v, rcond=None)[0]
    assert np.linalg.norm(T@reduced_v-v) < 1.e-13
    physical = reduced_v-Z@np.linalg.solve(Z.T@Z, Z.T@reduced_v)
    reduced_B = B@np.array([[Q(float(x)) for x in row] for row in T.toarray()], object)
    solved = solve_directional_velocity(K, Z, physical, reduced_B, factor)
    solved_full = T@solved
    D, H, active = _reference_directional(reference, Q(1.e5)*Q(length), solved_full)
    actual_D, actual_H, eD, eH = directional_operator(B, factor, solved_full)
    assert 0 < sum(b-a for a, b in active) < 1
    assert np.max(abs(actual_D-D)) <= eD+1.e-10
    assert np.max(abs(actual_H-H)) <= eH+1.e-9
    assert np.linalg.norm(Z.T@(K@solved+T.T@D)) <= 1.e-9*max(1., np.linalg.norm(K@solved), np.linalg.norm(D))
    assert np.linalg.norm(Mred@(solved-physical)) <= 1.e-12*max(1., np.linalg.norm(Mred@physical))
    b_ids = [dofs[node][2] for node in (1, 6, 2)]
    assert np.linalg.norm(H[np.ix_(b_ids, shell_ids)]) > 0
    # One-sided force rate tests the complete actual both-body work map.
    r0, contact_H, summary = manager.evaluate(u)
    assert summary.profile == PROFILE
    acceleration = np.arange(len(reduced_v), dtype=float)*1.e-3
    previous_velocity = .25*reduced_v
    dt = 1.e-5
    recovered_v, recovered_a = recover_directional_rates(manager, u, T, Mred,
        basis, tangent+contact_H, reduced_v, acceleration, previous_velocity,
        dt, summary)
    assert np.allclose(recovered_v, solved, rtol=1.e-12, atol=1.e-13)
    physical_a = acceleration-Z@np.linalg.solve(Z.T@Z, Z.T@acceleration)
    expected_a = physical_a+Z@np.linalg.solve(Z.T@Z,
        Z.T@(solved-previous_velocity)/dt)
    assert np.allclose(recovered_a, expected_a, rtol=1.e-12, atol=1.e-12)
    inertia_error = np.linalg.norm(Mred@(recovered_a-acceleration))
    assert inertia_error <= 1.e-10*max(1., np.linalg.norm(Mred@acceleration))
    errors = []
    force_scale = 1.e5*length*.01
    moment_scale = force_scale*float(height)
    translation = [i for node in refs for i in dofs[node][:3]]
    rotation = [i for node in refs for i in dofs[node][3:6]]
    for h in (1.e-6, 3.e-7, 1.e-7):
        r, _, _ = manager.evaluate(u+h*solved_full)
        difference = (r-r0)/h-D
        error = max(np.max(abs(difference[translation]))/force_scale,
                    np.max(abs(difference[rotation]))/moment_scale)
        errors.append(dict(h=h, normalized_force_rate_error=float(error)))
        assert error <= 1.e-5
    changed = u.copy()
    changed[dofs[1][2]] += 2.**-30
    assert not exact_gap_polynomial(manager, changed)[0]
    with pytest.raises(ValueError, match="genuinely exact"):
        directional_coefficients(manager, changed)
    report = dict(status="LOCAL_ADAPTER_ONLY_NOT_TRANSIENT_INITIALIZATION",
        target_slope=float(rise/X), exact_full_touch=True,
        uniform_ry_gap_coupling_m_per_rad=float(expected_coupling),
        mass_nullity=int(Z.shape[1]), null_origin_residual=float(origin_null_residual),
        directional_active_intervals=active, force_rate_errors=errors,
        component_error_D=float(eD), component_error_H=float(eH),
        connected_recovery_velocity_error=float(np.linalg.norm(recovered_v-solved)),
        connected_recovery_acceleration_error=float(np.linalg.norm(recovered_a-expected_a)),
        physical_acceleration_mass_action_error=float(inertia_error),
        structural_trial_discarded=True, material_commit=False, contact_null_rank=1, installed_route=bool(os.environ.get('CONTACT_INSTALLED_TARGET')))
    destination = os.environ.get("DIRECTIONAL_COUPLING_RESULT")
    if destination:
        Path(destination).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    manager.kernel.stiffness = 0.
    with pytest.raises(ValueError, match="mutat|binding|changed"):
        manager.guard()
