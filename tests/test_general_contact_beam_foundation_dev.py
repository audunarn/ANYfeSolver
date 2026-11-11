"""Independent continuum discriminator for the declared B2 round-tube owner.

Level A only: uniform dead nodal loading converges to the continuum load.
The reference uses small motion; this does not qualify general beam contact.
"""
import json
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"docs/d1_review_support"))
from d3_beam_foundation_reference import cantilever_foundation_reference

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
                       ContactSolverSettings, ContactSurface, LoadCase,
                       RigidContactBody, solve_static_nonlinear)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


RADIUS, YOUNG, NU, LENGTH = .01, 1.e7, .3, 1.
CLEARANCE, LOAD, LINE_STIFFNESS = .001, .001, 1.e4
AREA = math.pi*RADIUS**2
INERTIA = math.pi*RADIUS**4/4
BENDING = YOUNG*INERTIA
SHEAR = (5/6)*YOUNG/(2*(1+NU))*AREA


def reference(line_stiffness=LINE_STIFFNESS):
    return cantilever_foundation_reference(
        length=LENGTH, clearance=CLEARANCE, bending=BENDING, shear=SHEAR,
        load=LOAD, line_stiffness=line_stiffness)


def test_reference_open_polynomial_and_foundation_force():
    ref = reference()
    # Independent integration of Q'= -q, M'=-Q, theta'=M/D,
    # w'=theta+Q/S on the open branch, using the returned root resultants.
    m0, q0 = ref.state(0)[2:]
    for x in (0., ref.front*.37, ref.front):
        expected = (m0*x*x/(2*BENDING)-q0*x**3/(6*BENDING)
                    +LOAD*x**4/(24*BENDING)+(q0*x-LOAD*x*x/2)/SHEAR)
        assert abs(ref.state(x)[0]-expected) < 1.e-14
    reaction = quad(lambda x: LINE_STIFFNESS*(ref.state(x)[0]-CLEARANCE),
                    ref.front, LENGTH, epsabs=1.e-14, epsrel=1.e-11)[0]
    assert abs(reaction-ref.reaction)/ref.reaction < 1.e-9
    assert ref.boundary_residual < 1.e-9 and ref.front_residual < 1.e-9
    assert ref.contact_velocity_lower > 0


@pytest.mark.parametrize("line_stiffness", [1.e4, 1.e5])
def test_reference_decimal_transfer_resolution(line_stiffness):
    """Independent 80-digit Taylor/scaling-and-squaring at the solved front.

    This checks the numerical transfer/root result for these two inputs only.
    It does not certify expm or arbitrary active intervals generally.
    """
    decimal_transfer_control(reference(line_stiffness))


def decimal_transfer_control(ref):
    """Independent 80-digit transfer with optional declared endpoint work."""
    with localcontext() as ctx:
        ctx.prec = 80
        zero, one = Decimal(0), Decimal(1)
        d = Decimal.from_float
        L, c, D, S, q, k = map(d, (ref.length,ref.clearance,ref.bending,ref.shear,ref.load,ref.line_stiffness))
        endpoint = d(ref.endpoint_stiffness)*L**3/D
        sigma, beta, qbar = D/(S*L*L), k*L**4/D, q*L**4/(D*c)
        a = d(float(ref.front))/L

        def multiply(x, y):
            return [[sum((x[i][j]*y[j][l] for j in range(5)), zero)
                     for l in range(5)] for i in range(5)]

        def exponential(active, distance):
            A = [[zero for _ in range(5)] for _ in range(5)]
            A[0][1] = A[1][2] = one
            A[0][3], A[2][3] = sigma, -one
            A[3][0], A[3][4] = (beta if active else zero), -qbar-(beta if active else zero)
            A = [[v*distance for v in row] for row in A]
            norm = max(sum(abs(v) for v in row) for row in A)
            power = 0
            while norm > Decimal('.5'):
                A = [[v/2 for v in row] for row in A]
                norm /= 2
                power += 1
            term = [[one if i == j else zero for j in range(5)] for i in range(5)]
            result = [row[:] for row in term]
            for n in range(1, 301):
                term = [[v/n for v in row] for row in multiply(term, A)]
                result = [[result[i][j]+term[i][j] for j in range(5)] for i in range(5)]
                if max(abs(v) for row in term for v in row) < Decimal('1e-70'):
                    break
            else:
                raise AssertionError('independent exponential did not resolve')
            for _ in range(power):
                result = multiply(result, result)
            return result

        open_map, active_map = exponential(False, a), exponential(True, one-a)
        H = multiply(active_map, open_map)
        if endpoint == zero:
            det = H[2][2]*H[3][3]-H[2][3]*H[3][2]
            m = (-H[2][4]*H[3][3]+H[2][3]*H[3][4])/det
            Q = (-H[2][2]*H[3][4]+H[2][4]*H[3][2])/det
        else:
            tip_row = [H[3][j]+endpoint*H[0][j] for j in range(5)]
            det = H[2][2]*tip_row[3]-H[2][3]*tip_row[2]
            m = (-H[2][4]*tip_row[3]+H[2][3]*(tip_row[4]-endpoint))/det
            Q = (-H[2][2]*(tip_row[4]-endpoint)+H[2][4]*tip_row[2])/det
        seed = [zero, zero, m, Q, one]
        at_front = [sum((row[j]*seed[j] for j in range(5)), zero) for row in open_map]
        at_end = [sum((row[j]*at_front[j] for j in range(5)), zero) for row in active_map]
        reaction = q*L-Q*D*c/L**3
        assert abs(at_front[0]-one) < Decimal('1e-10')
        end_residual=max(abs(at_end[2]),abs(at_end[3]+endpoint*(at_end[0]-one)))
        assert end_residual < Decimal('1e-60')
        assert abs(float(reaction)-ref.reaction)/float(reaction) < 1.e-9
        tip_error=abs(float(at_end[0]*c)-ref.state(ref.length)[0])/float(at_end[0]*c)
        assert tip_error < 1.e-10
        return dict(front_residual_decimal=str(abs(at_front[0]-one)),
            endpoint_residual_decimal=str(end_residual),reaction=float(reaction),
            reference_reaction=ref.reaction,tip=float(at_end[0]*c),
            reference_tip=float(ref.state(ref.length)[0]),tip_relative_error=tip_error)


@pytest.mark.parametrize("cells,line_stiffness", [(16, 1.e4), (32, 1.e4),
                                                (64, 1.e4), (64, 1.e5)])
def test_b2_foundation_coupled_continuum_discriminator(cells, line_stiffness):
    model = FEModel("b2_foundation_continuum_discriminator")
    model.add_material("elastic", YOUNG, NU, density=1.)
    for node in range(1, cells+2):
        model.add_node(node, (node-1)*LENGTH/cells, 0., 0.)
    section = {"area": AREA, "Iy": INERTIA, "Iz": INERTIA,
               "J": 2*INERTIA, "shear_factor_y": 5/6, "shear_factor_z": 5/6}
    for cell in range(1, cells+1):
        model.add_element(cell, BeamElement(cell, [cell, cell+1], "elastic", section))
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,), {dof: 0. for dof in ("ux", "uy", "uz", "rx", "ry", "rz")}))
    model.add_boundary_condition(BoundaryCondition(
        "planar", tuple(range(2, cells+2)), {dof: 0. for dof in ("uy", "rx", "rz")}))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", tuple(range(1, cells+1)),
                                 beam_radius=RADIUS, beam_end_caps="none"),),
        rigid_bodies=(RigidContactBody("plane", "plane", (0., 0., -RADIUS-CLEARANCE),
                                      normal=(0., 0., 1.)),),
        interactions=(ContactInteraction("tube_plane", "tube", "plane", None,
                                         1.e-3*RADIUS, line_stiffness=line_stiffness),))
    load = LoadCase("uniform_dead_nodal_load")
    for node in range(1, cells+2):
        force = LOAD*LENGTH/cells*(.5 if node in (1, cells+1) else 1.)
        load.add_nodal_load(node, forces=(0., 0., -force))
    result = solve_static_nonlinear(
        model, load, contact=definition,
        contact_settings=ContactSolverSettings(pair_motion="general"),
        num_steps=4, max_iterations=18, tolerance=1.e-9,
        emit_restart_checkpoint=True)
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["gap_tolerance_met"]
    ref = reference(line_stiffness)
    q = np.asarray(result.displacements)
    tip = -float(q[model.mesh.dof_manager.get_node_dofs(cells+1)[2]])
    force = float(summary["beam_force"][2])
    force_error = abs(force-ref.reaction)/ref.reaction
    tip_error = abs(tip-ref.state(LENGTH)[0])/ref.state(LENGTH)[0]
    record = {"cells": cells, "line_stiffness_N_per_m2": line_stiffness,
              "tip_m": tip, "reference_tip_m": float(ref.state(LENGTH)[0]),
              "reaction_N": force, "reference_reaction_N": float(ref.reaction),
              "reaction_relative_error": float(force_error), "tip_relative_error": float(tip_error),
              "reference_front_m": float(ref.front), "contact": summary,
              "steps": [step.to_dict() for step in result.steps]}
    print("B2_FOUNDATION_RESPONSE "+json.dumps(record, sort_keys=True), flush=True)
    # Prospective development discriminator, not mesh/enforcement qualification.
    assert force_error <= (.02 if cells == 16 else .001)
    assert tip_error <= (.02 if cells == 16 else .001)
    assert np.linalg.norm(np.asarray(summary["beam_force"])+summary["rigid_force"])/LOAD < 1.e-6
