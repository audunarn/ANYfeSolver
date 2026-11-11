"""Tiny assembled reference solves; conditional Decimal checks, no qualification."""

from decimal import Decimal, ROUND_DOWN, localcontext

import numpy as np
import pytest

from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import HALVES, metrics
from docs.reference_cases.ge_beam3_curved_p5_chain_probe import ReferenceChainProbe, ChainSolveError, add_expansion
from docs.reference_cases.ge_beam3_curved_p5_precise_reference_probe import PreciseReferenceAction
from docs.reference_cases.ge_beam3_curved_p5_directional_probe import DirectionalResponseProbe
from test_ge_beam3_curved_p5_directional_probe import decimal_stationary_action
from test_ge_beam3_curved_p5_algebra_probe import reference, section
from test_ge_beam3_curved_p5_flexibility_probe import decimal_matrix, transpose, multiply, add, solve


def chain_references(count=2):
    base = reference(.4)
    stations = np.linspace(-1., 1., 2*count+1)
    nodes = np.array([base.position(t) for t in stations])
    frames = np.array([base.frame(t) for t in stations])
    return tuple(CurvedBeam3ReferenceGeometry(nodes[i:i+3], frames[i:i+3])
                 for i in range(0, 2*count, 2))


def end_load(size):
    loads = np.zeros(size)
    loads[-6:] = [.2, -.3, .1, .03, .02, -.04]
    return loads


def decimal_assembled_reference(references, elastic, loads):
    """Assemble original uncondensed rotations at 80 digits, then solve.

    Neither the directional kernel nor its dense preconditioner is imported
    here. Geometry and binary64 F/H/J metric inputs are shared. This is a
    separately reconstructed arithmetic check, not independent authorship,
    quadrature, engineering reference or assembled qualification authority.
    """
    with localcontext() as context:
        context.prec = 80
        nodal_size = len(loads)
        size = nodal_size+6*len(references)
        stiffness = [[Decimal(0) for _ in range(size)] for _ in range(size)]
        for element, ref in enumerate(references):
            coordinates = decimal_matrix(ref.coordinates)
            frames = [decimal_matrix(frame) for frame in ref.nodal_triads]
            indices = list(range(12*element, 12*element+18))+list(
                range(nodal_size+6*element, nodal_size+6*element+6))
            for cell, (left, right) in enumerate(HALVES):
                data = metrics(ref, elastic, cell)
                f, h, j = map(decimal_matrix, (data.force, data.compliance, data.coupling))
                dz = [[Decimal(0) for _ in range(24)] for _ in range(3)]
                ell = [[Decimal(0) for _ in range(24)] for _ in range(6)]
                x, y, z = [b-a for a, b in zip(coordinates[left], coordinates[right])]
                cross = [[Decimal(0), -z, y], [z, Decimal(0), -x], [-y, x, Decimal(0)]]
                lt, rt = transpose(frames[left]), transpose(frames[right])
                for i in range(3):
                    dz[i][6*left+i], dz[i][6*right+i] = Decimal(-1), Decimal(1)
                    for k in range(3):
                        dz[i][18+3*cell+k] = cross[i][k]
                        ell[i][6*left+3+k] = -lt[i][k]
                        ell[i][18+3*cell+k] = lt[i][k]
                        ell[i+3][6*right+3+k] = rt[i][k]
                        ell[i+3][18+3*cell+k] = -rt[i][k]
                de = add(multiply(j, dz), ell)
                local = add(multiply(transpose(dz), multiply(f, dz)),
                            multiply(transpose(de), solve(h, de)))
                for i, row in enumerate(indices):
                    for k, column in enumerate(indices):
                        stiffness[row][column] += local[i][k]
        rhs = [[Decimal.from_float(float(value))] for value in loads]+[[Decimal(0)] for _ in range(size-nodal_size)]
        free_solution = solve([row[6:] for row in stiffness[6:]], rhs[6:])
        solution = [[Decimal(0)] for _ in range(6)]+free_solution
        forces = multiply(stiffness, solution)
        reaction = [forces[i][0]-rhs[i][0] for i in range(nodal_size)]
        return [row[0] for row in solution[:nodal_size]], reaction


@pytest.mark.parametrize("rho", [1., 10000., 1000000.])
def test_two_element_curved_chain_displacement_reactions_and_work_match_decimal(rho):
    refs = chain_references()
    elastic = np.diag([rho*rho]*3+[1., 2., 3.])
    model = ReferenceChainProbe(refs, elastic)
    loads = end_load(model.size)
    made = model.solve(loads)
    expected, reactions = decimal_assembled_reference(refs, elastic, loads)
    with localcontext() as context:
        context.prec = 80
        displacement = [Decimal.from_float(float(a))+Decimal.from_float(float(b))
                        for a, b in zip(made.displacement_high, made.displacement_low)]
        error = np.array([float(a-b) for a, b in zip(displacement, expected)])
        work = sum(Decimal.from_float(float(load))*value for load, value in zip(loads, displacement))
        exact_work = sum(Decimal.from_float(float(load))*value for load, value in zip(loads, expected))
    assert np.linalg.norm(error) <= 1e-11*max(1., np.linalg.norm([float(v) for v in expected]))
    assert abs(float(work-exact_work)) <= 1e-11*max(1., abs(float(exact_work)))
    assert np.linalg.norm(made.reactions-np.array([float(v) for v in reactions])) <= 1e-11
    assert made.residual_norm <= 1e-11
    assert made.iterations <= 12
    total = (loads+made.reactions).reshape(-1, 6)
    assert np.linalg.norm(total[:, :3].sum(axis=0)) <= 1e-11
    assert np.linalg.norm((total[:, 3:]+np.cross(model.coordinates, total[:, :3])).sum(axis=0)) <= 1e-11


def test_high_contrast_coupled_chain_uses_complete_section():
    refs = chain_references()
    scaling = np.diag([1e6]*3+[1.]*3)
    elastic = scaling @ section() @ scaling
    model = ReferenceChainProbe(refs, elastic, action_backend="decimal-flexibility")
    loads = end_load(model.size)
    made = model.solve(loads)
    expected, reactions = decimal_assembled_reference(refs, elastic, loads)
    # Forward displacement check and reactions are separate from residual.
    assert np.linalg.norm(made.displacement_high-np.array([float(v) for v in expected])) <= 1e-11
    assert np.linalg.norm(made.reactions-np.array([float(v) for v in reactions])) <= 1e-11
    assert made.residual_norm <= 1e-11


def test_collapsing_displacement_expansion_loses_high_contrast_equilibrium():
    model = ReferenceChainProbe(chain_references(), np.diag([1e12]*3+[1., 2., 3.]))
    loads = end_load(model.size)
    made = model.solve(loads)
    retained = np.linalg.norm((loads-model.action(made.displacement_high, made.displacement_low))[6:], np.inf)
    collapsed = np.linalg.norm((loads-model.action(made.displacement_high+made.displacement_low))[6:], np.inf)
    assert retained <= 1e-11
    assert collapsed > 1e-8
    assert not made.displacement_high.flags.writeable
    assert not made.displacement_low.flags.writeable


def test_refinement_bounds_invalid_inputs_and_repeated_solution():
    refs = chain_references(1)
    model = ReferenceChainProbe(refs, section())
    loads = end_load(model.size)
    with pytest.raises(ChainSolveError, match="budget"):
        model.solve(loads, max_iterations=0)
    for limit in (-1, 13, True):
        with pytest.raises(ValueError, match="limit"):
            model.solve(loads, max_iterations=limit)
    for bad in (np.zeros(model.size-1), np.full(model.size, np.nan)):
        with pytest.raises(ValueError, match="nodal loads"):
            model.solve(bad)
    first, second = model.solve(loads), model.solve(loads)
    assert np.array_equal(first.displacement_high, second.displacement_high)
    assert np.array_equal(first.displacement_low, second.displacement_low)
    assert np.array_equal(first.reactions, second.reactions)
    with pytest.raises(ValueError, match="one or two"):
        ReferenceChainProbe(refs*3, section())
    with pytest.raises(ValueError, match="shared"):
        ReferenceChainProbe(refs*2, section())


def test_compensated_update_retains_sub_ulp_change_without_extra_dofs():
    high, low = add_expansion(np.array([1.]), np.array([0.]), np.array([2**-54]))
    assert high[0] == 1. and low[0] == 2**-54
    high, low = add_expansion(high, low, np.array([-1.]))
    assert high[0] == 2**-54 and low[0] == 0.


@pytest.mark.parametrize("rho", [1., 1000000.])
def test_precise_dual_action_matches_original_primal_equations(rho):
    ref = reference(.6, .2, .15)
    scaling = np.diag([rho]*3+[1.]*3)
    elastic = scaling @ section() @ scaling
    high = np.sin(np.arange(18)+.3)/7
    low = high*1e-18
    action = PreciseReferenceAction(ref, elastic).action(high, low)
    _, first = decimal_stationary_action(ref, elastic, high)
    _, second = decimal_stationary_action(ref, elastic, low)
    assert np.linalg.norm(action-first-second) <= 1e-11*max(1., np.linalg.norm(first))


def test_precise_backend_is_independent_of_ambient_decimal_context():
    ref = reference(.4)
    high = np.cos(np.arange(18))/8
    normal = PreciseReferenceAction(ref, section()).action(high)
    with localcontext() as context:
        context.prec = 7
        context.rounding = ROUND_DOWN
        context.Emax = 20
        changed_context = PreciseReferenceAction(ref, section()).action(high)
    assert np.array_equal(normal, changed_context)


def test_backend_selection_is_explicit_without_automatic_fallback(monkeypatch):
    refs = chain_references(1)
    with pytest.raises(ValueError, match="backend"):
        ReferenceChainProbe(refs, section(), action_backend="automatic")
    def unavailable_directional_action(*args):
        raise ChainSolveError("directional backend deliberately unavailable")
    monkeypatch.setattr(DirectionalResponseProbe, "tangent_action", unavailable_directional_action)
    chosen = ReferenceChainProbe(refs, section(), action_backend="decimal-flexibility")
    assert chosen.solve(end_load(chosen.size)).residual_norm <= 1e-11
    original = ReferenceChainProbe(refs, section())
    with pytest.raises(ChainSolveError, match="deliberately unavailable"):
        original.solve(end_load(original.size))
