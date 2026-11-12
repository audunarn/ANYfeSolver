"""Private kinematic entry to the unchanged constrained physical-fibre potential.

Minimize sum(w W(S x; origin)) subject to L.T x = k. The stationary multiplier
p is the retained resultant. This is an initial-trial construction, never a
history commit or a change to the globally retained stationary formulation.
"""
from decimal import Decimal as D, localcontext
from time import monotonic
from ._ge_beam3_fibre_cell import FibreCellConjugate, CellHistory
from ._ge_beam3_fibre_section import FibreHistory
from ._ge_beam3_station_resultant_cell import zero, ident, mul, tr, mv, dot, dec, pair, solve, cholesky
from ._ge_beam3_p5.compensated_coordinates import validate_pair


POLICY = 'GE_BEAM3_PHYSICAL_FIBRE_KINEMATIC_CONSTRAINED_ENTRY_V1'


def response(cell, kinematics, origin=None, *, kinematics_low=None, check=None):
    if type(cell) is not FibreCellConjugate: raise ValueError('exact native physical fibre cell required')
    started = monotonic()
    def guard():
        if check is not None: check()
        if monotonic()-started > 60.: raise TimeoutError('kinematic fibre cell deadline')
    guard(); origin = cell.virgin() if origin is None else origin
    if type(kinematics) not in (tuple, list) or len(kinematics) != 18: raise ValueError('eighteen kinematic components required')
    low = [0.]*18 if kinematics_low is None else kinematics_low
    if type(low) not in (tuple, list) or len(low) != 18: raise ValueError('eighteen low kinematic components required')
    for h, l in zip(kinematics, low): validate_pair(h, l)
    with localcontext() as context:
        context.prec = 80
        k = [dec(h)+dec(l) for h, l in zip(kinematics, low)]; origins = cell._origins(origin)
        load = [list(row) for row in cell.load_map]; transpose = tr(load); gram = mul(transpose, load)
        cholesky(gram); n = cell.size; null_load = [D(0)]*n
        # Geometry-only feasible start. A virgin elastic extrapolation may be
        # far beyond yield and is deliberately not the initial material state.
        coordinates = [row[0] for row in solve(gram, [[v] for v in k], guard)]
        x = mv(load, coordinates); evaluations = 0; line_trials = 0
        for iteration in range(49):
            guard(); potential, _, gradient, tangent, fields = cell._evaluate(x, null_load, origins, guard)
            evaluations += 1
            p = [row[0] for row in solve(gram, [[v] for v in mv(transpose, gradient)], guard)]
            residual = [a-b for a, b in zip(gradient, mv(load, p))]
            constraint = [a-b for a, b in zip(mv(transpose, x), k)]
            scale = max(D(1), *(abs(v) for v in gradient))
            if max(abs(v) for v in constraint) > D('1e-50')*max(D(1), *(abs(v) for v in k)):
                raise ValueError('kinematic fibre affine constraint drift')
            if max(abs(v) for v in residual) <= D('1e-28')*scale: break
            if iteration == 48: raise ValueError('kinematic fibre Newton update limit')
            cholesky(tangent)
            border = [row+[-v for v in forces] for row, forces in zip(tangent, load)]
            border += [row+[D(0)]*18 for row in transpose]
            direction = [row[0] for row in solve(border, [[-v] for v in residual]+[[D(0)] for _ in range(18)], guard)][:n]
            if dot(gradient, direction) >= 0: raise ValueError('nondecreasing constrained fibre direction')
            alpha, count = cell._line_minimum(x, direction, gradient, null_load, origins, guard)
            candidate = [v+alpha*d for v, d in zip(x, direction)]
            candidate_potential = cell._evaluate(candidate, null_load, origins, guard)[0]
            evaluations += count+1; line_trials += count
            if alpha <= 0 or candidate_potential >= potential: raise ValueError('nondecreasing constrained fibre line step')
            x = candidate
        cholesky(tangent)
        border = [row+[-v for v in forces] for row, forces in zip(tangent, load)]
        border += [row+[D(0)]*18 for row in transpose]
        sensitivity = solve(border, zero(n, 18)+ident(18), guard)
        hessian = sensitivity[n:]; cholesky(hessian)
        history = CellHistory(cell.identity, tuple(FibreHistory(cell.section.identity, row[2]) for row in fields))
        cell._origins(history)
        boundary = any(f.branch in ('YIELD_BOUNDARY', 'HARDENING_KNOT') for row in fields for f in row[3])
        result = dict(policy=POLICY, cell_identity=cell.identity, origin=origin, history=history,
            kinematics=pair(k), resultants=pair(p), potential=pair([potential]), hessian=[pair(row) for row in hessian],
            primal_variables=pair(x), stations=[dict(strain=pair(row[0]), resultants=pair(row[1]), fibres=row[3]) for row in fields],
            iterations=iteration, evaluations=evaluations, line_trials=line_trials,
            station_equilibrium_error=str(max(abs(v) for v in residual)), constraint_error=str(max(abs(v) for v in constraint)),
            derivative_kind='SEMISMOOTH_BRANCH_SELECTION' if boundary else 'CLASSICAL_SMOOTH_BRANCH',
            production_qualified=False)
    guard(); return result
