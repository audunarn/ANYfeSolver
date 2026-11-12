"""Private simultaneous station/fibre Legendre problem for two beam cells.

Psi*(p) = sup_x [p.L^T.x - sum_j w_j W(S_j.x; origin_j)].
The full physical-fibre potential, not an elastic tangent substitution, is
reevaluated at fixed origin throughout the coupled solve. Not a public route.
"""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
import hashlib
from math import isfinite, nextafter, ulp
from time import monotonic

from ._ge_beam3_fibre_section import (
    PhysicalFibreSection, FibreHistory, FibreRecovery, canonical, _scalar_pair,
)
from ._ge_beam3_station_resultant_cell import (
    zero, tr, mul, mv, dot, dec, pair, solve, cholesky, frozen,
)
from ._ge_beam3_effective_deadline import effective_deadline_seconds


POLICY = 'GE_BEAM3_COUPLED_PHYSICAL_FIBRE_CELL_CONJUGATE_V1'


@dataclass(frozen=True)
class CellHistory:
    cell_identity: str
    stations: tuple


@dataclass(frozen=True)
class FibreTrialState:
    """Owned local branch linearization; never accepted/checkpoint state."""
    cell_identity: str
    origin: CellHistory
    resultants: tuple
    strains: tuple
    strain_compliances: tuple


@dataclass(frozen=True)
class PairedFibreTrialState(FibreTrialState):
    """Actual paired input and smooth current branch; not a global affine law."""
    resultants_low: tuple
    smooth: bool


def _section_response(section, strain, origins, check, *, include_energy=False):
    """Decimal first/second variations and recovery of the declared potential."""
    stress = mv(section._background, strain)
    tangent = [list(row) for row in section._background]
    potential = dot(strain, stress)/2
    stored = potential; previous_hardening = D(0); dissipation = D(0)
    proposed = []; recovered = []
    for fibre, (z0, p0) in zip(section.fibres, origins):
        check(); b = section._map(fibre); area = D.from_float(fibre.area); young = D.from_float(fibre.young)
        ef = dot(b, strain); delta, et, branch = fibre.curve._return(young, ef, z0, p0)
        z = z0+delta; p = p0+abs(delta); elastic = ef-z; sigma = young*elastic
        potential += area*(young*elastic*elastic/2+fibre.curve._integral(p0, p))
        if include_energy:
            y0 = D.from_float(fibre.curve.flow_stress[0])
            stored += area*(young*elastic*elastic/2+
                           fibre.curve._integral(D(0), p)-y0*p)
            previous_hardening += area*(fibre.curve._integral(D(0), p0)-y0*p0)
            dissipation += area*y0*abs(delta)
        for i in range(6):
            stress[i] += area*b[i]*sigma
            for j in range(6): tangent[i][j] += area*et*b[i]*b[j]
        zh, zl = _scalar_pair(z); ph, pl = _scalar_pair(p)
        proposed.append((zh, zl, ph, pl))
        recovered.append(FibreRecovery(fibre.fibre_id, _scalar_pair(ef), _scalar_pair(elastic),
            _scalar_pair(sigma), _scalar_pair(et), _scalar_pair(delta), branch))
    fields = potential, stress, tangent, tuple(proposed), tuple(recovered)
    return fields + (stored, previous_hardening, dissipation) if include_energy else fields


class FibreCellConjugate:
    __slots__ = ('section', 'stations', 'identity', 'load_map', 'size', '_elastic', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('captured fibre cell is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, section, stations):
        if type(section) is not PhysicalFibreSection or len(section.fibres) > 32:
            raise ValueError('explicit bounded physical fibre section required')
        if type(stations) not in (list, tuple) or not 4 <= len(stations) <= 32:
            raise ValueError('bounded station inventory required')
        started = monotonic()
        def check():
            if monotonic()-started > effective_deadline_seconds(60.): raise TimeoutError('fibre cell compiler deadline')
        self.section = section; self.size = 6+3*len(stations)
        rows = []; capture = []
        with localcontext() as context:
            context.prec = 80
            load = zero(self.size, 18); elastic = zero(self.size, self.size)
            for i in range(6): load[i][i] = D(1)
            c = [list(row) for row in section._background]
            for fibre in section.fibres:
                b = section._map(fibre); weight = dec(fibre.area)*dec(fibre.young)
                for i in range(6):
                    for j in range(6): c[i][j] += weight*b[i]*b[j]
            for station, row in enumerate(stations):
                check()
                if type(row) is not dict or set(row) != {'cell', 't', 'weight', 'v'}:
                    raise ValueError('exact cell station schema required')
                cell = row['cell']; t = dec(row['t']); weight = dec(row['weight'])
                if type(cell) is not int or cell not in (0, 1) or weight <= 0 or not 0 < t < 1:
                    raise ValueError('positive interior half-cell station required')
                v = [[dec(x) for x in r] for r in row['v']]
                if len(v) != 3 or any(len(r) != 3 for r in v): raise ValueError('three by three station map required')
                mapping = zero(6, 6)
                for i in range(3):
                    mapping[i][:3] = v[i]
                    mapping[3+i][3+i] = D(1)
                    load[6+3*station+i][6+6*cell+i] = weight*(1-t)
                    load[6+3*station+i][9+6*cell+i] = weight*t
                indices = tuple(range(3*cell, 3*cell+3))+tuple(range(6+3*station, 9+3*station))
                block = mul(mul(tr(mapping), c), mapping)
                for i, a in enumerate(indices):
                    for j, b in enumerate(indices): elastic[a][b] += weight*block[i][j]
                rows.append((cell, weight, frozen(mapping), indices))
                capture.append(dict(cell=cell, t=float(t), weight=float(weight), v=[[float(x) for x in r] for r in v]))
            if {row[0] for row in rows} != {0, 1}: raise ValueError('both half-cells must be represented')
            if any(len({row['t'] for row in capture if row['cell'] == cell}) < 2 for cell in (0, 1)):
                raise ValueError('two distinct stations per half-cell required for endpoint moment work')
            cholesky(elastic)
            self.load_map = frozen(load); self._elastic = frozen(elastic); self.stations = tuple(rows)
            self.identity = hashlib.sha256(canonical(dict(policy=POLICY, section=section.identity,
                stations=capture))).hexdigest()
            check(); self._sealed = True

    def virgin(self):
        return CellHistory(self.identity, tuple(self.section.virgin() for _ in self.stations))

    def _origins(self, origin):
        if (type(origin) is not CellHistory or origin.cell_identity != self.identity
                or type(origin.stations) is not tuple or len(origin.stations) != len(self.stations)):
            raise ValueError('complete immutable cell-bound fibre history required')
        return tuple(self.section._origins(h) for h in origin.stations)

    def _evaluate(self, x, load, origins, check):
        residual = [-value for value in load]; tangent = zero(self.size, self.size)
        material_potential = D(0); fields = []
        for (_, weight, mapping, indices), history in zip(self.stations, origins):
            check(); strain = mv(mapping, [x[i] for i in indices])
            potential, stress, modulus, proposed, recovery = _section_response(self.section, strain, history, check)
            material_potential += weight*potential
            vector = mv(tr(mapping), stress); block = mul(mul(tr(mapping), modulus), mapping)
            for i, a in enumerate(indices):
                residual[a] += weight*vector[i]
                for j, b in enumerate(indices): tangent[a][b] += weight*block[i][j]
            fields.append((strain, stress, proposed, recovery))
        return material_potential-dot(x, load), material_potential, residual, tangent, fields

    def _line_minimum(self, x, direction, residual, load, origins, check):
        """Minimize the actual convex piecewise-quadratic potential on [0,1].

        Every yield/table-knot crossing is known from the affine fibre strain.
        This prevents geometric backtracking from crawling across narrow
        elastic intervals at high material contrast. No constitutive tuning.
        """
        cuts = {D(0), D(1)}
        for (_, _, mapping, indices), history in zip(self.stations, origins):
            strain = mv(mapping, [x[i] for i in indices])
            change = mv(mapping, [direction[i] for i in indices])
            for fibre, (z0, p0) in zip(self.section.fibres, history):
                check(); b = self.section._map(fibre); ef = dot(b, strain); de = dot(b, change)
                if de == 0: continue
                young = dec(fibre.young)
                levels = (p0,)+tuple(D.from_float(k) for k in fibre.curve.plastic_strain if D.from_float(k) > p0)
                for accumulated in levels:
                    check()
                    flow = fibre.curve._at(accumulated)[0]
                    for sign in (-1, 1):
                        boundary = z0+sign*(accumulated-p0+flow/young)
                        alpha = (boundary-ef)/de
                        if 0 < alpha < 1: cuts.add(alpha)
                    if len(cuts) > 4096: raise ValueError('coupled fibre line-partition limit')
        ordered = sorted(cuts); slope = dot(residual, direction)
        evaluations = 0
        for left, right in zip(ordered, ordered[1:]):
            check(); midpoint = (left+right)/2
            point = [v+midpoint*d for v, d in zip(x, direction)]
            tangent = self._evaluate(point, load, origins, check)[3]; evaluations += 1
            curvature = dot(direction, mv(tangent, direction))
            if curvature <= 0: raise ValueError('nonpositive fibre line curvature; strict conjugate unavailable')
            root = left-slope/curvature
            if left <= root <= right: return root, evaluations
            slope += curvature*(right-left)
        return D(1), evaluations

    def boundary_amount(self, trial, direction, *, origin, check=None):
        """First fixed-origin yield/table crossing, guarded for binary64 updates.

        Within the current affine material branch, dx/dp = T^-1 L. This is a
        local force-chart proposal; actual global merit still decides acceptance.
        """
        if (type(trial) is not FibreTrialState or trial.cell_identity != self.identity or
                trial.origin is not origin or len(trial.resultants) != 18):
            raise ValueError('current cell/origin-bound fibre trial linearization required')
        started = monotonic()
        def guard():
            if check is not None: check()
            if monotonic()-started > effective_deadline_seconds(60.): raise TimeoutError('fibre boundary proposal deadline')
        with localcontext() as context:
            context.prec = 80
            correction = tuple(direction)
            dp = tuple(dec(v) for v in correction)
            if len(dp) != 18: raise ValueError('18 retained resultant corrections required')
            origins = self._origins(origin)
            first = None
            for strain, compliance, history in zip(trial.strains, trial.strain_compliances, origins):
                guard()
                change = mv(compliance, dp)
                for fibre, (z0, p0) in zip(self.section.fibres, history):
                    guard(); b = self.section._map(fibre)
                    ef = dot(b, strain); de = dot(b, change)
                    if de == 0: continue
                    sensitivity = mv(tr(compliance), b)
                    levels = (p0,)+tuple(D.from_float(k) for k in fibre.curve.plastic_strain if D.from_float(k) > p0)
                    for accumulated in levels:
                        guard(); flow = fibre.curve._at(accumulated)[0]
                        for sign in (-1, 1):
                            boundary = z0+sign*(accumulated-p0+flow/dec(fibre.young))
                            amount = (boundary-ef)/de
                            if not 0 < amount < 1: continue
                            if first is None or amount < first[0]:
                                first = amount, de, sensitivity
            if first is None: return None
            amount, de, sensitivity = first
            # _advance performs amount*dp then p+product, each round-to-nearest.
            # Half an ULP bounds each operation; this uniform [0,1] bound avoids
            # fixture-dependent slack and includes subnormal arithmetic.
            strain_rounding = D(0)
            for p, d, coefficient in zip(trial.resultants, correction, sensitivity):
                upper = nextafter(abs(p)+abs(d), float('inf'))
                if not isfinite(upper): return None
                bound = (D.from_float(ulp(d))+D.from_float(ulp(upper)))/2
                strain_rounding += abs(coefficient)*bound
            proposed = nextafter(float(amount+strain_rounding/abs(de)), float('inf'))
            return proposed if isfinite(proposed) and 0 < proposed < 1 else None

    def paired_boundary_amount(self, trial, direction, *, origin, resultants,
                               resultants_low, check=None):
        """Heuristic first affine-branch event for the represented paired update.

        Fixed-origin FlowCurve laws are piecewise affine in strain. The cell
        inverse is locally affine only while all station branches stay fixed
        and its tangent is invertible. No global e(p) or acceptance claim follows.
        """
        if (type(trial) is not PairedFibreTrialState or
                trial.cell_identity != self.identity or trial.origin is not origin or
                trial.resultants != tuple(resultants) or
                trial.resultants_low != tuple(resultants_low)):
            raise ValueError('current paired cell/origin/resultant trial required')
        if not trial.smooth: return None
        from ._stiffened_shell_f06_increment_arithmetic import add_state
        import numpy as np
        guard = (lambda: None) if check is None else check
        with localcontext() as context:
            context.prec = 80
            dp = tuple(dec(float(v)) for v in direction)
            if len(dp) != 18: raise ValueError('18 retained resultant corrections required')
            first = None
            for strain, compliance, history in zip(trial.strains, trial.strain_compliances,
                                                    self._origins(origin)):
                guard(); change = mv(compliance, dp)
                for fibre, (z0, p0) in zip(self.section.fibres, history):
                    guard(); b = self.section._map(fibre)
                    ef = dot(b, strain); de = dot(b, change)
                    if de == 0: continue
                    sensitivity = mv(tr(compliance), b)
                    levels = (p0,)+tuple(D.from_float(k) for k in fibre.curve.plastic_strain if D.from_float(k)>p0)
                    for accumulated in levels:
                        for sign in (-1, 1):
                            boundary = z0+sign*(accumulated-p0+fibre.curve._at(accumulated)[0]/dec(fibre.young))
                            amount = (boundary-ef)/de
                            if 0 < amount < 1 and (first is None or amount < first[0]):
                                first = amount, de, sensitivity, boundary-ef
            if first is None: return None
            amount, de, sensitivity, distance = first
            # Product roundoff plus the low-word addition in Pair.__add__.
            # two_sum/renormalization retain their sum; nonfinite paths refuse.
            rounding = D(0)
            for h, l, d, coefficient in zip(trial.resultants, trial.resultants_low, direction, sensitivity):
                upper = nextafter(abs(h)+abs(d), float('inf'))
                if not isfinite(upper): return None
                low_upper = nextafter(ulp(upper)/2+abs(l), float('inf'))
                if not isfinite(low_upper): return None
                rounding += abs(coefficient)*(D.from_float(ulp(d))+D.from_float(ulp(low_upper)))/2
            proposed = nextafter(float(amount+rounding/abs(de)), float('inf'))
            if not isfinite(proposed) or not 0 < proposed < 1: return None
            guard()
            high, low = add_state(np.asarray(trial.resultants), np.asarray(trial.resultants_low),
                                  proposed*np.asarray(direction, dtype=float))
            change = [dec(float(h))+dec(float(l))-dec(a)-dec(b) for h,l,a,b in
                      zip(high,low,trial.resultants,trial.resultants_low)]
            represented = dot(sensitivity, change)
            # Verify the selected *linearized* event against the actual updater;
            # nonlinear constitutive evaluation and global merit remain decisive.
            if (represented-distance)*de < 0: return None
            return proposed

    def response(self, resultants, origin=None, *, check=None, include_energy=False,
                 capture_trial=False, resultants_low=None):
        if type(include_energy) is not bool:
            raise ValueError('explicit native energy recovery policy required')
        if type(capture_trial) is not bool:
            raise ValueError('explicit fibre trial capture policy required')
        started = monotonic()
        def guard():
            if check is not None: check()
            if monotonic()-started > effective_deadline_seconds(60.): raise TimeoutError('coupled fibre cell deadline')
        guard()
        origin = self.virgin() if origin is None else origin
        with localcontext() as context:
            context.prec = 80
            p = [dec(x) for x in resultants]
            if len(p) != 18: raise ValueError('18 retained resultants required')
            if resultants_low is not None:
                low = [dec(x) for x in resultants_low]
                if len(low) != 18:
                    raise ValueError('18 finite retained resultant corrections required')
                from ._ge_beam3_p5.compensated_coordinates import validate_pair
                for high, correction in zip(resultants, resultants_low):
                    validate_pair(high, correction)
                p = [high+correction for high, correction in zip(p, low)]
            origins = self._origins(origin); load = mv(self.load_map, p)
            # Virgin tangent is only a predictor. Origin plastic strains enter
            # its linear term; all actual trial states use the nonlinear law.
            predictor = list(load)
            for (_, weight, mapping, indices), history in zip(self.stations, origins):
                s = [D(0)]*6
                for fibre, (z0, _) in zip(self.section.fibres, history):
                    b = self.section._map(fibre); scale = dec(fibre.area)*dec(fibre.young)*z0
                    for i in range(6): s[i] += scale*b[i]
                for i, value in zip(indices, mv(tr(mapping), s)): predictor[i] += weight*value
            x = [row[0] for row in solve([list(row) for row in self._elastic], [[v] for v in predictor], guard)]
            threshold = D('1e-28')*max(D(1), *(abs(v) for v in load))
            evaluations = 0; line_trials = 0
            for iteration in range(49):
                guard(); objective, material, residual, tangent, fields = self._evaluate(x, load, origins, guard)
                evaluations += 1
                if max(abs(v) for v in residual) <= threshold: break
                if iteration == 48: raise ValueError('coupled fibre Newton update limit')
                # No pseudo-inverse, shift, stiffness floor or invented yield strength.
                try: cholesky(tangent)
                except ValueError as error: raise ValueError('nonpositive current fibre cell tangent; strict conjugate unavailable') from error
                direction = [row[0] for row in solve(tangent, [[-v] for v in residual], guard)]
                slope = dot(residual, direction)
                if slope >= 0: raise ValueError('nondecreasing coupled fibre Newton direction')
                alpha, count = self._line_minimum(x, direction, residual, load, origins, guard)
                proposed = [v+alpha*d for v, d in zip(x, direction)]
                candidate = self._evaluate(proposed, load, origins, guard)[0]
                evaluations += count+1; line_trials += count
                if alpha <= 0 or candidate >= objective:
                    raise ValueError('nondecreasing exact-partition fibre line step')
                x = proposed
            try: cholesky(tangent)
            except ValueError as error: raise ValueError('nonpositive accepted fibre cell tangent; smooth compliance unavailable') from error
            strain_compliance = solve(tangent, [list(row) for row in self.load_map], guard)
            compliance = mul(tr(self.load_map), strain_compliance)
            gradient = mv(tr(self.load_map), x)
            history = CellHistory(self.identity, tuple(FibreHistory(self.section.identity, row[2]) for row in fields))
            self._origins(history)
            station_rows = [dict(strain=pair(strain), resultants=pair(stress), fibres=recovery)
                            for strain, stress, _, recovery in fields]
            boundary = any(f.branch in ('YIELD_BOUNDARY', 'HARDENING_KNOT') for row in fields for f in row[3])
            response = dict(policy=POLICY, cell_identity=self.identity, origin=origin, history=history,
                potential=pair([-objective]), material_potential=pair([material]), gradient=pair(gradient),
                hessian=[pair(row) for row in compliance], stations=station_rows,
                iterations=iteration, evaluations=evaluations, line_trials=line_trials,
                residual=str(max(abs(v) for v in residual)),
                derivative_kind='SEMISMOOTH_BRANCH_SELECTION' if boundary else 'CLASSICAL_SMOOTH_BRANCH',
                production_qualified=False)
            if capture_trial:
                response['_trial_state'] = FibreTrialState(self.identity, origin, tuple(float(v) for v in p),
                    tuple(tuple(row[0]) for row in fields),
                    tuple(frozen(mul(mapping, [strain_compliance[i] for i in indices]))
                          for _, _, mapping, indices in self.stations))
            if capture_trial and resultants_low is not None:
                old = response['_trial_state']
                response['_trial_state'] = PairedFibreTrialState(self.identity, origin,
                    tuple(float(v) for v in resultants), old.strains, old.strain_compliances,
                    tuple(float(v) for v in resultants_low), not boundary)
            if include_energy:
                stored = D(0); previous_hardening = D(0); dissipation = D(0)
                for (_, weight, _, _), predecessor, fields_row, station in zip(
                        self.stations, origins, fields, station_rows):
                    strain, stress, proposed, recovery = fields_row
                    repeated = _section_response(self.section, strain, predecessor,
                                                 guard, include_energy=True)
                    if (repeated[1] != stress or repeated[3] != proposed or
                            repeated[4] != recovery):
                        raise ValueError('native energy recovery changed accepted station response')
                    stored += weight*repeated[5]
                    previous_hardening += weight*repeated[6]
                    dissipation += weight*repeated[7]
                    station['stored_energy_per_length'] = _scalar_pair(repeated[5])
                    station['previous_hardening_per_length'] = _scalar_pair(repeated[6])
                    station['incremental_dissipation_per_length'] = _scalar_pair(repeated[7])
                response['stored_energy'] = _scalar_pair(stored)
                response['previous_hardening_energy'] = _scalar_pair(previous_hardening)
                response['incremental_dissipation'] = _scalar_pair(dissipation)
                response['energy_recovery_station_updates'] = len(fields)
        guard()
        return response
