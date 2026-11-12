"""Private nonlinear cell conjugate compiled from constrained station stress.

Different construction from the source-equation audit: minimize the full
section complementary quadratic over station forces with the shared-cell
force constraint, then solve station plastic increments in the dual energy.
Standard-library Decimal is a bounded arithmetic backend, not certification.
"""
from decimal import Decimal as D, localcontext
from time import monotonic
from copy import deepcopy
from ._ge_beam3_effective_deadline import effective_deadline_seconds


def zero(n, m): return [[D(0) for _ in range(m)] for _ in range(n)]
def ident(n): return [[D(int(i == j)) for j in range(n)] for i in range(n)]
def tr(a): return list(map(list, zip(*a)))
def mul(a, b):
    columns = tr(b)
    return [[sum((x*y for x, y in zip(row, col)), D(0)) for col in columns] for row in a]
def mv(a, b): return [sum((x*y for x, y in zip(row, b)), D(0)) for row in a]
def dot(a, b): return sum((x*y for x, y in zip(a, b)), D(0))
def sub(a, rows, cols): return [[a[i][j] for j in cols] for i in rows]


def solve(a, b, check):
    """Bounded pivoted elimination with all right-hand sides together."""
    n = len(a); rhs = len(b[0]); x = [row[:]+r[:] for row, r in zip(a, b)]
    if not 1 <= n <= 150 or rhs > 66: raise ValueError('bounded compiler linear system')
    for col in range(n):
        check(); pivot = max(range(col, n), key=lambda i: abs(x[i][col]))
        if x[pivot][col] == 0: raise ValueError('singular constrained station system')
        x[col], x[pivot] = x[pivot], x[col]
        for row in range(col+1, n):
            if x[row][col] == 0: continue
            factor = x[row][col]/x[col][col]
            for j in range(col+1, n+rhs): x[row][j] -= factor*x[col][j]
            x[row][col] = D(0)
    out = zero(n, rhs)
    for row in range(n-1, -1, -1):
        for j in range(rhs):
            out[row][j] = (x[row][n+j]-sum((x[row][k]*out[k][j] for k in range(row+1, n)), D(0)))/x[row][row]
    return out


def dec(x):
    if type(x) is not float: raise ValueError('explicit binary64 material/quadrature input')
    value = D.from_float(x)
    if not value.is_finite(): raise ValueError('finite material/quadrature input')
    return value


def pair(values):
    if any(not x.is_finite() for x in values): raise ValueError('finite represented response required')
    high = [float(x) for x in values]
    if any(not D.from_float(h).is_finite() for h in high): raise ValueError('response exceeds binary64 range')
    low = [float(x-D.from_float(h)) for x, h in zip(values, high)]
    if any(x != 0 and h == 0 and l == 0 for x, h, l in zip(values, high, low)):
        raise ValueError('nonzero response underflows paired representation')
    return high, low


def history_row(row):
    """Explicit paired [z_high,z_low,p_high,p_low] history; never round to z,p."""
    if type(row) not in (list, tuple) or len(row) != 4: raise ValueError('paired station history shape')
    zh, zl, ph, pl = map(dec, row)
    z, p = zh+zl, ph+pl
    high, low = pair([z, p])
    if [high[0], low[0], high[1], low[1]] != list(row): raise ValueError('normalized paired station history')
    if p < abs(z): raise ValueError('admissible paired station history')
    return [z, p]


def cholesky(a):
    """Decimal positive-definiteness check; no mechanics imports."""
    n = len(a); lower = zero(n, n)
    for i in range(n):
        for j in range(i+1):
            x = a[i][j]-sum((lower[i][k]*lower[j][k] for k in range(j)), D(0))
            if i == j:
                if x <= 0: raise ValueError('nonpositive complementary Cholesky pivot')
                lower[i][j] = x.sqrt()
            else: lower[i][j] = x/lower[j][j]
    return lower


def frozen(value):
    return tuple(frozen(x) for x in value) if isinstance(value, (list, tuple)) else value


class StationResultantCell:
    __slots__ = ('digits', 'hardening', 'yield_force', 'weights', 'origins',
                 'stations', 'c0', 'g', 'k', 'q', 'force_map', 'inverse_section',
                 'direction', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False): raise AttributeError('captured cell is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, data, *, digits=80):
        if type(digits) is not int or not 60 <= digits <= 100: raise ValueError('bounded Decimal precision')
        self.digits = digits; data = deepcopy(data); started = monotonic()
        def check():
            if monotonic()-started > effective_deadline_seconds(60): raise ValueError('station compiler deadline')
        with localcontext() as ctx:
            ctx.prec = digits
            c = [[dec(x) for x in row] for row in data['elastic']]
            if len(c) != 6 or any(len(row) != 6 for row in c) or c != tr(c): raise ValueError('symmetric six-component elasticity')
            # Positive definiteness, not only invertibility, is required.
            cholesky(c)
            inv = solve(c, ident(6), check)
            direction = [dec(x) for x in data['direction']]
            if len(direction) != 6: raise ValueError('six-component plastic direction')
            self.hardening = dec(data['hardening']); self.yield_force = dec(data['yield_force'])
            if self.hardening <= 0 or self.yield_force <= 0: raise ValueError('strict positive hardening/yield')
            count = len(data['stations'])
            if not 2 <= count <= 48: raise ValueError('bounded station count')
            size = 3*count; variables = 18+count
            hn = zero(size, size); hm = zero(size, 12); mm = zero(12, 12)
            nt = zero(size, count); mt = zero(12, count); constraint = zero(6, size)
            self.weights = []; self.origins = []; self.stations = []
            for index, row in enumerate(data['stations']):
                check(); cell = row['cell']; w = dec(row['weight']); t = dec(row['t'])
                if type(cell) is not int or cell not in (0, 1) or w <= 0 or not 0 < t < 1:
                    raise ValueError('admitted cell/station data required')
                v = [[dec(x) for x in r] for r in row['v']]
                if len(v) != 3 or any(len(r) != 3 for r in v): raise ValueError('three by three station map')
                n = zero(3, 12)
                for i in range(3): n[i][6*cell+i] = 1-t; n[i][6*cell+3+i] = t
                cross = mul(sub(inv, range(3), range(3, 6)), n)
                moment = mul(mul(tr(n), sub(inv, range(3, 6), range(3, 6))), n)
                coupling = mv(tr(n), direction[3:])
                for i in range(3):
                    for j in range(3):
                        hn[3*index+i][3*index+j] = w*inv[i][j]
                        constraint[3*cell+i][3*index+j] = w*v[j][i]
                    for j in range(12): hm[3*index+i][j] = w*cross[i][j]
                    nt[3*index+i][index] = w*direction[i]
                for i in range(12):
                    mt[i][index] = w*coupling[i]
                    for j in range(12): mm[i][j] += w*moment[i][j]
                if len(row['origin']) != 2 or len(row.get('origin_low', [0., 0.])) != 2:
                    raise ValueError('station history shape')
                lows = row.get('origin_low', [0., 0.])
                old = history_row([row['origin'][0], lows[0], row['origin'][1], lows[1]])
                self.weights.append(w); self.origins.append(old)
                self.stations.append((cell, w, v, n))
            saddle = [row+[-x for x in column] for row, column in zip(hn, tr(constraint))]
            saddle += [row+[D(0)]*6 for row in constraint]
            rhs = [[D(0)]*6+[-x for x in row]+[-x for x in other] for row, other in zip(hm, nt)]
            rhs += [row+[D(0)]*(12+count) for row in ident(6)]
            mapping = solve(saddle, rhs, check); force_map = mapping[:size]
            # Substitute the stationary station forces into their complete
            # complementary quadratic, with m and tau retained as arguments.
            transform = force_map+zero(12+count, variables)
            for i in range(12): transform[size+i][6+i] = D(1)
            for i in range(count): transform[size+12+i][18+i] = D(1)
            full = [r+s+t for r, s, t in zip(hn, hm, nt)]
            full += [r+s+t for r, s, t in zip(tr(hm), mm, mt)]
            full += [r+s+[D(0)]*count for r, s in zip(tr(nt), tr(mt))]
            combined = mul(mul(tr(transform), full), transform)
            self.c0 = sub(combined, range(18), range(18))
            self.g = sub(combined, range(18, variables), range(18))
            self.k = [[-x for x in row] for row in sub(combined, range(18, variables), range(18, variables))]
            self.q = [row[:] for row in self.k]
            for i in range(count): self.q[i][i] += self.weights[i]*self.hardening
            cholesky(self.q)
            self.force_map = force_map; self.inverse_section = inv; self.direction = direction
            check()
            for name in ('weights', 'origins', 'stations', 'c0', 'g', 'k', 'q',
                         'force_map', 'inverse_section', 'direction'):
                setattr(self, name, frozen(getattr(self, name)))
            self._sealed = True

    def response(self, resultants, *, origins=None):
        started = monotonic()
        def check():
            if monotonic()-started > effective_deadline_seconds(60): raise ValueError('station conjugate deadline')
        with localcontext() as ctx:
            ctx.prec = self.digits
            p = [dec(x) for x in resultants]
            if len(p) != 18: raise ValueError('18 retained resultants')
            if origins is not None and (type(origins) not in (list, tuple) or len(origins) != len(self.origins)):
                raise ValueError('complete paired station history required')
            old_histories = self.origins if origins is None else [history_row(row) for row in origins]
            z0 = [x[0] for x in old_histories]; count = len(z0)
            radius = [w*(self.yield_force+self.hardening*old[1]) for w, old in zip(self.weights, old_histories)]
            drive = [x-y for x, y in zip(mv(self.g, p), mv(self.k, z0))]
            delta = [D(0)]*count; working = []; signs = [D(0)]*count
            threshold = D('1e-35')
            for iteration in range(33):
                check(); derivative = [x-y for x, y in zip(mv(self.q, delta), drive)]
                error = [abs(v+radius[i]*(1 if delta[i] > 0 else -1)) if delta[i] != 0
                         else max(abs(v)-radius[i], D(0)) for i, v in enumerate(derivative)]
                if max(error) <= threshold: break
                if iteration == 32: raise ValueError('station conjugate active-set limit')
                if not working or max(abs(derivative[i]+radius[i]*signs[i]) for i in working) <= threshold:
                    violations = [D(0) if i in working else max(abs(v)-radius[i], D(0)) for i, v in enumerate(derivative)]
                    release = max(range(count), key=lambda i: violations[i])
                    if violations[release] <= threshold: raise ValueError('unresolved station optimality')
                    working.append(release); working.sort(); signs[release] = D(-1 if derivative[release] > 0 else 1)
                answer = solve(sub(self.q, working, working), [[drive[i]-radius[i]*signs[i]] for i in working], check)
                target = [D(0)]*count
                for i, row in zip(working, answer): target[i] = row[0]
                crossing = [i for i in working if signs[i]*target[i] <= 0]
                if crossing:
                    ratios = {i: delta[i]/(delta[i]-target[i]) for i in crossing}
                    hit = min(crossing, key=lambda i: ratios[i]); alpha = ratios[hit]
                    delta = [x+alpha*(y-x) for x, y in zip(delta, target)]
                    delta[hit] = D(0); working.remove(hit); signs[hit] = D(0)
                else: delta = target
            plastic = [x+y for x, y in zip(z0, delta)]
            gradient = [x+y for x, y in zip(mv(self.c0, p), mv(tr(self.g), plastic))]
            hessian = [row[:] for row in self.c0]; active = [i for i, x in enumerate(delta) if x != 0]
            if active:
                ga = sub(self.g, active, range(18))
                term = mul(tr(ga), solve(sub(self.q, active, active), ga, check))
                hessian = [[x+y for x, y in zip(row, other)] for row, other in zip(hessian, term)]
            potential = dot(p, mv(self.c0, p))/2+dot(z0, mv(self.g, p))-dot(z0, mv(self.k, z0))/2
            potential += dot(delta, drive)-dot(delta, mv(self.q, delta))/2-dot(radius, [abs(x) for x in delta])
            stress = mv(self.force_map, p+plastic)
            stations = []; history = []
            for i, (cell, w, v, n) in enumerate(self.stations):
                s = stress[3*i:3*i+3]+mv(n, p[6:])
                elastic = mv(self.inverse_section, s)
                strain = [x+y*plastic[i] for x, y in zip(elastic, self.direction)]
                stations.append(dict(resultants=pair(s), elastic_strain=pair(elastic), strain=pair(strain),
                    plastic_coordinate=pair([plastic[i]]), accumulated=pair([old_histories[i][1]+abs(delta[i])])))
                high, low = pair([plastic[i], old_histories[i][1]+abs(delta[i])])
                history.append([high[0], low[0], high[1], low[1]])
            boundary = any(delta[i] == 0 and abs(derivative[i]) == radius[i] for i in range(count))
            return dict(potential=pair([potential]), gradient=pair(gradient),
                hessian=[pair(row) for row in hessian], increments=pair(delta),
                stations=stations, history=history, iterations=iteration, kkt_residual=str(max(error)),
                derivative_kind='SEMISMOOTH_ELASTIC_SELECTION' if boundary else 'CLASSICAL_SMOOTH_BRANCH',
                production_qualified=False)
