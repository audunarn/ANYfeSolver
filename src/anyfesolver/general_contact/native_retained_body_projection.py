"""Private exact endpoint-stationarity inventory for stored Q2 coordinates.

Each returned closed strip encloses a projection-feature breakpoint.  It is
not a sampled ownership certificate and must be charged before omission.
"""
from fractions import Fraction
from math import inf, nextafter


def charged_junction_omission(bulk, root_h, error, boundary_error, tolerances):
    """Charge a complete half-weighted physical30 integral, returning no work.

    Bulk already bounds the full represented directional integral, including
    its half weight. Root_h bounds all moving-owner Hessian in the band. Zero
    approximation has no returned-density error and needs no midpoint jet.
    """
    import numpy as np
    rows = tuple(np.asarray(row,dtype=float) for row in (bulk,error,boundary_error,tolerances))
    if (any(row.shape != (3,) or not np.all(np.isfinite(row)) or np.any(row < 0) for row in rows)
            or not np.isfinite(root_h) or root_h < 0):
        raise ValueError('native body junction omission requires finite nonnegative complete bounds')
    bulk,error,boundary_error,tolerances = rows
    def upper(value):
        result = float(value)
        return nextafter(result,inf) if Fraction(result) < value else result
    guard = Fraction(1)+Fraction(4096,2**52)
    bound = np.array([upper(Fraction(float(v))*guard+(Fraction(float(root_h)) if i==2 else 0))
                      for i,v in enumerate(bulk)])
    charged = np.array([upper(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
    boundary_charged = np.array([upper(Fraction(float(a))+Fraction(float(b))) for a,b in zip(boundary_error,bound)])
    if any(Fraction(float(v)) > Fraction(float(t))/2 for v,t in zip(boundary_charged,tolerances)) or np.any(charged > tolerances):
        raise MemoryError('native body junction omission reserve exhausted')
    return bound,charged,boundary_charged


def endpoint_projection_bands(source, target, left, right, *, maximum_cells):
    """Enclose all simple endpoint-phi roots; refuse tangency and plateaus.

    Phi(s)=(q_endpoint-p(s)).q'_endpoint is quadratic. Exact derivative
    critical points partition its domain into monotone intervals, so exact
    endpoint signs enumerate every root. Rational bisection terminates only
    once an outward binary64 bracket is at most four adjacent-float steps.
    Domain-edge roots retain a one-sided uncertainty strip.
    """
    rows = lambda data: tuple(tuple(Fraction(float(x)) for x in row) for row in data)
    p, q = rows(source), rows(target)
    linear = lambda a: tuple(-3*x+4*y-z for x,y,z in zip(*a))
    quadratic = lambda a: tuple(2*x-4*y+2*z for x,y,z in zip(*a))
    pb, pc, qb, qc = linear(p), quadratic(p), linear(q), quadratic(q)
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    low, high = Fraction(float(left)), Fraction(float(right))
    if not 0 <= low < high <= 1:
        raise ValueError('native body projection domain must lie in [0,1]')
    visited, bands = 0, []

    def outward(value, direction):
        result = float(value)
        if (direction < 0 and Fraction(result) > value) or (direction > 0 and Fraction(result) < value):
            result = nextafter(result, -inf if direction < 0 else inf)
        return result

    def strip(a,b):
        # One additional representable step covers an exactly representable
        # root too; no rounded scalar root is used as a zero-width cut.
        return (max(left,nextafter(outward(a,-1),-inf)),
                min(right,nextafter(outward(b,1),inf)))

    for endpoint in (0,1):
        point = q[0] if endpoint == 0 else q[2]
        tangent = qb if endpoint == 0 else tuple(x+2*y for x,y in zip(qb,qc))
        a = dot(tuple(x-y for x,y in zip(point,p[0])),tangent)
        b, c = -dot(pb,tangent), -dot(pc,tangent)
        value = lambda s: a+s*(b+s*c)
        if a == b == c == 0:
            raise ValueError('native body endpoint projection plateau lacks a classical tangent')
        stations = [low,high]
        if c:
            critical = -b/(2*c)
            if low <= critical <= high:
                if value(critical) == 0:
                    raise ValueError('native body tangent endpoint projection root unresolved')
                if low < critical < high:
                    stations.insert(1,critical)
        for station in stations:
            if value(station) == 0:
                bands.append(strip(station,station))
        for begin,stop in zip(stations[:-1],stations[1:]):
            first,last = value(begin),value(stop)
            if first*last >= 0:
                continue
            for _ in range(128):
                visited += 1
                if visited > maximum_cells:
                    raise MemoryError('native body endpoint projection inventory cell budget exhausted')
                band = strip(begin,stop)
                edge = band[0]
                for _ in range(4):
                    edge = nextafter(edge,inf)
                if edge >= band[1]:
                    bands.append(band)
                    break
                middle = (begin+stop)/2
                sample = value(middle)
                if sample == 0:
                    bands.append(strip(middle,middle))
                    break
                if first*sample > 0:
                    begin,first = middle,sample
                else:
                    stop = middle
            else:
                raise MemoryError('native body endpoint projection root precision exhausted')
    merged = []
    for begin,stop in sorted(bands):
        if begin >= stop:
            raise MemoryError('native body endpoint projection strip precision exhausted')
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0],max(stop,merged[-1][1]))
        else:
            merged.append((begin,stop))
    return tuple(merged),visited


def strict_projection_station(source, target, station):
    """Select constrained projection by exact endpoint KKT, never norm ties.

    Exact f_tt=(q')^2+(q-p).q'' is quadratic in t. A strictly positive
    minimum proves the sole stationary point is the global constrained min.
    The returned interior station is approximate; the private factored-gap
    residual certificate still accounts for its distance surplus.
    """
    from scipy.optimize import brentq
    rows = lambda data: tuple(tuple(Fraction(float(x)) for x in row) for row in data)
    p,q = rows(source),rows(target)
    lin = lambda r: tuple(-3*a+4*b-c for a,b,c in zip(*r))
    quad = lambda r: tuple(2*a-4*b+2*c for a,b,c in zip(*r))
    pb,pc,qb,qc = lin(p),quad(p),lin(q),quad(q)
    s = Fraction(float(station))
    point = tuple(a+s*(b+s*c) for a,b,c in zip(p[0],pb,pc))
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    offset = tuple(a-b for a,b in zip(q[0],point))
    coefficients = (dot(qb,qb)+2*dot(offset,qc),6*dot(qb,qc),6*dot(qc,qc))
    a,b,c = coefficients
    critical = max(Fraction(0),min(Fraction(1),-b/(2*c))) if c else Fraction(0)
    if min(a,a+b+c,a+critical*(b+critical*c)) <= 0:
        raise ValueError('native body private projection lacks exact strict convexity')
    def phi(t):
        t = Fraction(float(t))
        return dot(tuple(a+t*(b+t*c) for a,b,c in zip(offset,qb,qc)),
                   tuple(b+2*t*c for b,c in zip(qb,qc)))
    first,last = phi(0),phi(1)
    if first >= 0:
        return 0.
    if last <= 0:
        return 1.
    result = float(brentq(lambda t:float(phi(t)),0.,1.,xtol=5e-324,
                         rtol=8.881784197001252e-16,maxiter=64))
    if not 0 < result < 1:
        raise MemoryError('native body interior projection station precision exhausted')
    return result
