"""Independent literal seed-gap reference for the first captured Q4/S3 path.

Scope: D1_SEED_GAP_DISCRIMINATOR_DEV420_V1 (Mistral-owned discriminator).

This module is deliberately self-contained: stdlib only, no ``anyfesolver``
import, and no production gap/witness kernel is called for any oracle
value. It re-derives the physical source-normal gap of the captured pair
path from the literal endpoint geometry and certifies it with exact
rational interval arithmetic (``fractions.Fraction``; binary64 inputs are
exact dyadic rationals, subdivision happens at dyadic points, square
roots are enclosed with exact integer square roots), cross-checked by an
independently written directed-floating-point interval arithmetic
(``math.nextafter`` outward rounding).

Gap definition (derived independently from the literal Q4/S3 geometry;
the same mechanical contract the captured production witness serves):

    t in [0,1] path parameter, (u,v) in [0,1]^2 source skin parameters
    A_k(t) = (1-t) a0[k] + t a1[k]        affine physical-skin path (Q4)
    c_j(t) = (1-t) b0[j] + t b1[j]        affine physical-skin path (S3)
    s(t,u,v) = sum_k w_k(u,v) A_k(t)      bilinear Q4 skin
    T = (c1-c0) x (c2-c0)                 target normal
    P = (A1-A0) x (A3-A0)                 parent corner-0 source normal
    N = (s-c0) . T                        target-plane height
    r = N/(P.T)                           foot radius along P
    y = s - r P                           projected foot (on target plane)
    Ns = dX/du x dX/dv                    actual parametric Q4 normal
    g = r (P.Ns)/|Ns| = (s-y).Ns/|Ns|     physical source-normal gap
    dX/du = (1-v)(A1-A0) + v(A2-A3)
    dX/dv = (1-u)(A3-A0) + u(A2-A1)

Projected-overlap constraint (foot inside the S3 triangle):

    lambda1 = ((y-c0) x e2).T/|T|^2 >= 0
    lambda2 = (e1 x (y-c0)).T/|T|^2 >= 0
    lambda1 + lambda2 <= 1

Endpoint error boxes: every saved corner is enclosed by
``[saved - error, saved + error]``, blended affinely along the path; every
arithmetic operation propagates these enclosures.

Enclosure organization: the bilinear skin point ``s(t,u,v)`` is trilinear
in ``(t,u,v)`` and the skin derivatives are bilinear in ``(t,v)`` and
``(t,u)``, so over any dyadic box their coordinate ranges are enclosed by
the convex vertex form: the hull of the eight (respectively four) vertex
values, each vertex blending the corner enclosures at that vertex's own
path parameter with the exact vertex weights. This retains the
partition-of-unity correlation of the weights, which a natural interval
extension of the weighted sums loses. All remaining quantities (target
normal, parent normal, heights, barycentrics) are enclosed by interval
arithmetic over the corner boxes. Chart-uncertain boxes are refined inside
the bounded branch-and-bound budget; a box is reported as unresolved only
when that budget is exhausted.

Decision semantics of the captured contract: a violating point requires
``g < -gap_tolerance`` with the foot inside the target triangle; a safety
certificate requires ``min g >= -gap_tolerance`` over the whole path
domain (the foot-inside subset of the domain is covered by a whole-domain
certificate). A result that remains inconclusive reports the exact
remaining dyadic region. The certified scope is the single literal pair
path only; no statement is made about any other face, pose or the whole
solver path.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from fractions import Fraction

INPUT_SOURCE = (
    "C:/ANYworkspaceWork/contact_mistral_seed_gap_delivery_20261010/"
    "LITERAL_PATH_ZERO_INPUT.json"
)
INPUT_SHA256 = (
    "238a19cc6ac72445d97de8a44e3f44b75eb606bb42fb548ba18bfe7866952ec7"
)

CAPTURE_SOURCE = (
    "C:/ANYworkspaceWork/contact_mistral_admission_globalization_20261010/"
    "witness-export-repair-dev45-v1/attempt-01-witness-export-repair-v1/"
    "temp/initial-seed-gap-witness-capture-v1.json"
)
CAPTURE_SHA256 = (
    "f8e7dee46a3135c621492364f5f727d0eb34d57873afeb7146606c4bda31a114"
)

LITERAL_PATH_ZERO = {
    "input_source": 'C:\\ANYworkspaceWork\\contact_mistral_admission_globalization_20261010\\witness-export-repair-dev45-v1\\attempt-01-witness-export-repair-v1\\temp\\initial-seed-gap-witness-capture-v1.json',
    "input_sha256": 'f8e7dee46a3135c621492364f5f727d0eb34d57873afeb7146606c4bda31a114',
    "path_index": 0,
    "path": {
        "source_start_m": [[0.1000024999999875, 0.1, 0.014965000325099998], [0.6000024999999874, 0.1, 0.0150150003251], [0.6000024999999874, 0.6, 0.0150150003251], [0.1000024999999875, 0.6, 0.014965000325099998]],
        "source_end_m": [[0.10000244217862628, 0.09999994217863913, 0.014965000319317861], [0.6000025000334619, 0.09999423713764663, 0.0150150003251], [0.6000025058384252, 0.6, 0.014955928700950616], [0.09999673713769158, 0.6000000000000001, 0.01496499974881377]],
        "target_start_m": [[-1.4999999925000002e-06, -0.1, 0.014934999925000001], [1.2999985000000076, -0.1, 0.015064999925000003], [-1.4999999925000002e-06, 1.2, 0.014934999925000001]],
        "target_end_m": [[-1.4999999925000002e-06, -0.1, 0.014934999925000001], [1.2999997939053893, -0.1000004082786782, 0.015014292254476938], [-1.4999999925000002e-06, 1.2, 0.014934999925000001]],
        "endpoint_errors": [[[5.551253901003176e-17, 7.771561172376109e-17, 8.307243962224499e-18], [3.330682951663213e-16, 7.771561172376109e-17, 8.334999537840128e-18], [3.330682951663213e-16, 4.662936703425666e-16, 8.334999537840128e-18], [5.551253901003176e-17, 4.662936703425666e-16, 8.307243962224499e-18]], [[5.551257110733504e-17, 7.77156566599854e-17, 8.307243965434229e-18], [3.3306829518490334e-16, 7.772009036749181e-17, 8.334999537840128e-18], [3.3306829840730533e-16, 4.662936703425666e-16, 8.367790876456453e-18], [5.55157380412361e-17, 4.662936703425667e-16, 8.307244282127618e-18]], [[8.326672643055324e-22, 7.771561172376109e-17, 8.290590394755007e-18], [7.216441333390886e-16, 7.771561172376109e-17, 8.362754891355642e-18], [8.326672643055324e-22, 9.325873406851331e-16, 8.290590394755007e-18]], [[8.326672643055324e-22, 7.771561172376109e-17, 8.290590394755007e-18], [7.216448516008618e-16, 7.77159290201453e-17, 8.390903389015188e-18], [8.326672643055324e-22, 9.325873406851331e-16, 8.290590394755007e-18]]],
        "gap_tolerance_m": 5e-05,
    },
}

_F0 = Fraction(0)
_F1 = Fraction(1)


# ---------------------------------------------------------------------------
# Literal record loading and artifact binding
# ---------------------------------------------------------------------------

def load_literal_record():
    """Return the literal record, verifying the artifact when available.

    The embedded record is the canonical fallback; when the delivery file is
    present its sha256 must match and every embedded value must equal the
    artifact value exactly (binary64 equality), otherwise the loader refuses.
    """
    record = json.loads(json.dumps(LITERAL_PATH_ZERO))
    record["artifact_verified"] = False
    record["original_artifact_verified"] = False
    if os.path.isfile(INPUT_SOURCE):
        with open(INPUT_SOURCE, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != INPUT_SHA256:
            raise ValueError("literal artifact sha256 mismatch: " + digest)
        with open(INPUT_SOURCE, encoding="utf-8") as handle:
            source = json.load(handle)
        embedded = record["path"]
        artifact = source["path"]
        for key in ("source_start_m", "source_end_m", "target_start_m",
                    "target_end_m", "endpoint_errors"):
            if embedded[key] != artifact[key]:
                raise ValueError("embedded literal differs from artifact: " + key)
        if float(embedded["gap_tolerance_m"]) != float(artifact["gap_tolerance_m"]):
            raise ValueError("embedded literal differs from artifact: tolerance")
        if record["path_index"] != source["path_index"]:
            raise ValueError("embedded literal differs from artifact: path_index")
        record["artifact_verified"] = True
        original = record.get("input_source")
        if original and os.path.isfile(original):
            with open(original, "rb") as handle:
                original_digest = hashlib.sha256(handle.read()).hexdigest()
            if original_digest != record["input_sha256"]:
                raise ValueError("original capture artifact sha256 mismatch: "
                                 + original_digest)
            record["original_artifact_verified"] = True
    return record


def load_capture_records():
    """Load and verify the ten-record literal capture artifact.

    Returns one parameterized record per captured witness call, in capture
    order: the two endpoint witnesses first, then the eight path witnesses.
    Endpoint records carry degenerate poses (start == end) and no saved
    error boxes; the production endpoint gate evaluates them with zero
    endpoint errors, so explicit zero boxes are stored, never null. Every
    array is passed through unchanged (binary64 equality with the artifact);
    only the sha256 binding and record assembly happen here.
    """
    with open(CAPTURE_SOURCE, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != CAPTURE_SHA256:
        raise ValueError("capture artifact sha256 mismatch: " + digest)
    with open(CAPTURE_SOURCE, encoding="utf-8") as handle:
        capture = json.load(handle)
    tolerance = float(capture["fixture_gap_tolerance_m"])
    records = []
    for kind, key in (("endpoint", "endpoint_witnesses"),
                      ("path", "path_witnesses")):
        for index, witness in enumerate(capture[key]):
            if float(witness["gap_tolerance_m"]) != tolerance:
                raise ValueError("capture record tolerance disagrees")
            errors = witness.get("endpoint_errors")
            if errors is None:
                errors = ([[[0.0]*3 for _ in witness["source_start_m"]],]*2
                          + [[[0.0]*3 for _ in witness["target_start_m"]],]*2)
            if len(errors) != 4:
                raise ValueError("capture record needs four error boxes")
            records.append({
                "kind": kind,
                "record_index": index,
                "input_source": CAPTURE_SOURCE,
                "input_sha256": CAPTURE_SHA256,
                "path": {
                    "source_start_m": witness["source_start_m"],
                    "source_end_m": witness["source_end_m"],
                    "target_start_m": witness["target_start_m"],
                    "target_end_m": witness["target_end_m"],
                    "endpoint_errors": errors,
                    "gap_tolerance_m": tolerance,
                },
            })
    if len(records) != (int(capture["endpoint_witness_calls"])
                        + int(capture["path_witness_calls"])):
        raise ValueError("capture record counts disagree with witness calls")
    return records


class PairPathData:
    """Exact rational corner positions, error boxes and tolerance."""

    def __init__(self, record):
        path = record["path"] if "path" in record else record
        def corners(key):
            return [[Fraction(float(value)) for value in row]
                    for row in path[key]]
        self.a0 = corners("source_start_m")
        self.a1 = corners("source_end_m")
        self.b0 = corners("target_start_m")
        self.b1 = corners("target_end_m")
        errors = path["endpoint_errors"]
        self.e0 = [[Fraction(float(value)) for value in row] for row in errors[0]]
        self.e1 = [[Fraction(float(value)) for value in row] for row in errors[1]]
        self.e2 = [[Fraction(float(value)) for value in row] for row in errors[2]]
        self.e3 = [[Fraction(float(value)) for value in row] for row in errors[3]]
        self.tolerance = Fraction(float(path["gap_tolerance_m"]))


# ---------------------------------------------------------------------------
# Exact rational interval arithmetic (no rounding anywhere)
# ---------------------------------------------------------------------------

def _sqrt_rational(value):
    """Exact enclosure of the square root of a positive rational."""
    numerator, denominator = value.numerator, value.denominator
    root = math.isqrt(numerator * denominator)
    return Fraction(root, denominator), Fraction(root + 1, denominator)


def i_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def i_sub(a, b):
    return (a[0] - b[1], a[1] - b[0])


def i_mul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(products), max(products))


def i_div(a, b):
    """Directed division; the caller must guarantee b[0] > 0."""
    if b[0] <= 0:
        raise ValueError("directed division needs a strictly positive denominator")
    quotients = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    return (min(quotients), max(quotients))


def i_sqrt(a):
    if a[0] <= 0:
        raise ValueError("directed square root needs a strictly positive interval")
    low = _sqrt_rational(a[0])
    high = _sqrt_rational(a[1])
    return (low[0], high[1])


def v_add(a, b):
    return tuple(i_add(a[k], b[k]) for k in range(3))


def v_sub(a, b):
    return tuple(i_sub(a[k], b[k]) for k in range(3))


def v_scale(a, scalar):
    return tuple(i_mul(a[k], scalar) for k in range(3))


def v_dot(a, b):
    products = [i_mul(a[k], b[k]) for k in range(3)]
    return i_add(i_add(products[0], products[1]), products[2])


def v_cross(a, b):
    return (i_sub(i_mul(a[1], b[2]), i_mul(a[2], b[1])),
            i_sub(i_mul(a[2], b[0]), i_mul(a[0], b[2])),
            i_sub(i_mul(a[0], b[1]), i_mul(a[1], b[0])))


def _corner_boxes(start, end, error_start, error_end, t0, t1):
    """Enclose every corner position and its error box over t in [t0, t1]."""
    boxes = []
    for corner in range(len(start)):
        components = []
        for dimension in range(3):
            first = (1 - t0) * start[corner][dimension] + t0 * end[corner][dimension]
            last = (1 - t1) * start[corner][dimension] + t1 * end[corner][dimension]
            error_first = (1 - t0) * error_start[corner][dimension] + t0 * error_end[corner][dimension]
            error_last = (1 - t1) * error_start[corner][dimension] + t1 * error_end[corner][dimension]
            error = max(error_first, error_last)
            low = min(first, last) - error
            high = max(first, last) + error
            components.append((low, high))
        boxes.append(components)
    return boxes


def _corner_at(start, end, error_start, error_end, t):
    """Exact corner-component enclosure at one rational path parameter."""
    center = (1 - t) * start + t * end
    error = (1 - t) * error_start + t * error_end
    return (center - error, center + error)


def _vertex_point(data, t0, t1, u0, u1, v0, v1):
    """Multilinear vertex enclosure of the skin point over one box.

    ``s(t,u,v) = sum_k w_k(u,v) A_k(t)`` is trilinear in ``(t,u,v)``: over a
    box it is the convex trilinear combination of its eight vertex values,
    and each vertex value is enclosed by blending the corner enclosures at
    that vertex's own path parameter with the exact vertex weights. The
    per-component hull of the eight vertex intervals encloses ``s`` over
    the whole box and retains the partition-of-unity correlation of the
    weights that a natural interval extension of the weighted sum loses.
    """
    low = [None, None, None]
    high = [None, None, None]
    for t in (t0, t1):
        corner = [[_corner_at(data.a0[k][d], data.a1[k][d],
                              data.e0[k][d], data.e1[k][d], t)
                   for d in range(3)] for k in range(4)]
        for u in (u0, u1):
            for v in (v0, v1):
                weights = ((1 - u) * (1 - v), u * (1 - v),
                           u * v, (1 - u) * v)
                for d in range(3):
                    acc = i_mul((weights[0], weights[0]), corner[0][d])
                    for k in (1, 2, 3):
                        acc = i_add(acc, i_mul((weights[k], weights[k]),
                                               corner[k][d]))
                    if low[d] is None or acc[0] < low[d]:
                        low[d] = acc[0]
                    if high[d] is None or acc[1] > high[d]:
                        high[d] = acc[1]
    return tuple((low[d], high[d]) for d in range(3))


def _vertex_derivative(data, t0, t1, p0, p1, first):
    """Bilinear vertex enclosure of dX/du (first) or dX/dv over one box.

    ``dX/du = (1-v)(A1-A0) + v(A2-A3)`` is bilinear in ``(t,v)`` and
    ``dX/dv = (1-u)(A3-A0) + u(A2-A1)`` is bilinear in ``(t,u)``; over the
    box each is the convex combination of its four vertex values, so the
    hull of the four vertex intervals (corner enclosures at the vertex's
    own path parameter, exact vertex weights) encloses the derivative over
    the whole box.
    """
    edges = ((1, 0), (2, 3)) if first else ((3, 0), (2, 1))
    low = [None, None, None]
    high = [None, None, None]
    for t in (t0, t1):
        corner = [[_corner_at(data.a0[k][d], data.a1[k][d],
                              data.e0[k][d], data.e1[k][d], t)
                   for d in range(3)] for k in range(4)]
        for p in (p0, p1):
            for d in range(3):
                acc = i_add(i_mul((1 - p, 1 - p),
                                   i_sub(corner[edges[0][0]][d],
                                         corner[edges[0][1]][d])),
                            i_mul((p, p),
                                  i_sub(corner[edges[1][0]][d],
                                        corner[edges[1][1]][d])))
                if low[d] is None or acc[0] < low[d]:
                    low[d] = acc[0]
                if high[d] is None or acc[1] > high[d]:
                    high[d] = acc[1]
    return tuple((low[d], high[d]) for d in range(3))


def evaluate_box_exact(data, t_box, u_box, v_box):
    """Exact rational enclosure of the contract gap over one dyadic box."""
    t0, t1 = t_box
    u0, u1 = u_box
    v0, v1 = v_box
    source = _corner_boxes(data.a0, data.a1, data.e0, data.e1, t0, t1)
    target = _corner_boxes(data.b0, data.b1, data.e2, data.e3, t0, t1)
    point = _vertex_point(data, t0, t1, u0, u1, v0, v1)
    edge1 = v_sub(target[1], target[0])
    edge2 = v_sub(target[2], target[0])
    target_normal = v_cross(edge1, edge2)
    parent = v_cross(v_sub(source[1], source[0]), v_sub(source[3], source[0]))
    height = v_dot(v_sub(point, target[0]), target_normal)
    projected = v_dot(parent, target_normal)
    du = _vertex_derivative(data, t0, t1, v0, v1, True)
    dv = _vertex_derivative(data, t0, t1, u0, u1, False)
    actual = v_cross(du, dv)
    parent_actual = v_dot(parent, actual)
    target_squared = v_dot(target_normal, target_normal)
    actual_squared = v_dot(actual, actual)
    chart_ok = (projected[0] > 0 and parent_actual[0] > 0
                and actual_squared[0] > 0 and target_squared[0] > 0)
    if not chart_ok:
        return {"chart_ok": False}
    actual_norm = i_sqrt(actual_squared)
    target_norm = i_sqrt(target_squared)
    radius = i_div(height, projected)
    gap = i_div(i_mul(radius, parent_actual), actual_norm)
    foot = v_sub(point, v_scale(parent, radius))
    lambda1 = i_div(v_dot(v_cross(v_sub(foot, target[0]), edge2),
                          target_normal), target_squared)
    lambda2 = i_div(v_dot(v_cross(edge1, v_sub(foot, target[0])),
                          target_normal), target_squared)
    lambda_sum = i_add(lambda1, lambda2)
    if lambda1[0] >= 0 and lambda2[0] >= 0 and lambda_sum[1] <= 1:
        feasibility = "inside"
    elif lambda1[1] < 0 or lambda2[1] < 0 or lambda_sum[0] > 1:
        feasibility = "outside"
    else:
        feasibility = "uncertain"
    return {
        "chart_ok": True,
        "gap": gap,
        "point": point,
        "du": du,
        "dv": dv,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda_sum": lambda_sum,
        "feasibility": feasibility,
        "height": height,
        "projected": projected,
        "parent_actual": parent_actual,
        "actual_norm": actual_norm,
        "target_norm": target_norm,
        "plane_height": i_div(height, target_norm),
    }


def exact_point(data, t, u, v):
    """Exact quantities at one rational (t, u, v) of the saved geometry.

    Square roots are enclosed; every other quantity is an exact rational.
    """
    t = Fraction(t)
    u = Fraction(u)
    v = Fraction(v)
    def lerped(start, end):
        return [(1 - t) * start[k] + t * end[k] for k in range(3)]
    source = [lerped(data.a0[k], data.a1[k]) for k in range(4)]
    target = [lerped(data.b0[j], data.b1[j]) for j in range(3)]
    weights = ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
    point = [sum(weights[k] * source[k][d] for k in range(4)) for d in range(3)]
    edge1 = [target[1][d] - target[0][d] for d in range(3)]
    edge2 = [target[2][d] - target[0][d] for d in range(3)]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    target_normal = cross(edge1, edge2)
    parent = cross([source[1][d] - source[0][d] for d in range(3)],
                   [source[3][d] - source[0][d] for d in range(3)])
    offset = [point[d] - target[0][d] for d in range(3)]
    height = dot(offset, target_normal)
    projected = dot(parent, target_normal)
    du = [(1 - v) * (source[1][d] - source[0][d]) + v * (source[2][d] - source[3][d])
          for d in range(3)]
    dv = [(1 - u) * (source[3][d] - source[0][d]) + u * (source[2][d] - source[1][d])
          for d in range(3)]
    actual = cross(du, dv)
    parent_actual = dot(parent, actual)
    target_squared = dot(target_normal, target_normal)
    actual_squared = dot(actual, actual)
    if projected <= 0 or parent_actual <= 0 or actual_squared <= 0 or target_squared <= 0:
        raise ValueError("point chart is unresolved")
    actual_norm = i_sqrt((actual_squared, actual_squared))
    target_norm = i_sqrt((target_squared, target_squared))
    radius = height / projected
    scaled = (radius * parent_actual / actual_norm[0],
              radius * parent_actual / actual_norm[1])
    gap = (min(scaled), max(scaled))
    foot = [point[d] - radius * parent[d] for d in range(3)]
    foot_offset = [foot[d] - target[0][d] for d in range(3)]
    lambda1 = dot(cross(foot_offset, edge2), target_normal) / target_squared
    lambda2 = dot(cross(edge1, foot_offset), target_normal) / target_squared
    plane = (height / target_norm[0], height / target_norm[1])
    local = (height * actual_norm[0] / dot(actual, target_normal),
              height * actual_norm[1] / dot(actual, target_normal))
    return {
        "t": t, "u": u, "v": v,
        "point": point,
        "height": height,
        "projected": projected,
        "parent_actual": parent_actual,
        "actual_squared": actual_squared,
        "target_squared": target_squared,
        "radius": radius,
        "gap": gap,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda_sum": lambda1 + lambda2,
        "foot": foot,
        "plane_height": (min(plane), max(plane)),
        "local_normal_gap": (min(local), max(local)),
    }


def _point_witness_in_box(data, box):
    """Exact violating point inside a certified box, or None."""
    (t0, t1), (u0, u1), (v0, v1) = box
    candidates = [((t0 + t1) / 2, (u0 + u1) / 2, (v0 + v1) / 2)]
    for fraction in (1, 3):
        t = t0 + (t1 - t0) * fraction / 4
        for u in (u0 + (u1 - u0) / 4, u0 + 3 * (u1 - u0) / 4):
            for v in (v0 + (v1 - v0) / 4, v0 + 3 * (v1 - v0) / 4):
                candidates.append((t, u, v))
    for t, u, v in candidates:
        enclosure = evaluate_box_exact(data, (t, t), (u, u), (v, v))
        if (enclosure["chart_ok"]
                and enclosure["feasibility"] == "inside"
                and enclosure["gap"][1] < -data.tolerance):
            return {"t": t, "u": u, "v": v, "enclosure": enclosure}
    return None


def _children(box):
    """The eight dyadic sub-boxes of one box (dyadic midpoints)."""
    (t0, t1), (u0, u1), (v0, v1) = box
    tm = (t0 + t1) / 2
    um = (u0 + u1) / 2
    vm = (v0 + v1) / 2
    return [(tb, ub, vb)
            for tb in ((t0, tm), (tm, t1))
            for ub in ((u0, um), (um, u1))
            for vb in ((v0, vm), (vm, v1))]


def certify_exact(data, tolerance=None, max_depth=10, max_boxes=6000):
    """Exact rational branch-and-bound over the whole path domain.

    Returns a machine-readable certificate for min g over the foot-inside
    subset of [0,1]^3 (whole-domain boxes cover that subset; boxes whose
    foot is provably outside the target triangle are excluded).
    Chart-uncertain boxes are refined within the same bounded budget; a
    box is reported as unresolved only when the budget is exhausted.
    """
    tolerance = data.tolerance if tolerance is None else Fraction(tolerance)
    root = ((_F0, _F1), (_F0, _F1), (_F0, _F1))
    stack = [(root, 0)]
    visited = 0
    safe_lower = None
    remaining = []
    unresolved = []
    violation = None
    while stack:
        box, depth = stack.pop()
        visited += 1
        enclosure = evaluate_box_exact(data, *box)
        exhausted = depth >= max_depth or visited >= max_boxes
        if not enclosure["chart_ok"]:
            if exhausted:
                unresolved.append({"box": box, "depth": depth})
            else:
                for child in _children(box):
                    stack.append((child, depth + 1))
            continue
        if enclosure["feasibility"] == "outside":
            continue
        if enclosure["gap"][1] < -tolerance and enclosure["feasibility"] == "inside":
            violation = {
                "box": box,
                "gap": enclosure["gap"],
                "point_witness": _point_witness_in_box(data, box),
            }
            break
        if enclosure["gap"][0] >= -tolerance:
            value = enclosure["gap"][0]
            safe_lower = value if safe_lower is None else min(safe_lower, value)
            continue
        if exhausted:
            remaining.append({"box": box, "depth": depth,
                              "gap": enclosure["gap"],
                              "feasibility": enclosure["feasibility"]})
            continue
        for child in _children(box):
            stack.append((child, depth + 1))
    if violation is not None:
        status = "violation_proven"
        lower = None
    elif remaining or unresolved:
        status = "inconclusive"
        lower = safe_lower
    else:
        status = "safe_within_tolerance"
        lower = safe_lower
    return {
        "status": status,
        "visited_boxes": visited,
        "min_gap_lower_m": lower,
        "remaining_boxes": remaining,
        "unresolved_chart_boxes": unresolved,
        "violation": violation,
    }


# ---------------------------------------------------------------------------
# Independently written directed-floating-point interval arithmetic
# ---------------------------------------------------------------------------

def _down(value):
    return math.nextafter(value, -math.inf)


def _up(value):
    return math.nextafter(value, math.inf)


def f_add(a, b):
    return (_down(a[0] + b[0]), _up(a[1] + b[1]))


def f_sub(a, b):
    return (_down(a[0] - b[1]), _up(a[1] - b[0]))


def f_mul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (_down(min(products)), _up(max(products)))


def f_div(a, b):
    if not (b[0] > 0.0):
        raise ValueError("directed division needs a strictly positive denominator")
    quotients = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    return (_down(min(quotients)), _up(max(quotients)))


def f_sqrt(a):
    if not (a[0] > 0.0):
        raise ValueError("directed square root needs a strictly positive interval")
    return (_down(math.sqrt(a[0])), _up(math.sqrt(a[1])))


def f_point(value):
    """Rigorous float interval holding an exact rational input value."""
    nearest = float(value)
    if Fraction(nearest) == value:
        return (nearest, nearest)
    return (_down(nearest), _up(nearest))


def fv_add(a, b):
    return tuple(f_add(a[k], b[k]) for k in range(3))


def fv_sub(a, b):
    return tuple(f_sub(a[k], b[k]) for k in range(3))


def fv_scale(a, scalar):
    return tuple(f_mul(a[k], scalar) for k in range(3))


def fv_dot(a, b):
    products = [f_mul(a[k], b[k]) for k in range(3)]
    return f_add(f_add(products[0], products[1]), products[2])


def fv_cross(a, b):
    return (f_sub(f_mul(a[1], b[2]), f_mul(a[2], b[1])),
            f_sub(f_mul(a[2], b[0]), f_mul(a[0], b[2])),
            f_sub(f_mul(a[0], b[1]), f_mul(a[1], b[0])))


def _f_corner_boxes(start, end, error_start, error_end, t0, t1):
    one = (1.0, 1.0)
    one_minus_t0 = f_sub(one, f_point(t0))
    one_minus_t1 = f_sub(one, f_point(t1))
    t0i = f_point(t0)
    t1i = f_point(t1)
    boxes = []
    for corner in range(len(start)):
        components = []
        for dimension in range(3):
            first = f_add(f_mul(one_minus_t0, f_point(start[corner][dimension])),
                          f_mul(t0i, f_point(end[corner][dimension])))
            last = f_add(f_mul(one_minus_t1, f_point(start[corner][dimension])),
                         f_mul(t1i, f_point(end[corner][dimension])))
            error_first = f_add(f_mul(one_minus_t0, f_point(error_start[corner][dimension])),
                                f_mul(t0i, f_point(error_end[corner][dimension])))
            error_last = f_add(f_mul(one_minus_t1, f_point(error_start[corner][dimension])),
                               f_mul(t1i, f_point(error_end[corner][dimension])))
            error = max(error_first[1], error_last[1])
            components.append((_down(min(first[0], last[0]) - error),
                              _up(max(first[1], last[1]) + error)))
        boxes.append(components)
    return boxes


def _f_corner_at(start, end, error_start, error_end, t):
    """Directed-float corner-component enclosure at one rational t."""
    one = (1.0, 1.0)
    ti = f_point(t)
    one_minus_t = f_sub(one, ti)
    center = f_add(f_mul(one_minus_t, f_point(start)),
                   f_mul(ti, f_point(end)))
    error = f_add(f_mul(one_minus_t, f_point(error_start)),
                  f_mul(ti, f_point(error_end)))
    return (f_sub(center, error)[0], f_add(center, error)[1])


def _f_vertex_point(data, t0, t1, u0, u1, v0, v1):
    """Directed-float mirror of :func:`_vertex_point`."""
    one = (1.0, 1.0)
    low = [None, None, None]
    high = [None, None, None]
    for t in (t0, t1):
        corner = [[_f_corner_at(data.a0[k][d], data.a1[k][d],
                                data.e0[k][d], data.e1[k][d], t)
                   for d in range(3)] for k in range(4)]
        for u in (u0, u1):
            for v in (v0, v1):
                one_u = f_sub(one, f_point(u))
                one_v = f_sub(one, f_point(v))
                uu = f_point(u)
                vv = f_point(v)
                weights = (f_mul(one_u, one_v), f_mul(uu, one_v),
                           f_mul(uu, vv), f_mul(one_u, vv))
                for d in range(3):
                    acc = f_mul(weights[0], corner[0][d])
                    for k in (1, 2, 3):
                        acc = f_add(acc, f_mul(weights[k], corner[k][d]))
                    if low[d] is None or acc[0] < low[d]:
                        low[d] = acc[0]
                    if high[d] is None or acc[1] > high[d]:
                        high[d] = acc[1]
    return tuple((low[d], high[d]) for d in range(3))


def _f_vertex_derivative(data, t0, t1, p0, p1, first):
    """Directed-float mirror of :func:`_vertex_derivative`."""
    edges = ((1, 0), (2, 3)) if first else ((3, 0), (2, 1))
    one = (1.0, 1.0)
    low = [None, None, None]
    high = [None, None, None]
    for t in (t0, t1):
        corner = [[_f_corner_at(data.a0[k][d], data.a1[k][d],
                                data.e0[k][d], data.e1[k][d], t)
                   for d in range(3)] for k in range(4)]
        for p in (p0, p1):
            pi = f_point(p)
            one_p = f_sub(one, pi)
            for d in range(3):
                acc = f_add(f_mul(one_p,
                                  f_sub(corner[edges[0][0]][d],
                                        corner[edges[0][1]][d])),
                            f_mul(pi,
                                  f_sub(corner[edges[1][0]][d],
                                        corner[edges[1][1]][d])))
                if low[d] is None or acc[0] < low[d]:
                    low[d] = acc[0]
                if high[d] is None or acc[1] > high[d]:
                    high[d] = acc[1]
    return tuple((low[d], high[d]) for d in range(3))


def evaluate_box_float(data, t_box, u_box, v_box):
    """Directed-floating-point enclosure of the same contract gap."""
    source = _f_corner_boxes(data.a0, data.a1, data.e0, data.e1, t_box[0], t_box[1])
    target = _f_corner_boxes(data.b0, data.b1, data.e2, data.e3, t_box[0], t_box[1])
    point = _f_vertex_point(data, t_box[0], t_box[1],
                            u_box[0], u_box[1], v_box[0], v_box[1])
    edge1 = fv_sub(target[1], target[0])
    edge2 = fv_sub(target[2], target[0])
    target_normal = fv_cross(edge1, edge2)
    parent = fv_cross(fv_sub(source[1], source[0]), fv_sub(source[3], source[0]))
    height = fv_dot(fv_sub(point, target[0]), target_normal)
    projected = fv_dot(parent, target_normal)
    du = _f_vertex_derivative(data, t_box[0], t_box[1],
                              v_box[0], v_box[1], True)
    dv = _f_vertex_derivative(data, t_box[0], t_box[1],
                              u_box[0], u_box[1], False)
    actual = fv_cross(du, dv)
    parent_actual = fv_dot(parent, actual)
    target_squared = fv_dot(target_normal, target_normal)
    actual_squared = fv_dot(actual, actual)
    chart_ok = (projected[0] > 0.0 and parent_actual[0] > 0.0
                and actual_squared[0] > 0.0 and target_squared[0] > 0.0)
    if not chart_ok:
        return {"chart_ok": False}
    actual_norm = f_sqrt(actual_squared)
    target_norm = f_sqrt(target_squared)
    radius = f_div(height, projected)
    gap = f_div(f_mul(radius, parent_actual), actual_norm)
    foot = fv_sub(point, fv_scale(parent, radius))
    lambda1 = f_div(fv_dot(fv_cross(fv_sub(foot, target[0]), edge2),
                           target_normal), target_squared)
    lambda2 = f_div(fv_dot(fv_cross(edge1, fv_sub(foot, target[0])),
                           target_normal), target_squared)
    lambda_sum = f_add(lambda1, lambda2)
    if lambda1[0] >= 0.0 and lambda2[0] >= 0.0 and lambda_sum[1] <= 1.0:
        feasibility = "inside"
    elif lambda1[1] < 0.0 or lambda2[1] < 0.0 or lambda_sum[0] > 1.0:
        feasibility = "outside"
    else:
        feasibility = "uncertain"
    return {"chart_ok": True, "gap": gap, "point": point,
            "du": du, "dv": dv,
            "lambda1": lambda1,
            "lambda2": lambda2, "lambda_sum": lambda_sum,
            "height": height, "projected": projected,
            "parent_actual": parent_actual,
            "feasibility": feasibility}


def certify_float(data, tolerance=None, max_depth=10, max_boxes=6000):
    """Directed-float mirror of the exact branch-and-bound."""
    tolerance_value = (float(data.tolerance) if tolerance is None
                       else float(tolerance))
    root = ((_F0, _F1), (_F0, _F1), (_F0, _F1))
    stack = [(root, 0)]
    visited = 0
    safe_lower = None
    remaining = 0
    unresolved = 0
    violation = None
    while stack:
        box, depth = stack.pop()
        visited += 1
        enclosure = evaluate_box_float(data, *box)
        exhausted = depth >= max_depth or visited >= max_boxes
        if not enclosure["chart_ok"]:
            if exhausted:
                unresolved += 1
            else:
                for child in _children(box):
                    stack.append((child, depth + 1))
            continue
        if enclosure["feasibility"] == "outside":
            continue
        if (enclosure["gap"][1] < -tolerance_value
                and enclosure["feasibility"] == "inside"):
            violation = {"box": box, "gap": enclosure["gap"]}
            break
        if enclosure["gap"][0] >= -tolerance_value:
            value = enclosure["gap"][0]
            safe_lower = value if safe_lower is None else min(safe_lower, value)
            continue
        if exhausted:
            remaining += 1
            continue
        for child in _children(box):
            stack.append((child, depth + 1))
    if violation is not None:
        status = "violation_proven"
    elif remaining or unresolved:
        status = "inconclusive"
    else:
        status = "safe_within_tolerance"
    return {"status": status, "visited_boxes": visited,
            "min_gap_lower_m": safe_lower,
            "remaining_boxes": remaining,
            "unresolved_chart_boxes": unresolved,
            "violation": violation}


# ---------------------------------------------------------------------------
# Float locator for candidate minima (location only; rigor comes from the
# exact rational enclosures)
# ---------------------------------------------------------------------------

def float_point(data, t, u, v):
    """Plain float evaluation of the contract gap and foot barycentrics."""
    cache = getattr(data, "_float_corners", None)
    if cache is None:
        cache = tuple([[float(value) for value in row] for row in corners]
                      for corners in (data.a0, data.a1, data.b0, data.b1))
        data._float_corners = cache
    a0, a1, b0, b1 = cache
    def lerped(start, end):
        return [[(1 - t) * start[k][d] + t * end[k][d] for d in range(3)]
                for k in range(len(start))]
    source = lerped(a0, a1)
    target = lerped(b0, b1)
    weights = ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
    point = [sum(weights[k] * source[k][d] for k in range(4)) for d in range(3)]
    edge1 = [target[1][d] - target[0][d] for d in range(3)]
    edge2 = [target[2][d] - target[0][d] for d in range(3)]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    target_normal = cross(edge1, edge2)
    parent = cross([source[1][d] - source[0][d] for d in range(3)],
                   [source[3][d] - source[0][d] for d in range(3)])
    offset = [point[d] - target[0][d] for d in range(3)]
    height = dot(offset, target_normal)
    projected = dot(parent, target_normal)
    du = [(1 - v) * (source[1][d] - source[0][d]) + v * (source[2][d] - source[3][d])
          for d in range(3)]
    dv = [(1 - u) * (source[3][d] - source[0][d]) + u * (source[2][d] - source[1][d])
          for d in range(3)]
    actual = cross(du, dv)
    parent_actual = dot(parent, actual)
    target_squared = dot(target_normal, target_normal)
    actual_squared = dot(actual, actual)
    radius = height / projected
    gap = radius * parent_actual / math.sqrt(actual_squared)
    foot = [point[d] - radius * parent[d] for d in range(3)]
    foot_offset = [foot[d] - target[0][d] for d in range(3)]
    lambda1 = dot(cross(foot_offset, edge2), target_normal) / target_squared
    lambda2 = dot(cross(edge1, foot_offset), target_normal) / target_squared
    return {"gap": gap, "lambda1": lambda1, "lambda2": lambda2,
            "lambda_sum": lambda1 + lambda2}


def _feasible(evaluation):
    return (evaluation["lambda1"] >= 0.0 and evaluation["lambda2"] >= 0.0
            and evaluation["lambda_sum"] <= 1.0)


def locate_minimum(data):
    """Locate the constrained and unconstrained minimum candidates in floats."""
    best = None
    for i in range(9):
        t = i / 8
        for j in range(9):
            u = j / 8
            for k in range(9):
                v = k / 8
                evaluation = float_point(data, t, u, v)
                if _feasible(evaluation):
                    if best is None or evaluation["gap"] < best[0]:
                        best = (evaluation["gap"], t, u, v)
    unconstrained = None
    for i in range(9):
        t = i / 8
        for j in range(9):
            u = j / 8
            for k in range(9):
                v = k / 8
                evaluation = float_point(data, t, u, v)
                if unconstrained is None or evaluation["gap"] < unconstrained[0]:
                    unconstrained = (evaluation["gap"], t, u, v)
    if best is None:
        return {"constrained": None, "unconstrained": unconstrained}
    gap, t, u, v = best
    step = 0.125
    while step > 1e-9:
        improved = False
        for coordinate in range(3):
            for direction in (1.0, -1.0):
                candidate = [t, u, v]
                candidate[coordinate] = min(1.0, max(0.0, candidate[coordinate] + direction * step))
                evaluation = float_point(data, *candidate)
                if _feasible(evaluation) and evaluation["gap"] < gap:
                    gap, t, u, v = evaluation["gap"], *candidate
                    improved = True
        if not improved:
            step *= 0.5
    if unconstrained is not None:
        gap_u, t_u, u_u, v_u = unconstrained
        step = 0.125
        while step > 1e-9:
            improved = False
            for coordinate in range(3):
                for direction in (1.0, -1.0):
                    candidate = [t_u, u_u, v_u]
                    candidate[coordinate] = min(1.0, max(0.0, candidate[coordinate] + direction * step))
                    evaluation = float_point(data, *candidate)
                    if evaluation["gap"] < gap_u:
                        gap_u, t_u, u_u, v_u = evaluation["gap"], *candidate
                        improved = True
            if not improved:
                step *= 0.5
        unconstrained = (gap_u, t_u, u_u, v_u)
    return {"constrained": (gap, t, u, v), "unconstrained": unconstrained}


def _dyadic(value, levels=14):
    """Round down to the dyadic grid 2^-levels, as an exact Fraction."""
    scale = Fraction(2) ** levels
    return Fraction(math.floor(Fraction(value) * scale), 1) / scale


def feasible_upper_bound(data, located):
    """Best provably-feasible exact point near the located constrained min.

    Evaluates exact degenerate boxes (endpoint error boxes included) at a
    bounded dyadic candidate set; the smallest upper endpoint among
    provably-inside candidates is a rigorous upper bound of the feasible
    minimum gap.
    """
    candidates = []
    if located.get("constrained") is not None:
        _, t, u, v = located["constrained"]
        base = (_dyadic(t), _dyadic(u), _dyadic(v))
        for du in (0, 1, 2, 4):
            for dv in (0, 1, 2, 4):
                candidates.append((base[0], max(_F0, base[1] - du * Fraction(1, 2**14)),
                                   max(_F0, base[2] - dv * Fraction(1, 2**14))))
    for k in range(1, 17):
        delta = k * Fraction(1, 2**17)
        candidates.append((_F1, _F1 - delta, _F1))
        candidates.append((_F1, _F1, _F1 - delta))
    best = None
    for t, u, v in candidates:
        if not (0 <= t <= 1 and 0 <= u <= 1 and 0 <= v <= 1):
            continue
        enclosure = evaluate_box_exact(data, (t, t), (u, u), (v, v))
        if not enclosure["chart_ok"] or enclosure["feasibility"] != "inside":
            continue
        value = enclosure["gap"][1]
        if best is None or value < best["gap_upper_m"]:
            best = {
                "t": t, "u": u, "v": v,
                "gap_upper_m": value,
                "gap_interval": enclosure["gap"],
                "lambda1": enclosure["lambda1"],
                "lambda2": enclosure["lambda2"],
                "lambda_sum": enclosure["lambda_sum"],
            }
    return best


# ---------------------------------------------------------------------------
# Production witness sample grid, evaluated independently
# ---------------------------------------------------------------------------

_WITNESS_PARAMS = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
                   (0.5, 1.0), (0.0, 1.0), (0.0, 0.5), (0.5, 0.5))
_WITNESS_TIMES = (0.0, 0.5, 1.0)


def witness_sample_check(data):
    """Exact enclosure at the captured 27-sample witness grid."""
    entries = []
    for u, v in _WITNESS_PARAMS:
        for t in _WITNESS_TIMES:
            enclosure = evaluate_box_exact(data, (Fraction(t), Fraction(t)),
                                            (Fraction(u), Fraction(u)),
                                            (Fraction(v), Fraction(v)))
            if not enclosure["chart_ok"]:
                entries.append({"t": t, "u": u, "v": v, "chart_ok": False})
                continue
            entries.append({
                "t": t, "u": u, "v": v,
                "gap": enclosure["gap"],
                "feasibility": enclosure["feasibility"],
                "lambda_sum": enclosure["lambda_sum"],
                "violating": (enclosure["gap"][1] < -data.tolerance
                              and enclosure["feasibility"] == "inside"),
            })
    return entries


def end_pose_distinction(data):
    """Exact three-quantity distinction at the end pose t = 1.

    The contract gap g = r*(P.Ns)/|Ns|, the target-plane height N/|T| and
    the local-normal gap N*|Ns|/(Ns.T) are three distinct quantities for
    warped geometry; no equivalence between them is asserted anywhere. Each
    end-pose sample reports all three as exact rational enclosures with
    their exact strings retained (nearest-rounded floats are descriptive
    only and never consumed as bounds). A sample whose point chart is
    unresolved at the end pose is reported as such, never skipped silently.
    """
    entries = []
    for u, v in _WITNESS_PARAMS:
        try:
            point = exact_point(data, _F1, Fraction(u), Fraction(v))
        except ValueError as exc:
            entries.append({"u": u, "v": v, "chart_ok": False,
                           "reason": str(exc)})
            continue
        entries.append({
            "u": u, "v": v, "chart_ok": True,
            "contract_gap": [str(point["gap"][0]), str(point["gap"][1])],
            "plane_height": [str(point["plane_height"][0]),
                             str(point["plane_height"][1])],
            "local_normal_gap": [str(point["local_normal_gap"][0]),
                                 str(point["local_normal_gap"][1])],
            "contract_gap_m": [float(point["gap"][0]), float(point["gap"][1])],
            "plane_height_m": [float(point["plane_height"][0]),
                               float(point["plane_height"][1])],
            "local_normal_gap_m": [float(point["local_normal_gap"][0]),
                                   float(point["local_normal_gap"][1])],
            "lambda1": str(point["lambda1"]),
            "lambda2": str(point["lambda2"]),
            "lambda_sum": str(point["lambda_sum"]),
        })
    return entries


# ---------------------------------------------------------------------------
# Machine-readable discriminator result
# ---------------------------------------------------------------------------

def _box_float(box):
    return [float(value) for interval in box for value in interval]


def _interval_float(interval):
    return [float(interval[0]), float(interval[1])]


def run_discriminator(record=None, max_depth=10):
    """Full machine-readable independent result for one literal pair path."""
    if record is None:
        record = load_literal_record()
    data = PairPathData(record)
    tolerance = data.tolerance
    exact = certify_exact(data, tolerance, max_depth=max_depth)
    floated = certify_float(data, tolerance, max_depth=max_depth)
    try:
        located = locate_minimum(data)
    except ZeroDivisionError:
        # The float locator is location-only; a degenerate chart (zero
        # parent/target normal dot product) must not suppress the rigorous
        # certificate, which reports the unresolved chart on its own.
        located = {"constrained": None, "unconstrained": None}
    upper = feasible_upper_bound(data, located)
    samples = witness_sample_check(data)
    unconstrained = None
    if located.get("unconstrained") is not None:
        _, t, u, v = located["unconstrained"]
        enclosure = evaluate_box_exact(data, (Fraction(t), Fraction(t)),
                                        (Fraction(u), Fraction(u)),
                                        (Fraction(v), Fraction(v)))
        if enclosure["chart_ok"]:
            unconstrained = {
                "t": float(t), "u": float(u), "v": float(v),
                "gap_interval_m": _interval_float(enclosure["gap"]),
                "lambda_sum_interval": _interval_float(enclosure["lambda_sum"]),
                "feasibility": enclosure["feasibility"],
            }
    lower = exact["min_gap_lower_m"]
    result = {
        "reference": "general_contact_seed_gap_reference_v1",
        "input": {
            "source": record.get("input_source"),
            "sha256": record.get("input_sha256"),
            "delivery_sha256": INPUT_SHA256,
            "path_index": record.get("path_index"),
            "artifact_verified": bool(record.get("artifact_verified", False)),
            "original_artifact_verified": bool(
                record.get("original_artifact_verified", False)),
        },
        "gap_definition": ("g = r*(P.Ns)/|Ns| with r = N/(P.T), N = (s-c0).T, "
                           "P the parent corner-0 Q4 normal, Ns the actual "
                           "parametric Q4 normal, s the bilinear skin point on "
                           "the affine physical-skin path; foot y = s - r*P "
                           "inside the S3 triangle; endpoint error boxes "
                           "enclosed throughout; s, dX/du and dX/dv enclosed "
                           "by multilinear vertex bounds over each box"),
        "gap_tolerance_m": float(tolerance),
        "domain": {"t": [0.0, 1.0], "u": [0.0, 1.0], "v": [0.0, 1.0]},
        "exact_rational": {
            "status": exact["status"],
            "visited_boxes": exact["visited_boxes"],
            "min_gap_lower_m": None if lower is None else float(lower),
            "min_gap_lower_exact": None if lower is None else str(lower),
            "remaining_boxes": [
                {"box": _box_float(item["box"]), "depth": item["depth"],
                 "gap_interval_m": _interval_float(item["gap"]),
                 "gap_interval_exact": [str(item["gap"][0]),
                                       str(item["gap"][1])],
                 "feasibility": item["feasibility"]}
                for item in exact["remaining_boxes"]],
            "unresolved_chart_boxes": [
                {"box": _box_float(item["box"]), "depth": item["depth"]}
                for item in exact["unresolved_chart_boxes"]],
            "violation": None if exact["violation"] is None else {
                "box": _box_float(exact["violation"]["box"]),
                "gap_interval_m": _interval_float(exact["violation"]["gap"]),
                "gap_interval_exact": [str(exact["violation"]["gap"][0]),
                                       str(exact["violation"]["gap"][1])],
                "point_witness": None if exact["violation"]["point_witness"] is None else {
                    "t": float(exact["violation"]["point_witness"]["t"]),
                    "u": float(exact["violation"]["point_witness"]["u"]),
                    "v": float(exact["violation"]["point_witness"]["v"]),
                    "t_exact": str(exact["violation"]["point_witness"]["t"]),
                    "u_exact": str(exact["violation"]["point_witness"]["u"]),
                    "v_exact": str(exact["violation"]["point_witness"]["v"]),
                    "gap_interval_m": _interval_float(
                        exact["violation"]["point_witness"]["enclosure"]["gap"]),
                    "gap_interval_exact": [
                        str(exact["violation"]["point_witness"]["enclosure"]["gap"][0]),
                        str(exact["violation"]["point_witness"]["enclosure"]["gap"][1])],
                    "lambda1": _interval_float(
                        exact["violation"]["point_witness"]["enclosure"]["lambda1"]),
                    "lambda2": _interval_float(
                        exact["violation"]["point_witness"]["enclosure"]["lambda2"]),
                },
            },
        },
        "float_directed": {
            "status": floated["status"],
            "visited_boxes": floated["visited_boxes"],
            "min_gap_lower_m": None if floated["min_gap_lower_m"] is None
            else float(floated["min_gap_lower_m"]),
            "remaining_boxes": floated["remaining_boxes"],
            "unresolved_chart_boxes": floated["unresolved_chart_boxes"],
        },
        "located_minimum": {
            "unconstrained": unconstrained,
            "feasible_point": None if upper is None else {
                "t": float(upper["t"]), "u": float(upper["u"]), "v": float(upper["v"]),
                "t_exact": str(upper["t"]), "u_exact": str(upper["u"]),
                "v_exact": str(upper["v"]),
                "gap_upper_m": float(upper["gap_upper_m"]),
                "gap_upper_exact": str(upper["gap_upper_m"]),
                "gap_interval_m": _interval_float(upper["gap_interval"]),
                "gap_interval_exact": [str(upper["gap_interval"][0]),
                                       str(upper["gap_interval"][1])],
                "lambda1": _interval_float(upper["lambda1"]),
                "lambda2": _interval_float(upper["lambda2"]),
                "lambda_sum": _interval_float(upper["lambda_sum"]),
                "foot_inside": True,
            },
        },
        "witness_grid_27": {
            "violating_samples": sum(1 for item in samples if item.get("violating")),
            "min_gap_upper_m": min(
                (float(item["gap"][1]) for item in samples
                 if item.get("gap") is not None), default=None),
            "samples": [{"t": item["t"], "u": item["u"], "v": item["v"],
                         "gap_interval_m": _interval_float(item["gap"]),
                         "feasibility": item["feasibility"]}
                        if item.get("chart_ok", True) else item
                        for item in samples],
        },
        "end_pose_distinction": {
            "note": ("contract gap, target-plane height and local-normal "
                     "gap are distinct quantities for warped geometry; no "
                     "equivalence is asserted; exact strings are the "
                     "retained bounds, floats are descriptive only"),
            "samples": end_pose_distinction(data),
        },
        "certified_scope": ("single literal captured Q4/S3 pair path only; "
                            "no statement about other faces, poses or the "
                            "whole solver path"),
    }
    if lower is not None and upper is not None:
        result["min_gap_interval_m"] = [float(lower), float(upper["gap_upper_m"])]
        result["min_gap_interval_exact"] = [str(lower),
                                            str(upper["gap_upper_m"])]
    elif lower is not None:
        result["min_gap_interval_m"] = [float(lower), None]
        result["min_gap_interval_exact"] = [str(lower), None]
    if exact["status"] == "safe_within_tolerance":
        result["decision"] = {
            "path_safe_within_tolerance": True,
            "violation_proven": False,
            "refusal_classification": "none_safe_within_tolerance",
            # None exactly when the foot-inside domain is empty: the
            # record is vacuously safe and no finite gap is required.
            "min_gap_lower_plus_tolerance_m": (
                None if lower is None else float(lower + tolerance)),
        }
    elif exact["status"] == "violation_proven":
        result["decision"] = {"path_safe_within_tolerance": False,
                              "violation_proven": True,
                              "refusal_classification":
                                  "physical_violation_proven"}
    else:
        result["decision"] = {"path_safe_within_tolerance": False,
                              "violation_proven": False,
                              "inconclusive": True,
                              "refusal_classification":
                                  "chart_or_budget_inconclusive_refusal"}
    return result


def run_capture_record(record, max_depth=10, max_boxes=6000,
                       exact_max_boxes=6000):
    """Independent directed/exact result for one literal capture record.

    The directed-float branch-and-bound certifies the record first
    (rigorous outward rounding, fast), the exact rational feasible upper
    bound is always retained, and the slower exact rational
    branch-and-bound runs only where the directed certificate did not
    conclude safety (inconclusive or violation records), so the retained
    per-record result is always exact or directed, never sampled. Exact
    rational strings are retained for every reported bound.
    """
    data = PairPathData(record)
    tolerance = data.tolerance
    floated = certify_float(data, tolerance, max_depth=max_depth,
                            max_boxes=max_boxes)
    try:
        located = locate_minimum(data)
    except ZeroDivisionError:
        located = {"constrained": None, "unconstrained": None}
    upper = feasible_upper_bound(data, located)
    exact = None
    if floated["status"] != "safe_within_tolerance":
        exact = certify_exact(data, tolerance, max_depth=max_depth,
                              max_boxes=exact_max_boxes)
    entry = {
        "kind": record["kind"],
        "record_index": record["record_index"],
        "input_sha256": record["input_sha256"],
        "gap_tolerance_m": float(tolerance),
        "float_directed": {
            "status": floated["status"],
            "visited_boxes": floated["visited_boxes"],
            "min_gap_lower_m": None if floated["min_gap_lower_m"] is None
            else float(floated["min_gap_lower_m"]),
            "remaining_boxes": floated["remaining_boxes"],
            "unresolved_chart_boxes": floated["unresolved_chart_boxes"],
            "violation": None if floated["violation"] is None else {
                "box": _box_float(floated["violation"]["box"]),
                "gap_interval_m": _interval_float(
                    floated["violation"]["gap"]),
            },
        },
        "exact_rational": None if exact is None else {
            "status": exact["status"],
            "visited_boxes": exact["visited_boxes"],
            "min_gap_lower_m": None if exact["min_gap_lower_m"] is None
            else float(exact["min_gap_lower_m"]),
            "min_gap_lower_exact": None if exact["min_gap_lower_m"] is None
            else str(exact["min_gap_lower_m"]),
            "remaining_boxes": len(exact["remaining_boxes"]),
            "unresolved_chart_boxes": len(exact["unresolved_chart_boxes"]),
            "violation": None if exact["violation"] is None else {
                "box": _box_float(exact["violation"]["box"]),
                "gap_interval_m": _interval_float(exact["violation"]["gap"]),
                "gap_interval_exact": [str(exact["violation"]["gap"][0]),
                                       str(exact["violation"]["gap"][1])],
                "point_witness": None if exact["violation"]["point_witness"] is None else {
                    "t": float(exact["violation"]["point_witness"]["t"]),
                    "u": float(exact["violation"]["point_witness"]["u"]),
                    "v": float(exact["violation"]["point_witness"]["v"]),
                    "t_exact": str(exact["violation"]["point_witness"]["t"]),
                    "u_exact": str(exact["violation"]["point_witness"]["u"]),
                    "v_exact": str(exact["violation"]["point_witness"]["v"]),
                    "gap_interval_m": _interval_float(
                        exact["violation"]["point_witness"]["enclosure"]["gap"]),
                    "gap_interval_exact": [
                        str(exact["violation"]["point_witness"]["enclosure"]["gap"][0]),
                        str(exact["violation"]["point_witness"]["enclosure"]["gap"][1])],
                },
            },
        },
        "feasible_point": None if upper is None else {
            "t": float(upper["t"]), "u": float(upper["u"]),
            "v": float(upper["v"]),
            "t_exact": str(upper["t"]), "u_exact": str(upper["u"]),
            "v_exact": str(upper["v"]),
            "gap_upper_m": float(upper["gap_upper_m"]),
            "gap_upper_exact": str(upper["gap_upper_m"]),
            "foot_inside": True,
        },
        "end_pose_distinction": {
            "samples": end_pose_distinction(data)},
    }
    if upper is not None:
        entry["min_gap_upper_m"] = float(upper["gap_upper_m"])
        entry["min_gap_upper_exact"] = str(upper["gap_upper_m"])
    return entry


def run_capture_records(max_depth=10, max_boxes=6000,
                        exact_max_boxes=6000):
    """Independent directed/exact results for all literal capture records."""
    return [run_capture_record(record, max_depth=max_depth,
                               max_boxes=max_boxes,
                               exact_max_boxes=exact_max_boxes)
            for record in load_capture_records()]


if __name__ == "__main__":
    print(json.dumps(run_discriminator(), indent=2))
