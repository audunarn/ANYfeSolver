"""Development-only float/Jet2 agreement for a resolved artificial seam wedge.

This validates a local derivative; it is not an independent acceptance oracle.
The caller still checks exact tie ownership, area resolution and assembled
non-cancelling pressure-node value errors.
"""

from __future__ import annotations

import numpy as np

from .q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from .q4_source_overlap import projected_q4_source_face


def _clipping_signature(source, target):
    first, normal = projected_q4_source_face(source)
    projected = target-((target-first.corners[0]) @ normal)[:, None]*normal
    origin = first.corners[0]
    tangent = first.corners[1]-origin
    tangent /= np.linalg.norm(tangent)
    binormal = np.cross(normal, tangent)
    polygons = tuple(np.column_stack(((points-origin) @ tangent,
                                      (points-origin) @ binormal))
                     for points in (first.corners, projected))
    signature = []
    margin = float("inf")
    for side in (0, 1):
        vertices = polygons[side]
        edges = polygons[1-side]
        for vertex_index, point in enumerate(vertices):
            for edge_index, start in enumerate(edges):
                edge = edges[(edge_index+1) % len(edges)]-start
                squared = float(edge @ edge)
                fraction = float((point-start) @ edge/squared)
                if not -0.1 <= fraction <= 1.1:
                    continue
                signed = float(np.linalg.det(np.stack((edge, point-start))))
                signed /= np.sqrt(squared)
                signature.append((side, vertex_index, edge_index,
                                  int(np.sign(signed))))
                margin = min(margin, abs(signed))
    return tuple(signature), margin


def verify_tiny_wedge_derivatives(
        law, source, target, derivative, pressure, normalized_gap,
        active_local_coordinates, *, difference_step_m, force_scale_n,
        length_scale_m, maximum_evaluations):
    """Check every relevant local Hessian entry against float pair integrals.

    The work budgets are dimensional. Summing absolute nodal contributions
    prevents opposite pressure-node errors from cancelling in this check.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    active = np.asarray(active_local_coordinates, dtype=np.intp)
    local = np.concatenate((source, target)).reshape(-1)
    count = local.size
    if (len(active) == 0 or len(active) > 12
            or np.any(active < 0) or np.any(active >= count)):
        raise UnsupportedQ4S3Branch(
            "tiny seam wedge needs 1 to 12 resolved local tangent coordinates"
        )
    if (difference_step_m <= 0.0 or not np.isfinite(difference_step_m)
            or maximum_evaluations < 1):
        raise UnsupportedQ4S3Branch("tiny seam wedge difference budget is invalid")
    pressure = np.asarray(pressure, dtype=float)
    normalized_gap = np.asarray(normalized_gap, dtype=float)
    if (pressure.shape != (len(source),)
            or normalized_gap.shape != pressure.shape):
        raise UnsupportedQ4S3Branch("tiny seam wedge pressure map is invalid")

    values = []
    evaluations = 0
    for step in (difference_step_m, difference_step_m/2.0):
        cache = {}
        reference_signature, reference_margin = _clipping_signature(source, target)
        if reference_margin <= 2.0*step:
            raise UnsupportedQ4S3Branch(
                "tiny seam wedge has no resolved clipping-branch margin"
            )

        def at(shift):
            nonlocal evaluations
            key = tuple(np.asarray(shift, dtype=float))
            if key not in cache:
                if evaluations >= maximum_evaluations:
                    raise MemoryError("tiny seam wedge float derivative budget exhausted")
                trial = local+np.asarray(shift, dtype=float)
                trial_source = trial[:3*len(source)].reshape(-1, 3)
                trial_target = trial[3*len(source):].reshape(-1, 3)
                signature, margin = _clipping_signature(trial_source,
                                                        trial_target)
                if signature != reference_signature or margin <= 2.0*step:
                    raise UnsupportedQ4S3Branch(
                        "tiny seam wedge float trial changed clipping branch"
                    )
                candidate = law.evaluate(
                    trial_source, trial_target,
                )
                if candidate.reference_overlap_area <= 0.0:
                    raise UnsupportedQ4S3Branch(
                        "tiny seam wedge float derivative changed overlap branch"
                    )
                cache[key] = np.stack((candidate.reference_support,
                                       candidate.reference_weighted_gap))
                evaluations += 1
            return cache[key]

        zero = np.zeros(count)
        center = at(zero)
        basis = np.eye(count)[active]
        gradient = np.zeros((2, len(source), count))
        hessian = np.zeros((2, len(source), len(active), len(active)))
        for index in range(count):
            direction = np.eye(count)[index]
            plus = at(step*direction)
            minus = at(-step*direction)
            gradient[:, :, index] = (plus-minus)/(2.0*step)
        for i, vector in enumerate(basis):
            hessian[:, :, i, i] = (
                at(step*vector)-2.0*center+at(-step*vector)
            )/(step*step)
            for k in range(i+1, len(active)):
                other = basis[k]
                mixed = (
                    at(step*(vector+other))
                    - at(step*(vector-other))
                    - at(step*(-vector+other))
                    + at(-step*(vector+other))
                )/(4.0*step*step)
                hessian[:, :, i, k] = mixed
                hessian[:, :, k, i] = mixed
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            raise UnsupportedQ4S3Branch("tiny seam wedge float derivative is nonfinite")
        values.append((gradient, hessian))

    jet_gradient = np.stack((derivative.support_gradient,
                             derivative.weighted_gap_gradient))
    jet_hessian = np.stack((
        derivative.support_hessian[:, active][:, :, active],
        derivative.weighted_gap_hessian[:, active][:, :, active],
    ))
    force_budget = 1.0e-6*force_scale_n
    tangent_budget = 1.0e-5*force_scale_n/length_scale_m

    def check(first, second, *, label):
        gradient_error = first[0]-second[0]
        hessian_error = first[1]-second[1]
        nodal_force = pressure[:, None]*(
            gradient_error[1]-normalized_gap[:, None]*gradient_error[0]
        )
        nodal_tangent = pressure[:, None, None]*(
            hessian_error[1]
            - normalized_gap[:, None, None]*hessian_error[0]
        )
        force_error = float(np.sum(np.linalg.norm(nodal_force, axis=1)))
        tangent_error = float(np.sum(np.linalg.norm(
            nodal_tangent.reshape(len(source), -1), axis=1)))
        if (not np.isfinite(force_error) or not np.isfinite(tangent_error)
                or force_error > force_budget or tangent_error > tangent_budget):
            raise UnsupportedQ4S3Branch(
                f"tiny seam wedge {label} work derivative unresolved "
                f"(force_error_n={force_error:.12g}, "
                f"force_budget_n={force_budget:.12g}, "
                f"tangent_error_n_per_m={tangent_error:.12g}, "
                f"tangent_budget_n_per_m={tangent_budget:.12g})"
            )

    jet = (jet_gradient, jet_hessian)
    checks = (
        (values[0], jet, "full-step"),
        (values[1], jet, "half-step"),
        (values[0], values[1], "step-stability"),
    )
    for first, second, label in checks:
        check(first, second, label=label)
    return evaluations, tuple((first[0]-second[0], first[1]-second[1])
                              for first, second, _ in checks)
