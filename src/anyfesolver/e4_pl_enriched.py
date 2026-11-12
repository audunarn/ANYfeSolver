"""Experimental reference-linear E4-PL skin with four physical bubble DOFs.

This is a separate formulation, not a change to QualifiedE4PLShellElement.
Only the stress/strain stationary variables are eliminated.  All four added
displacement coordinates remain in the structural/contact system.  There is
no pressure stabilization, pressure filter, or claim of cut-uniform stability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .e4_pl_element import (
    QualifiedE4PLShellElement, _coefficients, _compatible, _global_transform,
    _jacobian, _solve_stationary_system, _source_fields, _stationary_blocks,
    equation7_frame,
)

FORMULATION = "E4PL_PHYSICAL_B4_REFERENCE_LINEAR_PUBLIC_S3_V2"
BASIS = ("(1-r*r)*(1-s*s)", "r*(1-r*r)*(1-s*s)",
         "s*(1-r*r)*(1-s*s)", "r*s*(1-r*r)*(1-s*s)")


class EnrichedProfileError(ValueError):
    """Geometry, material, state or caller is outside this explicit profile."""


def readonly(value, shape=None):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("real numeric values required")
        made = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EnrichedProfileError("real finite enriched numeric array required") from exc
    if (shape is not None and made.shape != shape) or not np.all(np.isfinite(made)):
        raise EnrichedProfileError("finite array with exact enriched layout required")
    return np.frombuffer(made.tobytes(order="C"), dtype=np.float64).reshape(made.shape)


def bubble_basis(r, s):
    """Values and natural derivatives; works for floats and second-order jets."""
    b = (1-r*r)*(1-s*s)
    br, bs = -2*r*(1-s*s), -2*s*(1-r*r)
    return ((b, r*b, s*b, r*s*b),
            (br, b+r*br, s*br, s*b+r*s*br),
            (bs, r*bs, b+s*bs, r*b+r*s*bs))


@dataclass(frozen=True)
class EnrichedCell:
    element_id: int
    node_ids: tuple
    reference: np.ndarray
    frame: np.ndarray
    stiffness: np.ndarray
    stationary: np.ndarray
    coupling: np.ndarray
    stationary_solution: np.ndarray
    identity: str
    diagnostics: dict

    def energy(self, coordinates):
        q = readonly(coordinates, (28,))
        return float(q @ self.stiffness @ q / 2)

    def recover(self, coordinates):
        """Actual 43 stationary amplitudes, with the same sign as condensation."""
        return readonly(-self.stationary_solution @ readonly(coordinates, (28,)))

    def recover_fields(self, coordinates, r, s):
        """Actual reference-frame mixed section fields, including all bubbles.

        These are the stationary independent stress/strain fields, not a
        claim that their pointwise difference is a constitutive-error oracle.
        """
        if not all(np.isfinite(x) and -1 <= x <= 1 for x in (r, s)):
            raise EnrichedProfileError("reference recovery point outside Q4")
        local = (self.reference-self.reference.mean(axis=0)) @ self.frame[:, :2]
        c = _coefficients(local)
        ns, ne = _source_fields(c, r, s)
        _, br, bs = bubble_basis(r, s)
        xr, xs, yr, ys, det = _jacobian(c, r, s)
        e = np.zeros((8, 4))
        e[6] = (ys*np.asarray(br)-yr*np.asarray(bs))/det
        e[7] = (-xs*np.asarray(br)+xr*np.asarray(bs))/det
        a = self.recover(coordinates)
        return dict(profile=FORMULATION, frame=self.frame,
                    generalized_stress=readonly(ns @ a[:14]+e @ a[35:39]),
                    generalized_strain=readonly(ne @ a[14:35]+e @ a[39:43]))


def compile_cell(element, mesh, material):
    """Extend the production mixed spaces by four compatible shear gradients.

    Pi(d,a,e)=a' L d - a' F e + .5 e' H e.  New stress and
    independent-strain functions are grad(B_a) in the physical shear rows.
    The old 35x35 block is used verbatim; all new blocks are physical
    integrals of these SAME spaces and the existing section constitutive C.
    Five-point tensor Gauss integrates the new polynomial products exactly
    on the admitted affine rectangle.  The old PL/drill term is unchanged.
    """
    if type(element) is not QualifiedE4PLShellElement:
        raise EnrichedProfileError("explicit qualified elastic E4-PL cell required")
    if (getattr(material, "yield_stress", 0.) != 0.
            or getattr(material, "j2_curve", None) is not None
            or getattr(material, "hardening_curve", None) is not None):
        raise EnrichedProfileError("enriched profile admits elastic material only")
    reference = np.asarray(element.get_node_coordinates(mesh), dtype=float)
    frame, local, warpage = equation7_frame(reference)
    if warpage > element.planar_tolerance:
        raise EnrichedProfileError("enriched profile requires a flat reference cell")
    c = _coefficients(local)
    scale = max(float(np.max(np.abs(local))), np.finfo(float).tiny)
    # Polynomial quadrature and the stated independent fields are admitted
    # only on an affine rectangle, not silently extended to warped/distorted Q4.
    edges = np.roll(local, -1, axis=0)-local
    if (abs(c["jr"])+abs(c["js"]) > 128*np.finfo(float).eps*scale*scale
            or abs(float(edges[0] @ edges[1])) > 128*np.finfo(float).eps*scale*scale
            or c["jc"] <= 0):
        raise EnrichedProfileError("enriched reference must be an affine rectangle")
    components = element.compute_stiffness_components(mesh, material)
    constitutive, _ = element._constitutive_and_drill_stiffness(material, frame)
    original, old_coupling, _ = _stationary_blocks(
        local, c, constitutive,
        compensated=element._elastic_increment_admitted(mesh, material, None))
    stationary = np.zeros((43, 43))
    coupling = np.zeros((28, 43))
    stationary[:35, :35] = original
    coupling[:24, :35] = old_coupling
    points, weights = np.polynomial.legendre.leggauss(5)
    for r, wr in zip(points, weights):
        for s, ws in zip(points, weights):
            _, dr, ds = bubble_basis(r, s)
            xr, xs, yr, ys, det = _jacobian(c, r, s)
            e = np.zeros((8, 4))
            e[6] = (ys*np.asarray(dr)-yr*np.asarray(ds))/det
            e[7] = (-xs*np.asarray(dr)+xr*np.asarray(ds))/det
            ns, ne = _source_fields(c, r, s)
            compatible = _compatible(local, c, r, s)
            w = wr*ws*det
            mass = w*(e.T @ e)
            # Original stress -> new independent strain; new stress -> old/new strain.
            stationary[:14, 39:43] -= w*(ns.T @ e)
            stationary[39:43, :14] -= w*(e.T @ ns)
            stationary[35:39, 14:35] -= w*(e.T @ ne)
            stationary[14:35, 35:39] -= w*(ne.T @ e)
            stationary[35:39, 39:43] -= mass
            stationary[39:43, 35:39] -= mass
            stationary[14:35, 39:43] += w*(ne.T @ constitutive @ e)
            stationary[39:43, 14:35] += w*(e.T @ constitutive @ ne)
            stationary[39:43, 39:43] += w*(e.T @ constitutive @ e)
            nodal = w*(compatible.T @ e)
            for node in range(4):
                coupling[6*node:6*node+5, 35:39] += nodal[5*node:5*node+5]
            coupling[24:28, :14] += w*(e.T @ ns)
            coupling[24:28, 35:39] += mass
    # Populate symmetric partners from the same contraction, including all
    # new/new cross terms; leave the entire original35 block byte-identical.
    for i in range(43):
        for j in range(max(35, i+1), 43):
            stationary[j, i] = stationary[i, j]
    solution, diagnostics = _solve_stationary_system(stationary, coupling)
    core_local = -coupling @ solution
    core_local = (core_local+core_local.T)/2
    transform = np.eye(28)
    transform[:24, :24] = _global_transform(frame)
    core = transform @ core_local @ transform.T
    core = (core+core.T)/2
    stiffness = core.copy()
    stiffness[:24, :24] += np.asarray(components["numerical"])
    # A model definition, not an acceptance threshold: refuse an algebraically
    # non-positive bubble block rather than manufacture a stiffness floor.
    if not np.all(np.isfinite(stiffness)) or np.linalg.eigvalsh(stiffness[24:, 24:])[0] <= 0:
        raise EnrichedProfileError("physical bubble energy is not positive")
    descriptor = {"formulation": FORMULATION, "basis": BASIS,
                  "element_id": int(element.element_id), "nodes": list(element.node_ids),
                  "reference": reference.tolist(), "stiffness_sha256":
                  hashlib.sha256(stiffness.tobytes()).hexdigest()}
    identity = hashlib.sha256(json.dumps(descriptor, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    diagnostics = dict(diagnostics, development_only=True,
                       stationary_fields=43, explicit_displacements=28,
                       nodal_core_difference=float(np.linalg.norm(
                           core[:24, :24]-components["core"])),
                       pressure_stability="UNPROVEN", kinematics="REFERENCE_LINEAR")
    return EnrichedCell(int(element.element_id), tuple(element.node_ids),
                        readonly(reference, (4, 3)), readonly(frame, (3, 3)),
                        readonly(stiffness, (28, 28)), readonly(stationary),
                        readonly(transform @ coupling), readonly(solution @ transform.T),
                        identity, diagnostics)
