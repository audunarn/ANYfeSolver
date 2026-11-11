"""Local physical chord metric, experimental and not programme-qualified.

Weights are explicit reference quadrature weights, frozen by the caller. No
nodal-count normalization or mass is inferred. This is a seminorm on the mixed
system: retained variables and multipliers still belong in equilibrium.
"""
from dataclasses import dataclass
import numpy as np
from ._ge_beam3_p5.algebra import skew


class ChordDomainError(ValueError):
    """A candidate leaves the local chord chart; malformed data is separate."""


def _owned(value):
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


@dataclass(frozen=True)
class PhysicalChordMetric:
    weights: np.ndarray
    length: float

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float)
        if (weights.ndim != 1 or not weights.size or not np.isfinite(weights).all()
                or np.any(weights <= 0.) or not np.isfinite(self.length)
                or self.length <= 0.):
            raise ValueError("positive finite reference weights and length required")
        squared=float(self.length)*float(self.length)
        if not np.isfinite(squared) or squared<=0.:
            raise ValueError("representable squared metric length required")
        object.__setattr__(self, "weights", _owned(weights))
        object.__setattr__(self, "length", float(self.length))

    def evaluate(self, displacement, frames, reference_frames):
        """Value and translation/spatial-left-rotation gradients.

        Physical translation differences are supplied directly, avoiding the
        subtraction of two large absolute positions. Rotation chord is local:
        relative angles >= pi/2 are rejected rather than traversing its fold.
        """
        n = len(self.weights)
        u = np.asarray(displacement, dtype=float)
        current = np.asarray(frames, dtype=float)
        reference = np.asarray(reference_frames, dtype=float)
        if (u.shape != (n, 3) or current.shape != (n, 3, 3)
                or reference.shape != current.shape or not np.isfinite(u).all()):
            raise ValueError("complete finite physical nodal poses required")
        for rotations in (current, reference):
            if (not np.isfinite(rotations).all()
                    or np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))) > 1e-10
                    or np.max(np.abs(np.linalg.det(rotations) - 1.)) > 1e-10):
                raise ValueError("proper physical rotation matrices required")
        relative = current @ reference.transpose(0, 2, 1)
        if np.any(np.trace(relative, axis1=1, axis2=2) <= 1.):
            raise ChordDomainError("local continuation rotation exceeds pi/2")
        difference = current - reference
        value = np.sum(self.weights * (
            np.sum((u / self.length)**2, axis=1)
            + .5*np.sum(difference*difference, axis=(1, 2))))
        translation = 2*self.weights[:, None]*u/self.length**2
        # d(3-tr(R R0.T)) / d spatial-left theta.
        spin = np.column_stack((relative[:, 2, 1]-relative[:, 1, 2],
                                relative[:, 0, 2]-relative[:, 2, 0],
                                relative[:, 1, 0]-relative[:, 0, 1]))
        return float(value), translation, self.weights[:, None]*spin

    def plane(self, displacement, frames, reference_frames, translations, spins):
        """Frozen physical tangent-plane value and actual spatial gradients.

        Caller supplies a unit tangent in this metric. Unlike squared distance,
        this signed projection has a nonzero row at the accepted predecessor.
        """
        self.evaluate(displacement,frames,reference_frames)  # same local domain
        translations=np.asarray(translations,dtype=float)
        spins=np.asarray(spins,dtype=float)
        if (translations.shape!=(len(self.weights),3) or spins.shape!=translations.shape
                or not np.isfinite(translations).all() or not np.isfinite(spins).all()):
            raise ValueError("finite physical tangent directions required")
        tangent=np.array([skew(v)@r for v,r in zip(spins,reference_frames)])
        value=np.sum(self.weights*(np.sum(displacement*translations,axis=1)/self.length**2
            +.5*np.sum((frames-reference_frames)*tangent,axis=(1,2))))
        product=frames@tangent.transpose(0,2,1)
        gradient=-.5*np.column_stack((product[:,2,1]-product[:,1,2],
            product[:,0,2]-product[:,2,0],product[:,1,0]-product[:,0,1]))
        return float(value),self.weights[:,None]*translations/self.length**2,self.weights[:,None]*gradient


DIRECTOR_METRIC_ID = "flat_native_bl5_position_director_v1"


@dataclass(frozen=True)
class NativeDirectorChordMetric:
    """Opt-in flat native BL5 shell position/director, beam full-frame metric.

    Shell nodes precede beam nodes. The flat native material director is the
    third column of each proper physical shell frame, R e_z. Its normal spin
    is a metric null direction only; no mechanical gauge invariance is claimed.
    Caller-supplied directors and inferred/renormalized weights are not admitted.
    """
    weights: np.ndarray
    length: float
    shell_nodes: int

    def __post_init__(self):
        physical = PhysicalChordMetric(self.weights, self.length)
        if type(self.shell_nodes) is not int or not 1 <= self.shell_nodes <= len(physical.weights):
            raise ValueError("exact positive shell node count required")
        object.__setattr__(self, "weights", physical.weights)
        object.__setattr__(self, "length", physical.length)

    def _poses(self, displacement, frames, reference_frames):
        n = len(self.weights)
        u = np.asarray(displacement, dtype=float)
        current = np.asarray(frames, dtype=float)
        reference = np.asarray(reference_frames, dtype=float)
        if (u.shape != (n, 3) or current.shape != (n, 3, 3)
                or reference.shape != current.shape or not np.isfinite(u).all()):
            raise ValueError("complete finite physical nodal poses required")
        for rotations in (current, reference):
            if (not np.isfinite(rotations).all()
                    or np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))) > 1e-10
                    or np.max(np.abs(np.linalg.det(rotations) - 1.)) > 1e-10):
                raise ValueError("proper physical rotation matrices required")
        s = self.shell_nodes
        d, d0 = current[:s, :, 2], reference[:s, :, 2]
        if np.any(np.sum(d*d0, axis=1) <= 0.):
            raise ChordDomainError("local shell director exceeds open hemisphere")
        if s < n:
            relative = current[s:] @ reference[s:].transpose(0, 2, 1)
            if np.any(np.trace(relative, axis1=1, axis2=2) <= 1.):
                raise ChordDomainError("local continuation beam rotation exceeds pi/2")
        return u, current, reference, d, d0

    def _directions(self, translations, spins):
        translations = np.asarray(translations, dtype=float)
        spins = np.asarray(spins, dtype=float)
        if (translations.shape != (len(self.weights), 3) or spins.shape != translations.shape
                or not np.isfinite(translations).all() or not np.isfinite(spins).all()):
            raise ValueError("finite physical tangent directions required")
        return translations, spins

    def evaluate(self, displacement, frames, reference_frames):
        """Squared chord with translation and spatial-left-rotation rows."""
        u, current, reference, d, d0 = self._poses(displacement, frames, reference_frames)
        s = self.shell_nodes
        w = self.weights[:s, None]
        value = np.sum(self.weights[:s]*(np.sum((u[:s]/self.length)**2, axis=1)
                                        + np.sum((d-d0)**2, axis=1)))
        translations = 2*self.weights[:, None]*u/self.length**2
        spins = np.zeros_like(u)
        spins[:s] = 2*w*np.cross(d0, d)
        if s < len(self.weights):
            beam = PhysicalChordMetric(self.weights[s:], self.length)
            extra, _, spins[s:] = beam.evaluate(u[s:], current[s:], reference[s:])
            value += extra
        if not np.isfinite(value) or not np.isfinite(translations).all() or not np.isfinite(spins).all():
            raise ValueError("representable director chord required")
        return float(value), translations, spins

    def plane(self, displacement, frames, reference_frames, translations, spins):
        """Signed projection onto the frozen predecessor physical tangent."""
        u, current, reference, d, d0 = self._poses(displacement, frames, reference_frames)
        translations, spins = self._directions(translations, spins)
        s = self.shell_nodes
        eta = np.cross(spins[:s], d0)
        value = np.sum(self.weights[:s]*(np.sum(u[:s]*translations[:s], axis=1)/self.length**2
                                        + np.sum((d-d0)*eta, axis=1)))
        row = np.zeros_like(u)
        row[:s] = self.weights[:s, None]*np.cross(d, eta)
        if s < len(self.weights):
            beam = PhysicalChordMetric(self.weights[s:], self.length)
            extra, _, row[s:] = beam.plane(u[s:], current[s:], reference[s:], translations[s:], spins[s:])
            value += extra
        translation_row = self.weights[:, None]*translations/self.length**2
        if not np.isfinite(value) or not np.isfinite(translation_row).all() or not np.isfinite(row).all():
            raise ValueError("representable director tangent plane required")
        return float(value), translation_row, row

    def tangent_norm_squared(self, reference_frames, translations, spins):
        """Shell tilt at the physical predecessor; beam full spatial spin."""
        _, _, _, _, d0 = self._poses(np.zeros((len(self.weights), 3)), reference_frames, reference_frames)
        translations, spins = self._directions(translations, spins)
        rotational = np.sum(spins*spins, axis=1)
        rotational[:self.shell_nodes] = np.sum(np.cross(spins[:self.shell_nodes], d0)**2, axis=1)
        value = np.sum(self.weights*(np.sum((translations/self.length)**2, axis=1) + rotational))
        if not np.isfinite(value):
            raise ValueError("representable director tangent norm required")
        return float(value)


def require_metric_owner(owner, metric):
    """Typed metric admission without changing the default physical metric."""
    if type(metric) not in (PhysicalChordMetric, NativeDirectorChordMetric):
        raise ValueError("explicit physical arc metric required")
    if type(metric) is PhysicalChordMetric:
        return  # Preserve historical owner admission and validation ordering.
    if len(metric.weights) != len(owner._shell_ids) + len(owner._beam_ids):
        raise ValueError("metric weights must follow compiled shell then beam nodes")
    if type(metric) is NativeDirectorChordMetric:
        from ._stiffened_shell_affine_endpoint import is_native_material_bl5_arc
        definition = owner._definition
        if (not is_native_material_bl5_arc(owner) or definition.shell_reference_policy != "flat"
                or getattr(owner, "_curved_reference", False) or definition.points
                or definition.footprints or definition.distributing_footprints
                or definition.finite_distributing_footprints
                or metric.shell_nodes != len(owner._shell_ids)):
            raise ValueError("director metric requires exact flat native material BL5 shell ownership")


def require_director_chart(q):
    """Stay in the connected nonsingular additive shell exponential chart."""
    rows = np.asarray(q, dtype=float).reshape(-1, 6)
    if not np.isfinite(rows).all():
        raise ValueError("finite shell chart coordinates required")
    # The spatial-left Jacobian loses two columns at |phi|=2*pi. A chord
    # domain guard alone cannot detect this (a complete drill has zero chord).
    if np.any(np.linalg.norm(rows[:, 3:], axis=1) >= 2*np.pi):
        raise ChordDomainError("director metric shell chart crosses 2*pi degeneracy")
