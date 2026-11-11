"""Private exact elastic law and constrained complementary cell; no routing."""
from dataclasses import dataclass
from hashlib import sha256
import json
import warnings
import numpy as np
from scipy.linalg import solve as _scipy_solve, LinAlgWarning
from .beam_sections import generalized_beam_stiffness

POLICY = "GE_BEAM3_G1_LINEAR_ELASTIC_SECTION_V1"


def solve(a, b, *, assume_a="gen"):
    """An unhealthy factorization is a failure, never a warning-only result."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            result = _scipy_solve(a, b, assume_a=assume_a, check_finite=True)
        except LinAlgWarning as exc:
            raise np.linalg.LinAlgError("G1 unhealthy factorization; cutback required") from exc
    if not np.isfinite(result).all():
        raise np.linalg.LinAlgError("G1 nonfinite factorization result")
    return result


def canonical(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return (json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(value):
    return sha256(canonical(value)).hexdigest()


def owned(value, shape=None):
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise ValueError("boolean is not a numeric mechanical value")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("finite numeric arrays required")
    a = np.asarray(raw, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError("finite array of exact shape required")
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class ElasticSection:
    stiffness: object
    name: str = "exact elastic"

    def __post_init__(self):
        c = owned(self.stiffness, (6, 6))
        if type(self.name) is not str or not self.name or len(self.name) > 256:
            raise ValueError("bounded section name required")
        if not np.array_equal(c, c.T) or np.any(np.diag(c) <= 0):
            raise ValueError("symmetric positive elastic matrix required")
        scale = np.sqrt(np.diag(c))
        np.linalg.cholesky(c / np.outer(scale, scale))
        object.__setattr__(self, "stiffness", c)

    @classmethod
    def isotropic(cls, *, E, G, area, Iy, Iz, J, shear_y, shear_z, name="isotropic"):
        values = owned([E, G, area, Iy, Iz, J, shear_y, shear_z], (8,))
        if np.any(values <= 0):
            raise ValueError("positive explicit section coefficients required")
        return cls(np.diag([E*area, shear_y*G*area, shear_z*G*area, G*J, E*Iy, E*Iz]), name)

    @classmethod
    def capture(cls, external):
        # Capture, never retain a mutable external object or a plastic surrogate.
        name = getattr(external, "name", None)
        c = generalized_beam_stiffness(external)
        if not np.array_equal(c, generalized_beam_stiffness(external)) or getattr(external, "name", None) != name:
            raise ValueError("external section changed during capture")
        return cls(c, name)

    def descriptor(self):
        return dict(policy=POLICY, name=self.name, stiffness=self.stiffness)

    @property
    def identity(self):
        return sha(self.descriptor())

    def response(self, strain, history=()):
        if type(history) is not tuple or history:
            raise ValueError("elastic material history must be empty")
        e = owned(strain, (6,))
        s = owned(self.stiffness @ e, (6,))
        potential = float(e @ s / 2)
        if not np.isfinite(potential):
            raise ValueError("nonfinite elastic work")
        return dict(resultants=s, tangent=self.stiffness,
                    potential=potential, history=())


class ElasticCell:
    """Constrained station-force minimization, with retained endpoint moments."""
    def __init__(self, section, stations):
        if type(section) is not ElasticSection:
            raise ValueError("exact elastic section required")
        self.section = section
        self.section_identity = section.identity
        self.stations = tuple((int(c), float(t), float(w), owned(v, (3, 3)))
                              for c, t, w, v in stations)
        if len(self.stations) not in (8, 16):
            raise ValueError("registered full station inventory required")
        n = 3*len(self.stations)
        compliance = solve(section.stiffness, np.eye(6), assume_a="pos")
        A, D, F, B = np.zeros((n, n)), np.zeros((n, 12)), np.zeros((12, 12)), np.zeros((6, n))
        maps = []
        for j, (cell, t, w, v) in enumerate(self.stations):
            if cell not in (0, 1) or not 0 < t < 1 or not np.isfinite(w) or w <= 0:
                raise ValueError("positive interior stations required")
            N = np.zeros((3, 12))
            N[:, 6*cell:6*cell+3] = (1-t)*np.eye(3)
            N[:, 6*cell+3:6*cell+6] = t*np.eye(3)
            k = slice(3*j, 3*j+3)
            A[k, k] = w*compliance[:3, :3]
            D[k] = w*compliance[:3, 3:] @ N
            F += w*N.T @ compliance[3:, 3:] @ N
            B[3*cell:3*cell+3, k] = w*v.T
            maps.append(owned(N))
        saddle = np.block([[A, -B.T], [B, np.zeros((6, 6))]])
        rhs = np.block([[np.zeros((n, 6)), -D], [np.eye(6), np.zeros((6, 12))]])
        lift = solve(saddle, rhs, assume_a="gen")[:n]
        P = np.c_[np.zeros((12, 6)), np.eye(12)]
        S = lift.T @ A @ lift + lift.T @ D @ P + P.T @ D.T @ lift + P.T @ F @ P
        if np.linalg.norm(S-S.T) > 1e-11*max(1., np.linalg.norm(S)):
            raise ValueError("complementary symmetry failed")
        np.linalg.cholesky(S)
        self.lift, self.compliance = owned(lift), owned(S)
        self.section_compliance, self.maps = owned(compliance), tuple(maps)
        self._capture = self._identity()

    def _identity(self):
        return sha(dict(section=self.section.identity, stations=self.stations,
                        lift=self.lift, compliance=self.compliance,
                        section_compliance=self.section_compliance, maps=self.maps))

    def guard(self):
        if self.section.identity != self.section_identity or self._identity() != self._capture:
            raise ValueError("elastic cell definition changed")

    def response(self, resultants):
        self.guard()
        p = owned(resultants, (18,))
        return float(p @ self.compliance @ p / 2), owned(self.compliance @ p), self.compliance

    def recover(self, resultants):
        self.guard()
        p = owned(resultants, (18,)); forces = self.lift @ p
        return tuple(dict(strain=owned(self.section_compliance @ np.r_[forces[3*j:3*j+3], N @ p[6:]]),
                          resultants=owned(np.r_[forces[3*j:3*j+3], N @ p[6:]]), history=())
                     for j, N in enumerate(self.maps))
