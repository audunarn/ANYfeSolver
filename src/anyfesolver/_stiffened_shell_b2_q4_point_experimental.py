"""One flat Q4 and physical B2 member joined by the existing F10 point law.

This is a bounded reference-linear experiment.  The six constraint rows are
compiled by F10/F11; this module only owns sparse assembly, affine reduction,
the solve, and physical equilibrium output.  It makes no line or footprint,
dynamic, stability, nonlinear or qualification claim.
"""

from __future__ import annotations

from .elements import create_shell_element

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy import sparse

from ._b2_f10_f11_point_attachment import LAW_ID, compile_reference_point
from ._b2_physical_linear_definition import (
    PROFILE_ID, CompiledPhysicalB2Member, PhysicalB2Member,
)
from .fe_core import FEModel
from .linalg import MatrixClass, factorize_cached
from ._stiffened_shell_b2_q4_panel_experimental import (
    _checked_affine_projection, _NumericalFailure as _AffineProjectionFailure,
)


class ExperimentalB2Q4PointError(RuntimeError):
    """The bounded point-coupled reference-linear problem cannot be used."""


class _NumericalFailure(RuntimeError):
    """A sparse solve or checked mixed equilibrium failed."""


@dataclass(frozen=True)
class ExperimentalB2Q4PointCapabilities:
    profile_id: str = PROFILE_ID
    point_law_id: str = LAW_ID
    status: str = "experimental"
    production_qualified: bool = False
    shell_domain: str = "one flat elastic Q4"
    beam_domain: str = "one physical B2 member"
    attachment: str = "one F10 reference point"
    operations: tuple[str, ...] = ("solve_linear", "close")


@dataclass(frozen=True)
class ExperimentalB2Q4PointResult:
    status: str
    displacements: tuple[float, ...] = ()
    multiplier: tuple[float, ...] = ()
    affine_reactions: tuple[float, ...] = ()
    point_gap: tuple[float, ...] = ()
    beam_resultants_local: tuple[tuple[float, ...], ...] = ()
    internal_strain_energy: float | None = None
    factorization_count: int | None = None
    solve_count: int | None = None
    failure_reason: str | None = None
    production_qualified: bool = False


class ExperimentalB2Q4PointAnalysis:
    """Compiled 36-coordinate Q4/B2 point system with six algebraic rows.

    The coordinate order is four shell vertices followed by two B2 member
    nodes, with six global translations/rotations per node.  Callers supply
    an audited affine plan ``u=T q+u0`` in that exact order. Returned affine
    reactions include all forces in the null space of ``T.T``; they are support
    reactions only for a pure fixed-coordinate plan.
    """

    def __init__(self, member: PhysicalB2Member | CompiledPhysicalB2Member,
                 shell_vertices_m, *, shell_E: float, shell_nu: float,
                 shell_thickness_m: float, station: float,
                 transformation: sparse.spmatrix, prescribed_offset,
                 offset_local_m=(0., 0., 0.), reference_normal=(0., 0., 1.)):
        if isinstance(member, CompiledPhysicalB2Member):
            compiled = member
            definition = member.definition
        elif isinstance(member, PhysicalB2Member):
            definition = member
            compiled = member.compile_local()
        else:
            raise ValueError("physical B2 member required")
        vertices = np.array(shell_vertices_m, dtype=float, copy=True)
        if vertices.shape != (4, 3) or not np.isfinite(vertices).all():
            raise ValueError("four finite Q4 shell vertices required")
        model = FEModel("experimental-B2-Q4-point")
        model.add_material("shell", shell_E, shell_nu)
        for index, point in enumerate(vertices, 1):
            model.add_node(index, *point)
        shell = create_shell_element(
            1, (1, 2, 3, 4), "shell", formulation="e4-pl", thickness=shell_thickness_m,
            reference_normal=np.asarray(reference_normal, dtype=float))
        model.add_element(1, shell)
        shell_k = shell.compute_stiffness_matrix(
            model.mesh, model.get_material("shell"))
        if shell_k.shape != (24, 24) or not np.isfinite(shell_k).all():
            raise ExperimentalB2Q4PointError("finite physical Q4 stiffness required")
        frame = compiled.local_dof_map()
        beam_k = frame.T @ compiled.operator.evaluate(np.zeros(12)).tangent @ frame
        if beam_k.shape != (12, 12) or not np.isfinite(beam_k).all():
            raise ExperimentalB2Q4PointError("finite physical B2 stiffness required")
        law = compile_reference_point(
            definition, vertices, station=station,
            offset_local_m=offset_local_m)
        try:
            T = transformation.tocsr(copy=True)
            u0 = np.array(prescribed_offset, dtype=float, copy=True)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("finite sparse Q4/B2 affine plan required") from exc
        if (T.shape[0] != 36 or T.shape[1] <= 6 or
                u0.shape != (36,) or not np.isfinite(T.data).all() or
                not np.isfinite(u0).all()):
            raise ValueError("finite sparse Q4/B2 affine plan required")
        K = sparse.block_diag((shell_k, beam_k), format="csr")
        C = sparse.csr_matrix(law.constraint)
        reduced_k = (T.T @ K @ T).tocsc()
        reduced_c = (C @ T).tocsc()
        if np.linalg.matrix_rank(reduced_c.toarray()) != 6:
            raise ValueError("six independent supported point rows required")
        kkt = sparse.bmat(
            ((reduced_k, reduced_c.T), (reduced_c, None)),
            format="csc")
        handle = factorize_cached(kkt, MatrixClass.SYMMETRIC_INDEFINITE)
        if handle.status != "ok":
            raise ExperimentalB2Q4PointError(
                "Q4/B2 point factorization failed: " + str(handle.failure_reason))
        self._member = compiled
        self._law = law
        self._K = K
        self._C = C
        self._T = T
        self._Tt = T.T.tocsr()
        self._projection_multi = np.flatnonzero(np.diff(self._Tt.indptr) > 1)
        self._projection_single = np.flatnonzero(np.diff(self._Tt.indptr) == 1)
        self._abs_K, self._abs_C = abs(K), abs(C)
        self._abs_Ct, self._abs_Tt = abs(C.T), abs(T.T)
        self._u0 = u0
        self._ku0 = np.asarray(K @ u0).reshape(-1)
        self._cu0 = np.asarray(C @ u0).reshape(-1)
        self._handle = handle
        self._closed = False

    @property
    def capabilities(self) -> ExperimentalB2Q4PointCapabilities:
        return ExperimentalB2Q4PointCapabilities()

    def solve_linear(self, load) -> ExperimentalB2Q4PointResult:
        if self._closed:
            raise ExperimentalB2Q4PointError("experimental Q4/B2 point analysis is closed")
        try:
            applied = np.array(load, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("finite 36-coordinate Q4/B2 load required") from exc
        if applied.shape != (36,) or not np.isfinite(applied).all():
            raise ValueError("finite 36-coordinate Q4/B2 load required")
        try:
            rhs = np.r_[self._project(applied-self._ku0), -self._cu0]
            answer = np.asarray(self._handle.solve(rhs), dtype=float)
        except (_NumericalFailure, RuntimeError) as exc:
            return ExperimentalB2Q4PointResult(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count, failure_reason=str(exc))
        try:
            if answer.shape != rhs.shape or not np.isfinite(answer).all():
                raise _NumericalFailure("finite Q4/B2 point solution required")
            q, lam = answer[:-6], answer[-6:]
            displacement = np.asarray(self._T @ q).reshape(-1) + self._u0
            gap = np.asarray(self._C @ displacement).reshape(-1)
            reaction = np.asarray(self._K @ displacement +
                                  self._C.T @ lam - applied).reshape(-1)
            if (not np.isfinite(displacement).all() or
                    not np.isfinite(gap).all() or not np.isfinite(reaction).all()):
                raise _NumericalFailure("finite Q4/B2 point recovery required")
            # Preserve numerical non-success; a finite solve alone is not an
            # equilibrium or constraint pass.
            reduced_reaction = self._project(reaction)
            # Same physical row-wise contract as the multi-Q4 panel owner:
            # no fixed N/Nm/m/rad floor can conceal a failed tiny-load solve.
            gap_scale = np.asarray(self._abs_C @ np.abs(displacement)).reshape(-1)
            physical_force_scale = np.asarray(
                self._abs_K @ np.abs(displacement)+
                self._abs_Ct @ np.abs(lam)+np.abs(applied)).reshape(-1)
            equilibrium_scale = np.asarray(
                self._abs_Tt @ physical_force_scale).reshape(-1)
            by_node = physical_force_scale.reshape(-1, 6)
            force_roundoff = float(np.max(by_node[:, :3]))
            moment_roundoff = float(np.max(by_node[:, 3:]))
            physical_roundoff = np.tile(
                (force_roundoff,)*3+(moment_roundoff,)*3,len(by_node))
            projected_roundoff = np.asarray(
                self._abs_Tt @ physical_roundoff).reshape(-1)
            roundoff_fraction = len(rhs)*np.finfo(float).eps
            explicit_rhs = np.abs(self._project(applied))
            resolution_floor = roundoff_fraction*projected_roundoff
            active = explicit_rhs > 0.
            if (roundoff_fraction >= 1e-9 or
                    np.any(resolution_floor[active] > 1e-9*explicit_rhs[active])):
                raise _NumericalFailure("Q4/B2 point mixed-load resolution insufficient")
            equilibrium_tolerance = (1e-9*equilibrium_scale+
                                     roundoff_fraction*projected_roundoff)
            if (not np.isfinite(gap_scale).all() or
                    not np.isfinite(equilibrium_tolerance).all() or
                    not np.isfinite(reduced_reaction).all() or
                    np.any(np.abs(gap) > 1e-9*gap_scale) or
                    np.any(np.abs(reduced_reaction) > equilibrium_tolerance)):
                raise _NumericalFailure("Q4/B2 point equilibrium or gap failed")
        except _NumericalFailure as exc:
            return ExperimentalB2Q4PointResult(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count, failure_reason=str(exc))
        local = self._member.local_dof_map() @ displacement[24:]
        if not np.isfinite(local).all():
            return ExperimentalB2Q4PointResult(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count,
                failure_reason="finite Q4/B2 local recovery coordinates required")
        recovered = self._member.operator.recover_sections(
            local, stations=(0., .5, 1.))
        energy = .5*float(displacement @ (self._K @ displacement))
        if (not isfinite(energy) or
                not np.isfinite(recovered.resultants).all() or
                not np.isfinite(recovered.strains).all()):
            return ExperimentalB2Q4PointResult(
                "failed", factorization_count=self._handle.factorization_count,
                solve_count=self._handle.solve_count,
                failure_reason="finite Q4/B2 physical recovery required")
        return ExperimentalB2Q4PointResult(
            "completed", tuple(map(float, displacement)), tuple(map(float, lam)),
            tuple(map(float, reaction)), tuple(map(float, gap)),
            tuple(tuple(map(float, row)) for row in recovered.resultants),
            energy, self._handle.factorization_count, self._handle.solve_count)

    def _project(self, values: np.ndarray) -> np.ndarray:
        try:
            return _checked_affine_projection(
                self._Tt, self._projection_multi, self._projection_single, values)
        except _AffineProjectionFailure as exc:
            raise _NumericalFailure("Q4/B2 point affine projection failed: "+str(exc)) from exc

    def close(self) -> None:
        self._closed = True
        self._handle = None


def create_experimental_b2_q4_point_analysis(*args, **kwargs):
    return ExperimentalB2Q4PointAnalysis(*args, **kwargs)
