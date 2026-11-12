"""Private B2 sparse affine batch feasibility; no public mixed-profile dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import math
from threading import Lock
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import sparse

from ._b2_physical_linear_definition import CompiledPhysicalB2Member
from .control import CancellationToken
from .linalg import FactorizationCache, MatrixClass, factorize_cached


class B2NumericalSolveFailure(RuntimeError):
    """A compiled B2 factorization or solve failed numerically."""


def _checked_affine_projection(rows, multi, singles, values: np.ndarray, *,
                               failure_type=B2NumericalSolveFailure,
                               label="physical B2") -> np.ndarray:
    """Preserve affine work, using exact stored-input dots on cancellation.

    Selection columns retain sparse multiplication. Only uncertain rows use
    exact binary64 products; a nonrepresentable result fails explicitly.
    """
    if not np.isfinite(values).all():
        raise failure_type(f"finite {label} affine projection required")
    result = np.asarray(rows @ values).reshape(-1)
    eps = np.finfo(float).eps
    for row in multi:
        start, stop = rows.indptr[row:row+2]
        terms = [(float(a), float(values[j])) for a, j in zip(
            rows.data[start:stop], rows.indices[start:stop])]
        magnitude = sum(abs(a*v) for a, v in terms)
        uncertainty = (2*len(terms)+2)*eps*magnitude
        if (not math.isfinite(float(result[row])) or
                not math.isfinite(magnitude) or
                (result[row] == 0. and
                 any(a != 0. and v != 0. for a, v in terms)) or
                uncertainty > 1e-9*abs(float(result[row]))):
            exact = sum((Fraction(a)*Fraction(v) for a, v in terms),
                        Fraction(0))
            try:
                result[row] = float(exact)
            except OverflowError as exc:
                raise failure_type(
                    f"{label} affine projection is not representable") from exc
            if exact != 0 and result[row] == 0.:
                raise failure_type(f"{label} affine projection is not representable")
    for row in singles[result[singles] == 0.]:
        slot = rows.indptr[row]
        a = float(rows.data[slot])
        v = float(values[rows.indices[slot]])
        if a != 0. and v != 0.:
            raise failure_type(f"{label} affine projection is not representable")
    if np.any((result != 0.) & (np.abs(result) < np.finfo(float).tiny)):
        raise failure_type(f"{label} affine projection is not representable")
    if not np.isfinite(result).all():
        raise failure_type(f"finite {label} affine projection required")
    return result


@dataclass(frozen=True)
class SelectedMemberRecovery:
    member_id: str
    stations: tuple[float, ...]
    resultants_local: tuple[tuple[float, ...], ...]
    strains_local: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SelectedCase:
    case_id: str
    values: tuple[float, ...]
    members: tuple[SelectedMemberRecovery, ...] = ()


@dataclass(frozen=True)
class BatchResult:
    status: str
    cases: tuple[SelectedCase, ...]
    factorization_count: int
    solve_count: int
    maximum_rhs_chunk: int
    production_qualified: bool = False


def _load_value_snapshot(value: np.ndarray, shape: tuple[int, ...]
                         ) -> tuple[np.ndarray, bytes]:
    """Copy one submitted load and fingerprint its finite binary64 values."""
    try:
        load = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite B2 load vector required") from exc
    if load.shape != shape or not np.isfinite(load).all():
        raise ValueError("finite B2 load vector required")
    return load, sha256(memoryview(load).cast("B")).digest()


def _accurate_sparse_residual(matrix: sparse.spmatrix, solution: np.ndarray,
                              rhs: np.ndarray, *, exact_products: bool = False) -> np.ndarray:
    """Residual of binary64 inputs with compensated products, on failure only.

    ``exact_products`` selects the existing rational path even where FMA is
    available, for the exceptional active-row resolution certificate.
    """
    rows = matrix.tocsr()
    result = np.empty_like(rhs)
    fused = None if exact_products else getattr(math, "fma", None)
    for column in range(rhs.shape[1]):
        for row in range(rows.shape[0]):
            start, stop = rows.indptr[row:row+2]
            terms = [-float(rhs[row, column])]
            exact = -Fraction(float(rhs[row, column])) if fused is None else None
            try:
                for a, k in zip(rows.data[start:stop], rows.indices[start:stop]):
                    x = float(solution[k, column])
                    product = float(a)*x
                    if not math.isfinite(product):
                        raise B2NumericalSolveFailure("physical B2 residual is not representable")
                    if fused is None:
                        exact += Fraction(float(a))*Fraction(x)
                    else:
                        terms.extend((product, fused(float(a), x, -product)))
                value = float(exact) if fused is None else math.fsum(terms)
            except (OverflowError, ValueError) as exc:
                raise B2NumericalSolveFailure(
                    "physical B2 residual is not representable") from exc
            result[row, column] = value
    if not np.isfinite(result).all():
        raise B2NumericalSolveFailure("physical B2 residual is not representable")
    return result


def _normalized_backward_limit(row_scale: np.ndarray,
                               equilibrium_scale: np.ndarray) -> np.ndarray:
    """Per-case backward bound in each reduced row's own physical units."""
    relative = np.zeros_like(row_scale)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        np.divide(row_scale, equilibrium_scale, out=relative,
                  where=equilibrium_scale > 0.)
        return (1e-9*np.max(relative, axis=0, keepdims=True)
                * equilibrium_scale)


def _fast_residual_within_margin(defect: np.ndarray, row_scale: np.ndarray,
                                 equilibrium_scale: np.ndarray,
                                 row_nnz: np.ndarray) -> bool:
    """Screen a rounded sparse dot against a conservative rowwise error bound.

    A row with ``n`` entries has at most ``2*n+2`` product/sum/subtraction
    roundings. The factor also accounts for rounding in the computed absolute
    row scale. Rows outside that bound take the accurate-product path.
    """
    if (not np.isfinite(defect).all() or not np.isfinite(row_scale).all() or
            not np.isfinite(equilibrium_scale).all() or
            np.any(row_scale <= np.finfo(float).tiny*1e9) or
            np.any(equilibrium_scale <= np.finfo(float).tiny*1e9)):
        return False
    unit_roundoff = np.finfo(float).eps/2
    mu = (2.*row_nnz+2.)*unit_roundoff
    if np.any(mu >= .25):
        return False
    # gamma_(2n+2)/(1-gamma_(2n+2)) = mu/(1-2*mu).
    # Fourfold allowance covers rounding of the bound arithmetic itself;
    # near-subnormal scales always use accurate products instead.
    factor = 4.*mu/(1.-2.*mu)
    smallest = np.nextafter(0., 1.)
    with np.errstate(over="ignore", invalid="ignore"):
        uncertainty = np.nextafter(factor[:, None]*row_scale, np.inf)
        uncertainty += (2.*row_nnz[:, None]+2.)*smallest
        margin = np.nextafter(np.abs(defect)+uncertainty, np.inf)
        backward_limit = _normalized_backward_limit(
            row_scale, equilibrium_scale)
    return bool(np.isfinite(margin).all() and
                np.isfinite(backward_limit).all() and
                np.all(margin <= backward_limit) and
                np.all(margin <= 1e-9*equilibrium_scale))


class CompiledB2AffineBatch:
    """Bounded graph/affine prototype with one owned sparse factorization."""

    def __init__(self, members: Sequence[CompiledPhysicalB2Member],
                 transformation: sparse.spmatrix, prescribed_offset,
                 selected_dofs, *, max_chunk: int = 8,
                 factorization_cache: FactorizationCache | None = None):
        compiled = tuple(members)
        if not compiled or any(not isinstance(m, CompiledPhysicalB2Member)
                               for m in compiled):
            raise ValueError("compiled physical B2 members required")
        if type(max_chunk) is not int or max_chunk <= 0:
            raise ValueError("positive bounded RHS chunk required")
        if factorization_cache is not None and not isinstance(
                factorization_cache, FactorizationCache):
            raise ValueError("local factorization cache required")
        ids = [m.definition.member_id for m in compiled]
        if len(set(ids)) != len(ids):
            raise ValueError("unique physical B2 member IDs required")
        points = {}
        for member in compiled:
            for node_id, point in zip(member.definition.node_ids,
                                      member.definition.reference_points_m):
                old = points.setdefault(node_id, point)
                if old != point:
                    raise ValueError("shared B2 node has inconsistent position")
        ordered = tuple(sorted(points))
        node_index = {node_id: i for i, node_id in enumerate(ordered)}
        size = 6*len(ordered)
        try:
            T = transformation.tocsr(copy=True)
            u0 = np.asarray(prescribed_offset, dtype=float).copy()
            selected_input = np.asarray(selected_dofs)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("finite sparse affine plan and selection required") from exc
        if selected_input.dtype.kind not in "iu":
            raise ValueError("integer selected DOFs required")
        selected = selected_input.astype(int, copy=True)
        if (T.shape[0] != size or T.shape[1] == 0 or
                u0.shape != (size,) or selected.ndim != 1 or
                np.any(selected < 0) or np.any(selected >= size) or
                not np.isfinite(T.data).all() or not np.isfinite(u0).all()):
            raise ValueError("finite sparse affine plan and selection required")
        rows, cols, values = [], [], []
        for member in compiled:
            local = member.operator.evaluate(np.zeros(12)).tangent
            frame = member.local_dof_map()
            stiffness = frame.T@local@frame
            dofs = [6*node_index[node]+j for node in member.definition.node_ids
                    for j in range(6)]
            for i, row in enumerate(dofs):
                for j, col in enumerate(dofs):
                    rows.append(row); cols.append(col); values.append(stiffness[i, j])
        K = sparse.coo_matrix((values, (rows, cols)),
                              shape=(size, size)).tocsr()
        reduced = (T.T@K@T).tocsc()
        row_nnz = np.diff(reduced.tocsr().indptr).astype(float)
        diagonal = reduced.diagonal()
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
            raise B2NumericalSolveFailure(
                "physical B2 batch factorization failed: positive finite diagonal required")
        with np.errstate(over="ignore", under="ignore", divide="ignore",
                         invalid="ignore"):
            scaling = 1./np.sqrt(diagonal)
            scaled = (sparse.diags(scaling)@reduced@sparse.diags(scaling)).tocsc()
        if (not np.isfinite(scaling).all() or np.any(scaling == 0.) or
                not np.isfinite(scaled.data).all() or
                np.any(scaled.diagonal() <= 0.)):
            raise B2NumericalSolveFailure(
                "physical B2 batch factorization failed: diagonal scaling is not representable")
        handle = factorize_cached(
            scaled, MatrixClass.SPD, cache=factorization_cache)
        if handle.status != "ok":
            raise B2NumericalSolveFailure("physical B2 batch factorization failed: " +
                               str(handle.failure_reason))
        self._K = K
        self._reduced = reduced
        row_nnz.setflags(write=False)
        self._row_nnz = row_nnz
        self._T = T
        # Compile the work map once; graph execution never imports a shell owner.
        self._work_rows = T.T.tocsr(copy=True)
        self._work_rows.eliminate_zeros()
        work_nnz = np.diff(self._work_rows.indptr)
        self._work_multi = np.flatnonzero(work_nnz > 1)
        self._work_singles = np.flatnonzero(work_nnz == 1)
        self._u0 = u0
        self._ku0 = np.asarray(K@u0).reshape(-1)
        self._length_scale = max(member.definition.length_m for member in compiled)
        self._selected = selected
        self._handle = handle
        scaling.setflags(write=False)
        self._scaling = scaling
        self._max_chunk = max_chunk
        self._closed = False
        self._operation_lock = Lock()
        self.node_ids = ordered
        self.members = compiled
        self._member_layout = {
            member.definition.member_id: (
                member,
                np.array([6*node_index[node]+j
                          for node in member.definition.node_ids
                          for j in range(6)], dtype=int),
                member.local_dof_map(),
            ) for member in compiled
        }

    @property
    def full_stiffness(self) -> sparse.csr_matrix:
        if self._closed:
            raise RuntimeError("physical B2 batch is closed")
        return self._K.copy()

    def close(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("physical B2 batch operation is active")
        try:
            self._closed = True
            self._handle = None
        finally:
            self._operation_lock.release()

    def solve_many(self, cases: Sequence[tuple[str, np.ndarray]], *,
                   cancellation_token: CancellationToken | None = None,
                   on_case: Callable[[str], None] | None = None,
                   recover_members: Mapping[str, Sequence[float]] | None = None,
                   ) -> BatchResult:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("physical B2 batch operation is active")
        try:
            return self._solve_many_locked(
                cases, cancellation_token=cancellation_token,
                on_case=on_case, recover_members=recover_members)
        finally:
            self._operation_lock.release()

    def _solve_many_locked(self, cases: Sequence[tuple[str, np.ndarray]], *,
                           cancellation_token: CancellationToken | None,
                           on_case: Callable[[str], None] | None,
                           recover_members: Mapping[str, Sequence[float]] | None,
                           ) -> BatchResult:
        if self._closed:
            raise RuntimeError("physical B2 batch is closed")
        # Freeze the case inventory before callbacks can change the caller's
        # sequence. Fingerprint each load without retaining an unbounded RHS
        # batch; each numerical load is copied again when its chunk begins.
        submitted = tuple((identity, value) for identity, value in cases)
        ids = [identity for identity, _ in submitted]
        if (any(not isinstance(identity, str) or not identity.strip()
                for identity in ids) or len(set(ids)) != len(ids)):
            raise ValueError("unique nonempty ordered B2 case IDs required")
        recovery = []
        if recover_members is not None:
            if not isinstance(recover_members, Mapping):
                raise ValueError("selected B2 recovery mapping required")
            for member_id, requested in recover_members.items():
                if member_id not in self._member_layout:
                    raise ValueError("selected B2 member ID is not compiled")
                try:
                    stations = np.asarray(requested, dtype=float).copy()
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("finite normalized B2 recovery stations required") from exc
                if (stations.ndim != 1 or stations.size == 0 or
                        not np.isfinite(stations).all() or
                        np.any(stations < 0.) or np.any(stations > 1.)):
                    raise ValueError("finite normalized B2 recovery stations required")
                recovery.append((member_id, tuple(float(x) for x in stations)))
        fingerprints = []
        for _identity, value in submitted:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                break
            snapshot, fingerprint = _load_value_snapshot(value, self._u0.shape)
            fingerprints.append(fingerprint)
            del snapshot
        completed = []
        largest = 0
        for start in range(0, len(submitted), self._max_chunk):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                break
            block = submitted[start:start+self._max_chunk]
            loads = []
            for j, (_identity, value) in enumerate(block):
                load, fingerprint = _load_value_snapshot(value, self._u0.shape)
                if fingerprint != fingerprints[start+j]:
                    raise ValueError("B2 load case changed after submission")
                loads.append(load)
            largest = max(largest, len(loads))
            driving = np.column_stack(loads)-self._ku0[:, None]
            rhs = np.column_stack([
                _checked_affine_projection(
                    self._work_rows, self._work_multi, self._work_singles,
                    driving[:, column])
                for column in range(len(loads))])
            # A small backward error can still conceal an unacceptable
            # physical imbalance when a weak section creates huge motions.
            # Scale force and moment rows from the applied generalized loads,
            # then pull those work-conjugate scales through the affine map.
            nodal_wrenches = np.abs(driving).reshape(-1, 6, len(loads))
            force_scale = np.max(nodal_wrenches[:, :3, :], axis=(0, 1))
            moment_scale = np.maximum(
                np.max(nodal_wrenches[:, 3:, :], axis=(0, 1)),
                self._length_scale*force_scale)
            force_scale = np.maximum(force_scale,
                                     moment_scale/self._length_scale)
            physical_scale = np.empty_like(driving)
            for component in range(6):
                physical_scale[component::6, :] = (
                    force_scale if component < 3 else moment_scale)
            equilibrium_scale = np.asarray(abs(self._T).T@physical_scale)
            if not np.isfinite(equilibrium_scale).all():
                raise B2NumericalSolveFailure("physical B2 equilibrium scale is not representable")
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_rhs = self._scaling[:, None]*rhs
            if (not np.isfinite(scaled_rhs).all() or
                    np.any((rhs != 0.) & (scaled_rhs == 0.))):
                raise B2NumericalSolveFailure("physical B2 scaled load is not representable")
            try:
                solved = np.asarray(self._handle.solve_many(scaled_rhs), dtype=float)
            except RuntimeError as exc:
                raise B2NumericalSolveFailure(
                    "physical B2 sparse solve failed: " + str(exc)) from exc
            with np.errstate(over="ignore", invalid="ignore"):
                solved = self._scaling[:, None]*solved
            if not np.isfinite(solved).all():
                raise B2NumericalSolveFailure("physical B2 solution is not representable")
            with np.errstate(over="ignore", invalid="ignore"):
                defect = np.asarray(self._reduced@solved-rhs)
                row_scale = (np.abs(self._reduced)@np.abs(solved) +
                             np.abs(rhs))
            def acceptable(error, backward_scale):
                backward_limit = _normalized_backward_limit(
                    backward_scale, equilibrium_scale)
                return (np.isfinite(error).all() and
                        np.isfinite(backward_scale).all() and
                        np.isfinite(backward_limit).all() and
                        np.all(np.abs(error) <= backward_limit) and
                        np.all(np.abs(error) <= 1e-9*equilibrium_scale))

            # A rounded sparse dot can hide many small terms. Trust it only
            # when a row-length bound fits below both acceptance margins.
            if not _fast_residual_within_margin(
                    defect, row_scale, equilibrium_scale, self._row_nnz):
                if not np.isfinite(row_scale).all():
                    raise B2NumericalSolveFailure("physical B2 batch solve residual failed")
                for _ in range(2):
                    defect = _accurate_sparse_residual(self._reduced, solved, rhs)
                    row_scale = (np.abs(self._reduced)@np.abs(solved) +
                                 np.abs(rhs))
                    if acceptable(defect, row_scale):
                        break
                    correction_rhs = self._scaling[:, None]*defect
                    if not np.isfinite(correction_rhs).all():
                        raise B2NumericalSolveFailure("physical B2 batch solve residual failed")
                    try:
                        correction = np.asarray(self._handle.solve_many(
                            correction_rhs), dtype=float)
                    except RuntimeError as exc:
                        raise B2NumericalSolveFailure(
                            "physical B2 sparse correction failed: " + str(exc)) from exc
                    solved -= self._scaling[:, None]*correction
                    if not np.isfinite(solved).all():
                        raise B2NumericalSolveFailure("physical B2 batch solve residual failed")
                else:
                    defect = _accurate_sparse_residual(self._reduced, solved, rhs)
                    row_scale = (np.abs(self._reduced)@np.abs(solved) +
                                 np.abs(rhs))
                    if not acceptable(defect, row_scale):
                        raise B2NumericalSolveFailure("physical B2 batch solve residual failed")
            selected = self._T[self._selected]@solved
            selected = np.asarray(selected)+self._u0[self._selected, None]
            if not np.isfinite(selected).all():
                raise B2NumericalSolveFailure("physical B2 selected solution is not finite")
            for j, (identity, _value) in enumerate(block):
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    break
                member_results = []
                for member_id, stations in recovery:
                    member, dofs, frame = self._member_layout[member_id]
                    physical = np.asarray(self._T[dofs]@solved[:, j]).reshape(-1)
                    physical += self._u0[dofs]
                    response = member.operator.recover_sections(
                        frame@physical, stations=stations)
                    member_results.append(SelectedMemberRecovery(
                        member_id, stations,
                        tuple(tuple(float(value) for value in row)
                              for row in response.resultants),
                        tuple(tuple(float(value) for value in row)
                              for row in response.strains)))
                    if cancellation_token is not None and cancellation_token.is_cancelled:
                        break
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    break
                completed.append(SelectedCase(
                    identity, tuple(float(v) for v in selected[:, j]),
                    tuple(member_results)))
                if on_case is not None:
                    on_case(identity)
            if cancellation_token is not None and cancellation_token.is_cancelled:
                break
        return BatchResult(
            "cancelled" if cancellation_token is not None and
            cancellation_token.is_cancelled else "completed",
            tuple(completed), self._handle.factorization_count,
            self._handle.solve_count, largest)
