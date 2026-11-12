"""Bounded physical-patch attachment elimination of the actual stored KKT.

Five-row physical laws remain unchanged. Six beam coordinates can be removed
only when existing grouped patch rows give independent local right/left blocks.
No global nullspace/inverse, symmetrization, new solver or physical threshold.
"""
from fractions import Fraction
from hashlib import sha256
import numpy as np
from scipy import sparse
from .linalg import MatrixClass, factorize_cached

MAX_PATCH_NODES = 16
MAX_LOCAL_SUPPORT = 6*MAX_PATCH_NODES
MAX_PREPARED_BYTES = 512*1024**2  # Eligibility/storage screen; legacy fallback.


def _independent_rows(block):
    """Exact represented local rank proof only, at most 80x6 coefficients."""
    basis, selected = {}, []
    for index, values in enumerate(block):
        row = [Fraction(float(value)) for value in values]
        for pivot in sorted(basis):
            value = row[pivot]
            if value:
                row = [a-value*b for a, b in zip(row, basis[pivot])]
        pivot = next((i for i, value in enumerate(row) if value), None)
        if pivot is not None:
            value = row[pivot]
            basis[pivot] = [entry/value for entry in row]
            selected.append(index)
        if len(selected) == 6:
            return tuple(selected)
    return ()


def _bytes(matrix):
    return matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes


def _row_product_upper(matrix, right_row_counts):
    return [sum(right_row_counts[int(j)] for j in matrix.indices[first:last])
            for first, last in zip(matrix.indptr[:-1], matrix.indptr[1:])]


def _reduced_operator(left, k, right, jbeta, cbeta):
    # The storage screen bounds K@P, not L@K. Keep this association explicit:
    # raw nonsymmetric J/C can give L and P different sparse supports.
    kp = k @ right
    return sparse.bmat(((left @ kp, left @ jbeta), (cbeta @ right, None)), format="csc")


class AttachmentElimination:
    """Immutable owner-local maps, with one reused existing global factor."""
    def __init__(self, k, j, c, primal_free, beam_columns, remaining_rows,
                 anchor_rows, right, left, inverse_right, inverse_left,
                 reduced, diagnostics):
        self.k, self.j, self.c = k, j, c
        self.free, self.beam = primal_free, beam_columns
        self.beta, self.alpha = remaining_rows, anchor_rows
        self.right, self.left = right, left
        self.inverse_right, self.inverse_left = inverse_right, inverse_left
        self.reduced, self.info = reduced, diagnostics
        self.primal_count, self.constraint_count = k.shape[0], c.shape[0]
        for array in (self.free, self.beam, self.beta, self.alpha,
                      *self.inverse_right, *self.inverse_left):
            array.setflags(write=False)

    def factorize(self):
        handle = factorize_cached(self.reduced, MatrixClass.GENERAL)
        return EliminatedHandle(self, handle)

    def _local_action(self, inverses, values):
        result = np.empty_like(values)
        for group, inverse in enumerate(inverses):
            rows = slice(6*group, 6*group+6)
            result[rows] = inverse @ values[rows]
        return result

    def solve(self, handle, rhs):
        rhs = np.asarray(rhs, dtype=float)
        total = self.primal_count+self.constraint_count
        if rhs.ndim not in (1, 2) or rhs.shape[0] != total or not np.isfinite(rhs).all():
            raise RuntimeError("finite full attachment right-hand side required")
        single = rhs.ndim == 1
        values = rhs[:, None] if single else rhs
        f, d = values[:self.primal_count], values[self.primal_count:]
        lift = np.zeros_like(f)
        lift[self.beam] = self._local_action(self.inverse_right, d[self.alpha])
        reduced_rhs = np.vstack((self.left @ (f-self.k @ lift),
                                 d[self.beta]-self.c[self.beta] @ lift))
        if not np.isfinite(lift).all() or not np.isfinite(reduced_rhs).all():
            raise RuntimeError("attachment affine elimination is not representable")
        solved = handle.solve(reduced_rhs[:, 0] if single else reduced_rhs)
        solved = np.asarray(solved, dtype=float)
        solved = solved[:, None] if single else solved
        primal = self.right @ solved[:len(self.free)]+lift
        multipliers = np.empty((self.constraint_count, values.shape[1]))
        multipliers[self.beta] = solved[len(self.free):]
        anchor_rhs = f[self.beam]-(self.k @ primal)[self.beam]-(self.j[:, self.beta] @ multipliers[self.beta])[self.beam]
        multipliers[self.alpha] = self._local_action(self.inverse_left, anchor_rhs)
        result = np.vstack((primal, multipliers))
        if not np.isfinite(result).all():
            raise RuntimeError("attachment recovery is not representable")
        return result[:, 0] if single else result


class EliminatedHandle:
    def __init__(self, maps, handle):
        self.maps, self.handle = maps, handle

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def solve(self, rhs):
        return self.maps.solve(self.handle, rhs)

    def solve_many(self, rhs):
        return self.solve(rhs)

    def diagnostics(self):
        return {**self.handle.diagnostics(), "attachment_elimination": dict(self.maps.info),
                "restored_shape": [self.maps.primal_count+self.maps.constraint_count]*2}


def prepare_attachment_elimination(matrix, primal_count, transformation, incidences):
    """Return maps/reason; ineligibility never narrows existing model admission.

    Caller admits only physical static, no points/H/body/spectral. Input bytes
    are copied and the operator/support signature is fixed once, never per load.
    Arbitrary constraint RHS, including original-target correction, is retained.
    """
    a = matrix.tocsr(copy=True)
    t = transformation.tocsr(copy=True)
    a.eliminate_zeros(); t.eliminate_zeros()
    a.sort_indices(); t.sort_indices()
    n, m = primal_count, a.shape[0]-primal_count
    if n <= 0 or a.shape != (n+m, n+m) or m <= 0 or t.shape[1] != n or not np.isfinite(a.data).all():
        return None, "invalid_local_layout"
    if np.any(a[n:, n:].data != 0.):
        return None, "nonzero_multiplier_compliance"
    groups = {}
    for item in incidences:
        if item.row_count != 5:
            return None, "not_five_row_physical_incidence"
        groups.setdefault(item.footprint_id, []).append(item)
    if not groups or sum(5*len(items) for items in groups.values()) != m:
        return None, "incomplete_physical_incidence"
    inventory = [i.row_start+v for items in groups.values() for i in items for v in range(5)]
    if set(inventory) != set(range(m)) or len(inventory) != len(set(inventory)):
        return None, "invalid_physical_row_inventory"
    k, j, c = a[:n, :n].tocsr(), a[:n, n:].tocsr(), a[n:, :n].tocsr()
    tc = t.tocsc()
    selected_beam, selected_rows, used_shell, used_beam = [], [], set(), set()
    inverses_b, inverses_d, local = [], [], []
    for items in groups.values():
        if len(items) > MAX_PATCH_NODES or len({i.shell_node for i in items}) != len(items):
            return None, "bounded_patch_incidence_required"
        shell_slots = {int(v) for i in items for v in i.shell_slots}
        beam_slots = tuple(map(int, items[0].beam_slots))
        if (len(beam_slots) != 6 or any(tuple(i.beam_slots) != beam_slots for i in items) or
                used_shell.intersection(shell_slots) or used_beam.intersection(beam_slots)):
            return None, "overlapping_attachment_blocks"
        columns = []
        for slot in beam_slots:
            if slot < 0 or slot >= t.shape[0]:
                return None, "invalid_beam_coordinate_inventory"
            start, stop = t.indptr[slot:slot+2]
            if stop-start != 1 or t.data[start] != 1.:
                return None, "fully_free_beam_selector_required"
            column = int(t.indices[start])
            if tc.indptr[column+1]-tc.indptr[column] != 1:
                return None, "beam_coordinate_shared_by_affine_map"
            columns.append(column)
        if len(set(columns)) != 6:
            return None, "distinct_beam_coordinates_required"
        rows = np.array([i.row_start+v for i in items for v in range(5)], dtype=int)
        if np.any(rows < 0) or np.any(rows >= m):
            return None, "invalid_physical_row_inventory"
        rank_rows = _independent_rows(c[rows][:, columns].toarray())
        if not rank_rows:
            return None, "local_position_director_rank_deficient"
        alpha = rows[np.array(rank_rows)]
        b = c[alpha][:, columns].toarray()
        d = j[columns][:, alpha].toarray()
        if not _independent_rows(d):
            return None, "local_left_block_rank_deficient"
        outside = np.setdiff1d(np.arange(m), rows, assume_unique=True)
        if c[outside][:, columns].nnz:
            return None, "foreign_right_attachment_coupling"
        if j[columns][:, outside].nnz:
            return None, "foreign_left_attachment_coupling"
        try:
            ib, id_ = np.linalg.solve(b, np.eye(6)), np.linalg.solve(d, np.eye(6))
        except np.linalg.LinAlgError:
            return None, "local_binary64_solve_unresolved"
        if not np.isfinite(ib).all() or not np.isfinite(id_).all():
            return None, "local_elimination_unrepresentable"
        selected_beam.extend(columns); selected_rows.extend(map(int, alpha))
        used_shell.update(shell_slots); used_beam.update(beam_slots)
        inverses_b.append(ib); inverses_d.append(id_)
        local.append((np.array(columns), alpha, rows))
    beam, alpha = np.array(selected_beam), np.array(selected_rows)
    if len(set(alpha)) != len(alpha):
        return None, "overlapping_constraint_rows"
    free = np.setdiff1d(np.arange(n), beam, assume_unique=True)
    beta = np.setdiff1d(np.arange(m), alpha, assume_unique=True)
    pr, pc, pv = list(map(int, free)), list(range(len(free))), [1.]*len(free)
    lr, lc, lv = list(range(len(free))), list(map(int, free)), [1.]*len(free)
    used_support = set()
    for (columns, anchors, rows), ib, id_ in zip(local, inverses_b, inverses_d):
        right_part = c[anchors][:, free].tocsr()
        touched_right = np.unique(right_part.indices)
        left_part = j[free][:, anchors].tocsr()
        touched_left = np.unique(left_part.tocoo().row)
        support = set(map(int, touched_right)) | set(map(int, touched_left))
        all_constraint_support = set(map(int, c[rows][:, free].indices))
        if len(support | all_constraint_support) > MAX_LOCAL_SUPPORT or used_support.intersection(support | all_constraint_support):
            return None, "overlapping_or_unbounded_affine_patch_support"
        used_support.update(support | all_constraint_support)
        z = -ib @ right_part[:, touched_right].toarray()
        y = -left_part[touched_left].toarray() @ id_
        for i, column in enumerate(columns):
            for at, value in zip(touched_right, z[i]):
                if value != 0.:
                    pr.append(int(column)); pc.append(int(at)); pv.append(float(value))
        for i, at in enumerate(touched_left):
            for column, value in zip(columns, y[i]):
                if value != 0.:
                    lr.append(int(at)); lc.append(int(column)); lv.append(float(value))
    right = sparse.coo_matrix((pv, (pr, pc)), shape=(n, len(free))).tocsr()
    left = sparse.coo_matrix((lv, (lr, lc)), shape=(len(free), n)).tocsr()
    if not np.isfinite(right.data).all() or not np.isfinite(left.data).all():
        return None, "affine_maps_unrepresentable"
    kp_rows = _row_product_upper(k, list(map(int, np.diff(right.indptr))))
    kk_upper = min(len(free)**2, sum(_row_product_upper(left, kp_rows)))
    jbeta = j[:, beta].tocsr(); cbeta = c[beta].tocsr()
    jj_upper = min(len(free)*len(beta), sum(_row_product_upper(left, list(map(int, np.diff(jbeta.indptr))))))
    cc_upper = min(len(free)*len(beta), sum(_row_product_upper(cbeta, list(map(int, np.diff(right.indptr))))))
    estimated_bytes = sum(map(_bytes, (a, k, j, c, right, left)))+24*(kk_upper+jj_upper+cc_upper+sum(kp_rows))
    if estimated_bytes > MAX_PREPARED_BYTES:
        return None, "bounded_sparse_preparation_storage_exceeded"
    reduced = _reduced_operator(left, k, right, jbeta, cbeta)
    reduced.eliminate_zeros(); reduced.sort_indices()
    if not np.isfinite(reduced.data).all():
        return None, "reduced_operator_unrepresentable"
    signature = sha256()
    for operand in (a, t):
        signature.update(repr(operand.shape).encode())
        for values in (operand.data, operand.indices, operand.indptr):
            signature.update(values.tobytes())
    info = {"groups": len(groups), "eliminated_primal": len(beam), "eliminated_multipliers": len(alpha),
            "retained_constraints": len(beta), "full_shape": a.shape, "reduced_shape": reduced.shape,
            "full_nnz": a.nnz, "reduced_nnz": reduced.nnz, "right_map_nnz": right.nnz,
            "left_map_nnz": left.nnz, "estimated_sparse_array_bytes_with_product_nnz_bound": estimated_bytes,
            "prepared_operator_bytes": sum(map(_bytes, (k, j, c, right, left, reduced))),
            "storage_scope": "sparse arrays/product nnz screen; excludes Python containers, allocator temporaries and factor fill; not a process RSS upper bound",
            "operator_support_signature": signature.hexdigest(), "raw_left_right_distinct": True,
            "factor_fill_bytes": "not estimated or measured", "qualification": False}
    return AttachmentElimination(k, j, c, free, beam, beta, alpha, right, left,
                                 tuple(inverses_b), tuple(inverses_d), reduced, info), "eligible"
