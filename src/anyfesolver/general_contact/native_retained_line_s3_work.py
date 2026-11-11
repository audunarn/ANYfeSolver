"""Local native B3-GE/S3 contact work on authoritative shell DOFs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver._ge_beam3_p5_seeded.core import sha

from .native_retained_line_s3 import (
    NativeRetainedLineS3Result,
    NativeRetainedRoundLineS3Interior,
)
from .native_retained_station import _readonly
from .s3_skin_work import S3SkinWorkAdapter


PROFILE = "NATIVE_B3_GE_S3_INTERIOR_AUTHORITATIVE_WORK_DEV_V0"


@dataclass(frozen=True)
class NativeRetainedLineS3WorkResult:
    contact: NativeRetainedLineS3Result
    residual: np.ndarray
    tangent: sparse.csr_matrix
    profile: str = PROFILE


class NativeRetainedLineS3Work:
    """Pull physical S3 skin-vertex work to the shell formulation's DOFs."""

    def __init__(self, contact: NativeRetainedRoundLineS3Interior,
                 shell: S3SkinWorkAdapter):
        if (type(contact) is not NativeRetainedRoundLineS3Interior
                or type(shell) is not S3SkinWorkAdapter):
            raise TypeError("native line-S3 work needs exact contact and S3 skin adapters")
        if (shell.body_id != contact.shell_reference.feature_id[0]
                or shell.element_id != contact.shell_element_id
                or shell.node_ids != contact.shell_node_ids
                or not np.array_equal(shell.reference_positions,
                                      contact.shell_reference.corners)):
            raise ValueError("native line-S3 physical shell reference does not bind adapter")
        if (not sparse.isspmatrix_csr(shell.work_map)
                or shell.work_map.shape[0] != 9):
            raise ValueError("native line-S3 shell work map must have nine physical rows")
        self.contact = contact
        self.shell = shell
        self.work_map = sparse.block_diag(
            (sparse.eye(42, format="csr"), shell.work_map), format="csr",
        )
        self.identity = self._fingerprint()

    def _fingerprint(self) -> str:
        mapping = self.shell.work_map
        return sha(dict(
            profile=PROFILE,
            contact=self.contact.identity,
            body=self.shell.body_id,
            element=self.shell.element_id,
            side=self.shell.side,
            nodes=self.shell.node_ids,
            director=self.shell.director,
            offset=self.shell.physical_offset,
            reference=self.shell.reference_positions,
            map_shape=mapping.shape,
            map_data=mapping.data,
            map_indices=mapping.indices,
            map_indptr=mapping.indptr,
        ))

    def evaluate(self, positions: np.ndarray, position_low: np.ndarray,
                 cell_rotations: np.ndarray,
                 shell_displacement: np.ndarray, *, _work_budget=None) -> NativeRetainedLineS3WorkResult:
        if _work_budget is not None:_work_budget.debit('geometry')
        if self._fingerprint() != self.identity:
            raise ValueError("native line-S3 shell work definition changed")
        physical = self.shell.current_positions(shell_displacement)
        if _work_budget is None:
            contact = self.contact.evaluate(positions, position_low, cell_rotations, physical)
        else:
            contact = self.contact.evaluate(positions, position_low, cell_rotations, physical,
                                            _work_budget=_work_budget, _geometry_predebited=True)
        residual = np.asarray(self.work_map.T @ contact.residual).reshape(-1)
        tangent = (self.work_map.T @ sparse.csr_matrix(contact.tangent)
                   @ self.work_map).tocsr()
        tangent.sort_indices()
        return NativeRetainedLineS3WorkResult(
            contact, _readonly(residual), tangent,
        )
