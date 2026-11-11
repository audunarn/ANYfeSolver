"""Experimental physical B2 linear graph session for the stiffened-shell facade.

This owns an elastic B2 graph and one sparse affine factorization.  Shell
attachment, inertia, spectra, buckling and production admission are separate
capabilities and are deliberately absent from this session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import sparse

from ._b2_physical_linear_batch_dev import (
    B2NumericalSolveFailure, CompiledB2AffineBatch, SelectedCase,
)
from ._b2_physical_linear_definition import (
    PROFILE_ID, CompiledPhysicalB2Member, PhysicalB2Member,
    PhysicalB2OperatorCache,
)
from .control import CancellationToken


class ExperimentalB2Error(RuntimeError):
    """An experimental B2 graph could not be prepared or used."""


@dataclass(frozen=True)
class ExperimentalB2Capabilities:
    profile_id: str = PROFILE_ID
    status: str = "experimental"
    production_qualified: bool = False
    shell_attachment_supported: bool = False
    domain: str = "connected physical B2 beam graph"
    operations: tuple[str, ...] = ("solve_linear_many", "close")


@dataclass(frozen=True)
class ExperimentalB2Result:
    status: str
    cases: tuple[SelectedCase, ...]
    factorization_count: int | None
    solve_count: int | None
    maximum_rhs_chunk: int | None
    failure_reason: str | None = None
    production_qualified: bool = False


def _connected(members: tuple[PhysicalB2Member, ...]) -> bool:
    adjacency: dict[int, set[int]] = {}
    for member in members:
        a, b = member.node_ids
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen = {members[0].node_ids[0]}
    pending = list(seen)
    while pending:
        for neighbour in adjacency[pending.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                pending.append(neighbour)
    return len(seen) == len(adjacency)


class ExperimentalB2Analysis:
    """One owned connected physical B2 graph with an explicit affine plan.

    Load vectors and selected DOFs use the sorted node-ID order, six global
    translations/rotations per node.  The affine plan is supplied by the
    existing constraint auditor; this class does not infer supports or ports.
    """

    def __init__(self, members: Sequence[PhysicalB2Member | CompiledPhysicalB2Member],
                 transformation: sparse.spmatrix, prescribed_offset,
                 selected_dofs, *, max_rhs_chunk: int = 8):
        supplied = tuple(members)
        if not supplied or any(not isinstance(member, (
                PhysicalB2Member, CompiledPhysicalB2Member)) for member in supplied):
            raise ValueError("physical B2 graph members required")
        definitions = tuple(member.definition if isinstance(
            member, CompiledPhysicalB2Member) else member for member in supplied)
        if not _connected(definitions):
            raise ValueError("connected physical B2 graph required")
        if len({member.member_id for member in definitions}) != len(definitions):
            raise ValueError("unique physical B2 member IDs required")
        # Definitions and compiled members own immutable canonical inputs.
        # Keep an already compiled operator; compile raw definitions once.
        cache = PhysicalB2OperatorCache()
        compiled = tuple(
            member if isinstance(member, CompiledPhysicalB2Member)
            else member.compile_local(operator_cache=cache)
            for member in supplied)
        try:
            batch = CompiledB2AffineBatch(
                compiled, transformation, prescribed_offset, selected_dofs,
                max_chunk=max_rhs_chunk)
        except B2NumericalSolveFailure as exc:
            raise ExperimentalB2Error(str(exc)) from exc
        self._members = compiled
        self._batch = batch
        self._closed = False

    @property
    def capabilities(self) -> ExperimentalB2Capabilities:
        return ExperimentalB2Capabilities()

    @property
    def members(self) -> tuple[CompiledPhysicalB2Member, ...]:
        return self._members

    @property
    def node_ids(self) -> tuple[int, ...]:
        return self._batch.node_ids

    def solve_linear_many(
            self, cases: Sequence[tuple[str, np.ndarray]], *,
            cancellation_token: CancellationToken | None = None,
            recover_members: Mapping[str, Sequence[float]] | None = None,
            on_case: Callable[[str], None] | None = None,
            ) -> ExperimentalB2Result:
        """Return complete cases only; preserve cancellation as an ordered prefix."""
        if self._closed:
            raise ExperimentalB2Error("experimental B2 analysis is closed")
        try:
            result = self._batch.solve_many(
                cases, cancellation_token=cancellation_token,
                recover_members=recover_members, on_case=on_case)
        except B2NumericalSolveFailure as exc:
            # A numerical failure cannot be reported as an offset-only or
            # partially solved case.  Unexpected callback/programming errors
            # propagate with their original details.
            return ExperimentalB2Result(
                "failed", (), None, None, None, str(exc))
        return ExperimentalB2Result(
            result.status, result.cases, result.factorization_count,
            result.solve_count, result.maximum_rhs_chunk)

    def close(self) -> None:
        if not self._closed:
            self._batch.close()
            self._closed = True


def create_experimental_b2_analysis(
        members: Sequence[PhysicalB2Member | CompiledPhysicalB2Member],
        transformation: sparse.spmatrix, prescribed_offset, selected_dofs,
        *, max_rhs_chunk: int = 8) -> ExperimentalB2Analysis:
    """Construct the bounded B2 session without changing legacy dispatch."""
    return ExperimentalB2Analysis(
        members, transformation, prescribed_offset, selected_dofs,
        max_rhs_chunk=max_rhs_chunk)
