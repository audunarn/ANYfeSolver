"""Bounded signed envelopes of physical case results.

Keys are caller-selected quantities, usually ``(field, entity, station,
component)``. Derived stresses must be evaluated for each physical case before
calling ``update``. Associated components are copied from that same case;
independent extrema are never presented as one simultaneous stress state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Hashable, Mapping

import numpy as np


@dataclass(frozen=True)
class GoverningValue:
    """One signed value and its physical case/station identity."""

    value: float
    case_id: object
    input_index: int
    time: float | None
    quantity: Hashable
    associated: Mapping[Hashable, float]


@dataclass(frozen=True)
class EnvelopeExtrema:
    minimum: GoverningValue
    maximum: GoverningValue
    absolute_maximum: GoverningValue


class EnvelopeReducer:
    """Retain three governing observations per selected scalar quantity.

    The first update freezes the selection. Later updates must contain exactly
    those keys, preventing an accidentally growing selection from retaining
    case-sized data. Equal values (or equal absolute values) use the earliest
    input position, independently of label uniqueness or update order.
    ``complete`` is false until the caller explicitly certifies exhaustion.
    A failed update or batch is atomic and marks the reducer incomplete.
    """

    def __init__(self) -> None:
        self._extrema: dict[Hashable, EnvelopeExtrema] = {}
        self._keys: frozenset[Hashable] | None = None
        self.case_count = 0
        self.complete = False

    @property
    def extrema(self) -> Mapping[Hashable, EnvelopeExtrema]:
        return MappingProxyType(self._extrema)

    def finish(self) -> None:
        """Certify that all intended physical cases were processed."""
        self.complete = True

    def update_batch(self, block) -> None:
        """Consume a solver or ANYfem selected-result block in input order.

        Array indices are retained as station/component coordinates. Every
        stress extremum retains the selected physical element components from
        its own governing case, including already evaluated derived measures.
        The reducer never computes von Mises from independently enveloped
        tensor components. Storage is bounded by the fixed output selection.

        ``reaction_node_ids`` defines the intentionally selected node
        universe, not all model nodes. Only selected nodes are enveloped;
        forces at nodes outside the selection are ignored, and the selection
        is never broadened. The producer supplies compact force maps that may
        omit zero entries: an omitted selected node contributes zeros, while
        every present entry must be a six-component real vector.

        The whole block is atomic: on any error the reducer keeps the extrema,
        frozen selection and case count it had before the block, and stays
        incomplete.
        """
        self.complete = False
        count = len(block.case_results)
        displacements = np.asarray(block.displacements)
        if (displacements.shape != (len(block.output_dofs), count)
                or any(len(items) != count for items in
                       (block.case_ids, block.input_indices, block.times))):
            raise ValueError('static block result/identity dimensions do not match')
        saved_extrema = self._extrema
        saved_keys = self._keys
        saved_count = self.case_count
        try:
            for column, result in enumerate(block.case_results):
                if result.get('status') != 'converged':
                    raise ValueError(
                        'cannot envelope an uncertified static case')
                values = {
                    ('displacement', int(dof)): float(value)
                    for dof, value in zip(block.output_dofs,
                                          displacements[:, column])
                }
                associated = {}
                for element, fields in result.get(
                        'element_stresses', {}).items():
                    physical = dict(_numeric_leaves(fields))
                    for coordinate, value in physical.items():
                        quantity = ('stress', int(element), *coordinate)
                        values[quantity] = value
                        associated[quantity] = physical
                forces = result.get('constraint_forces')
                if forces is None:
                    forces = {'reactions': result.get('reactions', {})}
                for field in ('reactions', 'support_reactions',
                              'mpc_slave_forces',
                              'mpc_master_equivalent_forces'):
                    if field not in forces:
                        continue
                    force_map = forces[field]
                    nodes = result.get('reaction_node_ids',
                                       tuple(force_map))
                    for node in nodes:
                        vector = _reaction_vector(force_map, node, field)
                        for component, value in enumerate(vector):
                            values[(field, int(node), component)] = float(value)
                self.update(values, case_id=block.case_ids[column],
                            input_index=_validated_input_index(
                                block.input_indices[column]),
                            time=block.times[column], associated=associated)
        except BaseException:
            self._extrema = saved_extrema
            self._keys = saved_keys
            self.case_count = saved_count
            raise

    def update(
        self,
        values: Mapping[Hashable, float],
        *,
        case_id: object,
        input_index: int,
        time: float | None = None,
        associated: Mapping[Hashable, Mapping[Hashable, float]] | None = None,
    ) -> None:
        self.complete = False
        input_index = _validated_input_index(input_index)
        instant = None if time is None else float(time)
        if instant is not None and not math.isfinite(instant):
            raise ValueError('case time must be finite')
        observed: dict[Hashable, GoverningValue] = {}
        shared_components: dict[int, Mapping[Hashable, float]] = {}
        for key, raw in values.items():
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f'non-finite envelope value for {key!r}')
            source = None if associated is None else associated.get(key)
            components = shared_components.get(id(source))
            if components is None:
                copied = {} if source is None else {
                    component: float(number) for component, number in source.items()
                }
                if any(not math.isfinite(number) for number in copied.values()):
                    raise ValueError(f'non-finite associated component for {key!r}')
                components = MappingProxyType(copied)
                shared_components[id(source)] = components
            observed[key] = GoverningValue(value, case_id, input_index, instant,
                                           key, components)
        keys = frozenset(observed)
        if self._keys is not None and keys != self._keys:
            raise ValueError('envelope quantity selection changed between cases')
        result = dict(self._extrema)
        for key, sample in observed.items():
            previous = result.get(key)
            if previous is None:
                result[key] = EnvelopeExtrema(sample, sample, sample)
                continue
            minimum = min((previous.minimum, sample),
                          key=lambda v: (v.value, v.input_index))
            maximum = min((previous.maximum, sample),
                          key=lambda v: (-v.value, v.input_index))
            absolute = min((previous.absolute_maximum, sample),
                           key=lambda v: (-abs(v.value), v.input_index))
            result[key] = EnvelopeExtrema(minimum, maximum, absolute)
        self._keys = keys
        self._extrema = result
        self.case_count += 1


def _validated_input_index(raw) -> int:
    """Return the exact integer index without silent truncation.

    Booleans are rejected even though ``bool`` subclasses the numeric types,
    and non-integral values such as ``2.9`` are rejected instead of being
    truncated. Exact integral floats and NumPy integers are accepted.
    """
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError('input_index must be an integral non-bool integer')
    if isinstance(raw, (int, np.integer)):
        index = int(raw)
    elif isinstance(raw, (float, np.floating)):
        if not math.isfinite(raw) or raw != int(raw):
            raise ValueError('input_index must be an integral non-bool integer')
        index = int(raw)
    else:
        raise ValueError('input_index must be an integral non-bool integer')
    if index < 0:
        raise ValueError('input_index must be a non-negative integer')
    return index


_ZERO_REACTION = np.zeros(6)


def _reaction_vector(force_map, node, field):
    """Return the six-component reaction of a selected node.

    The producer's maps are compact and may omit zero entries, so an omitted
    selected node contributes zeros. Any present entry must be a six-wide
    real vector; short, complex or non-numeric entries are rejected instead
    of being silently padded or truncated.
    """
    entry = force_map.get(node)
    if entry is None:
        return _ZERO_REACTION
    vector = np.asarray(entry)
    if vector.shape != (6,) or vector.dtype.kind not in 'fiu':
        raise ValueError(
            f'{field} entry for node {node!r} must be a six-component '
            'real vector')
    return vector


def _numeric_leaves(value, prefix=()):
    if isinstance(value, Mapping):
        for key, member in value.items():
            yield from _numeric_leaves(member, prefix + (key,))
    elif isinstance(value, (bool, np.bool_)):
        return
    elif isinstance(value, (complex, np.complexfloating)):
        raise ValueError(
            f'complex stress component at {prefix!r} is not physical')
    elif isinstance(value, (int, float, np.number)):
        yield prefix, float(value)
    elif isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value)
        if array.dtype.kind == 'c':
            raise ValueError(
                f'complex stress component at {prefix!r} is not physical')
        if array.dtype.kind in 'fiu':
            for index in np.ndindex(array.shape):
                yield prefix + index, float(array[index])
