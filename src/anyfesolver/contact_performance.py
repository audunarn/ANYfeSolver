"""Compact work storage for rigid-sphere contact assembly.

Contact geometry remains in :mod:`anyfesolver.contact`.  This module only owns
the reusable candidate arrays, deterministic active-contact reduction, direct
full-vector scatter, and lazy conversion to the public contact-record type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple

import numpy as np


_CLASSIFICATION_TO_CODE = {"face": 0, "edge": 1, "corner": 2, "beam": 3}
_CODE_TO_CLASSIFICATION = ("face", "edge", "corner", "beam")


@dataclass
class ContactWorkCounters:
    assembly_calls: int = 0
    candidate_contacts: int = 0
    selected_contacts: int = 0
    direct_full_scatter_count: int = 0
    full_scatter_nodal_entries: int = 0
    public_materialization_count: int = 0
    public_records_materialized: int = 0
    nodal_force_mappings_materialized: int = 0
    contact_buffer_growth_count: int = 0
    nodal_buffer_growth_count: int = 0

    def diagnostics(self) -> dict[str, int]:
        return {
            "assembly_calls": int(self.assembly_calls),
            "candidate_contacts": int(self.candidate_contacts),
            "selected_contacts": int(self.selected_contacts),
            "direct_full_scatter_count": int(self.direct_full_scatter_count),
            "full_scatter_nodal_entries": int(self.full_scatter_nodal_entries),
            "public_materialization_count": int(self.public_materialization_count),
            "public_records_materialized": int(self.public_records_materialized),
            "nodal_force_mappings_materialized": int(self.nodal_force_mappings_materialized),
            "contact_buffer_growth_count": int(self.contact_buffer_growth_count),
            "nodal_buffer_growth_count": int(self.nodal_buffer_growth_count),
        }


class ContactWorkBuffer:
    """Reusable structure-of-arrays storage for one contact assembly call."""

    def __init__(
        self,
        total_dofs: int,
        *,
        counters: ContactWorkCounters | None = None,
        initial_contact_capacity: int = 8,
        initial_nodal_capacity: int = 64,
    ) -> None:
        contact_capacity = max(int(initial_contact_capacity), 1)
        nodal_capacity = max(int(initial_nodal_capacity), 1)
        self.counters = counters if counters is not None else ContactWorkCounters()
        self.element_ids = np.empty(contact_capacity, dtype=np.int64)
        self.local_coordinates = np.empty((contact_capacity, 2), dtype=float)
        self.contact_points = np.empty((contact_capacity, 3), dtype=float)
        self.normals = np.empty((contact_capacity, 3), dtype=float)
        self.penetrations = np.empty(contact_capacity, dtype=float)
        self.normal_forces = np.empty(contact_capacity, dtype=float)
        self.sphere_forces = np.empty((contact_capacity, 3), dtype=float)
        self.structure_forces = np.empty((contact_capacity, 3), dtype=float)
        self.classification_codes = np.empty(contact_capacity, dtype=np.uint8)
        self.nodal_offsets = np.empty(contact_capacity + 1, dtype=np.intp)
        self.nodal_slots = np.empty(nodal_capacity, dtype=np.intp)
        self.nodal_forces = np.empty((nodal_capacity, 3), dtype=float)
        self.nodal_moments = np.empty((nodal_capacity, 3), dtype=float)
        self.load = np.zeros(max(int(total_dofs), 0), dtype=float)
        self.sphere_force = np.zeros(3, dtype=float)
        self.selected_indices = np.empty(0, dtype=np.intp)
        self.count = 0
        self.nodal_count = 0

    def _grow_contacts(self, required: int) -> None:
        current = int(self.element_ids.size)
        if required <= current:
            return
        capacity = max(required, current * 2)

        def grown(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            result = np.empty(shape, dtype=array.dtype)
            result[:current] = array[:current]
            return result

        self.element_ids = grown(self.element_ids, (capacity,))
        self.local_coordinates = grown(self.local_coordinates, (capacity, 2))
        self.contact_points = grown(self.contact_points, (capacity, 3))
        self.normals = grown(self.normals, (capacity, 3))
        self.penetrations = grown(self.penetrations, (capacity,))
        self.normal_forces = grown(self.normal_forces, (capacity,))
        self.sphere_forces = grown(self.sphere_forces, (capacity, 3))
        self.structure_forces = grown(self.structure_forces, (capacity, 3))
        self.classification_codes = grown(self.classification_codes, (capacity,))
        offsets = np.empty(capacity + 1, dtype=self.nodal_offsets.dtype)
        offsets[: current + 1] = self.nodal_offsets[: current + 1]
        self.nodal_offsets = offsets
        self.counters.contact_buffer_growth_count += 1

    def _grow_nodal(self, required: int) -> None:
        current = int(self.nodal_slots.size)
        if required <= current:
            return
        capacity = max(required, current * 2)
        slots = np.empty(capacity, dtype=self.nodal_slots.dtype)
        forces = np.empty((capacity, 3), dtype=float)
        moments = np.empty((capacity, 3), dtype=float)
        slots[: self.nodal_count] = self.nodal_slots[: self.nodal_count]
        forces[: self.nodal_count] = self.nodal_forces[: self.nodal_count]
        moments[: self.nodal_count] = self.nodal_moments[: self.nodal_count]
        self.nodal_slots = slots
        self.nodal_forces = forces
        self.nodal_moments = moments
        self.counters.nodal_buffer_growth_count += 1

    def reset(self, total_dofs: int) -> None:
        if self.load.size != int(total_dofs):
            self.load = np.zeros(max(int(total_dofs), 0), dtype=float)
        else:
            self.load.fill(0.0)
        self.sphere_force.fill(0.0)
        self.selected_indices = np.empty(0, dtype=np.intp)
        self.count = 0
        self.nodal_count = 0
        self.nodal_offsets[0] = 0
        self.counters.assembly_calls += 1

    def append(
        self,
        *,
        element_id: int,
        local_coordinates: Sequence[float],
        contact_point: np.ndarray,
        normal: np.ndarray,
        penetration: float,
        normal_force: float,
        sphere_force: np.ndarray,
        structure_force: np.ndarray,
        contact_classification: str,
        nodal_slots: Sequence[int],
        nodal_forces: np.ndarray,
        nodal_moments: np.ndarray | None = None,
    ) -> None:
        index = int(self.count)
        slot_array = np.asarray(nodal_slots, dtype=np.intp).reshape(-1)
        force_array = np.asarray(nodal_forces, dtype=float).reshape(-1, 3)
        if slot_array.size != force_array.shape[0]:
            raise ValueError("nodal_slots and nodal_forces must contain the same number of entries")
        moment_array = (
            np.zeros_like(force_array)
            if nodal_moments is None
            else np.asarray(nodal_moments, dtype=float).reshape(-1, 3)
        )
        if slot_array.size != moment_array.shape[0]:
            raise ValueError(
                "nodal_slots and nodal_moments must contain the same number of entries"
            )
        self._grow_contacts(index + 1)
        self._grow_nodal(self.nodal_count + int(slot_array.size))
        self.element_ids[index] = int(element_id)
        self.local_coordinates[index] = np.asarray(local_coordinates, dtype=float).reshape(2)
        self.contact_points[index] = np.asarray(contact_point, dtype=float).reshape(3)
        self.normals[index] = np.asarray(normal, dtype=float).reshape(3)
        self.penetrations[index] = float(penetration)
        self.normal_forces[index] = float(normal_force)
        self.sphere_forces[index] = np.asarray(sphere_force, dtype=float).reshape(3)
        self.structure_forces[index] = np.asarray(structure_force, dtype=float).reshape(3)
        self.classification_codes[index] = _CLASSIFICATION_TO_CODE[str(contact_classification)]
        start = int(self.nodal_count)
        stop = start + int(slot_array.size)
        self.nodal_slots[start:stop] = slot_array
        self.nodal_forces[start:stop] = force_array
        self.nodal_moments[start:stop] = moment_array
        self.nodal_count = stop
        self.nodal_offsets[index + 1] = stop
        self.count = index + 1
        self.counters.candidate_contacts += 1

    def select_and_scatter(
        self,
        *,
        max_active_contacts: int,
        preferred_element_ids: Iterable[int],
        node_dofs: np.ndarray,
        node_rotational_dofs: np.ndarray | None = None,
    ) -> None:
        limit = int(max_active_contacts)
        if self.count <= limit:
            selected = list(range(self.count))
        else:
            deepest = float(np.max(self.penetrations[: self.count]))
            tie_band = 0.95 * deepest
            preferred = {int(element_id) for element_id in preferred_element_ids}
            selected = sorted(
                range(self.count),
                key=lambda index: (
                    self.penetrations[index] >= tie_band
                    and int(self.element_ids[index]) in preferred,
                    self.penetrations[index],
                    self.normal_forces[index],
                ),
                reverse=True,
            )[:limit]
        self.selected_indices = np.asarray(selected, dtype=np.intp)
        self.load.fill(0.0)
        self.sphere_force.fill(0.0)
        scatter_entries = 0
        for index in selected:
            self.sphere_force += self.sphere_forces[index]
            start = int(self.nodal_offsets[index])
            stop = int(self.nodal_offsets[index + 1])
            scatter_entries += stop - start
            for entry in range(start, stop):
                self.load[node_dofs[self.nodal_slots[entry]]] += self.nodal_forces[entry]
                if node_rotational_dofs is not None:
                    self.load[node_rotational_dofs[self.nodal_slots[entry]]] += (
                        self.nodal_moments[entry]
                    )
        self.counters.selected_contacts += len(selected)
        self.counters.direct_full_scatter_count += 1
        self.counters.full_scatter_nodal_entries += scatter_entries

    @property
    def active_element_ids(self) -> Tuple[int, ...]:
        return tuple(int(self.element_ids[index]) for index in self.selected_indices)

    @property
    def active_classifications(self) -> Tuple[str, ...]:
        return tuple(
            _CODE_TO_CLASSIFICATION[int(self.classification_codes[index])]
            for index in self.selected_indices
        )

    @property
    def max_penetration(self) -> float:
        if self.selected_indices.size == 0:
            return 0.0
        return float(np.max(self.penetrations[self.selected_indices]))

    @property
    def peak_normal_force(self) -> float:
        if self.selected_indices.size == 0:
            return 0.0
        return float(np.max(self.normal_forces[self.selected_indices]))

    def materialize_records(
        self,
        record_type: Any,
        node_ids: np.ndarray,
    ) -> tuple[Any, ...]:
        """Create stable public records only when a caller needs them."""

        records = []
        for index_value in self.selected_indices:
            index = int(index_value)
            start = int(self.nodal_offsets[index])
            stop = int(self.nodal_offsets[index + 1])
            nodal_mapping = {
                int(node_ids[self.nodal_slots[entry]]): self.nodal_forces[entry].copy()
                for entry in range(start, stop)
            }
            moment_mapping = (
                {
                    int(node_ids[self.nodal_slots[entry]]): self.nodal_moments[
                        entry
                    ].copy()
                    for entry in range(start, stop)
                }
                if np.any(self.nodal_moments[start:stop] != 0.0)
                else {}
            )
            records.append(
                record_type(
                    element_id=int(self.element_ids[index]),
                    local_coordinates=(
                        float(self.local_coordinates[index, 0]),
                        float(self.local_coordinates[index, 1]),
                    ),
                    contact_point=self.contact_points[index].copy(),
                    normal=self.normals[index].copy(),
                    penetration=float(self.penetrations[index]),
                    normal_force=float(self.normal_forces[index]),
                    sphere_force=self.sphere_forces[index].copy(),
                    structure_force=self.structure_forces[index].copy(),
                    contact_classification=_CODE_TO_CLASSIFICATION[
                        int(self.classification_codes[index])
                    ],
                    nodal_forces=nodal_mapping,
                    nodal_moments=moment_mapping,
                )
            )
        self.counters.public_materialization_count += 1
        self.counters.public_records_materialized += len(records)
        self.counters.nodal_force_mappings_materialized += len(records)
        return tuple(records)
