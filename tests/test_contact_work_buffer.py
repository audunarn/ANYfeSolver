from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

from anyfesolver.contact import (
    RigidSphereImpact,
    SphereContactConfig,
    SphereContactRecord,
    _assemble_sphere_contact_work_buffer,
    _contact_geometry,
    _two_shell_contact_verification_panel,
    _verification_contact_panel,
    assemble_sphere_contact_load_vector,
)
from anyfesolver.contact_performance import ContactWorkBuffer, ContactWorkCounters


def _record_payload(records):
    return tuple(record.to_dict() for record in records)


def test_compact_contact_work_matches_public_load_force_and_records_exactly():
    model = _verification_contact_panel()
    sphere = RigidSphereImpact(
        "buffer_parity",
        radius=0.2,
        mass=1.0,
        start_point=(1.0, 0.5, 0.1),
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )
    config = SphereContactConfig(
        penalty_stiffness=1000.0,
        load_patch_radius_factor=1.25,
        min_load_patch_nodes=4,
    )
    position = np.array([1.0, 0.5, 0.1])
    velocity = np.zeros(3)
    public_load, public_sphere_force, public_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        position,
        velocity,
    )

    counters = ContactWorkCounters()
    work = ContactWorkBuffer(model.mesh.dof_manager.total_dofs, counters=counters)
    compact = _assemble_sphere_contact_work_buffer(
        model,
        sphere,
        config,
        position,
        velocity,
        work_buffer=work,
    )
    assert counters.public_materialization_count == 0
    records = compact.materialize_records(
        SphereContactRecord,
        _contact_geometry(model).node_ids,
    )

    np.testing.assert_array_equal(compact.load, public_load)
    np.testing.assert_array_equal(compact.sphere_force, public_sphere_force)
    assert _record_payload(records) == _record_payload(public_records)
    assert counters.direct_full_scatter_count == 1
    assert counters.public_materialization_count == 1
    assert counters.public_records_materialized == len(records)
    assert counters.nodal_force_mappings_materialized == len(records)

    # Public arrays and dictionaries must remain stable when the reusable
    # internal buffer is reset by the next trial iteration.
    saved_payload = _record_payload(records)
    saved_public_load = public_load.copy()
    _assemble_sphere_contact_work_buffer(
        model,
        sphere,
        config,
        np.array([0.5, 0.5, 2.0]),
        velocity,
        work_buffer=work,
    )
    assert _record_payload(records) == saved_payload
    assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        np.array([0.5, 0.5, 2.0]),
        velocity,
    )
    np.testing.assert_array_equal(public_load, saved_public_load)


def test_compact_contact_selection_is_deterministic_and_honors_sticky_ties():
    model = _two_shell_contact_verification_panel()
    sphere = RigidSphereImpact(
        "shared_edge",
        radius=0.2,
        mass=1.0,
        start_point=(1.0, 0.5, 0.1),
        travel_direction=(0.0, 0.0, -1.0),
        speed=0.0,
    )
    config = SphereContactConfig(penalty_stiffness=1000.0, max_active_contacts=1)
    position = np.array([1.0, 0.5, 0.1])
    velocity = np.zeros(3)
    work = ContactWorkBuffer(model.mesh.dof_manager.total_dofs)

    unpreferred_orders = []
    preferred_orders = []
    for _ in range(20):
        result = _assemble_sphere_contact_work_buffer(
            model,
            sphere,
            config,
            position,
            velocity,
            preferred_element_ids=(),
            work_buffer=work,
        )
        unpreferred_orders.append(result.active_element_ids)
        result = _assemble_sphere_contact_work_buffer(
            model,
            sphere,
            config,
            position,
            velocity,
            preferred_element_ids=(2,),
            work_buffer=work,
        )
        preferred_orders.append(result.active_element_ids)

    assert set(unpreferred_orders) == {(1,)}
    assert set(preferred_orders) == {(2,)}
    public_load, public_force, public_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        position,
        velocity,
        preferred_element_ids=(2,),
    )
    assert tuple(record.element_id for record in public_records) == (2,)
    np.testing.assert_array_equal(work.load, public_load)
    np.testing.assert_array_equal(work.sphere_force, public_force)


def test_public_contact_work_cache_is_thread_local_and_returns_independent_arrays():
    model = _verification_contact_panel()
    config = SphereContactConfig(penalty_stiffness=1000.0)
    positions = (np.array([0.5, 0.5, 0.1]), np.array([1.0, 0.5, 0.1]))
    expected = []
    for index, position in enumerate(positions):
        sphere = RigidSphereImpact(
            f"thread_{index}",
            radius=0.2,
            mass=1.0,
            start_point=position,
            travel_direction=(0.0, 0.0, -1.0),
            speed=0.0,
        )
        expected.append(
            assemble_sphere_contact_load_vector(
                model,
                sphere,
                config,
                position,
                np.zeros(3),
            )[:2]
        )

    barrier = threading.Barrier(2)

    def worker(index):
        position = positions[index]
        sphere = RigidSphereImpact(
            f"thread_{index}",
            radius=0.2,
            mass=1.0,
            start_point=position,
            travel_direction=(0.0, 0.0, -1.0),
            speed=0.0,
        )
        barrier.wait()
        result = None
        for _ in range(100):
            result = assemble_sphere_contact_load_vector(
                model,
                sphere,
                config,
                position,
                np.zeros(3),
            )
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(worker, range(2)))
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result[0], expected[index][0])
        np.testing.assert_array_equal(result[1], expected[index][1])
