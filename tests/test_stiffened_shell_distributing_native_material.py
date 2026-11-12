"""Native physical sections through the selected finite-area cyclic owner.

These bounded mechanics and lifecycle controls are not qualification evidence.
"""
from dataclasses import replace
import json
from math import fsum

import numpy as np
import pytest

from anymaterial import DNVC208MaterialCurve
from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC, decode, encode
from test_stiffened_shell_ge_distributing_finite_public import definition as elastic_definition


def definition(*, coupled=False, mixed=False, yielding=None, biaxial=False, native=True):
    base = elastic_definition()
    element, inertia = base.beams[0].instantiate()
    reference = element.operator.reference
    coordinates = reference.coordinates.copy()
    coordinates[:, 2] = 0.
    e, g = base.shell.young_pa, base.shell.young_pa/(2.*(1.+base.shell.poisson))
    arm = np.sqrt(8.333333333333333e-6/.01)
    fibres = tuple(b3_ge.Fibre(str(i), y*arm, z*arm, .0025, e,
        b3_ge.FlowCurve.linear(200e6 if yielding == "axial" else
            1e5 if yielding == "biaxial" and i == 0 else 1e20,
            200e9 if yielding == "axial" else 1e9))
        for i, (y, z) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))))
    background = np.diag(np.sqrt((0., g*.008, g*.008,
        g*1.6666666666666667e-5, 0., 0.)))
    section = b3_ge.PhysicalFibreSection(fibres, background)
    generalized = element.section
    if mixed and yielding == "axial":
        generalized = b3_ge.EllipsoidalGeneralizedSection(
            generalized.elastic, generalized.metric, 1e6, 1e11)
    def beam(eid, nodes, xyz, selected):
        return b3_ge.define_beam("b3-ge", eid, nodes, xyz, reference.nodal_triads,
            selected, inertia, quadrature=element.operator.order,
            arithmetic_policy=element.operator.arithmetic_policy)
    beams = (beam(1, element.node_ids, coordinates,
        generalized if mixed or not native else section),)
    node = 103
    if mixed:
        beams += (beam(2, (103, 104, 105), coordinates+np.array((2., 0., 0.)),
            section if native else generalized),)
        node = 105
    force = ((0., 20., 100.) if biaxial else
             (16e6 if yielding == "axial" else 1000., 0., 0.))
    actions = (ss.NodalAction("beam", node, (*force, 0., 0., 0.)),)
    curve = DNVC208MaterialCurve(200e6, 220e6, 230e6, .005, .01, 1e9, .2) if coupled else None
    policy = ((ss.GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1 if coupled else
               ss.GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1) if native else
              ss.GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1)
    return replace(base, beams=beams, shell=replace(base.shell, j2_curve=curve),
        finite_distributing_material_policy=policy, cyclic_reference_actions=actions)


def packed(owner, checkpoint=None):
    raw = owner.checkpoint() if checkpoint is None else checkpoint
    start = len(MAGIC)+4
    header_size = int.from_bytes(raw[len(MAGIC):start], "little")
    header = json.loads(raw[start:start+header_size])
    shapes = tuple((name, tuple(shape)) for name, shape in header["arrays"])
    return decode(raw, owner._identity, lambda kind, generation: shapes)


def checks(owner, result, previous=None):
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
    _, _, arrays = packed(owner)
    current, origin = arrays["accepted_fibres"], arrays["origin_fibres"]
    dz = np.array([fsum((*a[:2], *-b[:2])) for a, b in zip(current, origin)])
    dp = np.array([fsum((*a[2:], *-b[2:])) for a, b in zip(current, origin)])
    p = np.array([fsum(row[2:]) for row in current])
    z = np.array([fsum(row[:2]) for row in current])
    budget = 64.*np.finfo(float).eps*np.maximum(np.abs(p),
        np.array([abs(fsum(row[2:])) for row in origin]))
    assert np.all(dp >= -budget) and np.all(p >= np.abs(z)-budget)
    np.testing.assert_allclose(dp, np.abs(dz), rtol=1e-12, atol=float(np.max(budget)))
    increments = np.array([fsum(f.plastic_increment)
        for station in result.beam_fibre_material for f in station.fibres])
    np.testing.assert_allclose(increments, dz, rtol=1e-12, atol=float(np.max(budget)))
    sections = {e.element_id: e.section for e, _, _ in owner._beam_parts
                if type(e.section) is b3_ge.PhysicalFibreSection}
    dissipation = {eid: [] for eid in sections}
    for station in result.beam_fibre_material:
        fibres = {f.fibre_id: f for f in sections[station.element_id].fibres}
        for recovery in station.fibres:
            fibre = fibres[recovery.fibre_id]
            dissipation[station.element_id].append(station.measure*fibre.area*
                fibre.curve.flow_stress[0]*abs(fsum(recovery.plastic_increment)))
        # Observers receive detached deep copies; the owner retains immutable
        # arrays. Copy writeability does not weaken accepted-state ownership.
        actual_station = next(row for row in owner.recover().beam_fibre_material
            if (row.element_id, row.cell, row.station) ==
               (station.element_id, station.cell, station.station))
        assert not actual_station.strain.flags.writeable
        if station.strain.flags.writeable:
            assert not np.shares_memory(station.strain, actual_station.strain)
    for energy in result.beam_fibre_energy:
        actual = fsum(energy.incremental_dissipation_j)
        assert actual >= 0.
        np.testing.assert_allclose(actual, fsum(dissipation[energy.element_id]), rtol=1e-12, atol=0.)
    if previous is not None:
        for current_shell, old_shell in zip(result.shell_q4_material, previous.shell_q4_material):
            work = current_shell.layer_stress*(current_shell.plastic_strain-old_shell.plastic_strain)
            assert np.all(work.sum(axis=1) >= -64.*np.finfo(float).eps*np.abs(work).sum(axis=1))
    record, = result.distributing_footprint_wrenches
    assert record.configuration == "current" and record.host_element_ids == (2, 3)
    arm = ((record.beam_position_m-record.centroid_m) +
           (record.beam_position_low_m-record.centroid_low_m))
    scale = max(np.linalg.norm(record.beam_action), 1.)
    assert np.linalg.norm(record.shell_action[:3]+record.beam_action[:3]) <= 1e-8*scale
    assert np.linalg.norm(record.shell_action[3:]+record.beam_action[3:]+
                          np.cross(arm, record.beam_action[:3])) <= 1e-8*scale
    physical = owner._shell_count+6*len(owner._beam_ids)
    reaction = result.support_reactions[:physical].reshape(-1, 6)
    positions = np.vstack((np.asarray(owner._definition.shell.coordinates_m)+
        result.shell_displacement.reshape(-1, 6)[:, :3], result.beam_positions))
    action, = owner._definition.cyclic_reference_actions
    force = result.load_factor*np.asarray(action.wrench[:3])
    moment = np.cross(result.beam_positions[owner._beam_index[action.node]], force)
    norm = max(np.linalg.norm(action.wrench), 1.)
    assert np.linalg.norm(reaction[:, :3].sum(axis=0)+force) <= 1e-8*norm
    assert np.linalg.norm((reaction[:, 3:]+np.cross(positions, reaction[:, :3])).sum(axis=0)+moment) <= 1e-8*norm
    return arrays


@pytest.mark.parametrize("mixed,biaxial", ((False, False), (False, True), (True, True)))
def test_native_elastic_section_matches_generalized_full_area_response(mixed, biaxial):
    model = definition(mixed=mixed, biaxial=biaxial)
    owner = ss.create_analysis(definition=model)
    oracle = ss.create_analysis(definition=definition(mixed=mixed, biaxial=biaxial, native=False))
    try:
        result = owner.solve(model.cyclic_reference_actions, (.5, 1.), max_iterations=24, max_backtracks=16)
        reference = oracle.solve(model.cyclic_reference_actions, (.5, 1.), max_iterations=24, max_backtracks=16)
        arrays = checks(owner, result)
        assert not np.any(arrays["accepted_fibres"])
        for field in ("shell_displacement", "beam_positions", "cell_resultants", "support_reactions"):
            np.testing.assert_allclose(getattr(result, field), getattr(reference, field), rtol=1e-8, atol=1e-10)
        if biaxial:
            strain = np.array([r.strain+r.strain_low for r in result.beam_fibre_material])
            assert np.max(np.abs(strain[:, 4])) > 0. and np.max(np.abs(strain[:, 5])) > 0.
        assert owner.capabilities["increment_control"] == ("fixed",)
        assert "physical fibres" not in owner.capabilities["excluded"]
        assert owner.capabilities["beam_section_families"] == (
            ("RESULTANT_ELLIPSOID", "PHYSICAL_AXIAL_BIAXIAL_FIBRE") if mixed else
            ("PHYSICAL_AXIAL_BIAXIAL_FIBRE",))
    finally:
        owner.close(); oracle.close()


@pytest.mark.parametrize("coupled", (False, True))
def test_native_axial_yield_zero_signed_restart_and_canonical_tamper(coupled):
    model = definition(coupled=coupled, yielding="axial")
    owners = [ss.create_analysis(definition=model) for _ in range(3)]
    direct, resumed, untouched = owners
    history = (.25, .5, .75, 1., 0., -.25)
    zeros = []
    previous = []
    def observe(result):
        arrays = checks(direct, result, previous[-1] if previous else None)
        assert np.any(arrays["accepted_fibres"][:, 2:])
        if result.load_factor == 0.:
            zeros.append(direct.checkpoint())
        previous.append(result)
    try:
        final = direct.solve(model.cyclic_reference_actions, history, max_iterations=24,
            max_backtracks=16, on_accepted=observe)
        if coupled:
            assert np.max(final.shell_layer_alpha) > 0.
        zero, = zeros
        restored = resumed.restore(zero)
        assert restored.load_factor == 0. and restored.accepted_factors == history[:-1]
        assert resumed.checkpoint() == zero
        kind, generation, arrays = packed(resumed)
        assert kind == ("finite_distributing_cyclic_native_coupled" if coupled else
                        "finite_distributing_cyclic_native_material")
        assert ("shell_origin_alpha" in arrays) == coupled
        fields = ("origin_fibres", "accepted_fibres", "pattern") + (
            ("shell_origin_alpha", "shell_alpha") if coupled else ())
        for field in fields:
            changed = dict(arrays)
            changed[field] = arrays[field].copy()
            if field == "pattern":
                changed[field] *= 2.
            elif field.startswith("shell"):
                changed[field][0, 0] += 1e-6
            else:
                changed[field][0, 2] += 1e-6
            forged = encode(kind, resumed._identity, generation, tuple(changed.items()))
            with pytest.raises(ValueError):
                untouched.restore(forged)
            assert untouched._generation == 0 and untouched._finite_driver is None
        wrong_kind = "finite_distributing_cyclic_coupled" if coupled else "finite_distributing_cyclic_material"
        with pytest.raises(ValueError, match="exact mechanics kind"):
            untouched.restore(encode(wrong_kind, resumed._identity, generation, tuple(arrays.items())))
        assert untouched._generation == 0 and untouched._finite_driver is None
        completed = resumed.solve(model.cyclic_reference_actions, history[-1:],
            max_iterations=24, max_backtracks=16)
        assert resumed.checkpoint() == direct.checkpoint()
        for field in ("shell_displacement", "cell_resultants", "support_reactions"):
            np.testing.assert_array_equal(getattr(completed, field), getattr(final, field))
        assert completed.beam_fibre_energy == final.beam_fibre_energy
        assert [r.fibres for r in completed.beam_fibre_material] == [r.fibres for r in final.beam_fibre_material]
    finally:
        for owner in owners: owner.close()


def test_mixed_biaxial_yield_and_cancel_after_recovery_before_publication(monkeypatch):
    from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
    model = definition(mixed=True, yielding="biaxial", biaxial=True)
    owner = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    try:
        first = owner.solve(model.cyclic_reference_actions, (.25, .5), max_iterations=24, max_backtracks=16)
        arrays = checks(owner, first)
        assert arrays["accepted_plastic"].size and not np.any(arrays["accepted_plastic"])
        assert np.any(arrays["accepted_fibres"][:, 2:])
        checkpoint = owner.checkpoint()
        token = CancellationToken()
        original = finite.cancellation_safe_point
        def cancel(token_value, stage):
            if stage == "stiffened-shell.finite.material-publish":
                token.cancel("native area recovery before publication")
            return original(token_value, stage)
        with monkeypatch.context() as context:
            context.setattr(finite, "cancellation_safe_point", cancel)
            with pytest.raises(SolveCancelled, match="before publication"):
                owner.solve(model.cyclic_reference_actions, (1.,), max_iterations=24,
                    max_backtracks=16, cancellation_token=token)
        assert owner.recover() is first and owner.checkpoint() == checkpoint
        restored = resumed.restore(checkpoint)
        assert resumed.checkpoint() == checkpoint and restored.beam_fibre_energy == first.beam_fibre_energy
        direct = owner.solve(model.cyclic_reference_actions, (1., 0., -.25), max_iterations=24, max_backtracks=16)
        continued = resumed.solve(model.cyclic_reference_actions, (1., 0., -.25), max_iterations=24, max_backtracks=16)
        checks(owner, direct)
        assert owner.checkpoint() == resumed.checkpoint()
        np.testing.assert_array_equal(direct.cell_resultants, continued.cell_resultants)
    finally:
        owner.close(); resumed.close()


def test_simultaneous_generalized_and_fibre_yield_codec_restart():
    model = definition(mixed=True, yielding="axial")
    direct, resumed = (ss.create_analysis(definition=model) for _ in range(2))
    try:
        prefix = direct.solve(model.cyclic_reference_actions, (.25, .5, .75, 1., 0.),
            max_iterations=24, max_backtracks=16)
        arrays = checks(direct, prefix)
        assert np.any(arrays["accepted_accumulated"] > 0.)
        assert np.any(arrays["accepted_fibres"][:, 2:] > 0.)
        raw = direct.checkpoint()
        resumed.restore(raw)
        assert resumed.checkpoint() == raw
        expected = direct.solve(model.cyclic_reference_actions, (-.25,),
            max_iterations=24, max_backtracks=16)
        actual = resumed.solve(model.cyclic_reference_actions, (-.25,),
            max_iterations=24, max_backtracks=16)
        assert resumed.checkpoint() == direct.checkpoint()
        np.testing.assert_array_equal(actual.cell_resultants, expected.cell_resultants)
        assert actual.beam_fibre_energy == expected.beam_fibre_energy
    finally:
        direct.close(); resumed.close()


def test_native_policy_is_explicit_requires_fibres_and_preserves_old_guards():
    model = definition()
    old = definition(native=False)
    with pytest.raises(ValueError, match="requires a physical-fibre"):
        ss.create_analysis(definition=replace(old, finite_distributing_material_policy=
            ss.GE_FINITE_DISTRIBUTING_NATIVE_CYCLIC_MATERIAL_V1))
    with pytest.raises(ValueError, match="exclude physical-fibre"):
        ss.create_analysis(definition=replace(model, finite_distributing_material_policy=
            ss.GE_FINITE_DISTRIBUTING_GENERALIZED_CYCLIC_MATERIAL_V1))
    with pytest.raises(ValueError, match="matching shell J2"):
        ss.create_analysis(definition=replace(model, finite_distributing_material_policy=
            ss.GE_FINITE_DISTRIBUTING_NATIVE_COUPLED_J2_CYCLIC_MATERIAL_V1))
    writable = np.asarray(model.cyclic_reference_actions[0].wrench)
    mutable_action = replace(model.cyclic_reference_actions[0], wrench=writable)
    owner = ss.create_analysis(definition=replace(model, cyclic_reference_actions=(mutable_action,)))
    try:
        identity = owner._identity
        pattern = owner._cyclic_reference_pattern.copy()
        writable[:] = 0.
        assert owner._identity == identity and owner._definition.cyclic_reference_actions == model.cyclic_reference_actions
        np.testing.assert_array_equal(owner._cyclic_reference_pattern, pattern)
        with pytest.raises(ValueError, match="fixed increments"):
            owner.solve(model.cyclic_reference_actions, (.5,), increment_control=ss.AdaptiveIncrementControl())
        with pytest.raises(ValueError, match="model-bound reference"):
            owner.solve((replace(model.cyclic_reference_actions[0], wrench=(2., 0., 0., 0., 0., 0.)),), (.5,))
        assert owner._generation == 0 and owner._finite_driver is None
    finally: owner.close()
