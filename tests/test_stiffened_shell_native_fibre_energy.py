"""Accepted native energy, independently checked through the section entry.

This covers energy/recovery ownership, not complete coupled material acceptance.
"""
from decimal import Decimal as D, localcontext

import anyfesolver
import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_section import canonical
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from test_ge_beam3_fibre_cell import model
from test_ge_beam3_retained_fibre import make_model
from test_ge_beam3_retained_plastic import args
from test_stiffened_shell_ge_physical_fibre_finite import _yielding_definition


def exact(pair):
    return D.from_float(pair[0])+D.from_float(pair[1])


@pytest.mark.parametrize('contrast', (1.,1e4))
@pytest.mark.parametrize('tabular', (False,True))
def test_native_energy_capture_preserves_default_and_matches_section(contrast,tabular):
    section, _rows, cell = model(contrast,tabular)
    predecessor = cell.response((.3*np.sin(np.arange(18))).tolist())['history']
    saved = canonical(predecessor)
    p = (-.2*np.cos(np.arange(18))).tolist()
    default = cell.response(p,predecessor)
    response = cell.response(p,predecessor,include_energy=True)
    stripped = {key:value for key,value in response.items() if key not in (
        'stored_energy','previous_hardening_energy','incremental_dissipation',
        'energy_recovery_station_updates')}
    stripped['stations'] = [{key:value for key,value in row.items() if key not in (
        'stored_energy_per_length','previous_hardening_per_length',
        'incremental_dissipation_per_length')} for row in response['stations']]
    assert canonical(stripped) == canonical(default)
    assert canonical(predecessor) == saved
    assert response['energy_recovery_station_updates'] == len(cell.stations)
    with localcontext() as context:
        context.prec = 80
        expected_stored = D(0); expected_dissipation = D(0)
        expected_previous = D(0); expected_potential = D(0)
        for (_,weight,_,_), fields, origin in zip(
                cell.stations,response['stations'],predecessor.stations):
            oracle = section.response(fields['strain'][0],origin,
                                      strain_low=fields['strain'][1])
            stored = exact(oracle.stored_energy)
            dissipated = exact(oracle.dissipation)
            potential = exact(oracle.incremental_potential)
            expected_stored += weight*stored
            expected_dissipation += weight*dissipated
            expected_previous += weight*(stored+dissipated-potential)
            expected_potential += weight*potential
            assert exact(fields['incremental_dissipation_per_length']) >= 0
            assert abs(exact(fields['stored_energy_per_length'])-stored) <= D('1e-11')*max(D(1),abs(stored))
        for value,expected in ((response['stored_energy'],expected_stored),
                (response['incremental_dissipation'],expected_dissipation),
                (response['previous_hardening_energy'],expected_previous)):
            assert abs(exact(value)-expected) <= D('1e-11')*max(D(1),abs(expected))
        assert exact(response['stored_energy']) >= 0
        assert exact(response['incremental_dissipation']) >= 0
        potential = sum((D.from_float(row[0]) for row in response['material_potential']),D(0))
        difference = exact(response['stored_energy'])-exact(response['previous_hardening_energy'])+exact(response['incremental_dissipation'])-potential
        assert abs(difference) <= D('1e-11')*max(D(1),abs(expected_potential))


def test_legacy_station_recovery_and_new_material_recovery_agree():
    operator = make_model().mesh.elements[1].operator
    _x,_low,_q,u,p = args(operator)
    origin = operator.cell.response((p*.5).tolist())['history']
    legacy = operator.recover(u,p,origin=origin)
    material = operator.recover_material(u,p,origin=origin)
    assert canonical(legacy) == canonical(material.stations)
    assert tuple(row['history'] for row in legacy) == material.history.stations
    assert material.energy_recovery_station_updates == len(legacy)
    assert sum(material.stored_energy) > 0.
    for row in material.stations:
        assert not row['strain'].flags.writeable
        assert not row['current_frame'].flags.writeable


@pytest.mark.parametrize('mixed', (False,True))
def test_material_energy_survives_atomic_restore_and_unload(mixed):
    definition = _yielding_definition(mixed)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=definition)
    try:
        actions = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)
        first = owner.solve(actions,targets=(.5,1.),max_iterations=18)
        assert len(first.beam_fibre_energy) == (1 if mixed else 2)
        assert any(sum(row.incremental_dissipation_j)>0. for row in first.beam_fibre_energy)
        assert all(sum(row.stored_energy_j)>0. for row in first.beam_fibre_energy)
        saved = owner.checkpoint()
        restored = resumed.restore(saved)
        assert restored.beam_fibre_energy == first.beam_fibre_energy
        assert resumed.checkpoint() == saved
        direct = owner.solve(actions,targets=(.5,),max_iterations=18)
        continued = resumed.solve(actions,targets=(.5,),max_iterations=18)
        assert direct.beam_fibre_energy == continued.beam_fibre_energy
        assert owner.checkpoint() == resumed.checkpoint()
    finally:
        owner.close();resumed.close()
