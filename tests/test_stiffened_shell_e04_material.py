"""Declared roof material inputs; no engineering strength calibration."""
import numpy as np
import pytest
from anymaterial import DNVC208MaterialCurve
from anyfesolver import stiffened_shell as ss
from docs.reference_cases.stiffened_shell_e04_roof import build,definition,material_definition

LIMITS=(500.,300.,300.,1.,2.,.5)


def curve():
    p=3.10275e6
    return DNVC208MaterialCurve(p,1.05*p,1.1*p,.005,.01,4*p,.2)


@pytest.mark.parametrize('plastic_shell',[False,True])
def test_declared_roof_strength_preserves_reference_and_section(plastic_shell):
    c=build(4,4);old=definition(c,density_kg_m3=1.)
    limits=np.array(LIMITS)
    new=material_definition(c,density_kg_m3=1.,shell_curve=curve() if plastic_shell else None,
        beam_yield_limits=limits,beam_hardening_n=25000.)
    assert new.shell_reference_policy==ss.CURVED_Q4_POINT_MATERIAL_REFERENCE_V1
    assert new.retained_scale_policy=='native_force_moment_roles_v1'
    assert new.points==old.points and new.fixed==old.fixed
    np.testing.assert_array_equal(new.shell.coordinates_m,old.shell.coordinates_m)
    assert all(type(value) is float for row in new.shell.coordinates_m for value in row)
    assert new.shell.young_pa==old.shell.young_pa and new.shell.thickness_m==old.shell.thickness_m
    limits[:]=1.  # Caller mutation cannot change captured strength or section data.
    for a,b in zip(new.beams,old.beams):
        candidate,mass=a.instantiate();reference,old_mass=b.instantiate()
        np.testing.assert_array_equal(candidate.operator.reference.coordinates,reference.operator.reference.coordinates)
        np.testing.assert_array_equal(candidate.operator.reference.nodal_triads,reference.operator.reference.nodal_triads)
        np.testing.assert_array_equal(candidate.section.elastic,reference.section.elastic)
        np.testing.assert_array_equal(mass,old_mass)
        np.testing.assert_array_equal(np.diag(candidate.section.metric),(500/np.array(LIMITS))**2)
        assert candidate.section.yield_force==500 and candidate.section.hardening==25000
        origin=candidate.section.virgin()
        for i,limit in enumerate(LIMITS):
            stress=np.zeros(6);stress[i]=1.01*limit
            yielded=candidate.section.response(stress,origin=origin,control='resultant')
            assert yielded.branch=='PLASTIC' and sum(yielded.history.accumulated)>0
            assert sum(yielded.dissipation)>0
        assert candidate.section.virgin()==origin
    owner=ss.create_analysis(definition=new)
    try:
        assert owner.preflight()['production_qualified'] is False
        assert owner.capabilities['retained_coordinate_scale_policy']=='native_force_moment_roles_v1'
    finally:owner.close()
    assert tuple(b.raw for b in old.beams)==tuple(b.raw for b in definition(c,density_kg_m3=1.).beams)


@pytest.mark.parametrize('change',[
    {'beam_yield_limits':(500.,300.,300.,1.,2.)},
    {'beam_yield_limits':(500.,300.,300.,1.,2.,0.)},
    {'beam_yield_limits':(500.,300.,300.,1.,2.,float('nan'))},
    {'beam_yield_limits':(500.,300.,300.,1.,2.,True)},
    {'beam_hardening_n':0.}, {'beam_hardening_n':float('inf')},
    {'beam_hardening_n':'25000'}, {'shell_curve':object()},
])
def test_invalid_declared_strength_is_rejected(change):
    args=dict(density_kg_m3=1.,shell_curve=curve(),beam_yield_limits=LIMITS,beam_hardening_n=25000.)
    args.update(change)
    with pytest.raises(ValueError):material_definition(build(4,4),**args)
