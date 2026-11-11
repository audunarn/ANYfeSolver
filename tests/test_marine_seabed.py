"""Prospective development law checks; no trajectory or W6 acceptance."""
import copy
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from anyfesolver.marine_seabed import SeabedContact,SeabedContactError,seabed_contact_force


def test_closed_form_branches_energy_and_no_tensile_contact():
    bed=SeabedContact(-2.,1000.,20.)
    p=np.zeros((6,3));p[:,2]=[-1.8,-1.9,-2.,-2.,-2.,-2.]
    v=np.zeros_like(p);v[:,2]=[-1.,-1.,-.5,.5,10.,5.]
    response=seabed_contact_force(bed,.2,1.,p,v)
    np.testing.assert_allclose(response.force[:,2],[0.,0.,22.,18.,0.,0.],atol=1e-12)
    np.testing.assert_allclose(response.elastic_force[:,2],[0.,0.,20.,20.,20.,20.],atol=1e-12)
    assert not response.active[0] and not response.active[1] and not response.active[4]
    np.testing.assert_allclose(response.potential_energy,[0.,0.,1.,1.,1.,1.],atol=1e-13)
    np.testing.assert_allclose(response.dissipation,[0.,0.,1.,1.,200.,100.],atol=1e-12)
    assert np.all(response.force[:,2]>=0.) and np.all(response.dissipation>=0.)
    np.testing.assert_allclose(response.force[:,2]*v[:,2]-response.elastic_force[:,2]*v[:,2]+response.dissipation,0.,atol=1e-12)
    np.testing.assert_allclose(response.stiffness[:,2,2],[0.,0.,200.,200.,0.,0.],atol=1e-13)
    np.testing.assert_allclose(response.damping[:,2,2],[0.,0.,4.,4.,0.,0.],atol=1e-13)
    np.testing.assert_array_equal(response.force[:,:2],0.)
    response.force[:]=1.  # response storage is disposable, inputs/config are not
    np.testing.assert_array_equal(p[:,2],[-1.8,-1.9,-2.,-2.,-2.,-2.])


def test_exact_touch_and_raw_zero_use_inactive_branch():
    bed=SeabedContact(-2.,8.,4.)
    response=seabed_contact_force(bed,1.,1.,[[0.,0.,-1.5],[0.,0.,-2.]],[[0.,0.,-1.],[0.,0.,1.]])
    np.testing.assert_array_equal(response.force,0.);np.testing.assert_array_equal(response.stiffness,0.)
    np.testing.assert_array_equal(response.damping,0.);assert not response.active.any()
    assert response.potential_energy[1]==1. and response.dissipation[1]==4.


@pytest.mark.parametrize('failure',['elevation','stiffness','damping','bool','text','huge','version','unknown'])
def test_named_immutable_contact_and_config_refusals(failure):
    bed=SeabedContact(-2.,1000.,20.);data=bed.to_dict()
    assert SeabedContact.from_dict(copy.deepcopy(data))==bed
    with pytest.raises(FrozenInstanceError):bed.normal_stiffness=0.
    with pytest.raises(SeabedContactError):
        if failure=='elevation':data['elevation']=float('nan')
        elif failure=='stiffness':data['normal_stiffness']=0.
        elif failure=='damping':data['normal_damping']=-1.
        elif failure=='bool':data['elevation']=True
        elif failure=='text':data['normal_stiffness']='1000'
        elif failure=='huge':data['normal_stiffness']=10**400
        elif failure=='version':data['api_version']='wrong'
        else:data['execute']='unsafe'
        SeabedContact.from_dict(data)


@pytest.mark.parametrize('failure',['diameter','length','bool','shape','velocity','position','overflow'])
def test_endpoint_response_domain_refusals(failure):
    bed=SeabedContact(-2.,1000.,20.);D=.2;L=1.;p=[[0.,0.,-2.]];v=[[0.,0.,0.]]
    if failure=='diameter':D=0.
    elif failure=='length':L=-1.
    elif failure=='bool':p=[[False,0.,-2.]]
    elif failure=='shape':v=[[0.,0.,0.],[0.,0.,0.]]
    elif failure=='velocity':v=[[0.,0.,float('inf')]]
    elif failure=='position':p=[[0.,0.,float('nan')]]
    else:D=L=1e300
    with pytest.raises(SeabedContactError):seabed_contact_force(bed,D,L,p,v)
