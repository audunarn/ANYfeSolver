"""DEV proposal, ownership and caller schema checks; no material qualification."""
from dataclasses import replace,FrozenInstanceError
from types import SimpleNamespace
import math
import numpy as np
import pytest
from anyfesolver._stiffened_shell_q4_material_event import Q4MaterialTrial,boundary_amount,_origin_values
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver

def fixture(strain=.49,change=.1):
    origin=dict(plastic_strain=np.zeros((1,3)),alpha=np.zeros(1))
    jac=(((change,)+(0.,)*23,(0.,)*24,(0.,)*24),)
    trial=Q4MaterialTrial(origin,_origin_values(origin),(0.,)*24,((strain,0.,0.),),jac,
        ((200.,0.,0.),(0.,200.,0.),(0.,0.,100.)),(100.,),1e-10)
    return trial,origin,(1.,)+(0.,)*23

@pytest.mark.parametrize('strain,change,after',[(.49,.1,'plastic'),(-.49,-.1,'plastic'),(.6,-.2,'elastic')])
def test_actual_numerical_switch_and_reverse(strain,change,after):
    t,o,d=fixture(strain,change);amount=boundary_amount(t,d,origin=o)
    assert amount is not None and 0<amount<1
    actual=200*(strain+amount*change)
    switched=actual*actual/3-10000/3>1e-10*10000
    assert switched==(after=='plastic')
    assert boundary_amount(t,iter(d),origin=o)==amount
    with pytest.raises(FrozenInstanceError):t.flow=(1.,)

def test_absent_event_and_invalid_input_and_origin_mutation():
    t,o,d=fixture(0.,.01)
    assert boundary_amount(t,d,origin=o) is None
    for bad in (d[:-1],(float('nan'),)+d[1:],(True,)+d[1:]):
        with pytest.raises(ValueError):boundary_amount(t,bad,origin=o)
    with pytest.raises(ValueError):boundary_amount(replace(t,strain_differential=()),d,origin=o)
    with pytest.raises(ValueError):boundary_amount(t,d,origin=dict(o))
    o['alpha'][0]=1.
    with pytest.raises(ValueError):boundary_amount(t,d,origin=o)

def test_two_positive_unload_and_reverse_yield_roots_select_first():
    t,o,d=fixture(.6,-1.2)
    amount=boundary_amount(t,d,origin=o)
    assert .08<amount<.09
    assert 200*(.6-1.2*amount)>0.

def test_tangential_stress_direction_without_crossing():
    t,o,d=fixture()
    t=replace(t,strain=((0.,0.,1.),))
    assert boundary_amount(t,d,origin=o) is None

def test_initial_strain_offset_presence_and_mutation_binding():
    t,o,d=fixture();o['initial_membrane_prestrain']=np.zeros((1,3));t=replace(t,origin_values=_origin_values(o))
    assert boundary_amount(t,d,origin=o) is not None
    o['initial_membrane_prestrain'][0,0]=1e-6
    with pytest.raises(ValueError,match='origin-bound'):boundary_amount(t,d,origin=o)
    o['initial_membrane_prestrain'][0,0]=0.;del o['initial_membrane_prestrain']
    with pytest.raises(ValueError,match='origin-bound'):boundary_amount(t,d,origin=o)
    t,o,d=fixture();o['initial_bending_stress']=np.zeros((1,3))
    with pytest.raises(ValueError,match='origin-bound'):boundary_amount(t,d,origin=o)

def test_driver_shell_binding_mutation_and_legacy_dispatch():
    t,o,d=fixture();owner=SimpleNamespace(_bl5_native_material=True,_generation=5,_shell_count=24)
    driver=object.__new__(FinitePointDriver);driver.owner=owner;driver._beam_nodal=0
    driver._beam_origins=();driver._shell_origins=(o,)
    state=SimpleNamespace(q=np.zeros(24),resultants=np.empty((0,18)))
    driver._native_boundary_trial=(state,driver._beam_origins,5,())
    driver._q4_boundary_trial=(state,driver._shell_origins,5,((tuple(range(24)),t),))
    assert driver._native_boundary_amount(state,np.array(d)) is not None
    # A smaller native event owns the same combined scalar proposal.
    native=SimpleNamespace(resultants=(0.,)*18)
    native_origin=object();driver._beam_origins=(native_origin,)
    cell=SimpleNamespace(boundary_amount=lambda *args,**kwargs:.03)
    operator=SimpleNamespace(cell=cell,guard=lambda:None)
    owner._beam_parts=((SimpleNamespace(operator=operator),None,None),)
    state.resultants=np.zeros((1,18))
    driver._native_boundary_trial=(state,driver._beam_origins,5,((0,native),))
    assert driver._native_boundary_amount(state,np.r_[d,np.zeros(24)])==.03
    driver._q4_boundary_trial=(state,driver._shell_origins,4,())
    with pytest.raises(ValueError,match='stale'):driver._native_boundary_amount(state,np.array(d))
    driver._q4_boundary_trial=(state,driver._shell_origins,5,((tuple(range(24)),t),))
    state.q[0]=1.
    with pytest.raises(ValueError,match='changed'):driver._native_boundary_amount(state,np.array(d))
    owner._bl5_native_material=False
    assert driver._native_boundary_amount(state,np.array(d)) is None

def test_actual_q4_capture_preserves_three_response_fields():
    from anyfesolver import stiffened_shell as ss
    from examples.stiffened_shell_native_fibre_bl5_material import definition
    from anyfesolver._stiffened_shell_q4_algorithmic_pullback import response
    owner=ss.create_analysis(definition=definition())
    try:
        driver=FinitePointDriver(owner)
        eid,element,slots,_=owner._shell_parts[0]
        origin=driver._shell_origins[0];u=driver._state.q[slots]
        ordinary=response(owner._shell_model,element,u,origin,3)
        captured=response(owner._shell_model,element,u,origin,3,capture_material_trial=True)
        assert type(ordinary) is tuple and len(ordinary)==3 and len(captured)==4
        np.testing.assert_array_equal(ordinary[0],captured[0]);np.testing.assert_array_equal(ordinary[1],captured[1])
        for key in ('alpha','plastic_strain','layer_strain','layer_stress'):
            np.testing.assert_array_equal(ordinary[2][key],captured[2][key])
        assert captured[3].origin is origin and isinstance(captured[3].strain_differential,tuple)
        with pytest.raises(ValueError,match='capture policy'):
            response(owner._shell_model,element,u,origin,3,capture_material_trial=1)
    finally:owner.close()

def test_nonzero_rotation_and_von_karman_layer_direction():
    from anyfesolver import stiffened_shell as ss
    from examples.stiffened_shell_native_fibre_bl5_material import definition
    from anyfesolver._stiffened_shell_q4_algorithmic_pullback import response
    owner=ss.create_analysis(definition=definition())
    try:
        driver=FinitePointDriver(owner);eid,element,slots,_=owner._shell_parts[0];origin=driver._shell_origins[0]
        u=np.linspace(-1e-6,2e-6,24);u[3::6]=np.linspace(1e-4,3e-4,4);u[4::6]=np.linspace(-2e-4,1e-4,4)
        d=np.linspace(-2e-5,3e-5,24);h=1e-5
        selected=response(owner._shell_model,element,u,origin,3,capture_material_trial=True)
        positive=response(owner._shell_model,element,u+h*d,origin,3)[2]['layer_strain']
        negative=response(owner._shell_model,element,u-h*d,origin,3)[2]['layer_strain']
        expected=np.asarray(selected[3].strain_differential)@d
        error=np.linalg.norm((positive-negative)/(2*h)-expected)/np.linalg.norm(expected)
        assert error<=1e-5,error
    finally:owner.close()
