from copy import deepcopy
from math import fsum
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_spatial_stability_factors import capture,seed
from docs.reference_cases.ge_beam3_native_directional_work import evaluate,lift,adjudicate,STEPS


def test_actual_native_directional_variation_preserves_seed(seed):
    _,_,c,raw,_,packet,guard=capture(seed);p=json.loads(canonical(packet))
    direction=[float(.02*np.sin(i+.4)) for i in range(len(p['free_dofs']))]
    before=canonical(c.initial);events=[];saved=[]
    result=evaluate(c,c.initial,p,direction,progress=events.append,record=saved.append)
    assert c.checkpoint(())==raw and canonical(c.initial)==before
    assert result['max_stationary_lift_error']<=1e-11
    assert result['native_mixed_directional_work']>0
    assert [e['amount'] for e in events]==[0.,STEPS[0],-STEPS[0],STEPS[1],-STEPS[1]]
    for row in result['rows']:
        assert max(row[k] for k in ('potential_error','directional_work_error','full_residual_direction_error'))<=1e-7
    assert len(saved)==6
    assert canonical(saved)==canonical(json.loads(canonical(saved)))
    points={r['amount']:r for r in saved if r['kind']=='trial'}
    base=next(r for r in saved if r['kind']=='baseline')
    assert fsum(base['element_work'])==result['native_mixed_directional_work']
    for row in result['rows']:
        h=row['step'];plus=points[h];minus=points[-h]
        derivative=(np.array(plus['residual'])-np.array(minus['residual']))/(2*h)
        assert fsum((plus['potential'],minus['potential'],-2*points[0.]['potential']))/h**2==row['potential_second']
        assert fsum(float(a)*float(b) for a,b in zip(base['direction'],derivative))==row['residual_directional_work']
        assert float(np.linalg.norm(derivative-base['hessian_product']))/max(1.,float(np.linalg.norm(base['hessian_product'])))==row['full_residual_direction_error']
    guard()


@pytest.mark.parametrize('kind',('free_map','direction','internal','factors','zero'))
def test_direction_lift_mutation(seed,kind):
    _,_,c,_,_,packet,_=capture(seed);p=json.loads(canonical(packet));v=[.01]*len(p['free_dofs'])
    if kind=='free_map':p['free_dofs']=p['free_dofs'][1:]
    elif kind=='direction':v[0]=float('nan')
    elif kind=='internal':p['internal_layout'][0][1][0]+=1
    elif kind=='factors':p['right']=p['right'][:-1]
    else:v=[0.]*len(v)
    with pytest.raises(ValueError):lift(c.physical,p,v)


def good():
    return dict(elements=24,retained_dimension=870,original_free_dimension=426,
        trial_points=5,accepted_state_unchanged=True,material_history_committed=False,
        physical_control_constraint_added=False,max_stationary_lift_error=1e-15,native_mixed_directional_work=-.001,
        rows=[dict(step=h,potential_second=-.001,residual_directional_work=-.001,potential_error=1e-10,
            directional_work_error=1e-10,full_residual_direction_error=1e-9) for h in STEPS])


@pytest.mark.parametrize('kind',('coverage','commit','control','lift','sign','work','step','full_vector','nan','potential'))
def test_trial_adjudication_mutations(kind):
    v=good();adjudicate(v,-.001)
    if kind=='coverage':v['elements']=23
    elif kind=='commit':v['material_history_committed']=True
    elif kind=='control':v['physical_control_constraint_added']=True
    elif kind=='lift':v['max_stationary_lift_error']=1e-8
    elif kind=='sign':v['native_mixed_directional_work']=.001
    elif kind=='work':v['native_mixed_directional_work']=-.002
    elif kind=='step':v['rows'][0]['step']=.01
    elif kind=='full_vector':v['rows'][0]['full_residual_direction_error']=1e-4
    elif kind=='potential':v['rows'][0]['potential_second']=.001
    else:v['rows'][0]['potential_error']=float('nan')
    with pytest.raises(ValueError):adjudicate(v,-.001)


def test_failure_names_frozen_step_and_metric():
    v=good();v['rows'][0]['full_residual_direction_error']=2e-7
    with pytest.raises(ValueError,match='step 0.0001.*full_residual_direction_error'):
        adjudicate(v,-.001)
