"""New comparison authority/interpretation; no mechanics or base solutions."""
import ast
import copy
from pathlib import Path
import pytest
from docs.reference_cases import ge_beam3_saved_continuum_lateral_comparison as gate


def reference():
    r={k:0. for k in gate.REF_KEYS}
    p=sorted([-1.+i/128 for i in range(129)]+[-.12345])
    r.update(height=.1,axial=1000.,shear=400.,bending=.01,production_qualified=False,profile='BVP9',
        displacement=.04,load=.02,parameter=p,fields=[[float((j+1)*i) for i in range(130)] for j in range(4)])
    return r


def point():
    return dict(reference=reference(),row=dict(step=7,drop=.04,load=.021,reference_load=.02),recovered=[])


def observation():return dict(step=7,drop=.04,load=.021)


def test_exact_extraction_not_interpolation():
    r=reference();before=copy.deepcopy(r);v,indices=gate.fixed_grid(r)
    assert r==before and len(indices)==129 and v['parameter']==[-1.+i/128 for i in range(129)]
    assert all(v['fields'][j][i]==r['fields'][j][source] for i,source in enumerate(indices) for j in range(4))


@pytest.mark.parametrize('mutation',('missing','duplicate','reorder','nan','shape','nonfinite_field','boolean_field','extra_schema'))
def test_grid_mutations(mutation):
    r=reference()
    if mutation=='missing':r['parameter'][40]+=1e-6
    elif mutation=='duplicate':r['parameter'][40]=r['parameter'][39]
    elif mutation=='reorder':r['parameter'].reverse()
    elif mutation=='nan':r['parameter'][40]=float('nan')
    elif mutation=='shape':r['fields'][0].pop()
    elif mutation=='nonfinite_field':r['fields'][1][10]=float('inf')
    elif mutation=='boolean_field':r['fields'][1][10]=False
    else:r['extra']=None
    with pytest.raises(ValueError):gate.fixed_grid(r)


def test_matched_displacement_not_forced_load():
    r,_=gate.validate_point(point(),observation(),7)
    assert r['displacement']==observation()['drop'] and r['load']!=observation()['load']


@pytest.mark.parametrize('mutation',('material','profile','claim','drop','load','reference_load','boundary','differential','nan','step','extra'))
def test_point_mutations(mutation):
    p=point();o=observation()
    if mutation=='material':p['reference']['bending']=.02
    elif mutation=='profile':p['reference']['profile']='BVP7'
    elif mutation=='claim':p['reference']['production_qualified']=True
    elif mutation=='drop':o['drop']=.05
    elif mutation=='load':o['load']=.1
    elif mutation=='reference_load':p['row']['reference_load']=.021
    elif mutation=='boundary':p['reference']['boundary_error']=2e-11
    elif mutation=='differential':p['reference']['differential_error']=2e-8
    elif mutation=='nan':p['reference']['work_error']=float('nan')
    elif mutation=='step':o['step']=6
    else:p['extra']=None
    with pytest.raises(ValueError):gate.validate_point(p,o,7)


@pytest.mark.parametrize('det,count,agreement',((1.,0,True),(-1.,1,True),(1.,2,True),(-1.,3,True),(1.,1,False),(-1.,0,False)))
def test_parity_never_claims_full_inertia(det,count,agreement):
    result=gate.parity(det,count)
    assert result['parity_agreement'] is agreement and result['continuum_inertia_established'] is False
    assert result['root_uniqueness_established'] is False


@pytest.mark.parametrize('det,count',((0.,0),(float('nan'),0),(float('inf'),0),(1.,True),(1.,-1)))
def test_unresolved_parity(det,count):
    with pytest.raises(ValueError):gate.parity(det,count)


def test_hash_mutation_before_reference_evaluation(monkeypatch):
    monkeypatch.setattr(gate,'read',lambda p:b'changed')
    with pytest.raises(ValueError,match='count aggregate authority'):gate.inputs(4,6)


def test_no_production_import_or_base_solve():
    tree=ast.parse(Path(gate.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert all(not n.name.startswith('anyfesolver') for n in node.names)
        if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('anyfesolver')
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            assert name not in ('solve','solve_bvp','prepare','restore','stage','solve_modes')
