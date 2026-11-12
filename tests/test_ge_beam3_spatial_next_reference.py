import ast
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from scipy.interpolate import PPoly
from docs.reference_cases import ge_beam3_spatial_next_reference as reference
from docs.reference_cases import ge_beam3_spatial_next_comparison as comparison


def polynomial():
    c=np.sin(np.arange(4*3*52)).reshape(4,3,52)
    return PPoly.construct_fast(c,np.array([0.,.2,.65,1.]),extrapolate=False,axis=1)


def test_complete_polynomial_roundtrip():
    original=polynomial();saved=reference.pack(original);made=reference.unpack(saved)
    grid=np.r_[original.x,reference.base.validation_grid(original.x)]
    for derivative in (0,1,2,3):np.testing.assert_array_equal(made(grid,derivative),original(grid,derivative))
    assert np.isnan(made(-.01)).all() and np.isnan(made(1.01)).all()


@pytest.mark.parametrize('kind',('knots','shape','nan','axis','bool','extrapolate','extra'))
def test_polynomial_mutations(kind):
    v=reference.pack(polynomial())
    if kind=='knots':v['knots'][1]=0.
    elif kind=='shape':v['coefficients'][0][0].pop()
    elif kind=='nan':v['coefficients'][0][0][0]=float('nan')
    elif kind=='axis':v['axis']=0
    elif kind=='bool':v['axis']=True
    elif kind=='extrapolate':v['extrapolate']=True
    else:v['extra']=1
    with pytest.raises((ValueError,TypeError)):reference.unpack(v)


@pytest.mark.parametrize('target',(0.,.006,.007,float('nan'),True))
def test_no_unregistered_amplitude(target):
    with pytest.raises(ValueError):reference.solve({},target)


@pytest.mark.parametrize('sign',('plus','minus'))
def test_native_inputs_station_coverage_and_mutation(sign,tmp_path):
    row,recovery=comparison.inputs(sign)
    x,w,stress,strain=comparison.station_data(recovery)
    assert x.shape==w.shape==(192,) and stress.shape==strain.shape==(192,6)
    assert np.all(np.diff(x)>0) and np.all(w>0) and len(row['mechanical']['positions'])==49
    for kind in ('element','station','field','weight','xi'):
        changed=deepcopy(recovery)
        if kind=='element':changed[0]['element_id']=2
        elif kind=='station':changed[0]['stations'][0]['station']=1
        elif kind=='field':changed[0]['stations'][0]['strain'][0]=float('nan')
        elif kind=='weight':changed[0]['stations'][0]['measure']=-1.
        else:changed[0]['stations'][0]['xi']=1.
        with pytest.raises(ValueError):comparison.station_data(changed)
    (tmp_path/'manifest.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError):comparison.inputs(sign,tmp_path)


def test_frozen_continuum_source_and_no_production_imports():
    path=Path(reference.base.__file__)
    assert sha256(path.read_bytes()).hexdigest()==comparison.REFERENCE_SHA
    for module in (reference,reference.base):
        tree=ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(not x.name.startswith('anyfesolver') for x in node.names)
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('anyfesolver')
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','__import__') for n in ast.walk(tree))
