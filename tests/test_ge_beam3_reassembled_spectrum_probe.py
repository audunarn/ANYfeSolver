"""Small full-map reassembly tests. No beam qualification or frozen rerun."""
import json
from pathlib import Path
import numpy as np
import pytest
from docs.reference_cases.ge_beam3_reassembled_spectrum_probe import solve
from anyfesolver._native_reassembled_factor_modes import solve_reassembled_factor_modes as native_solve
from test_ge_beam3_seeded_reduction_diagnostic import test_capture_reference_factor_reduction as capture
from test_ge_beam3_curved_contrast_probe import ROTATION
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_decimal_factor_audit import factor_roots


@pytest.mark.parametrize('backend',[solve,native_solve],ids=['fma_probe','dyadic_native'])
def test_curved_full_pencil_matches_decimal_and_rotation_covariance(tmp_path,backend):
    root=tmp_path/'inputs'; root.mkdir(); capture(root)
    made=[]
    for case in ('E','R90'):
        with np.load(root/(case+'.npz')) as packet:
            f,b=packet['factor'],packet['kinetic']; free=tuple(int(x) for x in packet['free'])
            algebraic=tuple(int(x) for x in packet['algebraic'])
        result=backend(f,np.zeros((42,42)),b,free,algebraic,bounds=(-10000.,100000000.))
        exact=np.array(factor_roots(f.tolist(),b.tolist(),list(free),list(algebraic),digits=80)[:6],dtype=float)
        with (tmp_path/(case+'.json')).open('xb') as stream: stream.write(canonical(result))
        np.testing.assert_allclose(result.eigenvalues,exact,rtol=1e-11,atol=1e-11)
        made.append((result,b))
    a,b=made
    np.testing.assert_allclose(a[0].eigenvalues,b[0].eigenvalues,rtol=1e-11,atol=1e-11)
    transform=np.kron(np.eye(14),ROTATION)
    correlation=(transform@a[0].full_modes).T@(b[1].T@b[1])@b[0].full_modes
    np.testing.assert_allclose(np.abs(correlation),np.eye(6),rtol=1e-11,atol=1e-11)


def test_signed_coupling_retains_negative_mode():
    f=np.diag([1.,1e12]); g=np.array([[-3.,1e12],[1e12,0.]])
    result=solve(f,g,np.eye(2),(0,1),(),bounds=(-5.,0.),num_modes=1)
    assert abs(result.eigenvalues[0]+3.)<1e-10
    assert abs(result.full_modes[1,0]/result.full_modes[0,0]+1e-12)<1e-23
