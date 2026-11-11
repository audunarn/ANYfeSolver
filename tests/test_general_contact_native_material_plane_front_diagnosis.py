"""One saved-pose quadrature discriminator; not a qualification predicate."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from anyfesolver._ge_beam3_mixed_ad import Jet2
from anyfesolver.general_contact.native_surface_normal import _material_cell_normal_data
from test_general_contact_native_material_plane import _analysis


@pytest.mark.skipif(not os.environ.get('CONTACT_FRONT_INPUT') or not os.environ.get('CONTACT_FRONT_OUTPUT'),
                    reason='explicit saved-pose diagnostic paths required')
def test_saved_stronger_penalty_front_resolution():
    saved=json.loads(Path(os.environ['CONTACT_FRONT_INPUT']).read_text())
    kernel=_analysis(normal_stiffness=1e7).contact
    c=saved['diagnostics']['configuration']
    pose=kernel._configuration(np.array(c['positions']),np.array(c['position_low']),np.array(c['cell_rotations']))
    mapping=np.eye(9)
    q=[Jet2.variable(0.,i,9) for i in range(9)]
    integrand,_,cuts=_material_cell_normal_data(kernel.rule,pose,kernel.cell,
        *kernel.rule.definition['xi_range'],kernel.stiffness,q,
        [Jet2.constant(v,9) for v in kernel.normal],kernel.origin,
        [Jet2.constant(0.,9) for _ in range(3)],mapping,compensated_projection=True)
    pending=list(zip(cuts[:-1],cuts[1:]));rows=[];width=cuts[-1]-cuts[0]
    for _ in range(64):
        if not pending:break
        l,r=pending.pop();half=(r-l)/2;mid=l+half
        estimates=[]
        for count in (8,16):
            x,w=leggauss(count);estimate=Jet2.constant(0.,9)
            for point,weight in zip(x,w):estimate+=half*weight*integrand(mid+half*point)
            estimates.append(estimate)
        diff=estimates[1]-estimates[0]
        scale=np.r_[np.ones(6),np.full(3,.5)]
        err=np.array([abs(diff.value),max(abs(scale*diff.gradient)),
            np.max(abs(scale[:,None]*diff.hessian*scale[None,:]))])
        ratios=err/(np.array([1e-12,1e-10,1e-9])*(r-l)/width)
        rows.append(dict(left=l,right=r,error=err.tolist(),ratios=ratios.tolist(),
                         energy=estimates[1].value))
        if np.any(ratios>1):pending.extend(((l,mid),(mid,r)))
    Path(os.environ['CONTACT_FRONT_OUTPUT']).write_text(json.dumps(dict(cuts=cuts,rows=rows,
        pending=pending,scope='isolated saved pose; exact declared controls'),indent=2))
    assert rows
