"""One saved-fixture work/patch screen for a distinct projected density candidate.

No owner/product basis change or loaded solve. Projection changes force moments
and does not remove the existing quadratic trace interpolation mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_info
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator
from anyfesolver.corotational import rotation_matrix_from_vector

IDENTITY = 'BL5_REFERENCE_HALFCELL_AVERAGED_AFFINE_DENSITY_DEV_V1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compile_saved_projection(root):
    """Reconstruct the fixed local work once, without constructing an owner."""
    data = json.loads((root/'native/inputs.json').read_text())
    nx = data['layout']['nx']; ns = len(data['layout']['shell_node_ids'])
    physical = data['layout']['counts']['physical_dofs']; xyz = np.array(data['xyz_m'])
    supports = len(data['policy']['support_endpoints_m'])-1; m = 2*nx//supports
    assert (nx,supports) in ((16,8),(32,8),(32,16)) and m in (4,8)
    with np.load(root/'native/operators.npz', allow_pickle=False) as z:
        k, c = (sparse.csr_matrix((z[n+'_data'],z[n+'_indices'],z[n+'_indptr']),
            shape=tuple(z[n+'_shape'])) for n in ('K','C'))
        rigid = (z['shell_rigid_work_map']+z['beam_rigid_work_map']).T
        shell_work = z['shell_rigid_work_map'].copy()
        beam_work = z['beam_rigid_work_map'].copy()
        fixed = z['fixed'].copy()
    state = Mechanical(xyz[ns:], np.zeros_like(xyz[ns:]),
        np.tile(np.eye(3),(2*nx+1,1,1)),np.tile(np.eye(3),(nx,2,1,1)),np.zeros((nx,18)))
    records=[]; rr=[]; cc=[]; vv=[[],[]]
    force_integral=np.zeros(2); force_moment=np.zeros(2); gram=np.zeros(2)
    for record in data['station_traces']:
        b=record['beam_port']
        port=NativeHalfCellPort(tuple(b['node_indices']),b['member_index'],b['native_cell'],
            b['weight'],b['nodal_count'],b['retained_count'],np.array(b['lift_m']),np.array(b['frame']))
        local=BL5LocalOperator.at_reference(record['shell_reference_m'],np.zeros((2,6)),
            record['thickness_m'],port,state,station=record['shell_alpha'])
        np.testing.assert_allclose(local.reference_arm,record['reference_arm_m'],rtol=0.,atol=1e-13)
        jet=local.evaluate(np.zeros((2,6)),state,np.zeros(5))
        assert np.max(np.abs(jet.gap))<=1e-13
        rows=np.array(record['multiplier_rows']); slots=np.array(record['full_slots'])
        basis=np.array(record['density_basis']); projected=basis.copy()
        edge=2*b['member_index']+b['native_cell']; mean_xi=2*((edge % m)+.5)/m-1
        for channel in range(5): projected[channel,2*channel+1]=mean_xi
        weight=record['measure_m']
        rr.extend(np.repeat(rows,len(slots)));cc.extend(np.tile(slots,len(rows)))
        for j,phi in enumerate((basis,projected)):
            vv[j].extend((weight*phi.T@jet.gap_jacobian).ravel())
        records.append((local,slots,rows,projected,weight))
        if edge<m:
            x=(edge+b['weight'])/(2*nx)
            for j,phi in enumerate((basis[0,1],mean_xi)):
                force_integral[j]+=weight*phi;force_moment[j]+=weight*x*phi
                gram[j]+=weight*phi*phi
    original, projected=(sparse.coo_matrix((v,(rr,cc)),shape=c.shape).tocsr() for v in vv)
    return (data,k,c,original,projected,rigid,shell_work,beam_work,fixed,state,
            records,force_integral,force_moment,gram)


def run(root):
    (data,k,c,original,projected,rigid,shell_work,beam_work,fixed,state,
     records,force_integral,force_moment,gram)=compile_saved_projection(root)
    nx=data['layout']['nx'];ns=len(data['layout']['shell_node_ids'])
    physical=data['layout']['counts']['physical_dofs'];xyz=np.array(data['xyz_m'])
    supports=len(data['policy']['support_endpoints_m'])-1;m=2*nx//supports
    difference=original-c
    parity=float(np.linalg.norm(difference.data)/max(1.,np.linalg.norm(c.data)))
    assert parity<=1e-12
    assert np.max(np.abs(projected@rigid))<=1e-12
    H=1./supports;h=1./(2*nx);kappa=1e-4
    np.testing.assert_allclose(force_integral,0.,rtol=0.,atol=1e-14)
    np.testing.assert_allclose(force_moment[1]-force_moment[0],-H*H/(6*m*m),rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(gram,(H/3,H*(1-1/m**2)/3),rtol=1e-12,atol=1e-14)
    # Recover each projected affine channel's actual physical domain wrench.
    # The force acts at the shell top. Saved work maps use the BEAM ROOT origin,
    # not (0,0,0); preserve that same origin in the independent moment integral.
    origin=xyz[ns].copy()
    wrench_error=0.;wrench_details=[]
    for support,a in ((0,0.),(1,1.)):
        resultant=a*H
        first_moment=a*H*H*(support+.5)+H*H*(1-1/m**2)/6
        for channel in range(5):
            density=np.zeros(c.shape[0]);row=2*(channel*supports+support)
            density[row:row+2]=(a,1.)
            action=projected.T@density
            observed=beam_work@action;opposite=shell_work@action;expected=np.zeros(6)
            if channel<3:
                force=np.eye(3)[channel]
                expected[:3]=-resultant*force
                arm_integral=(first_moment-origin[0]*resultant,-origin[1]*resultant,
                              (data['shell']['thickness_m']/2-origin[2])*resultant)
                expected[3:]=-np.cross(arm_integral,force)
            else:
                expected[3+(1 if channel==3 else 0)]=(-1 if channel==3 else 1)*resultant
            wrench_error=max(wrench_error,float(np.max(np.abs(observed-expected))),
                             float(np.max(np.abs(observed+opposite))))
            wrench_details.append(dict(support=support,channel=channel,observed=observed.tolist(),expected=expected.tolist()))
    if wrench_error>1e-12:
        print(json.dumps(dict(wrench_error=wrench_error,controls=wrench_details)),flush=True)
    assert wrench_error<=1e-12
    for channel in range(3):
        q=np.zeros(k.shape[0]);q[6*ns:physical].reshape(-1,6)[:,channel]=1.
        gap=(projected@q).reshape(5,supports,2)
        expected=np.zeros_like(gap);expected[channel,:,0]=-H
        np.testing.assert_allclose(gap,expected,rtol=1e-12,atol=1e-13)
    # Common finite rigid motion uses physical rotations of retained cells too.
    spin=np.array((.13,-.08,.11));rotation=rotation_matrix_from_vector(spin);shift=np.array((.3,-.2,.4))
    moved=Mechanical(xyz[ns:]@rotation.T+shift,np.zeros_like(xyz[ns:]),
        np.tile(rotation,(2*nx+1,1,1)),np.tile(rotation,(nx,2,1,1)),np.zeros((nx,18)))
    shell=np.zeros((ns,6));shell[:,:3]=xyz[:ns]@rotation.T+shift-xyz[:ns];shell[:,3:]=spin
    objectivity=0.
    for local,slots,rows,phi,weight in records:
        value=local.evaluate(shell[(slots[:12].reshape(2,6)[:,0]//6)],moved,np.ones(5))
        objectivity=max(objectivity,float(np.max(np.abs(value.gap))),abs(value.value))
    assert objectivity<=1e-12
    nodal=np.zeros(k.shape[0]);s=nodal[:6*ns].reshape(-1,6)
    s[:,3]=kappa*xyz[:ns,0];s[:,1]=.0675*s[:,3];s[:,2]=xyz[:ns,1]*s[:,3]
    nodal[6*ns:physical].reshape(-1,6)[:,3]=kappa*xyz[ns:,0]
    psi=np.array([physical+24*b+3*j for b in range(nx) for j in (0,1)])
    moments=np.array([physical+24*b+12+3*j for b in range(nx) for j in range(4)])
    unknown=np.r_[psi,moments];block=k[unknown][:,unknown].tocsc();rhs=-(k@nodal)[unknown]
    rows=np.arange(8*supports,10*supports);d=projected[rows][:,unknown]
    value=spsolve(sparse.bmat([[block,d.T],[d,None]],format='csc'),np.r_[rhs,-(projected@nodal)[rows]])
    q=nodal.copy();q[unknown]=value[:len(unknown)];np.testing.assert_array_equal(q[fixed],0.)
    beam_k=k[6*ns:,6*ns:];energy=lambda v:float(.5*v[6*ns:]@(beam_k@v[6*ns:]))
    torsion=data['beams'][0]['section']['elastic'][3][3];continuum=.5*torsion*kappa*kappa
    ratio=energy(q)/continuum;np.testing.assert_allclose(ratio,1.,rtol=1e-10,atol=1e-12)
    assert np.max(np.abs(projected@q))<=1e-12
    # Compatible continuum bending is sampled on two different nodal grids.
    bend=np.zeros_like(q);s=bend[:6*ns].reshape(-1,6);s[:,4]=-kappa*xyz[:ns,0]
    s[:,0]=-.0675*s[:,4];s[:,2]=.5*kappa*xyz[:ns,0]**2
    b=bend[6*ns:physical].reshape(-1,6);b[:,4]=-kappa*xyz[ns:,0];b[:,2]=.5*kappa*xyz[ns:,0]**2
    bend[psi+1]=-kappa*(np.arange(2*nx)+.5)*h
    bending_gap=(projected@bend).reshape(5,supports,2)
    np.testing.assert_allclose(bending_gap[2,:,0]/H,kappa*h*h/4,rtol=1e-8,atol=1e-13)
    # Beam nodal alternation is invisible to BOTH interface rules but has energy.
    blind=np.zeros_like(q);blind[6*ns:physical].reshape(-1,6)[1:,3]=kappa*(-1.)**np.arange(1,2*nx+1)
    blind[moments]=spsolve(k[moments][:,moments].tocsc(),-(k@blind)[moments])
    assert np.max(np.abs(c@blind))<=1e-13 and np.max(np.abs(projected@blind))<=1e-13
    blind_energy=energy(blind);assert np.isfinite(blind_energy) and blind_energy>0.
    # Independent local work derivative uses the projected physical density.
    rng=np.random.default_rng(17);direction=rng.normal(size=len(q))*1e-2;density=rng.normal(size=c.shape[0])
    predicted=float(density@(projected@direction));actual=0.;step=1e-6
    for sign in (-1,1):
        v=sign*step*direction
        beam=xyz[ns:]+v[6*ns:physical].reshape(-1,6)[:,:3]
        cells=np.array([rotation_matrix_from_vector(v[physical+24*i+3*j:physical+24*i+3*j+3])
                        for i in range(nx) for j in (0,1)]).reshape(nx,2,3,3)
        trial=Mechanical(beam,np.zeros_like(beam),state.nodal_frames,cells,state.resultants)
        total=v[:6*ns].reshape(-1,6);work=0.
        for local,slots,rows,phi,weight in records:
            shellrows=slots[:12].reshape(2,6)[:,0]//6
            work+=weight*local.evaluate(total[shellrows],trial,phi@density[rows]).value
        actual+=sign*work/(2*step)
    derivative_error=abs(actual-predicted)/max(1.,abs(predicted));assert derivative_error<=1e-6
    return dict(m=m,original_operator_relative=parity,finite_objectivity=objectivity,
        projected_torsion_energy_ratio=ratio,normal_bending_mean_gap=(bending_gap[2,:,0]/H).tolist(),
        normal_bending_integral_gap=bending_gap[2,:,0].tolist(),
        expected_bending_mean=kappa*h*h/4,affine_force_first_moment=force_moment.tolist(),
        density_gram=gram.tolist(),interface_kernel_vector_moment_eliminated_energy_J=blind_energy,
        recovered_physical_wrench_error=wrench_error,
        physical_wrench_origin_m=origin.tolist(),
        local_dual_work_derivative_error=derivative_error,
        source_files={name:sha(root/'native'/name) for name in ('inputs.json','operators.npz')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixed',type=Path,required=True)
    p.add_argument('--growing',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    pools=threadpool_info();assert pools and all(v['num_threads']==1 for v in pools)
    result=dict(identity=IDENTITY,status='COMPLETED_DEVELOPMENT_ONLY',source_sha256=sha(__file__),
        scope='projected density work/patch screen, no product selection, loaded solve or F03 acceptance',
        pools=pools,cases=[run(a.fixed),run(a.growing)])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v['m']:v['projected_torsion_energy_ratio'] for v in result['cases']}))
