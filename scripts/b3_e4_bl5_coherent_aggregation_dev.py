"""Small uniform BL5 aggregation screen; no F03 or multiplier selection.

Complete P0 alternatives c_lambda=1/2 use the same rule for all five channels.
Geometry, supports, loads and physical output fields are fixed before execution.
Trace beta is deliberately absent until its physical dual norm is justified.
"""
from dataclasses import replace
from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
import numba
import anyfesolver
from threadpoolctl import threadpool_info
from anyfesolver import stiffened_shell
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition

LOADS={"fy":(0.,100.,0.,0.,0.,0.),"mx":(0.,0.,0.,10.,0.,0.),"mz":(0.,0.,0.,0.,0.,10.)}


def model(shell_edges, transverse_edges, halfcells, c_lambda, *, basis="p0"):
    if basis not in ("p0", "discontinuous_p1"):
        raise ValueError("explicit coherent development basis required")
    if basis == "discontinuous_p1":
        if c_lambda != 2:
            raise ValueError("affine candidate has one complete two-trace resolution rule")
        # Whole members prevent an affine mode on one constant-director
        # half-cell. All five channels share this support/basis rule.
        span=2*math.ceil(max(1.,halfcells/shell_edges))
    else:
        span=max(1,math.ceil(c_lambda*halfcells/shell_edges))
    if halfcells%span:
        raise ValueError("uniform coherent rule cannot infer a partial support")
    base=_bl5_joint_mesh_definition(shell_edges,halfcells,1)
    row=shell_edges+1
    xyz=tuple((float(x),float(y),0.) for y in np.linspace(0.,.2,transverse_edges+1)
              for x in np.linspace(0.,1.,row))
    cells=tuple((1+j*row+i,2+j*row+i,2+(j+1)*row+i,1+(j+1)*row+i)
                for j in range(transverse_edges) for i in range(shell_edges))
    aspect=max((1./shell_edges)/(.2/transverse_edges),(.2/transverse_edges)/(1./shell_edges))
    if aspect>4.:raise ValueError("development bond mesh exceeds registered aspect bound")
    shell=replace(base.shell,node_ids=tuple(range(1,len(xyz)+1)),coordinates_m=xyz,q4_elements=cells)
    line=replace(base.lines[0],multiplier_span=span,basis=basis)
    fixed=tuple(stiffened_shell.FixedCoordinate("shell",1+j*row,axis)
                for j in range(transverse_edges+1) for axis in range(6))
    fixed+=tuple(stiffened_shell.FixedCoordinate("beam",101,axis) for axis in range(5))
    definition=replace(base,shell=shell,lines=(line,),fixed=fixed)
    shell_x=np.array(xyz[:row])[:,0]
    beam_tables=[json.loads(beam.raw)['reference']['coordinates'] for beam in definition.beams]
    beam_x=np.array([beam_tables[i//2][i%2][0]
                     for i in range(halfcells)]+[beam_tables[-1][-1][0]])
    hs,hb=np.diff(shell_x),np.diff(beam_x)
    if not (np.allclose(hs,hs[0],rtol=1e-13,atol=1e-14) and
            np.allclose(hb,hb[0],rtol=1e-13,atol=1e-14)):
        raise ValueError("coherent development policy requires uniform projected traces")
    if span*hb[0]<c_lambda*hs[0]-1e-13:
        raise ValueError("physical support does not cover the declared shell resolution")
    policy=dict(c_lambda=c_lambda,span_halfcells=span,shape_aspect=aspect,
        shell_trace_segments=shell_edges,beam_halfcells=halfcells,
        support_membership=[list(range(a,a+span)) for a in range(0,halfcells,span)],
        support_endpoints_m=[float(x) for x in beam_x[::span]],
        shell_spacing_m=float(hs[0]),beam_spacing_m=float(hb[0]))
    if basis != "p0":
        policy.update(basis=basis, density_basis=("1", "2*(x-center)/support_width"),
                      qualification="EXPERIMENTAL_NOT_QUALIFIED")
    return definition,policy


def physical_transfer(owner,result):
    positions=np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
    frames=np.array([owner._beam_nodes[n][1] for n in owner._beam_ids])
    state=Mechanical(positions,np.zeros_like(positions),frames,
        np.tile(np.eye(3),(len(owner._beam_parts),2,1,1)),np.zeros((len(owner._beam_parts),18)))
    q=np.r_[result.shell_displacement,result.beam_coordinates]; root=positions[0]
    moments=np.zeros((3,6)); interior=np.zeros(6); gaps=[]; records=[]
    recovered=np.zeros(len(q)); parity=0.
    for sample,(rows,basis) in zip(owner._line_samples,owner._line_terms):
        local,slots,weight,_,_=sample; density=basis@result.multipliers[rows]
        high,low,rb=local.port.pose_pair(state); xb=high+low
        xs=np.array((1.-local.station,local.station))@local.reference
        director=rb@local.reference_director
        fb=rb@density[:3]
        couple=np.cross(director,rb@local.director_basis.T@density[3:])
        mb=np.cross(rb@local.reference_arm,fb)+couple
        ms=np.cross(.5*local.thickness*director,-fb)-couple
        ns=np.array((1.-local.station,local.station))
        pullback=np.r_[ns[0]*(-fb),ns[0]*ms,ns[1]*(-fb),ns[1]*ms,
            (1.-local.port.weight)*fb,local.port.weight*fb,
            mb+np.cross(rb@local.port.lift,fb)]
        evaluated=local.evaluate(np.zeros((2,6)),state,density)
        parity=max(parity,float(np.max(np.abs(pullback+evaluated.gradient))))
        np.add.at(recovered,slots,weight*pullback)
        g=evaluated.gap_jacobian@q[slots];gaps.append(g)
        wrench=np.r_[fb,mb+np.cross(xb-root,fb)]
        x=float(xs[0])
        for power in range(3):moments[power]+=weight*x**power*wrench
        if .25<=x<=.75:interior+=weight*wrench
        records.append(dict(x_m=x,measure_m=float(weight),beam_wrench_about_root_density=wrench.tolist(),
                            five_channel_gap=g.tolist()))
    np.testing.assert_allclose(recovered,-owner._C.T@result.multipliers,atol=1e-8,rtol=1e-10)
    assert parity<1e-8
    gaps=np.array(gaps)
    return dict(polynomial_transfer_moments=moments.tolist(),fixed_region_025_075_wrench=interior.tolist(),
        pointwise_translation_gap_max_m=float(np.max(np.abs(gaps[:,:3]))),
        pointwise_director_gap_max=float(np.max(np.abs(gaps[:,3:]))),
        local_pullback_max_abs=parity,station_records=records,
        measure="Integrals of beam-side physical exerted wrench density, Nm about beam root; not nodal split or weld stress")


def analyze(output, refine_from=None):
    if output.exists():raise FileExistsError("retain prior screen output")
    pools=threadpool_info()
    assert pools and all(p['num_threads']==1 for p in pools) and numba.config.DISABLE_JIT
    result={'classification':'development only; no physical inf-sup or F03 acceptance',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'imported_anyfesolver':anyfesolver.__file__,'threads':pools,'jit_disabled':True,
        'decision':'Check coherent support representability/work and transfer sensitivity before extending ratio/refinement envelope',
        'outcomes':'Work/map failure: repair. Nonfinite/singular actual KKT: investigate physical mode. Valid but different transfer: assess convergence under common fixed regions, no basis tuning.',
        'loads':LOADS,'cases':[],'operator_constructions':0}
    if refine_from is None:
        grids=((8,2,4),(8,2,16))
    else:
        prior=json.loads(refine_from.read_text())
        if (prior.get('status')!='COMPLETED_DEVELOPMENT_ONLY' or prior.get('loads')!={k:list(v) for k,v in LOADS.items()} or
                [(c['policy']['shell_trace_segments'],c['policy']['beam_halfcells'],c['policy']['c_lambda'])
                 for c in prior['cases']]!=[(8,4,1),(8,4,2),(8,16,1),(8,16,2)]):
            raise ValueError('unchanged completed reciprocal-ratio predecessor required')
        result['predecessor_sha256']=hashlib.sha256(refine_from.read_bytes()).hexdigest()
        result['decision']='Distinguish converging enforcement difference from persistent policy dependence under the observed normal-couple sensitivity'
        result['outcomes']='Common work/transfer limit: proceed complete norm/ratio envelope. Persistent policy-dependent limit or growing discrepancy: inspect saved work modes; no automatic further mesh.'
        # Three systematic two-direction levels include the retained8x2 case.
        # This is a diagnostic, not a frozen accuracy or F03 assessment.
        grids=((16,4,32),(32,8,64))
    owners={}
    try:
        for shell_edges,transverse_edges,halfcells in grids:
            for c_lambda in (1,2):
                definition,policy=model(shell_edges,transverse_edges,halfcells,c_lambda)
                key=(halfcells,policy['span_halfcells'])
                if key in owners:
                    result['cases'].append(dict(policy=policy,reused_case=owners[key]))
                    continue
                start=time.perf_counter();owner=stiffened_shell.create_analysis(definition=definition)
                result['operator_constructions']+=1
                try:
                    item=dict(policy=policy,identity=owner.preflight()['identity'],
                        setup_seconds=time.perf_counter()-start,matrix_order=owner._A.shape[0],
                        matrix_nnz=owner._A.nnz,constraint_rows=owner._C.shape[0],loads={})
                    owners[key]=len(result['cases'])
                    assert owner._C.shape[0]==5*halfcells//policy['span_halfcells']
                    for name,load in LOADS.items():
                        start=time.perf_counter()
                        solved=owner.solve_linear((stiffened_shell.NodalAction('beam',101+halfcells,load),))
                        assert solved.status=='completed' and solved.free_equilibrium_ratio<=1e-8 and solved.constraint_max<=1e-9
                        q=np.r_[solved.shell_displacement,solved.beam_coordinates]
                        work=float(q@owner._accepted[1])
                        np.testing.assert_allclose(q@(owner._K@q),work,atol=1e-12,rtol=1e-8)
                        xyz=np.vstack((owner._shell_xyz,np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])))
                        reactions=solved.reactions[:6*len(xyz)].reshape(-1,6)
                        root=xyz[len(owner._shell_ids)]
                        ground=np.r_[reactions[:,:3].sum(axis=0),
                            (reactions[:,3:]+np.cross(xyz-root,reactions[:,:3])).sum(axis=0)]
                        applied=np.r_[load[:3],np.array(load[3:])+np.cross(xyz[-1]-root,load[:3])]
                        np.testing.assert_allclose(ground,-applied,atol=1e-7,rtol=1e-8)
                        item['loads'][name]=dict(solve_seconds=time.perf_counter()-start,applied_work_j=work,
                            free_equilibrium=solved.free_equilibrium_ratio,weak_constraint=solved.constraint_max,
                            ground_wrench_about_beamroot=ground.tolist(),
                            beam_tip_pose=solved.beam_coordinates[6*(len(owner._beam_ids)-1):6*len(owner._beam_ids)].tolist(),
                            transfer=physical_transfer(owner,solved))
                    assert owner.preflight()['reference_linear_factorizations']==1
                    item['factorizations']=1
                    result['cases'].append(item)
                finally:owner.close()
        result['status']='COMPLETED_DEVELOPMENT_ONLY'
    finally:
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refine-from',type=Path)
    args=parser.parse_args();analyze(args.output,args.refine_from)
