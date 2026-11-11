"""New bounded connected-member cases, not a historical qualification rerun."""
import json
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_fibre_line_work import ReferenceLineForces
from anyfesolver import _ge_beam3_equilibrium_line_program as native
from anyfesolver import _ge_beam3_fibre_line_program as dense
from test_ge_beam3_spatial_nodal_moments import model as section_model
from test_ge_beam3_schur_line_program import save,numerical_difference

CASES=('straight-2-plastic','curved-2-clamped','curved-4-cantilever')


def inputs(case):
    if case not in CASES: raise ValueError('registered connected-member case')
    plastic=case=='straight-2-plastic'; clamped=case=='curved-2-clamped'
    macros=4 if case=='curved-4-cantilever' else 2
    height=0. if plastic else .15
    section=section_model(plastic=plastic,coupled=True).mesh.elements[1].section
    made=FEModel(case); frames=[]
    for i,x in enumerate(np.linspace(-1.,1.,2*macros+1),1):
        made.add_node(i,float(x),float(height*(1-x*x)),0.)
        tangent=np.array([1.,-2*height*x,0.]); tangent/=np.linalg.norm(tangent)
        second=np.array([0.,0.,1.]); frames.append(np.column_stack((tangent,second,np.cross(tangent,second))))
    for eid in range(1,macros+1):
        nodes=(2*eid-1,2*eid,2*eid+1)
        ref=Reference(np.array([made.mesh.nodes[i].coords() for i in nodes]),np.array([frames[i-1] for i in nodes]))
        element=NativeRetainedFibreElement(eid,nodes,ref,section,order=4)
        made.add_element(eid,element); made.materials[element.material_name]=section
    for name,node in [('root',1)]+([('tip',2*macros+1)] if clamped else []):
        made.add_boundary_condition(BoundaryCondition(name,[node],{k:0. for k in ('ux','uy','uz','rx','ry','rz')}))
    targets=(.2,.5,1.,.4,0.,-.4,0.) if plastic else (.2,.6,1.,.4,0.)
    force=(.32,.018,-.012) if plastic else ((.1,-.3,.12) if clamped else (.12,-.08,.04))
    return made,ForceProgram(targets,(),max_iterations=24),ReferenceLineForces(tuple((i,*force) for i in range(1,macros+1)))


def resultant_force(model,forces):
    """Independent Q2 derivative and declared 4-point-per-half load measure."""
    points,weights=np.polynomial.legendre.leggauss(4); total=np.zeros(3)
    for eid,*force in forces.rows:
        xyz=np.array([model.mesh.nodes[i].coords() for i in model.mesh.elements[eid].node_ids])
        a=(xyz[2]-xyz[0])/2; b=xyz[0]-2*xyz[1]+xyz[2]; length=0.
        for cell in (0,1):
            for point,weight in zip(points,weights):
                xi=cell-1+(point+1)/2
                length+=weight*np.linalg.norm(a+xi*b)/2
        total+=length*np.array(force)
    return total


@pytest.fixture(scope='module',params=CASES)
def paths(request,tmp_path_factory):
    root=tmp_path_factory.mktemp(request.param); results=[]; events=[]
    for module,label in ((dense,'dense'),(native,'equilibrium')):
        made,p,f=inputs(request.param)
        result=module.solve_force_program(made,p,line_forces=f,progress=events.append if label=='equilibrium' else None)
        save(root/(label+'.json'),result.checkpoint)
        save(root/(label+'-status.json'),dict(status=result.status,failure=result.failure,completed_targets=result.completed_targets))
        if label=='equilibrium': save(root/'progress.json',events)
        assert result.status=='completed',result.failure
        results.append(result)
    return request.param,root,results,events


def test_complete_connected_path_and_support_reactions(paths):
    case,root,(old,new),events=paths
    a=json.loads(old.checkpoint); b=json.loads(new.checkpoint); differences=[]; balances=[]
    assert len(a['records'])==len(b['records'])==new.completed_targets
    made,p,f=inputs(case); context=native.Context(made,p,line_forces=f)
    state,records=context.restore(new.checkpoint); load=resultant_force(made,f)
    for x,y in zip(a['records'],b['records']):
        assert x['target']==y['target'] and x['parameter']==y['parameter']
        differences.append({k:numerical_difference(x[k],y[k]) for k in ('mechanical','origins','histories','residual')})
        assert max(x['metrics']+y['metrics'])<=1e-11
        # Full residual at constrained translations is the support reaction;
        # free translations vanish at accepted equilibrium. Do not count cell
        # rotations or numerical force coordinates as physical external force.
        residual=np.array(y['residual'])[:context.layout.nodal_count].reshape(-1,6)
        net=residual[:,:3].sum(axis=0)+y['parameter']*load
        balance=float(np.linalg.norm(net)/max(1.,np.linalg.norm(y['parameter']*load)))
        assert balance<=1e-11; balances.append(balance)
    made,p,f=inputs(case); dc=dense.Context(made,p,line_forces=f); old_state,_=dc.restore(old.checkpoint)
    recovery=numerical_difference(context.recover(state),dc.recover(old_state))
    factors=[e['linear_solver'] for e in events if e['stage']=='fibre-equilibrium-line.factorized']
    assert factors and all(not v['full_mixed_fallback'] for v in factors)
    expected=6 if case=='curved-2-clamped' else 0
    assert all(v['compatibility_completion_rows']==expected for v in factors)
    assert context.checkpoint(records)==new.checkpoint
    if case=='straight-2-plastic':
        assert any(row[2]>0 for cell in state.histories for station in cell.stations for row in station.rows)
    save(root/'comparison.json',dict(case=case,records=differences,recovery_error=recovery,
        force_balance=balances,newton_factors=len(factors),compatibility_completion_rows=expected,
        dense_sha256=sha256(old.checkpoint).hexdigest(),equilibrium_sha256=sha256(new.checkpoint).hexdigest(),
        production_qualified=False))


def test_connected_pause_resume_and_support_binding(paths):
    case,root,(_,whole),_=paths
    made,p,f=inputs(case); paused=native.solve_force_program(made,p,line_forces=f,stop_after=2)
    save(root/'paused.json',paused.checkpoint); assert paused.status=='paused',paused.failure
    made,p,f=inputs(case)
    resumed=native.solve_force_program(made,p,line_forces=f,checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    save(root/'resumed.json',resumed.checkpoint)
    assert resumed.status=='completed' and resumed.checkpoint==whole.checkpoint,resumed.failure
    made,p,f=inputs(case)
    made.add_boundary_condition(BoundaryCondition('new-fix',[2],{'ux':0.}))
    with pytest.raises(ValueError): native.Context(made,p,line_forces=f).restore(whole.checkpoint)
