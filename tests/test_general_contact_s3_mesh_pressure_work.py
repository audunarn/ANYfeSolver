"""Independent pressure ramp and authenticated traction-inverted anchor diagnosis."""
import os,json,copy,hashlib
from pathlib import Path
from fractions import Fraction as F
from dataclasses import asdict
import numpy as np
import pytest
from examples.general_contact_s3_mesh_history_events import build_case
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.s3_friction_static import create_s3_friction_manager
from anyfesolver.general_contact.material_work_diagnostics import sliding_pressure_ramp_reference,reconstruct_material_history,diagnose_work_bundle,SCHEMA

OLD=Path('C:/ANYworkspaceWork/contact_s3_mesh_nontrivial_work_20261009')
CUTS={2:(2,4,6,8,10,12),4:(4,8,10,12,16,20,24)}
def root():return Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])
def save(value):
    root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if not root:return
    Path(root).write_text(json.dumps(value,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else v.item()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def test_ramp():
    results=[]
    for p0,p1 in ((1.,3.),(3.,1.)):
        for r in (2,4):
            m,d,s,f,p,_=build_case(r,'sliding');owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
            A=.01;mu=.3;k=1000.;S=.01;supports=np.array([A/6,A/3,A/6,A/3]);u=np.zeros(60)
            for node in (2,3,4,5):u[m.mesh.dof_manager.get_node_dofs(node)[2]]=-2e-7
            owner.commit(owner.validate_successor(u,p0*supports))
            seed=mu*p0/k
            for node in range(1,6):u[m.mesh.dof_manager.get_node_dofs(node)[0]]=seed
            prepared=owner.validate_successor(u,p0*supports);owner.commit(prepared);initial_storage=prepared[2]['storage_j']
            D=work=0.;rows=[]
            for j in range(1,r+1):
                pressure=p0+(p1-p0)*j/r
                for node in range(1,6):u[m.mesh.dof_manager.get_node_dofs(node)[0]]=seed+S*j/r
                prepared=owner.validate_successor(u,pressure*supports);report=prepared[2]
                assert all(b=='slip' for b in report['branch'])
                D+=report['dissipation_increment_j'];work-=report['material_relative_endpoint_contraction_j']
                rows.append(report);owner.commit(prepared)
            # Direct Fraction integration of (c0+dc*t)*(S-dc/k), independent of helper.
            a,muf,ks,sm=map(lambda x:F(float(x)),(A,mu,k,S));c0=muf*F(p0);dc=muf*(F(p1)-F(p0));gamma=sm-dc/ks
            exact_D=float(a*(c0*gamma+dc*gamma/F(2)));exact_work=float(a*(c0*sm+dc*sm/F(2)))
            endpoint_D=float(sum(a*(c0+dc*F(j,r))*gamma/F(r) for j in range(1,r+1)))
            endpoint_work=float(sum(a*(c0+dc*F(j,r))*sm/F(r) for j in range(1,r+1)))
            reference=sliding_pressure_ramp_reference(area_m2=A,mu=mu,stiffness_pa_per_m=k,motion_m=S,pressure_start_pa=p0,pressure_end_pa=p1,stages=r)
            assert abs(reference['exact_plastic_loss_j']-exact_D)<=1e-12
            assert abs(D-endpoint_D)<=1e-12 and abs(work-endpoint_work)<=1e-12
            assert abs(reference['endpoint_plastic_loss_j']-endpoint_D)<=1e-12
            assert abs(rows[-1]['storage_j']-initial_storage-reference['storage_change_j'])<=1e-12
            assert abs(D-endpoint_D)/exact_D<=.001
            results.append(dict(p0_pa=p0,p1_pa=p1,ratio=r,measured_D_j=D,exact_D_j=exact_D,measured_endpoint_work_j=work,exact_work_j=exact_work,reference=reference,rows=rows))
    assert results[0]['reference']['endpoint_bias_j']>0 and results[2]['reference']['endpoint_bias_j']<0
    assert abs(results[0]['reference']['endpoint_bias_j']-2*results[1]['reference']['endpoint_bias_j'])<=1e-12
    assert abs(results[2]['reference']['endpoint_bias_j']-2*results[3]['reference']['endpoint_bias_j'])<=1e-12
    save(dict(ramps=results,scope='independent_fixednormal_fullslip_reference; prescribed_local_pressure_not_coupled_equilibrium'))


@pytest.mark.skipif(not (OLD/'CLOSURE.json').is_file() or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='archived pressure-work inputs contact_s3_mesh_nontrivial_work_20261009 CLOSURE.json/outputs and AUTOMATIC_WORKFLOW_EVIDENCE absent: authenticated saved-output replay requires the original evidence root')
def test_prepare():
    closure=json.loads((OLD/'CLOSURE.json').read_text());bundle={};provenance=[]
    for ratio,key in ((2,'coarse'),(4,'fine')):
        m,d,s,f,p,_=build_case(ratio,'sliding');owner=create_s3_friction_manager(m,GeneralS3PatchManager(m,d,s),f)
        features=[]
        for j,(point,weight) in enumerate(zip(owner.points,owner.weights)):
            c=point.adapter.reference_positions;expected=.5*np.linalg.norm(np.cross(c[1]-c[0],c[2]-c[0]))/3
            assert abs(weight-expected)<1e-15
            features.append(dict(id=f'S3-{point.adapter.element_id}-point-{j}',source_element_id=point.adapter.element_id,source_node_ids=list(point.adapter.node_ids),barycentric=list(point.barycentric),reference_area_m2=float(weight)))
        ids=sorted({n for a in owner.target_adapters.values() for n in a.node_ids});faces={str(e):list(a.node_ids) for e,a in owner.target_adapters.items()}
        friction_contract=asdict(f);friction_contract.pop('physical_subincrements_per_user_increment')
        context=dict(model_name=m.name,reference_nodes={str(n):[q.x,q.y,q.z] for n,q in m.mesh.nodes.items()},
            element_topology=[[1,2,5],[2,3,5],[3,4,5],[6,7,10],[7,8,10],[8,9,10]],
            shell_contract=dict(formulation='e4-pl-s3-v2d',thickness_m=.05,source_offset_m=0.,target_offset_m=.01,director=[0.,0.,1.],E_pa=1e5,nu=.3),
            static_boundary_conditions=[asdict(x) for x in m.boundary_conditions],constraints=[asdict(x) for x in m.constraint_equations],
            definition=asdict(d),normal_settings=asdict(s),friction_settings=friction_contract,user_program=asdict(p),solver=dict(tolerance=1e-12,max_iterations=18))
        samples=[]
        for cut in CUTS[ratio]:
            path=OLD/'outputs'/f'R{ratio}-{cut}.json';assert sha(path)==closure['output_sha256'][path.name]
            saved=json.loads(path.read_text());provenance.append(dict(path=str(path),sha256=sha(path)))
            history=saved['checkpoint']['path_state']['history']
            for snap in saved['snapshots']:
                step=snap['step_index'];u=np.array(snap['displacement']);report=history[step-1]['friction']
                def skin(node,h):
                    q=m.mesh.get_node(node);dd=m.mesh.dof_manager.get_node_dofs(node)
                    return np.array([q.x,q.y,q.z])+u[dd[:3]]+h*(np.array([0.,0.,1.])+np.cross(u[dd[3:]],[0.,0.,1.]))
                source=[np.asarray(feature['barycentric'])@np.array([skin(n,-.025) for n in feature['source_node_ids']]) for feature in features]
                sample=dict(step_index=step,user_position=step/ratio,source_points_m=source,target_nodes_m=[skin(n,.015) for n in ids],
                    pressure_pa=report['pressure_pa'],traction_pa=report['traction_pa'],branch=report['branch'],target_elements=report['target_elements'],published_loss_j=report['dissipation_increment_j'])
                if step==cut:sample['checkpoint_anchors']=saved['checkpoint']['path_state']['friction_state']['history']
                samples.append(sample)
        assert [x['step_index'] for x in samples]==list(range(1,6*ratio+1))
        bundle[key]=dict(schema=SCHEMA,stiffness_pa_per_m=f.tangential_stiffness_Pa_per_m,mu=f.friction_coefficient,coordinate_tolerance_m=1e-10,
            features=features,target_node_ids=ids,target_faces=faces,samples=samples,declared_context=context)
    path=root()/'diagnostic-input.json';path.write_text(json.dumps(bundle,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else v.item()))
    save(dict(input_sha256=sha(path),producer_closure_sha256=sha(OLD/'CLOSURE.json'),provenance=provenance,scope='independent_literal_skin_reconstruction_from_authenticated_saved_u; prepared_material_locations_are_declared_inputs'))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'diagnostic-input.json').is_file(),
    reason='archived diagnostic input AUTOMATIC_WORKFLOW_EVIDENCE/diagnostic-input.json absent: prepared-bundle diagnosis requires the produced diagnostic input')
def test_diagnosis():
    bundle=json.loads((root()/'diagnostic-input.json').read_text());before=copy.deepcopy(bundle)
    output=diagnose_work_bundle(bundle);assert bundle==before and output==diagnose_work_bundle(bundle)
    for result in (output['coarse'],output['fine']):
        assert max(abs(row['loss_error_j']) for row in result['samples'])<=1e-12
        assert result['checkpoint_anchor_max_error_m']<=1e-10
        assert any(any(row['facet_transfer']) for row in result['samples'])
    assert max(abs(row['split_error_j']) for row in output['comparison']['intervals'])<=1e-12
    save(output)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'diagnostic-input.json').is_file(),
    reason='archived diagnostic input AUTOMATIC_WORKFLOW_EVIDENCE/diagnostic-input.json absent: consumer replay requires the produced diagnostic input')
def test_consumer(monkeypatch):
    from examples import general_contact_s3_mesh_history_events as example
    import sys
    def forbidden(*a,**kw):raise AssertionError('diagnostic attempted solver or model allocation')
    monkeypatch.setattr(example,'solve_static_nonlinear',forbidden);monkeypatch.setattr(example,'build_case',forbidden)
    output=root()/'consumer-output.json';input_path=root()/'diagnostic-input.json';input_before=input_path.read_bytes()
    monkeypatch.setattr(sys,'argv',['example','--diagnose-material-work',str(input_path),'--output',str(output)])
    example.main();actual=json.loads(output.read_text());expected=diagnose_work_bundle(json.loads(input_before))
    assert actual==expected and input_path.read_bytes()==input_before
    for option,value in (('--through-step','1'),('--ratio','2'),('--workflow','pressure_event'),('--restart','absent.json')):
        monkeypatch.setattr(sys,'argv',['example','--diagnose-material-work',str(input_path),option,value,'--output',str(output)])
        with pytest.raises(ValueError,match='solver controls'):example.main()
    from types import SimpleNamespace
    forwarded=[]
    def case_stub(ratio,workflow):
        forwarded.append((ratio,workflow));return (ratio,workflow)
    def solve_stub(case,through,incoming):
        assert case==forwarded[-1] and through is None and incoming is None
        return SimpleNamespace(status='completed',info={'stubbed_normal_caller':True},to_restart_checkpoint=lambda:None)
    monkeypatch.setattr(example,'build_case',case_stub);monkeypatch.setattr(example,'solve_case',solve_stub)
    for options,expected_pair in (([],(2,'pressure_event')),(['--ratio','4','--workflow','sliding'],(4,'sliding')),(['--ratio','0'],(0,'pressure_event'))):
        monkeypatch.setattr(sys,'argv',['example','--output',str(output),*options]);example.main();assert forwarded[-1]==expected_pair
    save(dict(normal_caller_defaults_and_explicit_arguments_preserved=True,stubbed_forwarding_not_solver_execution=True,actual_example_file_consumer=True,no_solver_or_model_allocation=True,input_unchanged=True,unused_solver_controls_refused=True,comparison=actual['comparison']))


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not (Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE'])/'diagnostic-input.json').is_file(),
    reason='archived diagnostic input AUTOMATIC_WORKFLOW_EVIDENCE/diagnostic-input.json absent: refusal replay requires the produced diagnostic input')
def test_refusal():
    bundle=json.loads((root()/'diagnostic-input.json').read_text());before=copy.deepcopy(bundle)
    bad=[]
    x=copy.deepcopy(bundle);x['coarse']['features'][0]['reference_area_m2']=0.;bad.append(x)
    x=copy.deepcopy(bundle);x['coarse']['samples'][0]['target_elements'][0]=999;bad.append(x)
    x=copy.deepcopy(bundle);x['coarse']['samples'][0]['traction_pa'][0][2]=1.;bad.append(x)
    x=copy.deepcopy(bundle);x['fine']['features'][0]['source_node_ids'].reverse();bad.append(x)
    x=copy.deepcopy(bundle);x['fine']['samples'][0]['step_index']=2;bad.append(x)
    x=copy.deepcopy(bundle);x['fine']['stiffness_pa_per_m']*=2;bad.append(x)
    x=copy.deepcopy(bundle);x['fine']['declared_context']['solver']['max_iterations']=19;bad.append(x)
    x=copy.deepcopy(bundle);x['fine']['target_node_ids'].reverse();bad.append(x)
    for value in (float('nan'),-1.):
        x=copy.deepcopy(bundle);x['coarse']['samples'][0]['published_loss_j']=value;bad.append(x)
    x=copy.deepcopy(bundle);next(row for row in x['coarse']['samples'] if 'checkpoint_anchors' in row)['checkpoint_anchors'][0]['barycentric']=[float('nan'),0.,1.];bad.append(x)
    for changed in bad:
        original=copy.deepcopy(changed)
        with pytest.raises(ValueError):diagnose_work_bundle(changed)
        assert json.dumps(changed,sort_keys=True)==json.dumps(original,sort_keys=True)
    assert bundle==before
    with pytest.raises(ValueError):sliding_pressure_ramp_reference(area_m2=.01,mu=.3,stiffness_pa_per_m=1000,motion_m=1e-5,pressure_start_pa=1,pressure_end_pa=3,stages=2)
    save(dict(refused_data_cases=len(bad),unsupported_ramp_refused=True,no_input_mutation=True))
