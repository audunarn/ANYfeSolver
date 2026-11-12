"""Deterministic light public-route exploration; not a qualification suite.

Run --manifest first, then one --case per externally budgeted process. Expectations
are saved before construction/solve. No automatic repairs or parameter tuning.
"""
from __future__ import annotations
import argparse,json,pathlib,time,traceback,hashlib,os,platform
import numpy as np
import anyfesolver as fs
from anyfesolver.elements import create_shell_element,BeamElement
from anyfesolver.boundary import BoundaryCondition,FixedSupport,LoadCase
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.matrix_assembly import assemble_load_vector
from threadpoolctl import threadpool_info


def cases():
    rows=[]
    def add(name,kind,**kw):rows.append(dict(name=name,kind=kind,**kw))
    for dof in ['uy','ux']:
        for sign in [1,-1]:add(f'plate-{dof}-{sign}','plate',dof=dof,target=sign*1e-6,linear=True)
    for name,kw in [('moderate',{'target':.001}),('large',{'target':.01}),('aspect-short',{'aspect':.1}),('aspect-long',{'aspect':10.}),('thin',{'thickness':.001}),('thick',{'thickness':.1}),('soft',{'E':1e6}),('stiff',{'E':1e12}),('nu-zero',{'nu':0.}),('nu-nearhalf',{'nu':.499}),('origin',{'origin':1e6}),('small-scale',{'scale':.01}),('reverse-order',{'reverse':True}),('mesh2',{'divisions':2})]:add('plate-'+name,'plate',**kw)
    for pressure in [1e-3,-1e-3,100.,-100.]:add('pressure-'+str(pressure),'plate',pressure=pressure)
    for axis in [0,2,3]:
        for sign in [1,-1]:add(f'beam-{axis}-{sign}','beam',axis=axis,load=sign*10.)
    for method in ['none','anderson','contact_newton']:
        for stiffness in [500.,400000.]:add(f'impact-{method}-{int(stiffness)}','impact',method=method,stiffness=stiffness)
    for method in ['anderson','contact_newton']:
        add('impact-damped-'+method,'impact',method=method,stiffness=400000.,damping=10.)
        add('impact-deforming-'+method,'impact',method=method,stiffness=40000.,deform=True)
    add('impact-open','impact',method='contact_newton',height=1.)
    for name in ['plate-nan-target','plate-negative-E','plate-half-nu','impact-nan-radius','impact-inf-speed','impact-nan-penalty','impact-zero-mass','unsupported-stability']:
        add(name,'invalid',expected_refusal=True)
    for row in rows:
        row['expectations']={'finite_success':True,'prescribed_error_atol':1e-12,'original_equilibrium_relative_limit':1e-6,'linear_comparison_relative_limit':1e-4 if row.get('linear') or row['kind']=='beam' else None,'invalid_must_not_report_completed':row.get('expected_refusal',False),'impact_oracle':'same-step two-mass only when rigid centered and completed; atol u2e-8,v3e-6,F3e-4' if row['kind']=='impact' and not row.get('deform') else None}
    return rows


def plate_model(c):
    m=fs.FEModel(c['name']);m.add_material('mat',c.get('E',210e9),c.get('nu',.3));n=c.get('divisions',1);scale=c.get('scale',1.);aspect=c.get('aspect',1.);origin=c.get('origin',0.)
    nodes=[(1+j*(n+1)+i,origin+scale*aspect*i/n,origin+scale*j/n,origin) for j in range(n+1) for i in range(n+1)]
    for node in nodes[::-1] if c.get('reverse') else nodes:m.add_node(*node)
    for j in range(n):
        for i in range(n):
            v=1+j*(n+1)+i;eid=1+j*n+i;m.add_element(eid,create_shell_element(eid,[v,v+1,v+n+2,v+n+1],'mat',thickness=c.get('thickness',.01)*scale))
    m.add_boundary_condition(FixedSupport('left',[1+j*(n+1) for j in range(n+1)]));load=None
    if 'pressure' in c:
        load=LoadCase('pressure')
        for eid in m.mesh.elements:load.add_pressure_load(eid,c['pressure'])
    else:m.add_boundary_condition(BoundaryCondition('driven',[(j+1)*(n+1) for j in range(n+1)],{c.get('dof','uy'):c.get('target',1e-6)*scale}))
    m.apply_boundary_conditions();return m,load


def beam_model(c):
    m=fs.FEModel(c['name']);m.add_material('mat',210e9,.3);m.add_node(1,0.,0.,0.);m.add_node(2,1.,0.,0.)
    m.add_element(1,BeamElement(1,[1,2],'mat',dict(area=.01,Iy=1e-6,Iz=1e-6,J=1e-6,orientation=(0.,0.,1.))))
    m.add_boundary_condition(FixedSupport('left',[1]));load=LoadCase('tip');v=np.zeros(6);v[c['axis']]=c['load'];load.add_nodal_load(2,v);m.apply_boundary_conditions();return m,load


def static(c):
    build_start=time.perf_counter()
    m,load=beam_model(c) if c['kind']=='beam' else plate_model(c);kin='von_karman' if c['kind']=='beam' else 'corotational';tangent='auto' if c['kind']=='beam' else 'work_conjugate'
    build_wall=time.perf_counter()-build_start
    linear_start=time.perf_counter();linear,_=fs.solve_linear(m,load);linear_wall=time.perf_counter()-linear_start
    messages=[]
    solve_start=time.perf_counter()
    r=fs.solve_static_nonlinear(m,load,num_steps=2,max_iterations=20,min_step_fraction=.0625,kinematics=kin,corotational_tangent=tangent,record_increment_snapshots=True,stability_control='plate_energy' if c.get('unsupported_stability') else None,status_callback=messages.append if c.get('trace') else None)
    solve_wall=time.perf_counter()-solve_start
    violations=[];u=np.asarray(r.displacements);prescribed={};names=['ux','uy','uz','rx','ry','rz']
    for bc in m.boundary_conditions:
        for node in bc.node_ids:
            for name,value in bc.dof_constraints.items():prescribed[m.mesh.nodes[node].dofs[names.index(name)]]=value*r.load_factor
    free=np.setdiff1d(np.arange(len(u)),list(prescribed));bcerror=max((abs(u[d]-v) for d,v in prescribed.items()),default=0.)
    data={'build_wall':build_wall,'linear_wall':linear_wall,'solve_wall':solve_wall,'status':r.status,'converged':r.converged,'factor':r.load_factor,'iterations':r.info.get('total_newton_iterations'),'accepted_steps':len(r.steps),'shell_rotation_stable':r.info.get('corotational_shell_rotation_stable',False),'snapshots':[{'factor':v.load_factor,'u':v.displacements.tolist()} for v in r.snapshots],'work_counters':r.info.get('nonlinear_performance'),'thread_policy':r.info.get('thread_policy'),'resolved_convergence':r.info.get('convergence_settings'),'adaptation':r.info.get('convergence_adaptation'),'u':u.tolist(),'node_displacements':{str(n):u[node.dofs].tolist() for n,node in m.mesh.nodes.items()},'boundary_error':bcerror,'finite':bool(np.all(np.isfinite(u))),'linear_relative_difference':float(np.linalg.norm(u-r.load_factor*linear)/max(np.linalg.norm(r.load_factor*linear),1e-30)),'reactions':r.steps[-1].support_reactions if r.steps else None}
    if r.converged:
        force,_,_=_assemble_nonlinear_system(m,u,{},5,tangent=False,require_full_coordinates=True,kinematics=kin,corotational_tangent=tangent,stable_shell_rotation=r.info.get('corotational_shell_rotation_stable',False))
        external,_=assemble_load_vector(m,load);residual=force-r.load_factor*external;reference=max(np.linalg.norm(external[free]),1.)
        data['free_residual_relative']=float(np.linalg.norm(residual[free])/reference)
        reactions=residual.copy();reactions[free]=0.;balance=(reactions+r.load_factor*external).reshape(-1,6)[:,:3].sum(axis=0);data['force_balance']=balance.tolist()
        if not data['finite']:violations.append('nonfinite successful field')
        if bcerror>1e-12:violations.append('prescribed displacement mismatch')
        if data['free_residual_relative']>1.00001e-6:violations.append('free equilibrium exceeds original tolerance')
        if c['expectations']['linear_comparison_relative_limit'] and data['linear_relative_difference']>1e-4:violations.append('tiny response differs from independent linear solve')
    if c.get('trace'):data.update(full_info=r.info,status_messages=messages)
    data['violations']=violations;return data


def impact(c):
    from anyfesolver.contact import _verification_contact_panel
    build_start=time.perf_counter();m=_verification_contact_panel()
    if c.get('deform'):m.add_boundary_condition(BoundaryCondition('pin',[1],{'uz':0.}))
    k=c.get('stiffness',40000.);damping=c.get('damping',0.);height=c.get('height',.205)
    build_wall=time.perf_counter()-build_start;solve_start=time.perf_counter()
    r=fs.solve_transient_sphere_impact(m,fs.TransientConfig(dt=.01,t_end=.03),fs.RigidSphereImpact('ball',radius=c.get('radius',.2),mass=c.get('mass',1.),start_point=(.5,.5,height),travel_direction=(0.,0.,-1.),speed=c.get('speed',1.)),fs.SphereContactConfig(penalty_stiffness=k,contact_damping=damping),nonlinear_config=fs.NonlinearTransientConfig(enabled=True,max_iterations=25,max_cutbacks=0,coupled_contact_acceleration=c.get('method','anderson')))
    data={'build_wall':build_wall,'solve_wall':time.perf_counter()-solve_start,'status':r.status,'finite':all(np.all(np.isfinite(v)) for v in [r.displacements,r.sphere_positions,r.sphere_velocities,r.contact_force_history]),'u':r.displacements.tolist(),'x':r.sphere_positions.tolist(),'v':r.sphere_velocities.tolist(),'f':r.contact_force_history.tolist(),'times':r.times.tolist(),'diagnostics':r.diagnostics,'violations':[]}
    if r.status in ('completed','no_contact'):
        if not data['finite']:data['violations'].append('nonfinite successful field')
        if not c.get('deform') and not c.get('expected_refusal'):
            x=np.array([height,0.]);v=np.array([-1.,0.]);a=np.zeros(2);positions=[x.copy()];velocities=[v.copy()];forces=[0.]
            for _ in range(3):
                xp=x+.01*v+.000025*a;vp=v+.005*a
                f=max((k*(.2-xp[0]+xp[1])-damping*(vp[0]-vp[1]))/(1+2*(k*.000025+damping*.005)),0.)
                if .2-xp[0]+xp[1]-.00005*f<=0:f=0.
                a=np.array([f,-f]);x=xp+.000025*a;v=vp+.005*a;positions.append(x.copy());velocities.append(v.copy());forces.append(f)
            errors={'x':float(np.max(abs(r.sphere_positions[:,2]-np.array(positions)[:,0]))),'v':float(np.max(abs(r.sphere_velocities[:,2]-np.array(velocities)[:,0]))),'target':float(np.max(abs(r.displacements[:,2::6]-np.array(positions)[:,1,None]))),'f':float(np.max(abs(r.contact_force_history[:,2]-forces)))};data['oracle_errors']=errors
            if errors['x']>2e-8 or errors['target']>2e-8 or errors['v']>3e-6 or errors['f']>3e-4:data['violations'].append('same-step rigid oracle mismatch')
    return data


def run(c,output):
    result={'input':c,'phase':'before construction','python':platform.python_version(),'numpy':np.__version__,'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},'script_hash':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    def save():output.write_text(json.dumps(result,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf8')
    save();start=time.perf_counter()
    try:
        effective=dict(c)
        if c['kind']=='invalid':
            name=c['name'];effective['kind']='impact' if name.startswith('impact') else 'plate'
            if name=='plate-nan-target':effective['target']=float('nan')
            if name=='plate-negative-E':effective['E']=-1.
            if name=='plate-half-nu':effective['nu']=.5
            if name=='impact-nan-radius':effective['radius']=float('nan')
            if name=='impact-inf-speed':effective['speed']=float('inf')
            if name=='impact-nan-penalty':effective['stiffness']=float('nan')
            if name=='impact-zero-mass':effective['mass']=0.
            if name=='unsupported-stability':effective.update(unsupported_stability=True,pressure=1.)
        result['effective_input']=effective;result['phase']='solving';save()
        data=impact(effective) if effective['kind']=='impact' else static(effective);result.update(data)
        complete=data.get('converged',data['status'] in ('completed','no_contact'))
        result['classification']='silent_bad_success' if complete and (data['violations'] or c.get('expected_refusal')) else ('completed_checked' if complete else ('invalid_failed_late' if c.get('expected_refusal') else 'ordinary_nonconvergence'))
    except Exception as exc:
        result.update(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc(),classification='invalid_exception_requires_triage' if c.get('expected_refusal') else 'exception_requires_triage')
    result.update(phase='finished',case_wall=time.perf_counter()-start,threadpools=threadpool_info());save();print(json.dumps({k:result.get(k) for k in ['classification','status','case_wall','exception','violations']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=pathlib.Path);p.add_argument('--case');p.add_argument('--output',type=pathlib.Path);p.add_argument('--trace',action='store_true');a=p.parse_args()
    if a.manifest:a.manifest.write_text(json.dumps(cases(),indent=2),encoding='utf8')
    else:run(dict(next(c for c in cases() if c['name']==a.case),trace=a.trace),a.output)
