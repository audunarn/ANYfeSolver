"""Prospective consolidated F13 cases. Import is inert; execution requires freeze."""
from __future__ import annotations
import hashlib,json,sys,weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/agent_plans/development/r8-f13-static-feasibility-20261009'
SITE=ROOT/'docs/agent_plans/development/r8-b2-work-guard-installed-20261009/site'
WHEEL_SHA='89cfba60de1d967507636eb46e711af52d825a51ce0c1286f22133bac357ded4'
GROUPS=('batch-1','batch-10','batch-100','ownership','cancellation','selected-recovery','refusal','failure-closure','cache')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

def validate_registration(registration):
    """Fail before any fixture construction; no legacy execution authority."""
    r=registration
    if r['status']!='FROZEN_BY_ASTRA' or r['schema']!='F13_STATIC_REGISTRATION_V1':
        raise ValueError('prospective F13 registration is not adopted/frozen')
    for rel,digest in r['candidate_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('frozen candidate changed:'+rel)
    if r['wheel_sha256']!=WHEEL_SHA or sha(SITE.parent/'dist/anysolver-0.4.7-py3-none-any.whl')!=WHEEL_SHA:
        raise ValueError('frozen installed wheel changed')
    if sha(r['python'])!=r['python_executable_sha256']:raise ValueError('frozen Python executable changed')
    childcap=40 if r.get('version')=='v3' else 30
    if r['resources']!=dict(budget_s=120,reserve_s=60,childcap_s=childcap,affinity=[0],numerical_threads=1,jit_disabled=True,rss_bytes=2147483648,jobs=1):
        raise ValueError('resource authority differs')
    if r.get('version')=='v3' and r['stage_caps_s']!=dict(source=30,installed=40):raise ValueError('v3 stage allocation differs')
    for rel,digest in r['installed_module_sha256'].items():
        if sha(SITE/rel)!=digest:raise ValueError('installed dependency changed:'+rel)
    for rel,record in r['direct_source_site_bindings'].items():
        source=ROOT/'src'/rel;site=SITE/rel
        if sha(source)!=record['source_sha256'] or sha(site)!=record['site_sha256']:
            raise ValueError('direct source/site binding changed:'+rel)
        if record['source_sha256']!=record['site_sha256']:
            if rel not in r['source_representation_exceptions']:
                raise ValueError('unadopted source/site byte mismatch:'+rel)
            if source.read_text()!=site.read_text():raise ValueError('source/site difference exceeds newline representation:'+rel)
    for name,record in r['tooling'].items():
        if sha(record['metadata_path'])!=record['metadata_sha256']:raise ValueError('pinned distribution metadata changed:'+name)
    if r['environment'].get('PYTHONOPTIMIZE')!='0':raise ValueError('assertion optimization must be explicitly disabled')
    expected=[f'{domain}/{group}' for domain in ('point','mixed','physical-footprint') for group in GROUPS]
    if r['case_inventory']!=expected:raise ValueError('case inventory incomplete or changed')
    return json.loads((DOC/'inputs-v1.json').read_text())

def runtime(registration):
    inputs=validate_registration(registration)
    sys.path.insert(0,str(SITE))
    import numpy as np
    from scipy import sparse
    from anyfesolver import stiffened_shell as api
    import anyfesolver._stiffened_shell_b2_q4_panel_experimental as module
    from anyfesolver.control import CancellationToken
    if not Path(api.__file__).resolve().is_relative_to(SITE):raise RuntimeError('source/site mixing')
    return SimpleNamespace(inputs=inputs,np=np,sparse=sparse,api=api,module=module,Token=CancellationToken)

def definition(ctx,domain,variant=None):
    """Construct solely the prospectively registered literal54-coordinate model."""
    np,sp,api,d=ctx.np,ctx.sparse,ctx.api,ctx.inputs
    nodes={n:np.array(x,float) for n,x in d['shell_nodes']}
    if variant=='geometry':
        for n in d['dependency_variants']['geometry']['nodes']:nodes[n][0]+=d['dependency_variants']['geometry']['dx_m']
    params=d['section'].copy()
    if variant=='section':params.update(d['dependency_variants']['section'])
    section=api.LinearSection.isotropic(**params)
    positions=d['beam_positions_footprint' if domain=='physical-footprint' else 'beam_positions_point']
    members=tuple(api.PhysicalB2Member('m'+str(i),'f13-source-'+str(i),tuple(d['beam_node_ids'][i:i+2]),
        tuple(tuple(x) for x in positions[i:i+2]),tuple(d['beam_local_y']),'f13-section',section) for i in range(2))
    cells=d['mixed_cells'] if domain=='mixed' else d['q4_cells'];s=d['shell']
    shells=tuple((api.Q4PanelShell if len(ids)==4 else api.S3V2DPanelShell)(
        's'+str(i),tuple(ids),s['E'],s['nu'],s['thickness_m'],tuple(s['normal'])) for i,ids in enumerate(cells))
    fixed=d['footprint_fixed_dofs' if domain=='physical-footprint' else 'point_fixed_dofs']
    free=[i for i in range(54) if i not in fixed]
    if variant=='support':free.remove(49)
    T=sp.eye(54,format='lil')[:,free]
    if variant=='support':T[49,free.index(25)]=.25
    T=T.tocsr();offset=np.zeros(54)
    if variant=='affine-offset':offset[36]=d['dependency_variants']['affine-offset']['value_m']
    if domain=='physical-footprint':
        f=d['physical_footprint'];pad=api.B2PhysicalRigidFootprint(f['id'],tuple(f['nodes']),f['beam_node'],
            tuple(f['director']),tuple(tuple(x) for x in f['tangent_basis']),tuple(tuple(x) for x in f['cells']))
        if variant=='attachment':pad=replace(pad,beam_node=21)
        result=api.B2PhysicalRigidFootprintDefinition(api.B2_E4_PL_FAST_LINEAR_V1,nodes,shells,members,(pad,),T,offset)
    else:
        points=tuple(api.Q4B2Point(p['id'],p['shell'],p['member'],
            .65 if variant=='attachment' and p['id']=='f13-right' else p['station'],tuple(p['offset_local_m'])) for p in d['points'])
        cls=api.B2MixedPanelDefinition if domain=='mixed' else api.B2Q4PanelDefinition
        result=cls(api.B2_E4_PL_FAST_LINEAR_V1,nodes,shells,members,points,T,offset)
    return result

def load(ctx):
    f=ctx.np.zeros(54)
    for dof,value in ctx.inputs['base_load_nonzero']:f[dof]=value
    return f

def selected(ctx,domain):
    if domain=='physical-footprint':return dict(selected_footprints=ctx.api.B2PhysicalRigidFootprintSelection(('f13-pad',)))
    return dict(selected_points=('f13-right',))

def full_selection(ctx,domain):
    return selected(ctx,domain) if domain=='physical-footprint' else {}

def norm_parity(np,value,expected,L):
    scales=np.tile([L,L,L,1.,1.,1.],len(expected)//6)
    denominator=float(np.linalg.norm(np.asarray(expected)/scales))
    if denominator==0:raise AssertionError('unresolved physical response denominator')
    error=float(np.linalg.norm((np.asarray(value)-expected)/scales))/denominator
    assert error<=1e-10,(error,'bounded response parity')
    return error

def selected_node_action_parity(np,actual,expected,force,length):
    """G0 v3: full ordered48-action norm, without nodal cancellation/floors."""
    a=np.asarray(actual,float);e=np.asarray(expected,float)
    if a.shape!=(48,) or e.shape!=(48,) or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise AssertionError('complete finite ordered48 node-action channels required')
    if not np.isfinite([force,length]).all() or force<=0 or length<=0:
        raise AssertionError('frozen positive finite force/length scales required')
    scales=np.tile([force]*3+[force*length]*3+[force]*3+[force*length]*3,4)
    denominator=float(np.linalg.norm(e/scales))
    if denominator==0 or not np.isfinite(denominator):raise AssertionError('nonzero finite full48 reference action norm required')
    error=float(np.linalg.norm((a-e)/scales)/denominator)
    if not np.isfinite(error) or error>1e-10:raise AssertionError('full48 selected node-action parity failed')
    return error

def physical_checks(ctx,owner,result,f):
    """Independent bounded KKT solve and physical work; no new physical law."""
    np=ctx.np;L=ctx.inputs['characteristic_length_m'];assert result.status=='completed',result.failure_reason
    K=owner._K.toarray();C=owner._C.toarray();T=owner._T.toarray();u0=np.array(owner._u0)
    A=np.block([[T.T@K@T,T.T@C.T],[C@T,np.zeros((len(C),len(C)))]])
    rhs=np.r_[T.T@(f-K@u0),-C@u0];answer=np.linalg.solve(A,rhs)
    expected=u0+T@answer[:T.shape[1]];u=np.array(result.displacements)
    parity=norm_parity(np,u,expected,L)
    lam=np.array(result.constraint_multipliers if hasattr(result,'constraint_multipliers') else
        [v for _,row in result.point_multipliers for v in row])
    reaction=np.array(result.affine_reactions);residual=K@u+C.T@lam-f-reaction
    F=float(np.linalg.norm(np.r_[f.reshape(-1,6)[:,:3].ravel(),f.reshape(-1,6)[:,3:].ravel()/L]))
    assert F>0
    action=np.tile([F,F,F,F*L,F*L,F*L],9)
    equation=float(np.max(np.abs(residual)/action))
    affine=float(np.max(np.abs(T.T@reaction)/(np.abs(T.T)@action)))
    pose=np.tile([L,L,L,1.,1.,1.],9);row_scale=np.abs(C)@pose
    assert np.all(row_scale>0)
    gap=float(np.max(np.abs(C@u)/row_scale));energy=.5*float(u@K@u)
    work=float(u@f+u@reaction);assert energy>0 and work!=0
    work_error=abs(2*energy-work)/abs(work)
    internal_error=abs(float(result.internal_strain_energy)-energy)/energy
    assert equation<=1e-8 and affine<=1e-8 and gap<=1e-9
    assert work_error<=1e-8 and internal_error<=1e-8
    positions=[owner._buckling_model.mesh.nodes[n].coords() for n in owner.shell_node_ids]
    beam_positions={n:p for _,(member,_) in owner._members for n,p in zip(member.definition.node_ids,member.definition.reference_points_m)}
    positions=np.array(positions+[beam_positions[n] for n in owner.beam_node_ids],float)
    actions=(f+reaction).reshape(-1,6)
    total_force=actions[:,:3].sum(axis=0)
    total_moment=(actions[:,3:]+np.cross(positions,actions[:,:3])).sum(axis=0)
    force_balance=float(np.linalg.norm(total_force)/F);moment_balance=float(np.linalg.norm(total_moment)/(F*L))
    assert force_balance<=1e-8 and moment_balance<=1e-8
    return dict(parity=parity,equilibrium=equation,affine_equilibrium=affine,constraint=gap,
        work_error=work_error,independent_operator_energy_error=internal_error,work_J=work,energy_J=energy,
        force_scale_N=F,moment_scale_Nm=F*L,length_scale_m=L,force_balance=force_balance,moment_balance=moment_balance,
        applied_plus_support_force_N=total_force.tolist(),applied_plus_support_moment_about_origin_Nm=total_moment.tolist())

def batch_case(ctx,domain,count):
    np=ctx.np;owner=ctx.api.create_analysis(definition=definition(ctx,domain));reference=None
    try:
        f=load(ctx);reference=ctx.api.create_analysis(definition=definition(ctx,domain))
        one=reference.solve_linear(f,recover_members=ctx.inputs['recovery'],**full_selection(ctx,domain))
        metrics=physical_checks(ctx,reference,one,f)
        cases=tuple((f'f13-{i:03d}',(1+(i%7)/8)*f) for i in range(count))
        result=owner.solve_linear_many(cases,selected_dofs=ctx.inputs['selected_dofs'],recover_members=ctx.inputs['recovery'],**selected(ctx,domain))
        assert result.status=='completed',result.failure_reason
        assert result.factorization_count==1 and result.solve_count==count and result.maximum_rhs_chunk==1
        assert result.selected_dofs==tuple(ctx.inputs['selected_dofs'])
        assert len(result.cases)==count and tuple(c.case_id for c in result.cases)==tuple(c[0] for c in cases)
        node_action_errors=[]
        for i,case in enumerate(result.cases):
            factor=1+(i%7)/8;expected=np.array(one.displacements)[ctx.inputs['selected_dofs']]*factor
            assert not hasattr(case,'displacements') and not hasattr(case,'affine_reactions')
            assert len(case.selected_displacements)==6 and len(case.selected_affine_reactions)==6
            assert len(case.beam_resultants_local)==1 and case.beam_resultants_local[0][0]=='m1'
            denominator=np.linalg.norm(expected);assert denominator>0
            assert np.linalg.norm(np.array(case.selected_displacements)-expected)/denominator<=1e-10
            reactions=np.array(one.affine_reactions)[ctx.inputs['selected_dofs']]*factor
            denominator=np.linalg.norm(reactions);assert denominator>0
            assert np.linalg.norm(np.array(case.selected_affine_reactions)-reactions)/denominator<=1e-10
            for actual,base in zip(case.beam_resultants_local[0][1],one.beam_resultants_local[0][1]):
                denominator=np.linalg.norm(base);assert denominator>0
                assert np.linalg.norm(np.array(actual)-factor*np.array(base))/ (factor*denominator)<=1e-10
            assert abs(case.internal_strain_energy-factor**2*one.internal_strain_energy)/(factor**2*one.internal_strain_energy)<=1e-8
            def action_parity(actual,expected,scales):
                actual=np.array(actual);expected=np.array(expected);scales=np.array(scales)
                denominator=np.linalg.norm(expected/scales);assert denominator>0
                assert np.linalg.norm((actual-expected)/scales)/denominator<=1e-10
            L=metrics['length_scale_m'];F=metrics['force_scale_N']
            row_scale=np.abs(reference._C.toarray())@np.tile([L,L,L,1.,1.,1.],9)
            if domain=='physical-footprint':
                assert result.selected_points==()
                assert result.selected_footprints==('f13-pad',) and result.constraint_rows==one.constraint_rows
                assert len(result.constraint_rows)==20 and len(case.constraint_multipliers)==len(case.constraint_gaps)==20
                assert case.point_multipliers==case.point_gaps==()
                action_parity(case.constraint_multipliers,factor*np.array(one.constraint_multipliers),F*L/row_scale)
                assert np.max(np.abs(np.array(case.constraint_gaps)-factor*np.array(one.constraint_gaps))/row_scale)<=1e-9
                assert len(case.footprint_wrenches)==len(one.footprint_wrenches)==1
                w=case.footprint_wrenches[0];base=one.footprint_wrenches[0]
                assert w.footprint_id==base.footprint_id=='f13-pad' and w.law_id==base.law_id==result.footprint_law_id
                assert w.common_reference_origin_m==base.common_reference_origin_m
                action_parity(w.beam_force_N+w.beam_moment_Nm,factor*np.array(base.beam_force_N+base.beam_moment_Nm),[F]*3+[F*L]*3)
                action_parity(w.shell_force_N+w.shell_moment_Nm,factor*np.array(base.shell_force_N+base.shell_moment_Nm),[F]*3+[F*L]*3)
                assert len(w.node_actions)==len(base.node_actions)==4
                actual_nodes=[];expected_nodes=[]
                for node,prior in zip(w.node_actions,base.node_actions):
                    assert (node.shell_node_id,node.beam_node_id,node.row_start)==(prior.shell_node_id,prior.beam_node_id,prior.row_start)
                    actual_nodes.extend(node.shell_force_N+node.shell_direct_couple_Nm+node.beam_force_N+node.beam_direct_couple_Nm)
                    expected_nodes.extend(factor*np.array(prior.shell_force_N+prior.shell_direct_couple_Nm+prior.beam_force_N+prior.beam_direct_couple_Nm))
                node_action_errors.append(selected_node_action_parity(np,actual_nodes,expected_nodes,F,L))
            else:
                assert result.selected_points==('f13-right',)
                assert len(case.point_multipliers)==len(case.point_gaps)==1
                assert case.point_multipliers[0][0]==case.point_gaps[0][0]=='f13-right'
                assert len(case.point_multipliers[0][1])==len(case.point_gaps[0][1])==6
                action_parity(case.point_multipliers[0][1],factor*np.array(dict(one.point_multipliers)['f13-right']),[F]*3+[F*L]*3)
                assert np.max(np.abs(np.array(case.point_gaps[0][1])-factor*np.array(dict(one.point_gaps)['f13-right']))/row_scale[6:12])<=1e-9
        for attr,envelopes in (('selected_displacements',result.displacement_envelopes),('selected_affine_reactions',result.affine_reaction_envelopes)):
            for j,e in enumerate(envelopes):
                values=[getattr(c,attr)[j] for c in result.cases]
                assert (e.minimum,e.maximum)==(min(values),max(values))
                assert (e.minimum_case_id,e.maximum_case_id)==(result.cases[values.index(min(values))].case_id,result.cases[values.index(max(values))].case_id)
        assert len(result.member_resultant_envelopes)==18
        assert reference._handle is not owner._handle
        caps=owner.capabilities
        capability_summary=dict(profile_id=str(caps.profile_id),point_law_id=str(caps.point_law_id),status=str(caps.status),
            production_qualified=bool(caps.production_qualified),operations=[str(x) for x in caps.operations],
            shell_domain=str(caps.shell_domain),beam_domain=str(caps.beam_domain),attachment=str(caps.attachment))
        return dict(case_count=count,factorization_count=1,solve_count=count,maximum_rhs_chunk=1,physical=metrics,
            capabilities=capability_summary,selected_port_ids=list(result.selected_footprints if domain=='physical-footprint' else result.selected_points),
            selected_node_action_parity_max=max(node_action_errors,default=None))
    finally:
        owner.close()
        if reference is not None:reference.close()

def ownership_case(ctx,domain):
    np=ctx.np;d=definition(ctx,domain);owner=ctx.api.create_analysis(definition=d)
    try:
        f=load(ctx);before=owner.solve_linear(f,**full_selection(ctx,domain));physical_checks(ctx,owner,before,f)
        original=tuple(x.copy() for x in (owner._K,owner._C,owner._T));handle=owner._handle
        for x in d.shell_nodes_m.values():x[:]+=17
        d.transformation.data[:]*=1.125;d.prescribed_offset[:]+=0.002
        after=owner.solve_linear(f);assert after.displacements==before.displacements
        assert owner._handle is handle and handle.factorization_count==1
        outcomes={}
        for variant in ctx.inputs['dependency_variants']:
            rebuilt=ctx.api.create_analysis(definition=definition(ctx,domain,variant))
            try:
                result=rebuilt.solve_linear(f,**full_selection(ctx,domain));metrics=physical_checks(ctx,rebuilt,result,f)
                assert rebuilt._handle is not handle and result.factorization_count==1
                assert any(a.shape!=b.shape or (a!=b).nnz for a,b in zip((rebuilt._K,rebuilt._C,rebuilt._T),original)) or variant=='affine-offset'
                assert result.displacements!=before.displacements
                outcomes[variant]=metrics
            finally:rebuilt.close()
        assert all((a!=b).nnz==0 for a,b in zip((owner._K,owner._C,owner._T),original))
        assert owner.solve_linear(f).displacements==before.displacements
        return dict(immutable_snapshots=True,distinct_factor_handles=True,variants=outcomes)
    finally:owner.close()

def cancellation_case(ctx,domain):
    outcomes={};f=load(ctx)
    for mode in ('pre-cancel','cancel-3','close-3','final-cancel','final-close'):
        owner=ctx.api.create_analysis(definition=definition(ctx,domain));token=ctx.Token();names=[]
        weak=weakref.ref(owner._handle)
        if mode=='pre-cancel':token.cancel('registered pre-cancel')
        def callback(name):
            names.append(name)
            if len(names)==(5 if mode.startswith('final') else 3):
                if 'close' in mode:owner.close();owner.close();assert weak() is None
                else:token.cancel('registered complete-prefix')
        try:
            result=owner.solve_linear_many(tuple((str(i),f) for i in range(5)),selected_dofs=(50,),on_case=callback,cancellation_token=token,**selected(ctx,domain))
            expected=0 if mode=='pre-cancel' else 5 if mode.startswith('final') else 3
            assert len(result.cases)==expected and result.solve_count==expected and result.factorization_count==1
            assert result.status==('completed' if expected==5 else 'cancelled')
            assert tuple(c.case_id for c in result.cases)==tuple(names)
            if expected==0:assert result.displacement_envelopes==result.affine_reaction_envelopes==()
            if 'close' in mode:
                assert owner._handle is None
                try:owner.solve_linear(f)
                except ctx.module.ExperimentalB2Q4PanelError:pass
                else:raise AssertionError('closed owner admitted solve')
            outcomes[mode]=dict(status=result.status,prefix=expected,solve_count=result.solve_count)
        finally:owner.close()
    return outcomes

def selected_recovery_case(ctx,domain):
    owner=ctx.api.create_analysis(definition=definition(ctx,domain));f=load(ctx)
    def forbidden(*a,**k):raise AssertionError('unselected recovery executed')
    try:
        operator=owner._members[0][1][0].operator
        with patch.object(type(operator),'recover_sections',forbidden),patch.object(ctx.module,'physical_selected_wrenches',forbidden):
            result=owner.solve_linear_many((('first',f),('tie',f)),selected_dofs=(50,))
        assert result.status=='completed' and result.member_resultant_envelopes==()
        assert all(c.beam_resultants_local==() for c in result.cases)
        for e in result.displacement_envelopes+result.affine_reaction_envelopes:assert e.minimum_case_id==e.maximum_case_id=='first'
        return dict(unselected_calls=0,tie_owner='first')
    finally:owner.close()

def refusal_case(ctx,domain):
    owner=ctx.api.create_analysis(definition=definition(ctx,domain));f=load(ctx)
    try:
        baseline=owner._handle.solve_count
        for cases,kwargs in (((('good',f),('bad',ctx.np.zeros(53))),{}),((('good',f),),dict(recover_members={'missing':(.2,)}))):
            try:owner.solve_linear_many(cases,selected_dofs=(50,),**kwargs)
            except ValueError:pass
            else:raise AssertionError('invalid complete batch admitted')
            assert owner._handle.solve_count==baseline
        later=f.copy()
        try:owner.solve_linear_many((('first',f),('later',later)),selected_dofs=(50,),on_case=lambda _:later.__setitem__(50,2*later[50]))
        except ValueError as exc:assert 'changed after submission' in str(exc)
        else:raise AssertionError('mutated RHS admitted')
        assert owner._handle.solve_count==baseline+1
        if domain=='mixed':
            unsupported=replace(definition(ctx,'physical-footprint'),shells=definition(ctx,'mixed').shells)
            with patch.object(ctx.module,'factorize_cached',side_effect=AssertionError('unsupported mixed footprint reached factorization')):
                try:ctx.api.create_analysis(definition=unsupported)
                except ValueError:pass
                else:raise AssertionError('mixed physical footprint silently admitted')
        return dict(prevalidated_invalid_count=2,mutated_load_prefix=1)
    finally:owner.close()

def failure_closure_case(ctx,domain):
    d=definition(ctx,domain,'affine-offset');calls=[]
    def bad_factor(*a,**k):calls.append(1);return SimpleNamespace(status='failed',failure_reason='F13 exact injected factor refusal')
    with patch.object(ctx.module,'factorize_cached',bad_factor):
        try:ctx.api.create_analysis(definition=d)
        except ctx.module.ExperimentalB2Q4PanelError as exc:assert 'F13 exact injected factor refusal' in str(exc)
        else:raise AssertionError('failed factor published offset-only owner')
    assert len(calls)==1
    owner=ctx.api.create_analysis(definition=definition(ctx,domain));original=owner._handle
    class FailSecond:
        @property
        def factorization_count(self):return original.factorization_count
        @property
        def solve_count(self):return original.solve_count
        def solve(self,rhs):
            if original.solve_count==1:raise RuntimeError('F13 second RHS failure')
            return original.solve(rhs)
    try:
        owner._handle=FailSecond()
        failed=owner.solve_linear_many(tuple((name,load(ctx)) for name in ('accepted','broken','later')),selected_dofs=(50,))
        assert failed.status=='failed' and tuple(x.case_id for x in failed.cases)==('accepted',)
        assert failed.solve_count==1 and failed.factorization_count==1 and 'F13 second RHS failure' in failed.failure_reason
    finally:owner.close();original=None
    for error_type in (RuntimeError,MemoryError):
        owner=ctx.api.create_analysis(definition=definition(ctx,domain));weak=weakref.ref(owner._handle);error=error_type('F13 original callback error')
        def callback(_):owner.close();assert weak() is None;raise error
        try:
            try:owner.solve_linear_many((('first',load(ctx)),('later',load(ctx))),selected_dofs=(50,),on_case=callback)
            except error_type as exc:assert exc is error
            else:raise AssertionError('callback failure swallowed')
            assert owner._handle is None and weak() is None
        finally:owner.close()
    return dict(failed_factor_owner_count=0,factor_attempts=1,failed_rhs_complete_prefix=1,original_callback_errors_preserved=2)

def cache_case(ctx,domain):
    """Existing local cache on actual small registered elastic KKT operators."""
    from anyfesolver.linalg import FactorizationCache,MatrixClass,factorize_cached
    from anyfesolver._b2_physical_linear_definition import PhysicalB2OperatorCache
    np=ctx.np;owners=[];matrices=[]
    try:
        for variant in (None,'section','geometry'):
            owner=ctx.api.create_analysis(definition=definition(ctx,domain,variant));owners.append(owner)
            T,C,K=owner._T,owner._C,owner._K
            reduced_k=(T.T@K@T).tocsc();reduced_c=(C@T).tocsc()
            A,_=ctx.module._scaled_point_kkt(reduced_k,reduced_c,None);matrices.append(A)
        member=definition(ctx,domain).members[0]
        localcache=PhysicalB2OperatorCache()
        # Exact binary lengths; never infer a hit from equal decimal0.9m spans.
        exact=replace(member,member_id='f13-exact',reference_points_m=((0.,0.,0.),(0.75,0.,0.)))
        turned=replace(exact,member_id='f13-turned',reference_points_m=((0.,0.,0.),(0.,0.75,0.)),local_y_global=(1.,0.,0.))
        a=exact.compile_local(operator_cache=localcache);b=turned.compile_local(operator_cache=localcache)
        assert a.operator is b.operator and not np.array_equal(a.local_frame_rows(),b.local_frame_rows())
        longer=replace(exact,member_id='f13-longer',reference_points_m=((0.,0.,0.),(0.875,0.,0.)))
        newsection=replace(exact,member_id='f13-section-owned',section=ctx.api.LinearSection.isotropic(**dict(ctx.inputs['section'],E=3960)))
        assert longer.compile_local(operator_cache=localcache).operator is not a.operator
        assert newsection.compile_local(operator_cache=localcache).operator is not a.operator
        equal_distinct=replace(exact,member_id='f13-equal-owned',section=ctx.api.LinearSection.isotropic(**ctx.inputs['section']))
        assert equal_distinct.compile_local(operator_cache=localcache).operator is not a.operator
        assert (localcache.hits,localcache.misses)==(1,4)
        q=np.array([0.,0.,0.,0.,0.,0.,0.0001,-0.0002,0.00015,0.00003,-0.00002,0.00001])
        for compiled,source in ((a,exact),(b,turned)):
            cached=compiled.operator.evaluate(compiled.local_dof_map()@q)
            fresh=source.local_operator().evaluate(source.local_dof_map()@q)
            assert np.array_equal(cached.tangent,fresh.tangent) and np.array_equal(cached.force,fresh.force)
        localcache.clear();assert exact.compile_local(operator_cache=localcache).operator is not a.operator
        cache=FactorizationCache(name='f13-owned-cache',max_entries=2)
        first=factorize_cached(matrices[0],MatrixClass.SYMMETRIC_INDEFINITE,cache=cache)
        assert first.status=='ok' and first.factorization_count==1
        assert factorize_cached(matrices[0].copy(),MatrixClass.SYMMETRIC_INDEFINITE,cache=cache) is first
        weak=weakref.ref(first)
        second=factorize_cached(matrices[1],MatrixClass.SYMMETRIC_INDEFINITE,cache=cache)
        assert second is not first and second.status=='ok';del first
        third=factorize_cached(matrices[2],MatrixClass.SYMMETRIC_INDEFINITE,cache=cache)
        assert third.status=='ok' and weak() is None
        assert len(cache._handles)==2 and cache.hits==1 and cache.misses==3
        rhs=np.ones(matrices[2].shape[0]);answer=third.solve(rhs)
        residual=np.max(np.abs(matrices[2]@answer-rhs)/(np.abs(matrices[2])@np.abs(answer)+np.abs(rhs)))
        assert residual<=1e-8
        second_weak=weakref.ref(second);third_weak=weakref.ref(third);del second,third;cache.clear()
        assert len(cache._handles)==0 and second_weak() is None and third_weak() is None
        return dict(cache_hits=1,cache_misses=3,max_entries=2,eviction_releases_handle=True,clear_releases_handles=True,equation=float(residual),
            local_operator_exact_length_hit=True,section_and_length_misses=True,member_frames_excluded=True,cached_fresh_response_exact=True)
    finally:
        for owner in owners:owner.close()

def run_case(ctx,domain,group):
    if domain not in ctx.inputs['domains'] or group not in GROUPS:raise ValueError('unregistered case')
    if group.startswith('batch-'):return batch_case(ctx,domain,int(group.split('-')[1]))
    return globals()[group.replace('-','_')+'_case'](ctx,domain)
