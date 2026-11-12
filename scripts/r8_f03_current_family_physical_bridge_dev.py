"""Current16/32 physical action/norm applicability; no quotient or solve on import.

Only the reviewed hS+hB constructors are used. Physical loads are independently
integrated continuous P0 densities, not reconstructed from candidate C. A
geometry-only view is not a compiled owner and never supplies an equilibrium.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from scipy import sparse
from scripts.r8_f03_sparse_signed_family_dev import sha_matrix

ELL=.4
BEAM_WIDTH=.1
CONSUMED=('K.npz','C.npz','compiled-layout.npz','states.npz','result.json','inputs.json')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def admit_saved_record(record,level):
    """Actual saved predicates, not an optional historical convenience summary."""
    require(type(level) is int and level in (16,32,64),'literal saved record level required')
    require(isinstance(record,dict) and all(isinstance(record.get(name),dict) for name in ('public_result','literal_inputs','operator_identity')),
            'missing required saved identity fields')
    require(record.get('family')==[level,level//4,level,2] and record.get('endpoint_node')==101+level and
            record.get('status')=='UNRESOLVED_KERNEL' and record.get('public_result',{}).get('status')=='completed',
            'literal completed parity case required')
    require(record.get('literal_inputs',{}).get('action_domain')=='beam' and
            record.get('literal_inputs',{}).get('action_node')==record['endpoint_node'] and
            record.get('literal_inputs',{}).get('action')==[0.,100.,0.,0.,0.,0.] and
            record.get('operator_identity',{}).get('actual_multiplier_space')=='BL5_P0_span_2','load or space mismatch')
    require('physical_parity_pass' not in record or record['physical_parity_pass'] is True,
            'contradictory or invalid physical parity summary')
    parity=record.get('parity');names=('shell','beam_nodes','retained_rotations','resultants','multipliers','reactions')
    require(isinstance(parity,dict) and set(names).issubset(parity),'missing required saved parity metrics')
    def finite_ratio(value):return type(value) in (float,int) and np.isfinite(value) and value>=0.
    require(all(finite_ratio(v) for v in parity.values()) and max(parity.values())<=1e-10,'saved physical/parity guards failed')
    limits=dict(equilibrium=1e-8,weak_gap=1e-9,energy_work_relative=1e-8,compatible_work_relative=1e-8,ground_work_relative=1e-8)
    for name in ('owner_metrics','reduced_metrics'):
        metrics=record.get(name)
        require(isinstance(metrics,dict) and set(limits).issubset(metrics),'missing required saved physical metrics')
        require(all(finite_ratio(metrics[key]) and metrics[key]<=limit for key,limit in limits.items()),'saved physical/parity guards failed')
    return True

def load_bound(directory,bindings,level):
    directory=Path(directory).resolve()
    require(type(level) is int and level in (16,32,64),'only saved current16/32/64')
    require(set(bindings)==set(CONSUMED),'complete saved input binding required')
    for name,h in bindings.items():
        p=directory/name
        require(p.resolve().parent==directory and p.is_file() and sha(p)==h,'saved input identity mismatch: '+name)
    r=json.loads((directory/'result.json').read_text())
    admit_saved_record(r,level)
    k=sparse.load_npz(directory/'K.npz');c=sparse.load_npz(directory/'C.npz')
    require(sha_matrix(k)==r['operator_identity']['k_sha256'] and sha_matrix(c)==r['operator_identity']['c_sha256'],'canonical operator identity mismatch')
    with np.load(directory/'compiled-layout.npz',allow_pickle=False) as z:layout={n:z[n].copy() for n in z.files}
    with np.load(directory/'states.npz',allow_pickle=False) as z:state={n:z[n].copy() for n in z.files}
    require(state['owner_q'].shape==(k.shape[0],) and state['owner_lm'].shape==(c.shape[0],) and np.isfinite(state['owner_q']).all(),'complete finite state required')
    for n in ('shell','beam_nodes','retained_rotations','resultants'):
        require(np.array_equal(layout[n],r['roles'][n]),'actual role mismatch')
    require(np.array_equal(np.sort(np.r_[layout['free'],layout['fixed']]),np.arange(k.shape[0])),'support coverage mismatch')
    return r,k,c,layout,state

def geometry_view(definition,record,k,c,layout):
    """Immutable literal geometry/member references, with saved operator layout."""
    ids=tuple(definition.shell.node_ids);xyz=np.array(definition.shell.coordinates_m,float)
    require(len(definition.lines)==1 and not definition.points and not definition.footprints,'one BL5 line only')
    cells=record['family'][2];line=definition.lines[0]
    require(line.basis=='p0' and line.multiplier_span==2 and line.multiplier_space=='BL5_P0_span_2','literal P0/span2 required')
    nodes={};parts=[]
    for member in definition.beams:
        element,_=member.instantiate();parts.append((element,None,None));ref=element.operator.reference
        for node,x,a in zip(element.node_ids,ref.coordinates,ref.nodal_triads):
            require(node not in nodes or (np.array_equal(nodes[node][0],x) and np.array_equal(nodes[node][1],a)),'inconsistent member reference')
            nodes[node]=(np.array(x,copy=True),np.array(a,copy=True))
    beamids=tuple(sorted(nodes));positions=np.array([nodes[n][0] for n in beamids])
    require(beamids==tuple(record['beam_node_ids'])==tuple(range(101,102+cells)),'actual beam IDs mismatch')
    require(ids==tuple(record['shell_node_ids']) and k.shape[0]==6*len(ids)+6*len(beamids)+24*len(parts),'actual coordinate layout mismatch')
    require(np.array_equal(xyz[:,2],np.zeros(len(xyz))) and np.allclose(positions[:,0],np.linspace(0.,1.,cells+1),rtol=0.,atol=2e-13),'flat unit-length geometry required')
    require(np.all(positions[:,1]==0.) and np.all(positions[:,2]==.0675) and all(np.array_equal(nodes[n][1],np.eye(3)) for n in beamids),'literal beam pose required')
    index={n:i for i,n in enumerate(ids)};bindex={n:i for i,n in enumerate(beamids)}
    fixed=np.array(sorted(6*index[s.node]+s.component if s.domain=='shell' else 6*len(ids)+6*bindex[s.node]+s.component for s in definition.fixed))
    require(np.array_equal(fixed,np.sort(layout['fixed'])),'actual supports differ from literal geometry')
    return SimpleNamespace(_definition=definition,_shell_ids=ids,_shell_xyz=xyz,_shell_index=index,
        _shell_count=6*len(ids),_beam_ids=beamids,_beam_nodes=nodes,_beam_parts=parts,
        _beam_count=k.shape[0]-6*len(ids),_K=k,_C=c,_fixed=layout['fixed'],_free=layout['free'])

def trace_norm(view,arm):
    """Same supported shell and beam constructors, explicit saved-layout scatter."""
    from scripts.r8_bl5_mixed_shell_h1_dev import supported_shell_h1
    from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm
    d=view._definition;index=view._shell_index;cells=len(view._beam_ids)-1
    elements=tuple(tuple(index[n] for n in e) for e in (*d.shell.q4_elements,*d.shell.s3_elements))
    fixed=tuple((index[s.node],s.component) for s in d.fixed if s.domain=='shell')
    hs,_,es=supported_shell_h1(view._shell_xyz[:,:2],elements,ELL,fixed)
    _,hb,active,_=beam_norm(1.,ELL,cells,1,arm,BEAM_WIDTH)
    nodal=6*len(view._beam_ids);rows=[];columns=[]
    for node in range(len(view._beam_ids)):
        for a in range(3):rows.append(6*node+a);columns.append(3*node+a)
    for cell in range(cells):
        for a in range(3):rows.append(nodal+24*(cell//2)+3*(cell%2)+a);columns.append(3*len(view._beam_ids)+3*cell+a)
    eb=sparse.coo_matrix((np.ones(len(rows)),(rows,columns)),shape=(view._beam_count,3*len(view._beam_ids)+3*cells)).tocsc()[:,active]
    e=sparse.block_diag((es,eb),format='csc');h=sparse.block_diag((hs,hb),format='csc')
    require(not e[view._fixed].nnz,'fixed coordinates enter physical norm')
    absent=np.intersect1d(view._free,np.flatnonzero(np.diff(e.tocsr().indptr)==0))
    omitted=view._C[:,absent].copy();omitted.eliminate_zeros()
    require(not omitted.nnz or np.max(np.abs(omitted.data))<=2e-13,'omitted coordinates carry BL5 work')
    return h,e,(view._C@e).tocsc(),absent

def verify_full_action(c,loads,lambdas):
    """All physical coordinates and density columns, including fixed rows."""
    expected=-np.asarray(c.T@lambdas)
    error=float(np.max(np.abs(loads-expected)))
    require(np.allclose(loads,expected,rtol=1e-11,atol=2e-12),'full independent physical transpose action differs')
    return error

def independent_action(view,e,work,frame,director,basis,arm):
    """All force-equivalent columns; no Riesz/dual inverse or native solve."""
    from scripts.b3_e4_bl5_reference_actions_dev import physical_loads,spatial_action
    from scripts.r8_bl5_actual_mref_dev import _candidate_rhs,_complete_rectangular_partition
    sx=np.unique(view._shell_xyz[:,0]);sy=np.unique(view._shell_xyz[:,1]);cells=len(view._beam_ids)-1;supports=cells//2
    _complete_rectangular_partition(view,sx,sy)
    lambdas=np.diag(np.repeat((1.,1.,1.,ELL,ELL),supports));arm_s=.5*view._definition.shell.thickness_m*director
    physical=[];wrench_errors=[]
    for column in lambdas.T:
        density=column.reshape(5,supports).T
        br,sr,wrenches=physical_loads(density,frame,director,basis,arm,arm_s,np.linspace(0.,1.,cells+1),sx,len(sy)-1,cells,2)
        physical.append(_candidate_rhs(view,sx,sy,br,sr))
        # Action couples are about each reference axis: use actual axis origins.
        fb,mb,fs,ms=wrenches
        axis_b=np.array((0.,0.,.0675));axis_s=np.zeros(3)
        wrench_errors.append(np.max(np.abs(mb+ms+np.cross(axis_b,fb)+np.cross(axis_s,fs))))
    loads=np.column_stack(physical);actual=-np.asarray(work.T@lambdas);expected=np.asarray(e.T@loads)
    full_error=verify_full_action(view._C,loads,lambdas)
    error=float(np.max(np.abs(actual-expected)))
    require(np.allclose(actual,expected,rtol=1e-11,atol=2e-12),'independent physical transpose action differs')
    require(max(wrench_errors)<=2e-12,'reference moment origin/arm map differs')
    return dict(columns=len(physical),maximum_pairing_error=error,maximum_full_coordinate_pairing_error=full_error,maximum_local_wrench_error=float(max(wrench_errors)),
        physical_density_units='forceN/m and coupleNm/m; mu rotations converted by ell=.4m'),loads,lambdas

def common_profiles(coarse,fine):
    """Physical common nodes plus declared cell-average projection of fine directors.

    Native directors are cell-constant: a coarse midpoint is a fine interface,
    so it must not be treated as a unique fine pointwise director.
    """
    ca,cq=coarse;fa,fq=fine
    def pose(view,q):
        ns=len(view._shell_ids);nb=len(view._beam_ids);cells=nb-1
        shell=q[:6*ns].reshape(ns,6);beam=q[6*ns:6*(ns+nb)].reshape(nb,6)
        base=6*(ns+nb);rot=np.array([q[base+24*(j//2)+3*(j%2):base+24*(j//2)+3*(j%2)+3] for j in range(cells)])
        return shell,beam,rot
    cs,cb,cr=pose(ca,cq);fs,fb,fr=pose(fa,fq)
    lookup={tuple(x):i for i,x in enumerate(fa._shell_xyz)}
    require(all(tuple(x) in lookup for x in ca._shell_xyz),'no exact common physical nodes')
    indices=np.array([lookup[tuple(x)] for x in ca._shell_xyz]);fcommon=fs[indices]
    averaged=fr.reshape(len(cr),2,3).mean(axis=1)
    fields=dict(shell_translation=(cs[:,:3],fcommon[:,:3]),shell_director_change=(np.cross(cs[:,3:],(0.,0.,1.)),np.cross(fcommon[:,3:],(0.,0.,1.))),
        beam_translation=(cb[:,:3],fb[::2,:3]),beam_cell_director_average=(np.cross(cr,(0.,0.,1.)),np.cross(averaged,(0.,0.,1.))))
    return {name:dict(coarse=a.tolist(),fine_common=b.tolist(),signed_change=(b-a).tolist(),maximum_absolute_change=float(np.max(np.abs(b-a))),
        meaning='physical nodal observable' if 'average' not in name else 'fine two-cell mean over same coarse material interval; not pointwise director') for name,(a,b) in fields.items()}

def source_reconciliation(root,old_manifest):
    m=json.loads(Path(old_manifest).read_text());pairs={}
    for rel,h in m['source_sha256'].items():
        p=Path(root)/rel
        pairs[rel]=dict(old_sha256=h,current_sha256=sha(p) if p.is_file() else None,byte_equal=p.is_file() and sha(p)==h)
    relevant={}
    for rel,h in m.get('files',{}).items():
        if not any(tag in rel for tag in ('nx16-ny4-half16/','nx32-ny8-half32/')):continue
        if '/scripts/' in rel:
            current='scripts/'+rel.split('/scripts/',1)[1]
        elif '/fixtures/' in rel:
            current='tests/'+rel.split('/fixtures/',1)[1]
        elif rel.endswith(('case.json','scalar.json','identity.json','result.json')):
            relevant[rel]=dict(old_sha256=h,kind='historical case/reference metadata; no matrix/input equivalence inferred');continue
        else:continue
        p=Path(root)/current
        relevant[rel]=dict(old_sha256=h,current_path=str(p),current_sha256=sha(p) if p.is_file() else None,byte_equal=p.is_file() and sha(p)==h)
    return dict(status='NO_NUMERICAL_FLOOR_TRANSFER',sources=pairs,relevant_16_32_identities=relevant,old_floor_reused=False,
        blocker='old21 source identities must be reconciled by affected semantics; raw differences are not operator failure')

def reference_linear_weak_gaps(loads,q,lambdas):
    """Independent density transpose evaluated on saved reference-linear q.

    With loads=-C.T@lambdas and diagonal lambdas this equals Cq. It is the
    integrated weak gap, not a sided transfer/compliance observable. In SI the
    first three channel groups integrate translation gap over ds (m²); the last
    two integrate dimensionless director-change gap over ds (m). These are not
    normalized physical accuracy measures and no numerical floor is applied.
    """
    return -(loads.T@q)/np.diag(lambdas)

def exerted_from_density(loads,lm,scales):
    """loads=-C.T diag(scales), hence +loads@(lm/scales)=-C.T lm."""
    return loads@(lm/scales)

def rigid_reference_generators(view,origin):
    """Spatial reference variations; independent cell frames rotate with the body.

    Native 42-slot members retain two spatial cell rotations, then 18 algebraic
    resultants. The latter have zero rigid variation; they are not display forces.
    Virgin nodal/cell frames are identity, so the reference chart derivative is I.
    """
    origin=np.asarray(origin,float);require(origin.shape==(3,) and np.isfinite(origin).all(),'finite origin required')
    g=np.zeros((view._K.shape[0],6));ns=view._shell_count;nb=6*len(view._beam_ids)
    for start,x in [(6*i,x) for i,x in enumerate(view._shell_xyz)]+[(ns+6*i,view._beam_nodes[n][0]) for i,n in enumerate(view._beam_ids)]:
        g[start:start+3,:3]=np.eye(3)
        g[start:start+3,3:]=np.column_stack([np.cross(e,x-origin) for e in np.eye(3)])
        g[start+3:start+6,3:]=np.eye(3)
    for i in range(len(view._beam_parts)):
        for cell in (0,1):g[ns+nb+24*i+3*cell:ns+nb+24*i+3*cell+3,3:]=np.eye(3)
    return g

def reference_wrench(view,action,origin,domain='all'):
    """Independent nodal force/arm and nodal+retained spatial couple sum."""
    a=np.asarray(action,float);require(a.shape==(view._K.shape[0],) and np.isfinite(a).all(),'finite complete action required')
    require(domain in ('all','shell','beam'),'physical domain required');o=np.asarray(origin,float)
    f=np.zeros(3);m=np.zeros(3);ns=view._shell_count;nb=6*len(view._beam_ids)
    nodes=[]
    if domain!='beam':nodes.extend((6*i,x) for i,x in enumerate(view._shell_xyz))
    if domain!='shell':nodes.extend((ns+6*i,view._beam_nodes[n][0]) for i,n in enumerate(view._beam_ids))
    for start,x in nodes:
        f+=a[start:start+3];m+=np.cross(x-o,a[start:start+3])+a[start+3:start+6]
    if domain!='shell':
        for i in range(len(view._beam_parts)):
            m+=a[ns+nb+24*i:ns+nb+24*i+6].reshape(2,3).sum(axis=0)
    return np.r_[f,m]

def absolute_product_ratio(residual,denominator):
    """No floor: same-unit absolute products bound the action magnitude."""
    r=np.abs(np.asarray(residual));d=np.asarray(denominator)
    require(r.shape==d.shape and np.isfinite(r).all() and np.isfinite(d).all() and np.all(d>=0),'invalid objectivity action')
    require(not np.any((d==0)&(r!=0)),'nonzero action with zero absolute-product denominator')
    return float(np.max(np.divide(r,d,out=np.zeros_like(r),where=d!=0),initial=0.))

def reference_objectivity(k,c,g,scale):
    """Sparse all-column work/objectivity, never a coercivity or inertia test."""
    s=np.asarray(scale,float);require(s.shape==(k.shape[0],) and np.isfinite(s).all() and np.all(s>0),'positive compiled coordinate scales required')
    a=sparse.diags(s)@k@sparse.diags(s);r=g/s[:,None]
    kr=np.asarray(a@r);kr_abs=np.asarray(abs(a)@abs(r))
    rk=np.asarray(a.T@r).T;rk_abs=np.asarray(abs(a).T@abs(r)).T
    cg=np.asarray(c@g);cg_abs=np.asarray(abs(c)@abs(g))
    metrics=dict(right=absolute_product_ratio(kr,kr_abs),left=absolute_product_ratio(rk,rk_abs),constraint=absolute_product_ratio(cg,cg_abs))
    return metrics,dict(weighted_right=kr,absolute_right=kr_abs,weighted_left=rk,absolute_left=rk_abs,constraint_rigid=cg,constraint_absolute=cg_abs)

def reference_density_columns(view):
    """Independent continuous P0 actions, no supported norm or owner construction."""
    from scripts.b3_e4_bl5_reference_actions_dev import physical_loads
    from scripts.r8_bl5_actual_mref_dev import _candidate_rhs,_complete_rectangular_partition
    sx=np.unique(view._shell_xyz[:,0]);sy=np.unique(view._shell_xyz[:,1]);cells=len(view._beam_ids)-1;supports=cells//2
    _complete_rectangular_partition(view,sx,sy)
    frame=np.eye(3);director=np.array((0.,0.,1.));basis=np.array(((1.,0.,0.),(0.,1.,0.)))
    arm=np.array((0.,0.,-.05));arm_s=.5*view._definition.shell.thickness_m*director
    require(abs(.0675+arm[2]-arm_s[2])<=2e-13,'physical interface arms differ')
    scales=np.repeat((1.,1.,1.,ELL,ELL),supports);columns=[]
    for col in np.diag(scales).T:
        br,sr,_=physical_loads(col.reshape(5,supports).T,frame,director,basis,arm,arm_s,np.linspace(0.,1.,cells+1),sx,len(sy)-1,cells,2)
        columns.append(_candidate_rhs(view,sx,sy,br,sr))
    return np.column_stack(columns),scales

def run_wrench_recovery(manifest,output):
    """Explicit saved levels: no owner, assembly, norm, factor or equilibrium call."""
    import traceback
    from scripts.b3_e4_bl5_coherent_aggregation_dev import model
    out=Path(output);out.mkdir(exist_ok=False);result=dict(status='FAILED',qualification=False,cases={})
    origin=np.zeros(3);shift=np.array((.23,-.17,.11));units=np.array((100.,100.,100.,100.,100.,100.))
    try:
        m=json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
        for p,h in m['source_inputs'].items():require(sha(p)==h,'source binding changed')
        levels=recovery_levels(m.get('levels',[16,32]))
        for level in levels:
            r,k,c,layout,state=load_bound(m['saved'][str(level)]['directory'],m['saved'][str(level)]['files'],level)
            view=geometry_view(model(level,level//4,level,2)[0],r,k,c,layout)
            g=rigid_reference_generators(view,origin);gs=rigid_reference_generators(view,shift)
            obj,arrays=reference_objectivity(k,c,g,layout['coordinate_scale'])
            loads,scales=reference_density_columns(view);q=state['owner_q'];lm=state['owner_lm'];f=state['load']
            expected=np.zeros_like(f);expected[view._shell_count+6*(len(view._beam_ids)-1)+1]=100.
            require(np.array_equal(f,expected),'actual endpoint Fy100 load differs')
            interface=exerted_from_density(loads,lm,scales);native=-np.asarray(c.T@lm)
            internal=np.asarray(k@q);residual=internal-native-f;support=np.zeros_like(f);support[layout['fixed']]=residual[layout['fixed']]
            actions=dict(external=f,support=support,internal=internal,interface=interface)
            wrenches={name:{domain:reference_wrench(view,a,origin,domain).tolist() for domain in ('all','shell','beam')} for name,a in actions.items()}
            work_errors={name:float(np.max(np.abs((g.T@a-np.asarray(wrenches[name]['all']))/units))) for name,a in actions.items()}
            shift_errors={name:float(np.max(np.abs((gs.T@a-np.r_[np.asarray(wrenches[name]['all'])[:3],np.asarray(wrenches[name]['all'])[3:]-np.cross(shift,np.asarray(wrenches[name]['all'])[:3])])/units))) for name,a in actions.items()}
            balance=np.asarray(wrenches['external']['all'])+np.asarray(wrenches['support']['all'])
            internal_w=np.asarray(wrenches['internal']['all']);interface_w=np.asarray(wrenches['interface']['all'])
            domain_balance={d:(np.asarray(wrenches['external'][d])+np.asarray(wrenches['support'][d])+np.asarray(wrenches['interface'][d])-np.asarray(wrenches['internal'][d])).tolist() for d in ('shell','beam')}
            rec=dict(objectivity=obj,wrenches=wrenches,rigid_work_coefficient_errors=work_errors,origin_shift_errors=shift_errors,
                external_support_balance=balance.tolist(),maximum_balance_ratio=float(np.max(np.abs(balance/units))),
                internal_rigid_work_ratio=float(np.max(np.abs(internal_w/units))),interface_cancellation_ratio=float(np.max(np.abs(interface_w/units))),domain_balance=domain_balance,
                full_action_delta=float(np.max(np.abs(interface-native))),coefficient_delta=float(np.max(np.abs(loads+(c.T@sparse.diags(scales)).toarray()))),
                root_support_work=float(q@support),applied_work=float(q@f),resultant_actions_not_physical_wrenches=True,
                normalization='F=100N L=1m; moment=100Nm; rigid coefficient action work scale=100J; no response floor',operator_identity=r['operator_identity'])
            result['cases'][str(level)]=rec
            np.savez(out/f'wrenches{level}.npz',**arrays,generator=g,shift_generator=gs,loads=loads,scales=scales,q=q,lm=lm,
                external=f,support=support,internal=internal,interface=interface,native_interface=native,residual=residual,
                coordinate_scale=layout['coordinate_scale'],fixed=layout['fixed'],resultants=layout['resultants'],shell_xyz=view._shell_xyz,
                beam_xyz=np.array([view._beam_nodes[n][0] for n in view._beam_ids]))
            # Every actual diagnostic above survives a refusal below.
            require(max(obj.values())<=1e-9,'reference rigid tangent/constraint objectivity failed')
            verify_full_action(c,loads,np.diag(scales))
            require(np.allclose(interface,native,rtol=1e-11,atol=2e-12),'actual exerted interface action differs')
            require(max(work_errors.values())<=1e-8 and max(shift_errors.values())<=1e-8,'rigid virtual-work/origin map failed')
            require(max(rec['maximum_balance_ratio'],rec['internal_rigid_work_ratio'],rec['interface_cancellation_ratio'])<=1e-8,'global physical wrench balance failed')
            require(max(np.max(np.abs(np.asarray(v)/units)) for v in domain_balance.values())<=1e-8,'physical domain wrench equilibrium failed')
            require(abs(q@support)/abs(q@f)<=1e-8,'actual support work failed')
        result.update(status='REFERENCE_LINEAR_WRENCH_RECOVERY_PASS',compiled_owners=0,equilibrium_solves=0,riesz_solves=0,
            scope='stored reference-linear rigid work and physical wrenches only; no torque stiffness, drilling, locking, finite chart, inf-sup or F03 qualification')
        return result
    except BaseException as exc:
        result.update(exception=repr(exc),traceback=traceback.format_exc());raise
    finally:(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')

def recovery_levels(levels):
    """Default consumed16/32 dispatch unchanged; sole64 is prospectively explicit."""
    require(isinstance(levels,(list,tuple)) and all(type(n) is int for n in levels) and tuple(levels) in ((16,32),(64,)),
            'only default saved16/32 or explicit sole64 recovery')
    return tuple(levels)

def three_level_observations(records,profiles):
    """Observed increments only; no fitted settling/accuracy criterion."""
    require(set(records)=={'16','32','64'},'complete three-level records required')
    values={name:np.array([records[str(n)][name] for n in (16,32,64)]) for name in ('applied_work','root_support_work')}
    for action,domain in (('support','all'),('interface','shell'),('interface','beam')):
        values[action+'_'+domain]=np.array([records[str(n)]['wrenches'][action][domain] for n in (16,32,64)])
    changes={name:dict(values=v.tolist(),signed_increments=np.diff(v,axis=0).tolist(),absolute_increments=np.abs(np.diff(v,axis=0)).tolist()) for name,v in values.items()}
    return dict(levels=[16,32,64],quantities=changes,common_reference_linear_profiles=profiles,
        qualification=False,accuracy_estimate=None,convergence_acceptance=None,
        interpretation='three observed levels only; no fitted threshold, asymptotic error or continuum/reference matching claim',
        profile_meaning='same physical common nodal coordinates; independent half-cell director changes use labeled two-cell interval averages, not interface point values')

def run_three_level_recovery(manifest,output):
    """One saved64 recovery plus bound old observations; zero owner or solve calls."""
    from scripts.b3_e4_bl5_coherent_aggregation_dev import model
    m=json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    require(recovery_levels(m['levels'])==(64,),'three-level stage recovers sole64 only')
    prior=m['prior_wrenches'];require(sha(prior['path'])==prior['sha256'],'prior wrench observation identity mismatch')
    old=json.loads(Path(prior['path']).read_text());require(old['status']=='REFERENCE_LINEAR_WRENCH_RECOVERY_PASS' and set(old['cases'])=={'16','32'},'bound prior16/32 observations required')
    result=run_wrench_recovery(manifest,output);out=Path(output)
    try:
        records={**old['cases'],**result['cases']};views={}
        for level in (16,32,64):
            record,k,c,layout,state=load_bound(m['saved'][str(level)]['directory'],m['saved'][str(level)]['files'],level)
            views[level]=(geometry_view(model(level,level//4,level,2)[0],record,k,c,layout),state['owner_q'])
        profiles={str(a)+'_to_'+str(b):common_profiles(views[a],views[b]) for a,b in ((16,32),(32,64))}
        result.update(status='THREE_LEVEL_REFERENCE_LINEAR_OBSERVATIONS_PASS',observations=three_level_observations(records,profiles),
            prior_wrenches=prior,recovered_levels=[64],reused_observation_levels=[16,32])
        return result
    except BaseException as exc:
        import traceback
        result.update(status='FAILED_THREE_LEVEL_OBSERVATIONS',exception=repr(exc),traceback=traceback.format_exc());raise
    finally:(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')

def p0_region_masks(policy,cells):
    """Exact union of bound P0 supports, channel-major; never mask nodal rows."""
    require(type(cells) is int and cells in (16,32,64),'literal regional native level required')
    require(policy.get('span_halfcells')==2 and policy.get('beam_halfcells')==cells and policy.get('c_lambda')==2,
            'regional P0 span/membership identity differs')
    require(policy.get('support_membership')==[list(range(j,j+2)) for j in range(0,cells,2)],'regional support membership/order differs')
    e=np.asarray(policy.get('support_endpoints_m'),float);expected=np.arange(cells//2+1)/(cells//2)
    require(e.shape==expected.shape and np.array_equal(e,expected),'regional support endpoints differ')
    regions=dict(Q1=(0.,.25),Q2=(.25,.5),Q3=(.5,.75),Q4=(.75,1.),interior=(.25,.75),whole=(0.,1.))
    return {name:np.tile((e[:-1]>=a)&(e[1:]<=b),5) for name,(a,b) in regions.items()}

def p0_regional_actions(loads,lm,scales,masks):
    loads=np.asarray(loads,float);lm=np.asarray(lm,float);scales=np.asarray(scales,float)
    require(loads.ndim==2 and lm.shape==scales.shape==(loads.shape[1],) and np.isfinite(loads).all() and
            np.isfinite(lm).all() and np.isfinite(scales).all() and np.all(scales>0),'complete finite regional density inputs required')
    require(all(v.dtype==bool and v.shape==lm.shape for v in masks.values()),'regional column mask differs')
    density=lm/scales
    return {name:loads[:,mask]@density[mask] for name,mask in masks.items()}

def run_regional_wrenches(manifest,output):
    """Saved arrays only. No model, reference-element, owner, K or norm creation."""
    import traceback
    out=Path(output);out.mkdir(exist_ok=False);result=dict(status='FAILED',qualification=False,cases={})
    shift=np.array((.23,-.17,.11));units=np.full(6,100.)
    try:
        m=json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
        for p,h in m['source_inputs'].items():require(sha(p)==h,'regional source binding changed')
        require(set(m['saved'])=={'16','32','64'} and set(m['wrench_arrays'])=={'16','32','64'},'all three bound regional levels required')
        for level in (16,32,64):
            item=m['saved'][str(level)];directory=Path(item['directory']).resolve()
            require(set(item['files'])==set(CONSUMED),'complete saved regional binding required')
            for name,h in item['files'].items():require(sha(directory/name)==h,'regional saved input changed')
            record=json.loads((directory/'result.json').read_text())
            admit_saved_record(record,level)
            binding=m['wrench_arrays'][str(level)];require(sha(binding['path'])==binding['sha256'],'regional wrench array identity differs')
            with np.load(binding['path'],allow_pickle=False) as z:data={n:z[n].copy() for n in z.files}
            with np.load(directory/'states.npz',allow_pickle=False) as z:
                require(np.array_equal(data['q'],z['owner_q']) and np.array_equal(data['lm'],z['owner_lm']),'regional native state differs')
            shellxyz=data['shell_xyz'];beamxyz=data['beam_xyz'];ns=6*len(shellxyz);nb=6*len(beamxyz);n=len(data['q'])
            require(n==record['shape']['full'] and data['loads'].shape==(n,5*level//2) and len(beamxyz)==level+1 and
                record['beam_node_ids']==list(range(101,102+level)) and (n-ns-nb)==24*(level//2),'regional physical layout differs')
            require(np.array_equal(data['scales'],np.repeat((1.,1.,1.,ELL,ELL),level//2)),'regional channel scaling differs')
            # Shape-only descriptor and saved reference coordinates: no K/model construction.
            view=SimpleNamespace(_K=SimpleNamespace(shape=(n,n)),_shell_count=ns,_shell_xyz=shellxyz,
                _beam_ids=tuple(record['beam_node_ids']),_beam_nodes={node:(x,None) for node,x in zip(record['beam_node_ids'],beamxyz)},_beam_parts=(None,)*(level//2))
            g=rigid_reference_generators(view,np.zeros(3));require(np.array_equal(g,data['generator']),'regional saved rigid map differs')
            masks=p0_region_masks(record['input_policy'],level);actions=p0_regional_actions(data['loads'],data['lm'],data['scales'],masks)
            wr={name:{domain:reference_wrench(view,a,np.zeros(3),domain).tolist() for domain in ('all','shell','beam')} for name,a in actions.items()}
            errors={name:float(np.max(np.abs((g.T@a-np.asarray(wr[name]['all']))/units))) for name,a in actions.items()}
            shifted={name:reference_wrench(view,a,shift).tolist() for name,a in actions.items()}
            originerrors={name:float(np.max(np.abs((np.asarray(shifted[name])-np.r_[np.asarray(wr[name]['all'])[:3],np.asarray(wr[name]['all'])[3:]-np.cross(shift,np.asarray(wr[name]['all'])[:3])])/units))) for name in actions}
            composition=sum(actions[name] for name in ('Q1','Q2','Q3','Q4'))-actions['whole']
            interior=actions['Q2']+actions['Q3']-actions['interior']
            rec=dict(wrenches=wr,origin_shift_wrenches=shifted,rigid_work_coefficient_errors=errors,origin_shift_errors=originerrors,
                maximum_quarter_action_recomposition_error=float(np.max(np.abs(composition))),maximum_interior_action_recomposition_error=float(np.max(np.abs(interior))),
                masks={name:np.flatnonzero(mask).tolist() for name,mask in masks.items()},support_endpoints_m=record['input_policy']['support_endpoints_m'],
                meaning='integrated exerted interface action by physical P0 support union, common reference origin0; no nodal-region cropping',
                density_row_order='channel*(level/2)+halfcell//2; all five channels retained',normalization='F100N L1m, moment100Nm; unchanged1e-8 wrench/work mapping')
            result['cases'][str(level)]=rec
            np.savez(out/f'regions{level}.npz',**{'action_'+name:a for name,a in actions.items()},**{'mask_'+name:mask for name,mask in masks.items()},
                loads=data['loads'],lm=data['lm'],scales=data['scales'],generator=g,shell_xyz=shellxyz,beam_xyz=beamxyz,
                quarter_recomposition=composition,interior_recomposition=interior,whole_native_action=data['native_interface'])
            require(np.allclose(actions['whole'],data['native_interface'],rtol=1e-11,atol=2e-12),'regional whole action differs from verified native action')
            require(np.allclose(sum(actions[name] for name in ('Q1','Q2','Q3','Q4')),actions['whole'],rtol=1e-11,atol=2e-12) and
                np.allclose(actions['Q2']+actions['Q3'],actions['interior'],rtol=1e-11,atol=2e-12),'regional action recomposition failed')
            require(max(errors.values())<=1e-8 and max(originerrors.values())<=1e-8 and max(np.max(np.abs(np.asarray(v['all'])/units)) for v in wr.values())<=1e-8,
                    'regional physical wrench/work map failed')
            for domain in ('shell','beam'):
                require(np.max(np.abs((sum(np.asarray(wr[name][domain]) for name in ('Q1','Q2','Q3','Q4'))-np.asarray(wr['whole'][domain]))/units))<=1e-8 and
                    np.max(np.abs((np.asarray(wr['Q2'][domain])+np.asarray(wr['Q3'][domain])-np.asarray(wr['interior'][domain]))/units))<=1e-8,'regional wrench recomposition failed')
        trends={}
        for region in ('Q1','Q2','Q3','Q4','interior','whole'):
            values=np.array([result['cases'][str(n)]['wrenches'][region]['shell'] for n in (16,32,64)])
            trends[region]=dict(shell_wrench_values=values.tolist(),signed_increments=np.diff(values,axis=0).tolist(),absolute_increments=np.abs(np.diff(values,axis=0)).tolist())
        result.update(status='SAVED_P0_REGIONAL_WRENCH_OBSERVATIONS_PASS',trends=trends,model_constructions=0,equilibrium_solves=0,
            scope='three fixed-region observations only; no causal/accuracy/convergence/F03 acceptance or threshold fitted')
        return result
    except BaseException as exc:
        result.update(exception=repr(exc),traceback=traceback.format_exc());raise
    finally:(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')

def run_bridge(manifest,output):
    """ONE current32 compilation, saved16 geometry only; no equilibrium/Riesz solve."""
    import traceback
    from anyfesolver import stiffened_shell
    from scripts.b3_e4_bl5_coherent_aggregation_dev import model
    from scripts.r8_bl5_actual_primal_work_dev import actual_flat_primal_work
    out=Path(output);out.mkdir(exist_ok=False);result=dict(status='FAILED',qualification=False);owner=None
    try:
        m=json.loads(Path(manifest).read_text(encoding='utf-8-sig'));root=Path(m['root'])
        for p,h in m['source_inputs'].items():require(sha(p)==h,'source binding changed')
        cases={level:load_bound(m['saved'][str(level)]['directory'],m['saved'][str(level)]['files'],level) for level in (16,32)}
        d32,_=model(32,8,32,2);owner=stiffened_shell.create_analysis(definition=d32)
        r,k,c,layout,state=cases[32]
        require(sha_matrix(owner._K)==sha_matrix(k) and sha_matrix(owner._C)==sha_matrix(c) and str(owner._identity)==r['operator_identity']['owner_identity'],'current32 compiled owner differs from saved operator')
        local=owner._line_samples[0][0];frame=local.port.frame;director=frame@local.reference_director;basis=local.director_basis;arm=frame@local.reference_arm
        require(np.array_equal(frame,np.eye(3)) and np.array_equal(director,(0.,0.,1.)) and np.allclose(arm,(0.,0.,-.05),rtol=0.,atol=2e-13),'literal flat force/moment frame differs')
        records={};views={}
        for level in (16,32):
            r,k,c,layout,state=cases[level];d=d32 if level==32 else model(16,4,16,2)[0]
            view=geometry_view(d,r,k,c,layout);views[level]=(view,state['owner_q'])
            require(abs(.5*d.shell.thickness_m-.0675-arm[2])<=2e-13,'physical shell/beam arm identity mismatch')
            h,e,w,absent=trace_norm(view,arm)
            if level==32:
                oh,oe,ow=actual_flat_primal_work(owner,ELL,BEAM_WIDTH)
                for a,b in ((h,oh),(e,oe),(w,ow)):
                    diff=(a-b).tocoo();require(not diff.nnz or np.max(np.abs(diff.data))<=2e-12,'saved norm/scatter differs from actual adopted constructor')
            action,loads,lambdas=independent_action(view,e,w,frame,director,basis,arm)
            q=state['owner_q'];lm=state['owner_lm'];physical=exerted_from_density(loads,lm,np.diag(lambdas))
            native=-np.asarray(c.T@lm);delta=physical-native;threshold=2e-12+1e-11*np.abs(native)
            # Preserve actual actions before the unchanged combined-action refusal.
            np.savez(out/f'action{level}.npz',loads=loads,lambdas=lambdas,lm=lm,q=q,physical_action=physical,native_action=native,signed_delta=delta,threshold=threshold)
            result.setdefault('action_diagnostics',{})[str(level)]=dict(maximum_absolute_delta=float(np.max(np.abs(delta))),
                failed_rows=np.flatnonzero(np.abs(delta)>threshold).tolist(),coefficient_checks=action)
            require(np.allclose(physical,-c.T@lm,rtol=1e-11,atol=2e-12),'complete physical action including supported rows differs')
            records[str(level)]=dict(**action,primal_size=h.shape[0],primal_nnz=h.nnz,embedding_shape=e.shape,off_line_shell_fields_retained=True,
                omitted_columns=absent.tolist(),omitted_resultants=len(layout['resultants']),operator_identity=r['operator_identity'],
                actual_physical_pairing=float(q@physical),negative_constraint_pairing=float(-lm@(c@q)),
                reference_linear_weak_gaps=reference_linear_weak_gaps(loads,q,lambdas).tolist(),
                weak_gap_provenance='equals Cq from independent density transpose; integrated weak gap only, not sided transfer/compliance',
                weak_gap_units='three channel groups m^2, two director-change channel groups m; reference-linear SI',
                reference=dict(status='MATRIX_REUSE_UNRESOLVED',reason='no bound exact continuous-dual matrix matching current row layout consumed; no Riesz/majorant/refinement run',ell=ELL,beam_width=BEAM_WIDTH))
            sparse.save_npz(out/f'H{level}.npz',h);sparse.save_npz(out/f'E{level}.npz',e);sparse.save_npz(out/f'work{level}.npz',w)
        result.update(status='PHYSICAL_ACTION_NORM_BRIDGE_PASS_REFERENCE_UNRESOLVED',cases=records,profiles=common_profiles(views[16],views[32]),
            historical_reconciliation=source_reconciliation(root,m['old_manifest']),compiled_owners=1,equilibrium_solves=0,riesz_solves=0,
            scope='physical action/norm applicability and matched observations only; no beta/floor/locking/accuracy/F03 acceptance')
        return result
    except BaseException as exc:
        result.update(exception=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if owner is not None:owner.close()
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
