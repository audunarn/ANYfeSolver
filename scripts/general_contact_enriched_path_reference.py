"""Independent multi-target contact/work/reference for the ORIGINAL V16 path.

Shared actual structural FE only. Geometry, inverse, quadrature, normalized
gaps and finite work are independently reconstructed. No fixed active node
or matched184/max4 assumption is used.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.general_contact_enriched_path_geometry import ContactOracle,require,finite,ReferenceRefused,q4
from scripts.general_contact_enriched_path_bounds import global_gap_certificate
from scripts.general_contact_enriched_reference import norms,passed_norms,rigid_wrench,publish

STEPS=(1e-5,3e-6)
ORDERS=(4,8)
SCALE=.018
CANONICAL_COORDINATES=tuple(range(6))+tuple(range(150,156))+tuple(range(300,304))


def canonical_parity(value,canonical):
    reference_norm=float(np.linalg.norm(canonical['support']))
    require(reference_norm>0,'nonzero canonical support, no relative nearzero floor')
    errors=dict(support=float(np.linalg.norm(value['support']-canonical['support']))/reference_norm,
        gap=float(np.linalg.norm(value['gap']-canonical['gap']))/50e-6,
        work=abs(value['work']-canonical['work'])/SCALE,
        signature_equal=value['signature']==canonical['signature'])
    errors['predicate']=bool(errors['signature_equal'] and max(errors['support'],errors['gap'],errors['work'])<=1e-13)
    return errors


def inventory(system):
    rows={};names=('ux','uy','uz','rx','ry','rz')
    for node in system._nodal.model.mesh.nodes:
        for index,name in zip(system._nodal.model.mesh.dof_manager.get_node_dofs(node),names,strict=True):
            rows[int(index)]=dict(kind='nodal',node=int(node),axis=name,unit='m' if name[0]=='u' else 'rad',
                conjugate='N' if name[0]=='u' else 'Nm',normalization=.018)
    for cell,indices in zip(system.cells,system.cell_indices,strict=True):
        for index,name in zip(indices[24:],('B','Br','Bs','Brs'),strict=True):
            rows[int(index)]=dict(kind='physical_beta',element=int(cell.element_id),axis=name,unit='m',conjugate='N',normalization=.018)
    require(set(rows)==set(range(system.offset.size)),'complete original plus all added work coordinates')
    return rows


def full_work(model,system,definitions,u,z,force,*,cancel=lambda:False,progress=None,reuse=None):
    n=system.offset.size;force=finite(force,(n,));labels=inventory(system)
    rows=[]
    for h in STEPS:
        orders=[]
        for order in ORDERS:
            if cancel():
                from anyfesolver.control import SolveCancelled
                raise SolveCancelled('independent full path work cancelled')
            oracle=None if reuse is None else reuse.get((h,order))
            if oracle is None:oracle=ContactOracle(model,definitions,u,z,order,cancel=cancel)
            else:oracle.require_exact_reuse(model,definitions,u,z,order)
            base=oracle.base_value;central=np.empty(n);positive=np.empty(n);negative=np.empty(n)
            gradient=np.empty((len(z),n));changed=[]
            fresh=ContactOracle(model,definitions,u,z,order,cancel=cancel).base_value
            baseline_parity=canonical_parity(base,fresh)
            require(baseline_parity['predicate'],'original baseline canonical arithmetic parity failed')
            sampled=[]
            for i in range(n):
                if cancel():
                    from anyfesolver.control import SolveCancelled
                    raise SolveCancelled('independent path work cancelled')
                p,m=np.array(u,copy=True),np.array(u,copy=True);p[i]+=h;m[i]-=h
                plus=oracle.evaluate(p)
                if cancel():
                    from anyfesolver.control import SolveCancelled
                    raise SolveCancelled('independent path work cancelled')
                minus=oracle.evaluate(m)
                central[i]=(plus['work']-minus['work'])/(2*h)
                positive[i]=(plus['work']-base['work'])/h;negative[i]=(base['work']-minus['work'])/h
                gradient[:,i]=(plus['gap']-minus['gap'])/(2*h)
                if i in CANONICAL_COORDINATES:
                    cp=ContactOracle(model,definitions,p,z,order,cancel=cancel).base_value
                    cm=ContactOracle(model,definitions,m,z,order,cancel=cancel).base_value
                    pp,pm=canonical_parity(plus,cp),canonical_parity(minus,cm)
                    derivative=abs(central[i]-(cp['work']-cm['work'])/(2*h))/SCALE
                    parity=dict(coordinate=i,plus=pp,minus=pm,derivative=derivative,
                        predicate=bool(pp['predicate'] and pm['predicate'] and derivative<=1e-7))
                    sampled.append(parity)
                    if not parity['predicate']:
                        if progress:progress(dict(kind='FAILED_ORIGINAL_CANONICAL_PARITY',h=h,order=order,evidence=parity))
                        raise ReferenceRefused('original sampled canonical value/derivative parity failed')
                if plus['signature']!=base['signature'] or minus['signature']!=base['signature']:changed.append(i)
                if progress and (i%50==49 or i==n-1):progress(dict(kind='full_physical_work',h=h,order=order,completed=i+1,total=n))
            oracle.verify()
            errors={key:norms((value-force)/SCALE) for key,value in (('central',central),('positive',positive),('negative',negative))}
            orders.append(dict(order=order,central=central.tolist(),positive=positive.tolist(),negative=negative.tolist(),
                gap_gradient=gradient.tolist(),errors=errors,changed_coordinates=changed,
                support=base['support'].tolist(),gap=base['gap'].tolist(),signature=base['signature'],
                baseline_canonical_parity=baseline_parity,sampled_canonical_parity=sampled,
                all_finite=bool(all(np.all(np.isfinite(v)) for v in (central,positive,negative,gradient)))))
        mutual={key:norms((np.array(orders[0][key])-orders[1][key])/SCALE) for key in ('central','positive','negative')}
        rows.append(dict(h=h,orders=orders,mutual=mutual))
    numeric=all(o['all_finite'] and all(passed_norms(v,1e-6) for v in o['errors'].values()) for row in rows for o in row['orders'])
    quadrature=all(passed_norms(v,1e-7) for row in rows for v in row['mutual'].values())
    return dict(authoritative_coordinates=n,inventory=labels,records=rows,
        numeric_predicate=numeric,quadrature_predicate=quadrature,predicate=bool(numeric and quadrature))


def gap_gradient(oracle,u,h=3e-6,cancel=lambda:False):
    g=np.empty((len(oracle.z),len(u)))
    for i in range(len(u)):
        if cancel():
            from anyfesolver.control import SolveCancelled
            raise SolveCancelled('independent path reference cancelled')
        p,m=np.array(u,copy=True),np.array(u,copy=True);p[i]+=h;m[i]-=h
        g[:,i]=(oracle.evaluate(p)['gap']-oracle.evaluate(m)['gap'])/(2*h)
    return finite(g,(len(oracle.z),len(u)))


def reference_scales(oracle,stiffness):
    """Full reference hat support independently integrated, never current M."""
    roots,weights=np.polynomial.legendre.leggauss(4);support=np.zeros(len(oracle.z))
    for sid,(x,director,offset,dofs,beta) in oracle.source_data.items():
        corners=x+offset*director
        indices=[oracle.mode[n] for n in oracle.model.mesh.elements[sid].node_ids]
        for r,wr in zip(roots,weights):
            for s,ws in zip(roots,weights):
                hats,nr,ns=q4(r,s)
                support[indices]+=wr*ws*np.linalg.norm(np.cross(nr @ corners,ns @ corners))*hats
    require(np.all(support>0),'every independent reference pressure mode supported')
    return support*stiffness


def independent_response(model,system,definitions,load,criteria,*,cancel=lambda:False,progress=None):
    """Independent normalized-gap modified Newton, ALL modes, zero free start.

    The reference uses shared structural tangent as a preconditioner, not a
    production contact Hessian/active set. Bounded18iterations/8halvings.
    No guarantee of convergence or positive definite Schur complement.
    """
    t=system.transform;n=t.shape[1];m=len(system.pressure_ids)
    reduced,z=np.zeros(n),np.zeros(m);trace=[]
    scales=None;last=None
    def evaluate(rq,rz,derivatives):
        u=system.offset+t @ rq
        oracle=ContactOracle(model,definitions,u,rz,8,cancel=cancel,cache_pose_results=derivatives)
        g=gap_gradient(oracle,u,cancel=cancel) if derivatives else None
        return u,oracle,g
    for iteration in range(19):
        u,oracle,g=evaluate(reduced,z,True)
        if scales is None:
            scales=reference_scales(oracle,system.normal_stiffness)
            require(np.max(np.abs(scales-system.scales)/scales)<=1e-13,'independent full-reference NCP scale parity')
        internal,k,_=system.structural_response(u,cancel=cancel)
        r=t.T @ (internal-g.T @ z-load);gap=oracle.base_value['gap']
        c=z-np.maximum(0,z-scales*gap)
        eq=float(np.linalg.norm(r)/criteria.force_scale_n);ncp=float(np.sum(np.abs(c))/criteria.force_scale_n)
        require(np.isfinite(eq+ncp),'finite independent path reference metrics')
        trace.append(dict(iteration=iteration,equilibrium=eq,NCP_L1=ncp,positive_modes=np.flatnonzero(z>0).tolist()))
        if progress:progress(dict(kind='independent_response',trace=trace))
        if eq<=1e-6 and ncp<=1e-8 and np.all(z>=0):
            return dict(converged=True,coordinates=u,nodal_forces=z,trace=trace,oracle=oracle,
                gap_gradient=g,initialization='ZERO_FREE_COORDINATES_ZERO_FORCES_AT_ACTUAL_PRESCRIBED_LIFT',
                active_mode_rule='ALL25_GENERAL_NCP_NO_FIXED_NODE',max_iterations=18,max_halvings=8)
        if iteration==18:break
        active=z-scales*gap>0;G=g @ t;lower=np.zeros_like(G);lower[active]=scales[active,None]*G[active]
        matrix=np.block([[t.T @ k @ t,-G.T],[lower,np.diag((~active).astype(float))]])
        try:delta=finite(np.linalg.solve(matrix,-np.r_[r,c]),(n+m,))
        except np.linalg.LinAlgError as exc:raise ReferenceRefused('independent reference linear system singular') from exc
        merit=max(eq/1e-6,ncp/1e-8)
        for half in range(9):
            rq=reduced+2.**(-half)*delta[:n];rz=z+2.**(-half)*delta[n:]
            trial,trial_oracle,_=evaluate(rq,rz,False)
            # A trial needs its OWN independent G for physical equilibrium;
            # never freeze the prior normalized-gap/contact force derivative.
            tg=gap_gradient(trial_oracle,trial,cancel=cancel)
            tf,_,_=system.structural_response(trial,cancel=cancel)
            rr=t.T @ (tf-tg.T @ rz-load)
            cc=rz-np.maximum(0,rz-scales*trial_oracle.base_value['gap'])
            ee=float(np.linalg.norm(rr)/criteria.force_scale_n);nn=float(np.sum(np.abs(cc))/criteria.force_scale_n)
            require(np.isfinite(ee+nn),'finite independent reference trial')
            if max(ee/1e-6,nn/1e-8)<merit:
                reduced,z=rq,rz;break
        else:raise ReferenceRefused('independent reference backtracking exhausted')
    return dict(converged=False,trace=trace,max_iterations=18,max_halvings=8,
                initialization='ZERO_FREE_COORDINATES_ZERO_FORCES_AT_ACTUAL_PRESCRIBED_LIFT')


def assess_endpoint(owner,*,output,cancel=lambda:False):
    """Complete current AND new-reference original predicate families."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    payload=json.loads(owner.checkpoint());row=payload['records'][-1];system=owner._last_system
    require(system is not None and system.checkpoint().decode()==payload['terminal_step_checkpoint'],'actual accepted-step source/endpoint correspondence')
    model=owner.model;definitions=tuple(p.definition for p in system.pairs)
    u=finite(row['coordinates'],system.offset.shape);z=finite(row['nodal_forces'],system.reference_support.shape)
    force=finite(row['contact_force'],system.offset.shape);load=owner.load_at(row['step_index'])
    result=dict(status='PARTIAL',gates={name:None for name in ('current_equilibrium','current_NCP_global_gap',
        'physical_wrench','independent_integration','full364_current_work','independent_current_KKT','independent_response')},
        coordinate_count=len(u),free_count=system.transform.shape[1],pressure_count=len(z),scientific_acceptance=False)
    def progress(event):publish(output/'PROGRESS.json',event)
    def gate(name,predicate,**evidence):
        result.update(evidence);result['gates'][name]=bool(predicate)
        result['status']='PARTIAL' if predicate else 'NONPASS';publish(output/'ASSESSMENT_PARTIAL.json',result)
        return bool(predicate)
    internal,_,_=system.structural_response(u,cancel=cancel)
    q,_=np.linalg.qr(system.transform,mode='reduced')
    eq=float(np.linalg.norm(q.T @ (internal+force-load))/SCALE)
    if not gate('current_equilibrium',eq<=1e-6,current_equilibrium=eq):return result
    oracle=ContactOracle(model,definitions,u,z,8,cancel=cancel)
    c=z-np.maximum(0,z-reference_scales(oracle,system.normal_stiffness)*oracle.base_value['gap'])
    gap=global_gap_certificate(oracle,u);ncp=float(np.sum(np.abs(c))/owner.criteria.force_scale_n)
    if not gate('current_NCP_global_gap',ncp<=1e-8 and gap['lower_m']>=-50e-6,NCP=ncp,gap_certificate=gap):return result
    if not gate('physical_wrench',max(row['force_balance'],row['moment_balance'])<=1e-6,wrench=dict(force=row['force_balance'],moment=row['moment_balance'])):return result
    controls=[]
    for order in ORDERS:
        independent=ContactOracle(model,definitions,u,z,order,cancel=cancel)
        sf,sm=rigid_wrench(independent,'source');tf,tm=rigid_wrench(independent,'target')
        support=float(np.max(np.abs(independent.base_value['support']-row['support'])/independent.base_value['support']))
        error=max(float(np.linalg.norm(sf-row['source_force'])/SCALE),float(np.linalg.norm(sm-row['source_moment'])/SCALE),
                  float(np.linalg.norm(tf-row['target_force'])/SCALE),float(np.linalg.norm(tm-row['target_moment'])/SCALE))
        controls.append(dict(order=order,support=support,wrench=error))
    if not gate('independent_integration',all(v['support']<=1e-11 and v['wrench']<=1e-6 for v in controls),integration=controls):return result
    work=full_work(model,system,definitions,u,z,force,cancel=cancel,progress=progress)
    if not gate('full364_current_work',work['predicate'],current_work=work):return result
    # Original disposable probe is a CONTROL REACTION, never a changed z.
    # Eliminating controls into an affine lift leaves contact W(u,z) unchanged.
    multipliers=np.asarray(row['control_reactions_N']);mu_probe=multipliers.copy();mu_probe[0]+=.00018
    rbase=system._base_transform.T @ (internal+force-load)
    reaction_delta=system._control_reduced.T @ (mu_probe-multipliers)
    result['original_pre_Newton_control_probe']=dict(offset_N=.00018,normal_forces_unchanged=True,
        accepted_checkpoint_unchanged=system.checkpoint().decode()==payload['terminal_step_checkpoint'],
        physical_pose_unchanged=np.array_equal(u,row['coordinates']),control_multipliers=multipliers.tolist(),
        disposable_control_multipliers=mu_probe.tolist(),base_residual_at_probe=(rbase-system._control_reduced.T @ mu_probe).tolist(),
        expected_residual_delta=(-reaction_delta).tolist(),profile='ELIMINATED_CONTROL_EQUIVALENT_KKT_NO_NEWTON_OR_COMMIT')
    G=np.array(work['records'][-1]['orders'][-1]['gap_gradient'])
    kkt=float(np.linalg.norm(q.T @ (internal-G.T @ z-load))/SCALE)
    if not gate('independent_current_KKT',kkt<=1e-6,independent_KKT=kkt):return result
    reference=independent_response(model,system,definitions,load,owner.criteria,cancel=cancel,progress=progress)
    if not reference['converged']:
        gate('independent_response',False,reference=reference);return result
    ru,rz=reference['coordinates'],reference['nodal_forces'];ro=reference.pop('oracle')
    reference.pop('gap_gradient')
    # The reference-endpoint WORK oracle is compared to the actual product
    # physical gradient there, never to the same FD used to solve reference.
    reference_raw=system.linearize(ru,rz,cancel=cancel)
    rf=reference_raw.force
    rw=full_work(model,system,definitions,ru,rz,rf,cancel=cancel,progress=progress,reuse={(3e-6,8):ro})
    reference.update(coordinates=ru.tolist(),nodal_forces=rz.tolist(),full_work=rw)
    if not rw['predicate']:
        gate('independent_response',False,reference=reference,later_reference_comparisons='UNRUN');return result
    reference_gap=global_gap_certificate(ro,ru)
    response=float(np.linalg.norm(ru-u)/max(np.linalg.norm(ru),np.finfo(float).tiny))
    force_error=norms((rz-z)/SCALE)
    contact_error=norms((rf-force)/SCALE)
    all_reference_force_controls=[norms((np.array(o['central'])-force)/SCALE) for r in rw['records'] for o in r['orders']]
    sf,_=rigid_wrench(ro,'source');physical_force=float(np.linalg.norm(sf-row['source_force'])/SCALE)
    target=owner.program.target_at(row['step_index'])
    scaled_control=float(np.max(np.abs(system._control_matrix @ ru[:system.physical_dofs]-target)/owner.control_scales))
    predicate=bool(reference_gap['lower_m']>=-50e-6 and response<=1e-6 and physical_force<=1e-6
        and scaled_control<=1e-6 and passed_norms(force_error,1e-6)
        and len(all_reference_force_controls)==4 and all(passed_norms(v,1e-6) for v in all_reference_force_controls))
    gate('independent_response',predicate,reference=reference,reference_gap=reference_gap,
        response_relative=response,all25_force_error=force_error,all364_contact_error=contact_error,
        complete_reference_contact_force_controls=all_reference_force_controls,
        physical_source_force=physical_force,reference_scaled_control=scaled_control)
    result['status']='PASS' if all(v is True for v in result['gates'].values()) else 'NONPASS'
    publish(output/'ASSESSMENT.json',result)
    return result
