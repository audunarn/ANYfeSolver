"""Separate conforming multi-target nonlinear physical profile, V16.

All physical 37-variable integrand and structural equations are protected
reuse. Only explicit multi-edge selection and affine control layout are new.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space
from anyfesolver.e4_pl_enriched import EnrichedProfileError,readonly
from .enriched_nonlinear import NonlinearEnrichedSystem
from .enriched_static import (AcceptedState,ContactState,check_cancel,digest,array_digest)
from .enriched_multitarget_topology import certify_disk
from ._enriched_multitarget_trial import AcceptedTrial

FORMULATION='E4_PL_PHYSICAL_B4_MULTITARGET_AFFINE_PATH_V16'


@dataclass(frozen=True)
class MultiTargetContactState(ContactState):
    physical_wrench: dict


def physical_wrench(rows,support,gap,z):
    sf,sm,tf,tm=(np.zeros(3) for _ in range(4))
    for pair,x,dm,db in rows:
        modes=np.asarray(pair.mode_indices)
        if np.any(support[modes]<=0):
            raise EnrichedProfileError('whole target disk lost supported physical hats')
        local=-np.sum((z[modes]/support[modes])[:,None]*(db-gap[modes,None]*dm),axis=0)
        fq,ft,fm=local[:12].reshape(4,3),local[12:21].reshape(3,3),local[25:].reshape(4,3)
        sf-=np.sum(fq+fm,axis=0);tf-=np.sum(ft,axis=0)
        sm-=np.sum(np.cross(x[:12].reshape(4,3),fq)+np.cross(x[25:].reshape(4,3),fm),axis=0)
        tm-=np.sum(np.cross(x[12:21].reshape(3,3),ft),axis=0)
    return dict(source_force=readonly(sf),source_moment=readonly(sm),target_force=readonly(tf),target_moment=readonly(tm),
        complete_pairs=len(rows),global_normalization=True,beta_cartesian_wrench_added=False)


class MultiTargetSystem(NonlinearEnrichedSystem):
    @classmethod
    def from_model(cls,model,definitions,*,normal_stiffness,quadrature,control_matrix):
        base=NonlinearEnrichedSystem.from_model(model,definitions,
            normal_stiffness=normal_stiffness,quadrature=quadrature)
        made=object.__new__(cls)
        for name,value in vars(base).items(): object.__setattr__(made,name,value)
        object.__setattr__(made,'normal_stiffness',float(normal_stiffness))
        c=readonly(control_matrix)
        if c.ndim!=2 or c.shape[1]!=base.physical_dofs or not len(c):
            raise EnrichedProfileError('immutable original nodal path-control rows required')
        beta=4*len(base.cells);nodal_free=base.transform.shape[1]-beta
        old=base.transform[:base.physical_dofs,:nodal_free]
        a=c @ old
        if np.linalg.matrix_rank(a)!=len(c):
            raise EnrichedProfileError('dependent/incompatible path-control rows')
        null=null_space(a)
        lift=old @ np.linalg.pinv(a)
        t=np.zeros((base.offset.size,null.shape[1]+beta))
        t[:base.physical_dofs,:null.shape[1]]=old @ null
        t[base.physical_dofs:,null.shape[1]:]=np.eye(beta)
        for name,value in (('transform',readonly(t)),('_base_offset',base.offset),
            ('_control_matrix',c),('_control_lift',readonly(lift)),('_start',None),
            ('_base_transform',base.transform),('_control_reduced',readonly(np.column_stack((a,np.zeros((len(c),beta))))))):
            object.__setattr__(made,name,value)
        descriptor=base.definition
        descriptor.update(profile=FORMULATION,normal_stiffness=float(normal_stiffness),immutable_control_rows=array_digest(c),
            transform=array_digest(t),affine_lift=array_digest(lift),
            target_union='ORIENTED_CONFORMING_PL_DISK_PHYSICAL_WORK_TRACES',
            tie_matrix='SELECTED_PIECEWISE_NEWTON_NOT_SMOOTH_HESSIAN')
        object.__setattr__(made,'_definition_json',json.dumps(descriptor,sort_keys=True,allow_nan=False))
        object.__setattr__(made,'identity',digest(descriptor))
        object.__setattr__(made,'_template_identity',made.identity)
        return made

    def at(self,values,*,start=None):
        """A new private step owner; never mutate a fixed owner's RHS/identity."""
        self._nodal.verify()
        rhs=readonly(values,(len(self._control_matrix),))
        offset=np.array(self._base_offset,copy=True)
        offset[:self.physical_dofs]+=self._control_lift @ (rhs-self._control_matrix @ offset[:self.physical_dofs])
        scales=np.maximum(1e-9,np.abs(rhs))
        if np.max(np.abs(self._control_matrix @ offset[:self.physical_dofs]-rhs)/scales)>1e-6:
            raise EnrichedProfileError('path-control affine lift failed original scaled criterion')
        made=object.__new__(type(self))
        for name,value in vars(self).items(): object.__setattr__(made,name,value)
        object.__setattr__(made,'offset',readonly(offset))
        if start is not None:
            reduced=readonly(start.reduced,(made.transform.shape[1],))
            force=readonly(start.nodal_forces,made.reference_support.shape)
            start=(reduced,force)
        object.__setattr__(made,'_start',start)
        object.__setattr__(made,'_accepted',None)
        object.__setattr__(made,'_busy',False)
        descriptor=self.definition
        descriptor.update(template_identity=self._template_identity,control_values=rhs.tolist(),offset=array_digest(offset))
        object.__setattr__(made,'_definition_json',json.dumps(descriptor,sort_keys=True,allow_nan=False))
        object.__setattr__(made,'identity',digest(descriptor))
        return made

    def _pair_maps_and_certificates(self,u):
        maps=[p.map(u) for p in self.pairs]
        groups={}
        for i,pair in enumerate(self.pairs): groups.setdefault(pair.definition.source_element,[]).append(i)
        certificates={}
        for source,indices in groups.items():
            if len(indices)==1: continue
            rows=certify_disk(tuple(self.pairs[i] for i in indices),tuple(maps[i] for i in indices))
            for i,certificate in zip(indices,rows): certificates[i]=certificate
        return maps,certificates

    def _mixed_direction(self,reduced,z,raw,residual,complementarity,jacobian,*,iteration,counts,cancel):
        # Same certified optional cold direction law, now explicitly this profile.
        if iteration==0 and self._accepted is None and self._start is None and np.all(reduced==0) and np.all(z==0):
            supported=raw.support>0
            if np.any(supported) and np.all(raw.normalized_gap[supported]>=0):
                delta,_=self._cold_direction(reduced,z,raw,residual,jacobian,counts=counts,cancel=cancel)
                if delta is not None: return delta
        from .enriched_static import EnrichedStaticSystem
        return EnrichedStaticSystem._mixed_direction(self,reduced,z,raw,residual,complementarity,jacobian,
            iteration=iteration,counts=counts,cancel=cancel)


    def linearize(self,coordinates,nodal_forces,*,cancel=lambda:False):
        """Original physical observer; paired C0 first variation at internal ties.

        AD retains moving cut/domain chains. At an admitted tie the reported
        second derivative is only a selected piecewise Newton model; no smooth
        Hessian certificate is inferred. The residual/strict NCP gate is exact.
        """
        from .enriched_moment_batch import batch_moments as moments
        from .enriched_multitarget_moments import selected_batch_moments as selected_pair_moments
        self._nodal.verify()
        u=readonly(coordinates,self.offset.shape);z=readonly(nodal_forces,self.reference_support.shape)
        maps,certificates=self._pair_maps_and_certificates(u)
        n,m=u.size,z.size
        support,b=np.zeros(m),np.zeros(m)
        dm,db=np.zeros((m,n)),np.zeros((m,n))
        hm,hb=np.zeros((m,n,n)),np.zeros((m,n,n))
        gram=np.zeros((m,m));branches=[];count=0;local_rows=[]
        for index,(pair,mapping)in enumerate(zip(self.pairs,maps)):
            check_cancel(cancel)
            x,a,h=mapping
            args=(pair.reference_source,x[:12].reshape(4,3),x[12:21].reshape(3,3),
                  x[21:25],pair.reference_directors[0],self.quadrature)
            kwargs=dict(current_midsurface=x[25:37].reshape(4,3),cancel=cancel)
            if index in certificates:
                data=selected_pair_moments(*args,internal_edges=certificates[index]['internal_edges'],
                    target_node_ids=pair.target_node_ids,**kwargs)
                branches.append(('C0_UNION_SELECTED_MATRIX',data.branch))
            else:
                data=moments(*args,**kwargs);branches.append(('SMOOTH_PAIR',data.branch))
            count+=data.triangles
            local_rows.append((pair,x,data.support_gradient,data.gap_integral_gradient))
            ix=np.asarray(pair.indices);modes=np.asarray(pair.mode_indices)
            gram[np.ix_(modes,modes)]+=data.pressure_gram
            for local,mode in enumerate(modes):
                support[mode]+=data.support[local];b[mode]+=data.weighted_gap[local]
                for g,hh,gradient,hessian in ((dm,hm,data.support_gradient[local],data.support_hessian[local]),
                        (db,hb,data.gap_integral_gradient[local],data.gap_integral_hessian[local])):
                    g[mode,ix]+=gradient @ a
                    hh[mode][np.ix_(ix,ix)]+=a.T @ hessian @ a+np.einsum('i,ijk->jk',gradient,h)
        if certificates and np.any(support<=0):
            raise EnrichedProfileError('certified whole union must retain all positive Q1 supports')
        supported=support>0;gap,pressure=np.zeros(m),np.zeros(m)
        gap[supported]=b[supported]/support[supported];pressure[supported]=z[supported]/support[supported]
        g,hg=np.zeros((m,n)),np.zeros((m,n,n))
        for i in np.flatnonzero(supported):
            g[i]=(db[i]-gap[i]*dm[i])/support[i]
            hg[i]=(hb[i]-gap[i]*hm[i]-np.outer(g[i],dm[i])-np.outer(dm[i],g[i]))/support[i]
        force=-g.T @ z;tangent=-np.einsum('i,ijk->jk',z,hg)
        wrench=physical_wrench(local_rows,support,gap,z)
        check_cancel(cancel);self._nodal.verify()
        return MultiTargetContactState(*[readonly(x)for x in (support,b,gap,pressure,g,force,tangent,gram)],tuple(branches),count,wrench)

    def solve(self, load, *, criteria, control, cancel=lambda: False):
        """Development NCP solve; exact physical observer, no global acceptance.

        Iterates and rejected trials are local.  Only a fully converged NCP
        candidate is published; cancellation/refusal leaves accepted identity
        and checkpoint bytes intact.  Physical-gap extrema/reference/pressure
        stability qualification remains an independent, mandatory open gate.
        """
        if self._busy:
            raise EnrichedProfileError("reentrant enriched transaction refused")
        criteria.validate()
        control.validate()
        load = readonly(load, self.offset.shape)
        self._busy = True
        try:
            check_cancel(cancel)
            if self._accepted is None:
                reduced = np.zeros(self.transform.shape[1]) if self._start is None else np.array(self._start[0])
                z = np.zeros(len(self.pressure_ids)) if self._start is None else np.array(self._start[1])
            else:
                if self._accepted.criteria != tuple(vars(criteria).values()):
                    raise EnrichedProfileError("accepted criteria identity changed")
                reduced = np.array(self._accepted.reduced)
                z = np.array(self._accepted.nodal_forces)
            counts = dict(evaluations=0, logical_base_evaluations=0, accepted_trial_reuses=0,
                          linear_solves=0, rejected_trials=0)
            pending = None
            for iteration in range(control.maximum_iterations+1):
                if pending is None:
                    u, raw, r, c, j = self.equations(reduced, z, load, cancel=cancel)
                    counts["evaluations"] += 1
                else:
                    reused = pending.consume(self, reduced, z, load, criteria, control, cancel)
                    pending = None
                    if reused is None:
                        reused = self.equations(reduced, z, load, cancel=cancel)
                        counts["evaluations"] += 1
                    else:
                        counts["accepted_trial_reuses"] += 1
                    u, raw, r, c, j = reused
                counts["logical_base_evaluations"] += 1
                eq, ncp = self.metrics(r, c, criteria)
                if eq <= criteria.equilibrium_relative and ncp <= criteria.complementarity_relative:
                    if np.any(z < 0):
                        raise EnrichedProfileError("negative converged nodal force")
                    check_cancel(cancel)
                    generation = 1 if self._accepted is None else self._accepted.generation+1
                    self._accepted = AcceptedState(readonly(u), readonly(reduced), readonly(z),
                                                   load, generation, tuple(vars(criteria).values()))
                    return dict(status="DEVELOPMENT_NCP_CONVERGED", criteria=vars(criteria),
                                equilibrium=eq, complementarity=ncp, counts=counts,
                                generation=generation, _raw=raw, physical_gap_extrema="UNVERIFIED",
                                pressure_stability="UNPROVEN", full_D1_acceptance=False)
                if iteration == control.maximum_iterations:
                    break
                check_cancel(cancel)
                delta = self._mixed_direction(reduced,z,raw,r,c,j,
                    iteration=iteration,counts=counts,cancel=cancel)
                if not np.all(np.isfinite(delta)):
                    raise EnrichedProfileError("nonfinite mixed Newton correction")
                merit = max(eq/criteria.equilibrium_relative, ncp/criteria.complementarity_relative)
                for half in range(control.maximum_halvings+1):
                    check_cancel(cancel)
                    alpha = 2.**(-half)
                    rq = reduced+alpha*delta[:reduced.size]
                    rz = z+alpha*delta[reduced.size:]
                    dependencies = AcceptedTrial.dependencies(self,rq,rz,load)
                    trial = self.equations(rq, rz, load, cancel=cancel)
                    _, _, rr, cc, _ = trial
                    counts["evaluations"] += 1
                    ee, nn = self.metrics(rr, cc, criteria)
                    if max(ee/criteria.equilibrium_relative, nn/criteria.complementarity_relative) < merit:
                        pending = AcceptedTrial.capture(self, rq, rz, load, criteria, control, trial, cancel, dependencies)
                        reduced, z = rq, rz
                        break
                    counts["rejected_trials"] += 1
                else:
                    raise EnrichedProfileError("mixed backtracking exhausted")
            raise EnrichedProfileError("mixed iteration budget exhausted")
        finally:
            self._busy = False

    def checkpoint(self):
        if self._accepted is None:
            raise EnrichedProfileError('no accepted nonlinear enriched state')
        self._nodal.verify()
        a=self._accepted
        payload=dict(profile=FORMULATION,definition=self.identity,coordinates=a.coordinates.tolist(),
            nodal_forces=a.nodal_forces.tolist(),reduced=a.reduced.tolist(),criteria=list(a.criteria),
            load=a.load.tolist(),generation=a.generation)
        return json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

    def restore(self,checkpoint,*,criteria,cancel=lambda:False):
        if self._busy or type(checkpoint)is not bytes or len(checkpoint)>2*1024*1024:
            raise EnrichedProfileError('invalid nonlinear checkpoint/transaction envelope')
        criteria.validate()
        try:
            p=json.loads(checkpoint)
            canonical=json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        except (ValueError,UnicodeError) as exc:
            raise EnrichedProfileError('malformed nonlinear enriched checkpoint') from exc
        keys={'profile','definition','coordinates','nodal_forces','reduced','criteria','load','generation'}
        if (type(p)is not dict or set(p)!=keys or p['profile']!=FORMULATION or p['definition']!=self.identity
                or type(p['generation'])is not int or p['generation']<1
                or p['criteria']!=list(vars(criteria).values()) or canonical!=checkpoint):
            raise EnrichedProfileError('wrong nonlinear enriched profile/definition/layout')
        u=readonly(p['coordinates'],self.offset.shape)
        reduced=readonly(p['reduced'],(self.transform.shape[1],))
        z=readonly(p['nodal_forces'],self.reference_support.shape)
        load=readonly(p['load'],self.offset.shape)
        if not np.array_equal(self.offset+self.transform @ reduced,u):
            raise EnrichedProfileError('nonlinear checkpoint violates original constraints')
        self._busy=True
        try:
            _,_,r,c,_=self.equations(reduced,z,load,cancel=cancel)
            eq,ncp=self.metrics(r,c,criteria)
            if eq>criteria.equilibrium_relative or ncp>criteria.complementarity_relative or np.any(z<0):
                raise EnrichedProfileError('nonlinear checkpoint physical residual admission failed')
            check_cancel(cancel)
            self._accepted=AcceptedState(u,reduced,z,load,p['generation'],tuple(vars(criteria).values()))
        finally:
            self._busy=False


# Exact built-in pure graph admitted for duplicate-value reuse. Unknown overrides
# keep complete evaluation; no arbitrary __wrapped__ metadata is trusted.
from .enriched_nonlinear import CurrentNodalBaseline as _Baseline
from anyfesolver.e4_pl_enriched_nonlinear import NonlinearEnrichedCell as _Cell
from . import enriched_multitarget_moments as _Selected, enriched_moment_batch as _Batch
_REUSE_OPERATOR_FUNCTIONS=tuple((f,f.__code__) for f in (
    MultiTargetSystem.equations,MultiTargetSystem.structural_response,
    MultiTargetSystem.linearize,MultiTargetSystem.metrics,MultiTargetSystem._mixed_direction,
    _Baseline.verify,_Baseline.evaluate,_Cell.response,
    _Selected.selected_batch_moments,_Batch.batch_moments))
