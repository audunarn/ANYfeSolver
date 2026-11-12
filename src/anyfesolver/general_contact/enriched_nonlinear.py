"""Versioned elastic nonlinear enriched static owner, development only.

The nodal baseline is the ACTUAL current nonlinear assembly, including its
public S3 station/state dispatch. It is never replaced by a Q4 approximation.
The protected V2 physical contact geometry/NCP and all pressures are reused.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from scipy.linalg import null_space

from anyfesolver.e4_pl_enriched import EnrichedProfileError, readonly
from anyfesolver.e4_pl_enriched_nonlinear import FORMULATION, compile_nonlinear_cell
from anyfesolver.nonlinear_restart import model_configuration_fingerprint
from anyfesolver.nonlinear_static import (
    _prepare_initial_states, _prepare_qualified_q4_states_for_nonlinear_solve,
    _activate_nonlinear_state_storage, _assemble_nonlinear_system,
    _discard_nonlinear_state_candidate,
)
from .enriched_static import (
    EnrichedStaticSystem, AcceptedState, ContactState, CompiledPair, PairDefinition, MixedCriteria,
    IterationControl, check_cancel, digest, array_digest,
)

# Every map component is affine in the46 local coordinates. Immutable shared
# zero second derivative avoids32 duplicate37x46x46 arrays in matched174.
_AFFINE_MAP_HESSIAN=readonly(np.zeros((37,46,46)))


@dataclass(frozen=True)
class AffinePhysicalPair(CompiledPair):
    """Original matched skin/work map, including nonzero reference offsets.

    This is deliberately additive rotation, not V2's SO(3) offset map. The
    nodal physical positions and complete constant work rows match the actual
    Q4SkinWorkAdapter/S3SkinWorkAdapter at arbitrary admitted finite q.
    """
    target_node_ids: tuple

    def map(self,u):
        local=readonly(np.asarray(u)[np.asarray(self.indices)],(46,))
        points=[];a=np.zeros((37,46));h=_AFFINE_MAP_HESSIAN
        for node in range(7):
            offset=self.definition.source_offset if node<4 else self.definition.target_offset
            director=self.reference_directors[node]
            points.extend(self.reference_corners[node]+offset*director+local[6*node:6*node+3]
                          +offset*np.cross(local[6*node+3:6*node+6],director))
            a[3*node:3*node+3,6*node:6*node+3]=np.eye(3)
            for k in range(3):
                a[3*node:3*node+3,6*node+3+k]=offset*np.cross(np.eye(3)[k],director)
        points.extend(local[42:46]);a[21:25,42:46]=np.eye(4)
        for node in range(4):
            points.extend(self.reference_corners[node]+local[6*node:6*node+3])
            a[25+3*node:28+3*node,6*node:6*node+3]=np.eye(3)
        return np.asarray(points),a,h


def _compile_physical_definition(model,definitions,normal_stiffness,quadrature):
    """New-profile admission; physical directors/offsets are NOT chart normals.

    A target may have the same reference orientation as the source. Reverse
    only its integration vertex/DOF ordering, preserving its actual director
    and the caller's physical surface offset. No element or model is edited.
    """
    from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
    from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    from anyfesolver.e4_pl_enriched import compile_cell
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.assembly import build_constraint_transformation
    definitions=tuple(definitions)
    allowed=(QualifiedE4PLShellElement,QualifiedE4PLS3ShellElement,NativeParityE4PLS3V2DShellElement)
    if (not definitions or any(type(d)is not PairDefinition for d in definitions)
            or len({(d.source_element,d.target_element) for d in definitions})!=len(definitions)):
        raise EnrichedProfileError('nonduplicate explicit nonlinear Q4/S3 pairs required')
    for e in model.mesh.elements.values():
        if type(e)not in allowed:
            raise EnrichedProfileError('nonlinear enriched profile requires exact elastic Q4/public S3')
        m=model.materials[e.material_name]
        if (m.yield_stress!=0 or m.hardening_curve is not None or getattr(m,'j2_curve',None)is not None
                or getattr(e,'shell_section',None)is not None):
            raise EnrichedProfileError('whole model scalar elastic domain required')
        if type(e)is NativeParityE4PLS3V2DShellElement and (
                not e._is_v2c_elastic_overlap(m) or e.director_polarity!=1):
            raise EnrichedProfileError('exact public S3 elastic scalar domain required')
    source_ids=sorted({d.source_element for d in definitions})
    cells=tuple(compile_cell(model.mesh.elements[i],model.mesh,
        model.materials[model.mesh.elements[i].material_name]) for i in source_ids)
    n=model.mesh.dof_manager.total_dofs
    old=assemble_stiffness_matrix(model)[0]
    _,_,t,offset,_,_=build_constraint_transformation(old,np.zeros(n),model)
    size=n+4*len(cells)
    stiffness=np.zeros((size,size));stiffness[:n,:n]=old.toarray()
    indices=[]
    for ordinal,cell in enumerate(cells):
        ix=tuple(d for node in cell.node_ids for d in model.mesh.dof_manager.get_node_dofs(node))
        ix+=tuple(range(n+4*ordinal,n+4*ordinal+4));indices.append(ix)
        e=model.mesh.elements[cell.element_id]
        delta=np.array(cell.stiffness)
        delta[:24,:24]-=e.compute_stiffness_components(model.mesh,model.materials[e.material_name])['total']
        stiffness[np.ix_(ix,ix)]+=delta
    stiffness=(stiffness+stiffness.T)/2
    transform=np.zeros((size,t.shape[1]+4*len(cells)))
    transform[:n,:t.shape[1]]=t.toarray();transform[n:,t.shape[1]:]=np.eye(4*len(cells))
    pressure_ids=tuple(sorted({node for cell in cells for node in cell.node_ids}))
    mode={node:i for i,node in enumerate(pressure_ids)};support=np.zeros(len(mode));pairs=[]
    for cell in cells:
        area=np.linalg.norm(np.cross(cell.reference[1]-cell.reference[0],cell.reference[3]-cell.reference[0]))
        for node in cell.node_ids:support[mode[node]]+=area/4
    for d in definitions:
        if not np.all(np.isfinite((d.source_offset,d.target_offset))):
            raise EnrichedProfileError('finite actual physical offsets required')
        cell=cells[source_ids.index(d.source_element)];target=model.mesh.elements[d.target_element]
        source_element=model.mesh.elements[d.source_element]
        source_director=source_element.physical_reference_director
        source_director=np.asarray(cell.frame[:,2] if source_director is None else source_director,dtype=float)
        if type(target)not in (QualifiedE4PLS3ShellElement,NativeParityE4PLS3V2DShellElement):
            raise EnrichedProfileError('physical target must be exact qualified S3')
        target_nodes=tuple(target.node_ids)
        ref=np.asarray(target.get_node_coordinates(model.mesh),dtype=float)
        geometric=np.cross(ref[1]-ref[0],ref[2]-ref[0]);geometric/=np.linalg.norm(geometric)
        actual_director=getattr(target,'physical_reference_director',None)
        director=np.asarray(geometric if actual_director is None else actual_director,dtype=float)
        if director.shape!=(3,) or not np.all(np.isfinite(director)) or abs(np.linalg.norm(director)-1)>1e-10:
            raise EnrichedProfileError('finite actual unit target director required')
        if geometric @ cell.frame[:,2]>0:
            order=(0,2,1);ref=ref[list(order)];target_nodes=tuple(target_nodes[i] for i in order)
        if len(set((*cell.node_ids,*target_nodes)))!=7:
            raise EnrichedProfileError('source/target physical nodes must be disjoint')
        ix=tuple(dof for node in (*cell.node_ids,*target_nodes) for dof in model.mesh.dof_manager.get_node_dofs(node))
        ix+=indices[source_ids.index(d.source_element)][24:]
        pairs.append(AffinePhysicalPair(d,ix,readonly(np.vstack((cell.reference,ref))),
            readonly(np.array([source_director]*4+[director]*3)),
            readonly(cell.reference+d.source_offset*source_director),tuple(mode[node] for node in cell.node_ids),target_nodes))
    return EnrichedStaticSystem(stiffness=stiffness,transform=transform,
        offset=np.concatenate((offset,np.zeros(4*len(cells)))),physical_dofs=n,cells=cells,
        cell_indices=indices,pairs=pairs,pressure_ids=pressure_ids,reference_support=support,
        normal_stiffness=normal_stiffness,quadrature=quadrature,
        element_formulations=[dict(element_id=int(i),class_module=type(e).__module__,
            class_name=type(e).__qualname__,formulation_id=e.formulation_id,
            reference_surface_offset=float(getattr(e,'reference_surface_offset',0.)))
            for i,e in sorted(model.mesh.elements.items())])


class CurrentNodalBaseline:
    """One fixed unloaded elastic history; every trial is discarded, not committed."""
    def __init__(self, model):
        self.model = model
        self.fingerprint = model_configuration_fingerprint(model)
        self.elements = tuple((int(i),e) for i,e in model.mesh.elements.items())
        n = model.mesh.dof_manager.total_dofs
        zero,info = np.zeros(n),{}
        states,_ = _prepare_initial_states(model,None,None,5)
        states = _prepare_qualified_q4_states_for_nonlinear_solve(model,zero,states,5,info,
            supplied_element_ids=(),ordinary_restart=False)
        self.states = _activate_nonlinear_state_storage(model,states,5,info,
            kinematics='von_karman',committed_displacements=zero)

    def verify(self):
        if (model_configuration_fingerprint(self.model)!=self.fingerprint
                or tuple(self.model.mesh.elements)!=tuple(i for i,_ in self.elements)
                or any(self.model.mesh.elements[i] is not e for i,e in self.elements)):
            raise EnrichedProfileError('nonlinear compiled model definition changed')

    def evaluate(self, coordinates, *, cancel=lambda:False):
        from .runtime import active_contact_manager
        self.verify()
        if active_contact_manager() is not None:
            raise EnrichedProfileError('nested contact manager would double-count physical contact')
        check_cancel(cancel)
        q = readonly(coordinates,(self.model.mesh.dof_manager.total_dofs,))
        try:
            f,k,_ = _assemble_nonlinear_system(self.model,q,self.states,5,tangent=True,
                kinematics='von_karman',require_full_coordinates=True)
            force = readonly(np.asarray(f).reshape(-1),q.shape)
            tangent = readonly(k.toarray() if hasattr(k,'toarray') else k,(q.size,q.size))
            check_cancel(cancel)
            self.verify()
            return force,tangent
        finally:
            _discard_nonlinear_state_candidate(self.states)


class NonlinearEnrichedSystem(EnrichedStaticSystem):
    """All original nodal coordinates, all explicit beta and original Q1 z.

    The inherited immutable ``stiffness`` is a reference-linear layout input,
    not this profile's response operator. Use structural_response(q) for the
    actual force/tangent. equations() never applies the stored linear matrix.
    """
    @classmethod
    def from_model(cls,model,definitions,*,normal_stiffness,quadrature,
                   control_matrix=None,control_values=None):
        # Original typed whole-model elastic admission, exact public S3,
        # constraints, geometry, pressure identities and reference supports.
        base = _compile_physical_definition(model,definitions,normal_stiffness,quadrature)
        made = object.__new__(cls)
        for name,value in vars(base).items():
            object.__setattr__(made,name,value)
        if control_matrix is not None:
            c = readonly(control_matrix)
            if c.ndim!=2 or c.shape[1]!=base.physical_dofs:
                raise EnrichedProfileError('exact original nodal control rows required')
            values = readonly(control_values,(c.shape[0],))
            # This bounded one-press route retains the ORIGINAL zero controls.
            if np.any(values!=0) or np.any(c @ base.offset[:base.physical_dofs]!=values):
                raise EnrichedProfileError('nonzero path controls are not admitted by nonlinear V4')
            nodal_columns = base.transform.shape[1]-4*len(base.cells)
            old_t = base.transform[:base.physical_dofs,:nodal_columns]
            null = null_space(c @ old_t)
            t = np.zeros((base.offset.size,null.shape[1]+4*len(base.cells)))
            t[:base.physical_dofs,:null.shape[1]] = old_t @ null
            t[base.physical_dofs:,null.shape[1]:] = np.eye(4*len(base.cells))
            object.__setattr__(made,'transform',readonly(t))
            control_descriptor=dict(matrix=array_digest(c),values=values.tolist(),
                                    nodal_free=int(null.shape[1]))
        else:
            if control_values is not None:
                raise EnrichedProfileError('control values require their actual control rows')
            control_descriptor=None
        cells = tuple(compile_nonlinear_cell(model.mesh.elements[c.element_id],model.mesh,
            model.materials[model.mesh.elements[c.element_id].material_name]) for c in base.cells)
        baseline = CurrentNodalBaseline(model)
        object.__setattr__(made,'_nodal',baseline)
        object.__setattr__(made,'_nonlinear_cells',cells)
        descriptor = base.definition
        descriptor.update(profile=FORMULATION,structural_kinematics='ACTUAL_NODAL_PLUS_PSI43_DIFFERENCE',
            stored_stiffness_scope='REFERENCE_LINEAR_LAYOUT_ONLY_NOT_NONLINEAR_RESPONSE',
            physical_offset_kinematics='ORIGINAL_AFFINE_OFFSET_EXACT_WORKMAP',
            transform=array_digest(made.transform),controls=control_descriptor,
            actual_model_fingerprint=baseline.fingerprint,
            nonlinear_cells=[c.identity for c in cells],
            committed_material_history='UNLOADED_ELASTIC_FIXED; EVERY_TRIAL_DISCARDED')
        object.__setattr__(made,'identity',digest(descriptor))
        object.__setattr__(made,'_definition_json',json.dumps(descriptor,sort_keys=True,allow_nan=False))
        return made

    def _mixed_direction(self,reduced,z,raw,residual,complementarity,jacobian,
                         *,iteration,counts,cancel):
        # Only the exact cold nonlinear profile uses the predictor. Warm,
        # subsequent-iteration and subclass routes retain original Newton.
        cold=(type(self)is NonlinearEnrichedSystem and iteration==0
            and self._accepted is None and np.all(reduced==0)and np.all(z==0))
        if not cold:
            return super()._mixed_direction(reduced,z,raw,residual,complementarity,jacobian,
                iteration=iteration,counts=counts,cancel=cancel)
        supported=raw.support>0
        if not(np.any(supported)and np.all(raw.normalized_gap[supported]>=0)):
            return super()._mixed_direction(reduced,z,raw,residual,complementarity,jacobian,
                iteration=iteration,counts=counts,cancel=cancel)
        delta,_report=self._cold_direction(reduced,z,raw,residual,jacobian,
            counts=counts,cancel=cancel)
        if delta is not None:return delta
        return super()._mixed_direction(reduced,z,raw,residual,complementarity,jacobian,
            iteration=iteration,counts=counts,cancel=cancel)

    def _cold_direction(self,reduced,z,raw,residual,jacobian,*,counts,cancel):
        """Private diagnostic result; no accepted-state or model publication."""
        from .enriched_cold_predictor import cold_unilateral_direction,ColdPredictorDeclined
        self._nodal.verify()
        n=reduced.size;supported=raw.support>0
        h=jacobian[:n,:n]
        g=raw.gap_gradient[supported] @ self.transform
        try:
            d,p,certificate=cold_unilateral_direction(h,g,raw.normalized_gap[supported],
                residual,z[supported],counts=counts,cancel=cancel)
        except ColdPredictorDeclined as decline:
            # Preserve original Newton outside certified algebraic domain.
            # A changed physical model or cancellation can NEVER become fallback.
            self._nodal.verify();check_cancel(cancel)
            return None,dict(status='DECLINED',reason=str(decline),supported_modes=int(np.count_nonzero(supported)))
        predicted=np.zeros_like(z);predicted[supported]=p
        self._nodal.verify();check_cancel(cancel)
        return np.concatenate((d,predicted-z)),dict(certificate,status='CERTIFIED',
            unsupported_forces_zero=True)

    def structural_response(self,coordinates,*,cancel=lambda:False):
        q = readonly(coordinates,self.offset.shape)
        self._nodal.verify()
        nodal_f,nodal_k = self._nodal.evaluate(q[:self.physical_dofs],cancel=cancel)
        force,tangent = np.zeros(q.size),np.zeros((q.size,q.size))
        force[:self.physical_dofs] = nodal_f
        tangent[:self.physical_dofs,:self.physical_dofs] = nodal_k
        enrichment_energy = 0.
        for cell,indices in zip(self._nonlinear_cells,self.cell_indices):
            check_cancel(cancel)
            ix = np.asarray(indices)
            value,f,k = cell.response(q[ix])
            enrichment_energy += value
            force[ix] += f
            tangent[np.ix_(ix,ix)] += k
        check_cancel(cancel)
        self._nodal.verify()
        return readonly(force),readonly(tangent),float(enrichment_energy)

    def recover_element_states(self,coordinates=None,*,cancel=lambda:False):
        if coordinates is None:
            if self._accepted is None:raise EnrichedProfileError('no accepted nonlinear state to recover')
            coordinates=self._accepted.coordinates
        q=readonly(coordinates,self.offset.shape);self._nodal.verify();states={}
        for cell,indices in zip(self._nonlinear_cells,self.cell_indices):
            check_cancel(cancel)
            states[str(cell.linear.element_id)]=dict(profile=FORMULATION,identity=cell.identity,
                coordinates=q[np.asarray(indices)].tolist(),
                stationary_amplitudes43=cell.recover(q[np.asarray(indices)]).tolist(),
                history='ELASTIC_ALGEBRAIC_FUNCTION_OF_COMPLETE_COORDINATES')
        check_cancel(cancel);self._nodal.verify()
        return states

    def _pair_maps_and_certificates(self,u):
        from .enriched_union import certify_union
        maps=[p.map(u)for p in self.pairs]
        groups={}
        for i,p in enumerate(self.pairs):groups.setdefault(p.definition.source_element,[]).append(i)
        certificates={}
        for source,indices in groups.items():
            if len(indices)==1:continue
            if len(indices)!=2:
                raise EnrichedProfileError('only one smooth target or certified paired target union admitted')
            i,j=indices
            cert=certify_union(self.pairs[i],self.pairs[j],maps[i],maps[j])
            certificates[i]=cert;certificates[j]=cert
        return maps,certificates

    def union_certificates(self,coordinates):
        self._nodal.verify()
        u=readonly(coordinates,self.offset.shape)
        _,certificates=self._pair_maps_and_certificates(u)
        return {str(self.pairs[i].definition.source_element):c for i,c in certificates.items()}

    def linearize(self,coordinates,nodal_forces,*,cancel=lambda:False):
        """Original physical observer; paired C0 first variation at internal ties.

        AD retains moving cut/domain chains. At an admitted tie the reported
        second derivative is only a selected piecewise Newton model; no smooth
        Hessian certificate is inferred. The residual/strict NCP gate is exact.
        """
        from .enriched_moment_batch import batch_moments as moments
        from .enriched_moment_batch import selected_batch_moments as selected_pair_moments
        self._nodal.verify()
        u=readonly(coordinates,self.offset.shape);z=readonly(nodal_forces,self.reference_support.shape)
        maps,certificates=self._pair_maps_and_certificates(u)
        n,m=u.size,z.size
        support,b=np.zeros(m),np.zeros(m)
        dm,db=np.zeros((m,n)),np.zeros((m,n))
        hm,hb=np.zeros((m,n,n)),np.zeros((m,n,n))
        gram=np.zeros((m,m));branches=[];count=0
        for index,(pair,mapping)in enumerate(zip(self.pairs,maps)):
            check_cancel(cancel)
            x,a,h=mapping
            args=(pair.reference_source,x[:12].reshape(4,3),x[12:21].reshape(3,3),
                  x[21:25],pair.reference_directors[0],self.quadrature)
            kwargs=dict(current_midsurface=x[25:37].reshape(4,3),cancel=cancel)
            if index in certificates:
                data=selected_pair_moments(*args,internal_edge_nodes=certificates[index]['shared_nodes'],
                    target_node_ids=pair.target_node_ids,**kwargs)
                branches.append(('C0_UNION_SELECTED_MATRIX',data.branch))
            else:
                data=moments(*args,**kwargs);branches.append(('SMOOTH_PAIR',data.branch))
            count+=data.triangles
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
        check_cancel(cancel);self._nodal.verify()
        return ContactState(*[readonly(x)for x in (support,b,gap,pressure,g,force,tangent,gram)],tuple(branches),count)

    def equations(self,reduced,z,load,*,cancel=lambda:False):
        reduced = readonly(reduced,(self.transform.shape[1],))
        load = readonly(load,self.offset.shape)
        z = readonly(z,self.reference_support.shape)
        u = self.offset+self.transform @ reduced
        internal,stiffness,_ = self.structural_response(u,cancel=cancel)
        raw = self.linearize(u,z,cancel=cancel)
        active = (raw.support>0)&(z-self.scales*raw.normalized_gap>0)
        c = np.where(active,self.scales*raw.normalized_gap,z)
        residual = self.transform.T @ (internal+raw.force-load)
        g = raw.gap_gradient @ self.transform
        lower = np.zeros_like(g)
        lower[active] = self.scales[active,None]*g[active]
        j = np.block([[self.transform.T @ (stiffness+raw.tangent) @ self.transform,-g.T],
                      [lower,np.diag((~active).astype(float))]])
        if not all(np.all(np.isfinite(x)) for x in (residual,c,j)):
            raise EnrichedProfileError('nonfinite nonlinear enriched equations')
        return u,raw,residual,c,j

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
