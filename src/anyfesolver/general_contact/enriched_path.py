"""Versioned public force/displacement path and complete atomic state, V16."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
import numpy as np
from .enriched_multitarget import MultiTargetSystem,FORMULATION
from .enriched_multitarget_gap import global_gap_certificate
from .enriched_static import PairDefinition,MixedCriteria,IterationControl,check_cancel,digest
from anyfesolver.e4_pl_enriched import EnrichedProfileError,readonly

PATH_PROFILE='ENRICHED_MULTITARGET_CONTACT_PATH_V16'
DOF={'ux':0,'uy':1,'uz':2,'rx':3,'ry':4,'rz':5}
_DUMPS=json.dumps


def canonical(value):
    return _DUMPS(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class PhysicalPose:
    """Physical37 compiled affine field, independent of moment quadrature."""
    def __init__(self,system,cancel=lambda:False):
        self.system,self.cancel=system,cancel
    def verify(self):
        check_cancel(self.cancel);self.system._nodal.verify()
    def vertices(self,pose):
        self.verify();u=readonly(pose,self.system.offset.shape)
        sources,targets={},{}
        for pair in self.system.pairs:
            x,_,_=pair.map(u)
            sid,tid=pair.definition.source_element,pair.definition.target_element
            if sid not in sources:
                mid=x[25:37].reshape(4,3)
                normal=np.cross(mid[1]-mid[0],mid[3]-mid[0]);normal/=np.linalg.norm(normal)
                sources[sid]=(x[:12].reshape(4,3),x[21:25],normal)
            if tid not in targets:
                actual=tuple(self.system._nodal.model.mesh.elements[tid].node_ids)
                bynode={node:x[12+3*j:15+3*j] for j,node in enumerate(pair.target_node_ids)}
                targets[tid]=np.array([bynode[n] for n in actual])
        self.verify();return sources,targets


class EnrichedPathOwner:
    """Immutable program/rows; private step owners with varying RHS/lift.

    Only accepted complete checkpoint bytes are published. The trusted prefix
    hash binds earlier history; terminal residual admission alone cannot
    authenticate arbitrary caller-supplied earlier poses.
    """
    def __setattr__(self,name,value):
        if hasattr(self,name) and name not in ('_publication','_busy'):
            raise EnrichedProfileError('compiled enriched path/program is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,model,definition,program,*,quadrature,force_scale_n,
                 physical_force_scale_n=.018,control=IterationControl(18,8)):
        from .static_path import ContactPathProgram
        from .core import ContactDefinition,ContactSurface,ContactInteraction
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
        from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
        from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
        if type(program) is not ContactPathProgram or program.control_profile!='translation_v1':
            raise EnrichedProfileError('original immutable translation ContactPathProgram required')
        self.program=program
        self.program_bytes=canonical(program.contract())
        self.model=model
        n=model.mesh.dof_manager.total_dofs
        if (type(definition) is not ContactDefinition or len(definition.surfaces)!=2
            or len(definition.interactions)!=1 or definition.rigid_bodies
            or any(type(s) is not ContactSurface or s.beam_radius is not None or s.beam_end_caps!='none' for s in definition.surfaces)):
            raise EnrichedProfileError('one original physical source/target interaction required')
        interaction=definition.interactions[0]
        if (type(interaction) is not ContactInteraction or interaction.normal_law!='augmented'
            or interaction.normal_stiffness is None
            or interaction.line_stiffness is not None or interaction.endpoint_stiffness is not None
            or interaction.gap_tolerance!=50e-6):
            raise EnrichedProfileError('original augmented normal law/50um gap/no line or endpoint law required')
        surfaces={s.name:s for s in definition.surfaces}
        if interaction.surface not in surfaces or interaction.target not in surfaces or interaction.surface==interaction.target:
            raise EnrichedProfileError('actual named source and deformable target participants required')
        source,target=surfaces[interaction.surface],surfaces[interaction.target]
        if (any(sid not in model.mesh.elements or type(model.mesh.elements[sid])is not QualifiedE4PLShellElement for sid in source.element_ids)
            or any(tid not in model.mesh.elements or type(model.mesh.elements[tid])not in
                (QualifiedE4PLS3ShellElement,NativeParityE4PLS3V2DShellElement) for tid in target.element_ids)):
            raise EnrichedProfileError('exact original Q4 source and supported elastic physical S3 targets required')
        self.contact_definition=definition
        self.definition_bytes=canonical(asdict(definition))
        sides={'bottom':-1.,'midsurface':0.,'top':1.}
        if source.side not in sides or target.side not in sides:
            raise EnrichedProfileError('original physical contact sides required')
        pairs=[]
        for sid in source.element_ids:
            element=model.mesh.elements[sid]
            so=sides[source.side]*element.thickness/2
            for tid in target.element_ids:
                other=model.mesh.elements[tid]
                to=other.material_surface_offset_from_reference(sides[target.side])
                pairs.append(PairDefinition(sid,tid,float(so),float(to)))
        c=np.zeros((len(program.controls),n))
        for row,control_row in enumerate(program.controls):
            if control_row.coordinate_profile!='translation_v1':
                raise EnrichedProfileError('original additive translational controls required')
            for node,name,weight in control_row.terms:
                c[row,model.mesh.dof_manager.get_node_dofs(node)[DOF[name]]]+=weight
        self.template=MultiTargetSystem.from_model(model,pairs,normal_stiffness=definition.interactions[0].normal_stiffness,
            quadrature=quadrature,control_matrix=c)
        self.criteria=MixedCriteria(float(force_scale_n),1e-6,1e-8);self.criteria.validate()
        control.validate();self.control=control
        self.physical_scale=float(physical_force_scale_n)
        if not np.isfinite(self.physical_scale) or self.physical_scale<=0:
            raise EnrichedProfileError('positive original physical force/moment scale required')
        self.control_scales=readonly(np.maximum(1e-9,np.max(np.abs(np.asarray(program.waypoints)),axis=0)))
        self.identity=digest(dict(profile=PATH_PROFILE,physical_profile=FORMULATION,template=self.template.identity,
            program=program.contract(),contact_definition=asdict(definition),criteria=vars(self.criteria),iterations=vars(control),
            physical_force_moment_scale=self.physical_scale,control_scales=self.control_scales.tolist()))
        self._publication=None;self._busy=False

    @property
    def _state(self): return None if self._publication is None else self._publication[0]

    @property
    def _last_system(self): return None if self._publication is None else self._publication[1]

    @property
    def _imported_prefix(self): return None if self._publication is None else self._publication[2]

    def verify(self):
        self.template._nodal.verify()
        if canonical(self.program.contract())!=self.program_bytes:
            raise EnrichedProfileError('path program/schedule changed')
        if canonical(asdict(self.contact_definition))!=self.definition_bytes:
            raise EnrichedProfileError('path contact definition/participant/law changed')

    def load_at(self,step):
        load=np.zeros(self.template.offset.size)
        for force,value in zip(self.program.nodal_forces,self.program.force_values_at(step),strict=True):
            load[self.model.mesh.dof_manager.get_node_dofs(force.node_id)[DOF[force.dof]]]+=value
        return readonly(load)

    @property
    def accepted_step(self):
        return 0 if self._state is None else json.loads(self._state)['accepted_step']

    def checkpoint(self):
        self.verify()
        if self._state is None: raise EnrichedProfileError('no accepted public path state')
        return self._state

    def advance(self,*,cancel=lambda:False,before_publish=None):
        if self._busy: raise EnrichedProfileError('reentrant enriched path transaction')
        self._busy=True
        try:
            self.verify();check_cancel(cancel)
            old=None if self._state is None else json.loads(self._state)
            step=1 if old is None else old['accepted_step']+1
            if step>self.program.total_steps: raise EnrichedProfileError('path already completed')
            target=self.program.target_at(step)
            start=None
            if old is not None:
                from .enriched_static import AcceptedState
                prior=old['records'][-1]
                start=AcceptedState(readonly(prior['coordinates']),readonly(prior['reduced']),
                    readonly(prior['nodal_forces']),readonly(prior['load']),step-1,tuple(vars(self.criteria).values()))
            system=self.template.at(target,start=start)
            result=system.solve(self.load_at(step),criteria=self.criteria,control=self.control,cancel=cancel)
            raw=result.pop('_raw');accepted=system._accepted
            error=float(np.max(np.abs(system._control_matrix @ accepted.coordinates[:system.physical_dofs]-target)/self.control_scales))
            if not np.isfinite(error) or error>1e-6: raise EnrichedProfileError('original scaled path-control predicate failed')
            wrench=raw.physical_wrench
            force=float(np.linalg.norm(wrench['source_force']+wrench['target_force'])/self.physical_scale)
            moment=float(np.linalg.norm(wrench['source_moment']+wrench['target_moment'])/self.physical_scale)
            if not np.isfinite(force+moment) or max(force,moment)>1e-6:
                raise EnrichedProfileError('original physical path-wrench predicate failed')
            internal,_,_=system.structural_response(accepted.coordinates,cancel=cancel)
            base_residual=system._base_transform.T @ (internal+raw.force-accepted.load)
            reactions,_,rank,_=np.linalg.lstsq(system._control_reduced.T,base_residual,rcond=None)
            old_kkt=float(np.linalg.norm(base_residual-system._control_reduced.T @ reactions)/self.physical_scale)
            if rank!=len(target) or not np.isfinite(old_kkt) or old_kkt>1e-6:
                raise EnrichedProfileError('original physical control-reaction KKT predicate failed')
            gap=global_gap_certificate(PhysicalPose(system,cancel),accepted.coordinates)
            states=system.recover_element_states(cancel=cancel)
            record=dict(step_index=step,coordinates=accepted.coordinates.tolist(),reduced=accepted.reduced.tolist(),
                nodal_forces=accepted.nodal_forces.tolist(),load=accepted.load.tolist(),target=target.tolist(),
                active_pressure_modes=int(np.count_nonzero(accepted.nodal_forces>0)),
                support=raw.support.tolist(),normalized_gap=raw.normalized_gap.tolist(),pressure=raw.pressure.tolist(),
                contact_force=raw.force.tolist(),
                control_scaled_error=error,equilibrium=result['equilibrium'],NCP=result['complementarity'],
                control_reactions_N=reactions.tolist(),base_KKT_relative=old_kkt,
                force_balance=force,moment_balance=moment,physical_gap_certificate=gap,
                source_force=wrench['source_force'].tolist(),source_moment=wrench['source_moment'].tolist(),
                target_force=wrench['target_force'].tolist(),target_moment=wrench['target_moment'].tolist(),
                element_states=states,counts=result['counts'],total_coordinates=system.offset.size,
                free_coordinates=system.transform.shape[1],pressure_ids=list(system.pressure_ids))
            records=[] if old is None else list(old['records'])
            records.append(record)
            payload=dict(profile=PATH_PROFILE,definition=self.identity,program_sha256=hashlib.sha256(self.program_bytes).hexdigest(),
                accepted_step=step,total_steps=self.program.total_steps,records=records,
                terminal_step_checkpoint=system.checkpoint().decode('utf-8'),scientific_acceptance=False)
            candidate=canonical(payload)
            if before_publish is not None: before_publish(candidate)
            check_cancel(cancel);self.verify()
            self._publication=(candidate,system,self._imported_prefix)
            return record
        finally:
            self._busy=False

    def run(self,*,through=None,cancel=lambda:False,before_publish=None):
        stop=self.program.total_steps if through is None else through
        if type(stop) is not int or not self.accepted_step<=stop<=self.program.total_steps:
            raise EnrichedProfileError('invalid public path terminal step')
        while self.accepted_step<stop: self.advance(cancel=cancel,before_publish=before_publish)
        return json.loads(self.checkpoint())

    def restore(self,prefix,*,expected_sha256,cancel=lambda:False):
        if self._busy or self._state is not None or type(prefix) is not bytes or len(prefix)>8*1024*1024:
            raise EnrichedProfileError('fresh owner and bounded authentic path-prefix bytes required')
        self._busy=True
        try:
            self.verify();check_cancel(cancel)
            if hashlib.sha256(prefix).hexdigest()!=expected_sha256:
                raise EnrichedProfileError('authentic prefix-byte binding mismatch')
            try: p=json.loads(prefix)
            except (ValueError,UnicodeError) as exc: raise EnrichedProfileError('malformed public path-prefix') from exc
            required={'profile','definition','program_sha256','accepted_step','total_steps','records','terminal_step_checkpoint','scientific_acceptance'}
            if (type(p) is not dict or set(p)!=required or canonical(p)!=prefix or p['profile']!=PATH_PROFILE
                or p['definition']!=self.identity or p['program_sha256']!=hashlib.sha256(self.program_bytes).hexdigest()
                or type(p['accepted_step']) is not int or not 1<=p['accepted_step']<=self.program.total_steps
                or p['total_steps']!=self.program.total_steps or p['scientific_acceptance'] is not False
                or type(p['records']) is not list or len(p['records'])!=p['accepted_step']):
                raise EnrichedProfileError('incompatible public path-prefix identity/layout')
            for index,row in enumerate(p['records'],1):
                if (row['step_index']!=index or not np.array_equal(readonly(row['load'],self.template.offset.shape),self.load_at(index))
                    or not np.array_equal(readonly(row['target'],self.control_scales.shape),self.program.target_at(index))):
                    raise EnrichedProfileError('prefix history/program correspondence failed')
                readonly(row['coordinates'],self.template.offset.shape)
                readonly(row['reduced'],(self.template.transform.shape[1],))
                readonly(row['nodal_forces'],self.template.reference_support.shape)
            system=self.template.at(self.program.target_at(p['accepted_step']))
            system.restore(p['terminal_step_checkpoint'].encode('utf-8'),criteria=self.criteria,cancel=cancel)
            terminal=p['records'][-1];a=system._accepted
            if (not np.array_equal(a.coordinates,terminal['coordinates']) or not np.array_equal(a.reduced,terminal['reduced'])
                or not np.array_equal(a.nodal_forces,terminal['nodal_forces']) or not np.array_equal(a.load,terminal['load'])
                or canonical(system.recover_element_states(cancel=cancel))!=canonical(terminal['element_states'])):
                raise EnrichedProfileError('prefix/terminal complete physical-state correspondence failed')
            global_gap_certificate(PhysicalPose(system,cancel),a.coordinates)
            check_cancel(cancel);self.verify()
            self._publication=(prefix,system,expected_sha256)
        except (KeyError,TypeError,ValueError) as exc:
            if isinstance(exc,EnrichedProfileError): raise
            raise EnrichedProfileError('malformed public path-prefix fields') from exc
        finally:
            self._busy=False
