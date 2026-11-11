"""Physical-fibre retained work adapter for the separate contact analysis owner.

The family owns its real conjugate fibre returns and history validation. Its
24 internal coordinates remain in the coupled system; no element condensation
or generalized-section history conversion occurs here.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from threading import Lock

import numpy as np

from anyfesolver.control import cancellation_safe_point
from anyfesolver.fe_core import FEModel
from anyfesolver.constraint_audit import require_valid_constraints
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from anyfesolver._ge_beam3_retained_fibre_state import Context as FibreContext
from anyfesolver._ge_beam3_retained_nodal_loading import Program
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha

POLICY='PHYSICAL_FIBRE_RETAINED_CONTACT_CONTEXT_DEV_V1'


class NativeFibreContactContext:
    def __init__(self,native,program,*,max_coordinates=256):
        if (type(native) is not NativeBeamAnalysis or native._family!='PHYSICAL_FIBRE_NODAL'
                or type(program) is not Program):
            raise TypeError('exact physical-fibre native owner and nodal program required')
        native._guard();program.__post_init__();program.pattern.require(native.model.mesh)
        program.nodal_forces.require(native.model.mesh)
        if program.pattern.line.rows or program.pattern.couples:
            raise ValueError('physical-fibre contact currently requires pure spatial nodal dead forces')
        if type(max_coordinates) is not int or max_coordinates not in (256,512):
            raise ValueError('native fibre contact requires an admitted coordinate budget')
        self.max_coordinates=max_coordinates
        self.native,self.program=native,program
        self._token=None;self._cancel_active=False;self._lock=Lock()
        # The existing fibre family has a distinct retained container. Its
        # reference/material definitions come from the immutable native owner.
        model=FEModel('native-fibre-contact-retained')
        for node,value in sorted(native.model.mesh.nodes.items()):model.add_node(node,*value.coords())
        for eid,element in sorted(native.model.mesh.elements.items()):
            if type(element) is not NativeFibreStaticElement:
                raise ValueError('homogeneous exact physical-fibre definitions required')
            retained=NativeRetainedFibreElement(eid,element.node_ids,element.operator.reference,
                element.section,order=element.operator.order)
            model.add_element(eid,retained);model.materials[retained.material_name]=retained.section
        for boundary in native.model.boundary_conditions:model.add_boundary_condition(deepcopy(boundary))
        force=ForceProgram(program.targets,program.nodal_forces.rows,program.max_iterations,program.max_backtracks)
        self._backend=FibreContext(model,force,check=self._safe_point,max_coordinates=max_coordinates)
        # Preserve the existing construction-to-expiry resource contract. An
        # operation, pause or checkpoint replay never renews this allowance.
        self._started=self._backend.started
        self.support_identity=canonical(require_valid_constraints(native.model))
        self.length=max(float(np.linalg.norm(e.operator.reference.coordinates[-1]-
            e.operator.reference.coordinates[0])) for _,e in self.elements)
        self.initial=self._backend.initial
        self.identity=self._fingerprint()
        self.guard()

    def _safe_point(self):
        if self._cancel_active:
            cancellation_safe_point(self._token,'physical-fibre contact material candidate')

    def _fingerprint(self):
        layout=self._backend.layout
        values=dict(policy=POLICY,native=self.native.identity,program=self.program,
            backend=self._backend.identity,support=sha256(self.support_identity).hexdigest(),length=self.length,
            count=layout.count,nodal_count=layout.nodal_count,node_ids=layout.node_ids,
            free=layout.free,fixed=layout.fixed,nodes=layout.nodes,slots=layout.slots,
            scale=layout.scales,equilibrium=layout.equilibrium,compatibility=layout.compatibility,
            force=layout.force,node_index=layout.node_index,
            reference_positions=layout.reference_positions,reference_frames=layout.reference_frames,
            operator_ids=[probe.identity for probe in layout.probes],
            backend_program=self._backend.program,program_data=self._backend.program_data,
            initial=self.initial,backend_initial=self._backend.initial,layout_initial=layout.initial)
        if self.max_coordinates != 256:values['max_coordinates']=self.max_coordinates
        return sha(values)

    def guard(self):
        layout=self._backend.layout
        if type(self.max_coordinates) is not int or self.max_coordinates not in (256,512):
            raise ValueError('native fibre contact coordinate budget changed')
        if (self._backend.started!=self._started or
                len(layout.probes)!=len(layout.elements) or
                len(self._backend.probes)!=len(layout.probes) or
                any(probe is not element.operator for probe,(_,element) in zip(layout.probes,layout.elements)) or
                any(a is not b for a,b in zip(self._backend.probes,layout.probes))):
            raise ValueError('physical-fibre contact captured operator ownership or deadline changed')
        self._safe_point();self.native._guard();self._backend.guard()
        if (self.native._family!='PHYSICAL_FIBRE_NODAL' or self._fingerprint()!=self.identity or
                canonical(require_valid_constraints(self.native.model))!=self.support_identity):
            raise ValueError('physical-fibre contact context/model/program/layout changed')

    @property
    def elements(self):return self._backend.layout.elements
    @property
    def count(self):return self._backend.layout.count
    @property
    def nodal_count(self):return self._backend.layout.nodal_count
    @property
    def node_ids(self):return self._backend.layout.node_ids
    @property
    def nodes(self):return self._backend.layout.nodes
    @property
    def slots(self):return self._backend.layout.slots
    @property
    def scale(self):return self._backend.layout.scales
    @property
    def free(self):return self._backend.layout.free
    @property
    def equilibrium(self):return self._backend.layout.equilibrium
    @property
    def compatibility(self):return self._backend.layout.compatibility

    def make(self,data,*,decoded=False):
        self.guard();return self._backend.layout.make(data,decoded=decoded)

    def advance(self,state,step):
        self.guard();return self._backend.layout.advance(state,step)

    def step(self,state,residual,tangent):
        from anyfesolver._native_retained_step import retained_newton_step
        self.guard()
        result=retained_newton_step(self,state,residual,tangent,family='physical-fibre')
        self.guard();return result

    def histories(self,values,*,decoded=False):
        self.guard();return self._backend.history(values,decoded=decoded)

    def recover(self,mechanical,origins,histories,completed_targets):
        """Recover exact accepted fibre fields using this family's state owner."""
        from anyfesolver._ge_beam3_retained_fibre_state import State
        state=State(self.make(mechanical.descriptor()),self.histories(origins),
            self.histories(histories),completed_targets,self._backend.identity)
        return self._backend.recover(state)

    def assemble(self,mechanical,parameter,origins):
        self.guard()
        residual,hessian,metrics,responses=self._backend.assemble(mechanical,parameter,origins)
        # Fibre evaluations return conservative chart Hessians. Convert each
        # native block once to the same spatial residual derivative used by the
        # coupled contact driver, retaining the compensated Hessian low part.
        tangent=hessian.copy()
        for slots,response in zip(self.slots,responses):
            chart=response.hessian+response.hessian_low
            correction=spatial_jacobian(response.residual,chart)-chart
            rows,cols=np.meshgrid(slots,slots,indexing='ij')
            np.add.at(tangent,(rows,cols),correction)
        self.guard()
        return residual,tangent,metrics,responses,None

    @contextmanager
    def operation(self,cancellation_token):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('physical-fibre contact context already in operation')
        self._token=cancellation_token
        try:
            self.guard();yield
        finally:
            self._token=None;self._cancel_active=False;self._lock.release()

    def activate_cancellation(self,cancellation_token):
        # Validate/replay the committed prefix first. Candidate material work
        # then observes cancellation inside the shared driver's rollback guard.
        if not self._lock.locked() or cancellation_token is not self._token:
            raise ValueError('physical-fibre cancellation requires its bound operation')
        self._cancel_active=True
