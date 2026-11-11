"""Explicit physical-fibre grouped-body static owner; full native transaction."""
import numpy as np
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_p5.algebra import skew
from .native_fibre_context import NativeFibreContactContext
from .native_retained_pair_static import NativeRetainedRoundPairAnalysis, NativeRetainedPairAccepted
from .native_retained_body_pair import NativeRetainedBodyPair


class NativeRetainedBodyPairAnalysis(NativeRetainedRoundPairAnalysis):
    def __init__(self,definitions,boundaries,program,bodies,*,radius_m=.05,
                 line_stiffness_n_per_m2=4000.,gap_tolerance_m=5e-5,
                 equilibrium_tolerance=1e-8,maximum_swept_intervals=128,max_coordinates=256,**kernel_controls):
        self.native=NativeBeamAnalysis(definitions,boundaries)
        if self.native._family!='PHYSICAL_FIBRE_NODAL':raise ValueError('native bodies require physical-fibre native definitions')
        if (not np.isfinite(gap_tolerance_m) or not np.isfinite(equilibrium_tolerance)
                or min(gap_tolerance_m,equilibrium_tolerance)<=0
                or type(maximum_swept_intervals) is not int or maximum_swept_intervals<1):
            raise ValueError('native body static controls require positive finite bounds')
        self.context=NativeFibreContactContext(self.native,program,max_coordinates=max_coordinates)
        self.contact=NativeRetainedBodyPair(self.context,bodies,radius_m=radius_m,
            line_stiffness_n_per_m2=line_stiffness_n_per_m2,**kernel_controls)
        self.profile='NATIVE_CONNECTED_BODY_PAIR_PHYSICAL_FIBRE_STATIC_DEV_V1'
        self.checkpoint_schema='NATIVE_CONNECTED_BODY_PAIR_CHECKPOINT_DEV_V1'
        self.program=program;self.indices=tuple(range(len(self.context.elements)))
        self.gap_tolerance=float(gap_tolerance_m);self.equilibrium_tolerance=float(equilibrium_tolerance)
        self.maximum_swept_intervals=maximum_swept_intervals
        self.gap_step_control=True;self.reuse_target_trials=False;self.initial_trial=None;self._trial_state=None
        self.identity=self._identity()
        first=self._contact(self.context.initial.mechanical)
        if first.conservative_lower_gap_m < -256*np.finfo(float).eps:
            raise ValueError('initial native body gap is not certified')
        self.initial=NativeRetainedPairAccepted(self.context.initial.mechanical,self.context.initial.histories,0,first)

    def _guard(self):
        if self.indices != tuple(range(len(self.context.elements))):
            raise ValueError('native body bound element indices changed')
        super()._guard()

    def _uses_physical_gap(self):return True

    def _contact(self,mechanical):return self.contact.evaluate(mechanical)

    def _physical_gap(self,mechanical):
        self._guard();self.context._safe_point()
        value=self.contact.physical_gap_lower(mechanical)
        self._guard();self.context._safe_point();return value

    def _assemble(self,mechanical,parameter,origins,*,prepared_contact=None):
        self._guard()
        if prepared_contact is not None and prepared_contact[0] is not mechanical:
            raise ValueError('native body prepared contact requires exact candidate')
        residual,tangent,_,responses,_=self.context.assemble(mechanical,parameter,origins)
        contact=self._contact(mechanical) if prepared_contact is None else prepared_contact[1]
        residual+=contact.residual;block=contact.tangent.copy()
        rotations=[6*node+3 for node in range(len(self.context.node_ids))]
        rotations += [self.context.nodal_count+24*element+3*cell
                      for element in range(len(self.context.elements)) for cell in (0,1)]
        for start in rotations:block[start:start+3,start:start+3]-=.5*skew(contact.residual[start:start+3])
        tangent+=block;scaled=residual/self.context.scale
        metrics=(float(np.linalg.norm(scaled[self.context.equilibrium])),float(np.linalg.norm(scaled[self.context.compatibility])))
        return residual,tangent,metrics,responses,contact
