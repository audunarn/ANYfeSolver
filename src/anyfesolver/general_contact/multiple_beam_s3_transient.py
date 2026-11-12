"""Explicit elastic multiple Beam/S3 transient composition; no new line law.

Children own their physical geometry, Jet2 work and complete path proofs. The
ordinary Newmark driver owns the single structural/mass assembly and commits.
This is an opt-in development profile; verification does not confer acceptance.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from anymaterial.isotropic import IsotropicMaterial
from anyfesolver.elements import BeamElement,QuadraticBeamElement
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.nonlinear_restart import model_configuration_fingerprint
from .core import ContactDefinition,ContactSolverSettings,_contact_settings_contract
from .beam_shell import validate_beam_shell_contact
from .deformable_beam_s3_static import PROFILE as CHILD_PROFILE
from .beam_s3_transient import BeamS3TransientManager,hermite_gap_bounds
from .runtime import _ACTIVE_CONTACT

PROFILE='ELASTIC_MULTIPLE_B2_B3_DEFORMABLE_FIXED_DIRECTOR_S3_NEWMARK_DEV_V1'
RESTART_SCHEMA='ANYSOLVER_MULTIPLE_BEAM_DEFORMABLE_S3_TRANSIENT_DEV_CHECKPOINT_V21'


def pair_definitions(model,definition,settings):
    """Canonical complete explicit inventory; refuse before child allocation."""
    if type(definition) is not ContactDefinition or type(settings) is not ContactSolverSettings:
        raise ValueError('multiple beam-S3 requires exact contact definitions/settings')
    count=len(definition.interactions)
    if definition.rigid_bodies or count<2:
        raise ValueError('multiple beam-S3 requires at least two explicit deformable pairs')
    if count>settings.maximum_candidate_pairs:
        raise ValueError('multiple beam-S3 candidate pair budget exceeded')
    surfaces={};physical={};radii={}
    for surface in definition.surfaces:
        if type(surface.name) is not str or not surface.name or surface.name in surfaces or len(surface.element_ids)!=1:
            raise ValueError('multiple beam-S3 requires unique named single-element surfaces')
        eid=surface.element_ids[0]
        if surface.beam_radius is not None:
            if eid in radii and radii[eid]!=surface.beam_radius:
                raise ValueError('multiple beam-S3 conflicting beam radius')
            radii[eid]=surface.beam_radius;key=('beam',eid)
        else:key=('shell',eid,surface.side)
        if key in physical:raise ValueError('multiple beam-S3 duplicate physical surface alias')
        physical[key]=surface.name;surfaces[surface.name]=surface
    names=set();used=set();features=set();plans=[]
    for pair in definition.interactions:
        if type(pair.name) is not str or not pair.name or pair.name in names:
            raise ValueError('multiple beam-S3 requires unique stable interaction names')
        names.add(pair.name)
        if pair.surface not in surfaces or pair.target not in surfaces:
            raise ValueError('multiple beam-S3 references an unknown participant')
        beam,shell=surfaces[pair.surface],surfaces[pair.target]
        if beam.beam_radius is None or shell.beam_radius is not None:
            raise ValueError('multiple beam-S3 requires directed round beam to physical S3 skin')
        feature=(beam.element_ids[0],shell.element_ids[0],shell.side)
        if feature in features:raise ValueError('multiple beam-S3 duplicate physical pair')
        features.add(feature);used.update((beam.name,shell.name))
        child=ContactDefinition((beam,shell),(),(pair,))
        report=validate_beam_shell_contact(model,child,settings)
        if not report.valid or report.profile!=CHILD_PROFILE:
            raise ValueError('multiple beam-S3 unsupported child '+pair.name+': '+'; '.join(report.issues))
        plans.append(child)
    if used!=set(surfaces):raise ValueError('multiple beam-S3 has unreferenced surfaces')
    selected={eid for surface in surfaces.values() for eid in surface.element_ids}
    if selected!=set(model.mesh.elements):raise ValueError('multiple beam-S3 requires every model element explicitly selected')
    nodes=set()
    for eid in selected:
        element=model.mesh.elements[eid];nodes.update(element.node_ids)
        if type(element) not in (BeamElement,QuadraticBeamElement,NativeParityE4PLS3V2DShellElement):
            raise ValueError('multiple beam-S3 element type unsupported')
        material=model.get_material(element.material_name)
        if type(material) is not IsotropicMaterial or material.yield_stress!=0. or material.hardening_curve is not None:
            raise ValueError('multiple beam-S3 requires exact elastic isotropic materials')
    if nodes!=set(model.mesh.nodes):raise ValueError('multiple beam-S3 has unreferenced material nodes')
    return tuple(sorted(plans,key=lambda d:d.interactions[0].name))


def _model_binding(model):
    # Established descriptor includes public section/skin/material/constraint
    # configuration and deliberately excludes derived private element caches.
    # Bind actual node routing too: total width alone does not identify a map.
    return (model_configuration_fingerprint(model),_beam_operator_binding(model),
        tuple((nid,tuple(model.mesh.dof_manager.get_node_dofs(nid)))
              for nid in sorted(model.mesh.nodes)))


def _beam_operator_binding(model):
    rows=[]
    for eid,element in sorted(model.mesh.elements.items()):
        if type(element) not in (BeamElement,QuadraticBeamElement):continue
        if element._geometric_nonlinearity!='von_karman' or element._fiber_plasticity is not None:
            raise ValueError('multiple beam-S3 actual kinematic/constitutive operator unsupported or mutated')
        values=[]
        for key in ('_A','_Iy','_Iz','_J','_ky','_kz'):
            value=getattr(element,key)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:
                raise ValueError('multiple beam-S3 actual section operator malformed')
            values.append((key,float(value)))
        orientation=element._orientation
        if orientation is not None:
            array=np.asarray(orientation)
            if array.shape!=(3,) or array.dtype.kind not in 'fi' or not np.isfinite(array).all():
                raise ValueError('multiple beam-S3 actual orientation operator malformed')
            orientation=tuple(float(x) for x in array)
        rows.append((eid,tuple(values),orientation,'von_karman',None))
    return tuple(rows)


@dataclass(frozen=True)
class MultipleBeamS3Results:
    profile: str
    pairs: tuple
    pair_gap_tolerances: tuple

    @property
    def minimum_gap(self):return min(r.minimum_gap for _,r in self.pairs)
    @property
    def gap_tolerance_met(self):return all(r.gap_tolerance_met for _,r in self.pairs)
    @property
    def line_contact_energy(self):return sum(r.line_contact_energy for _,r in self.pairs)
    @property
    def active_line_length(self):return sum(r.active_line_length for _,r in self.pairs)
    @property
    def minimum_rim_clearance(self):return min(r.minimum_rim_clearance for _,r in self.pairs)
    @property
    def integration_points(self):return sum(r.integration_points for _,r in self.pairs)
    @property
    def active_intervals(self):return sum(r.active_intervals for _,r in self.pairs)

    def _wrenches(self,kind):
        rows={}
        for _,r in self.pairs:
            eid=r.beam_element if kind=='beam' else r.shell_element
            wrench=np.r_[r.beam_force,r.beam_moment] if kind=='beam' else np.r_[r.shell_force,r.shell_moment]
            rows.setdefault(eid,np.zeros(6));rows[eid]+=wrench
        return {str(eid):{'force_n':tuple(float(x) for x in w[:3]),'moment_nm':tuple(float(x) for x in w[3:]),'moment_origin_m':(0.,0.,0.)} for eid,w in sorted(rows.items())}

    def to_dict(self):
        return dict(profile=self.profile,pair_count=len(self.pairs),minimum_gap=self.minimum_gap,
            line_contact_energy=self.line_contact_energy,active_line_length=self.active_line_length,
            active_intervals=self.active_intervals,integration_points=self.integration_points,
            minimum_rim_clearance=self.minimum_rim_clearance,gap_tolerance_met=self.gap_tolerance_met,
            pairs={name:r.to_dict() for name,r in self.pairs},
            pair_gap_tolerances_m=dict(self.pair_gap_tolerances),
            beam_element_wrenches=self._wrenches('beam'),shell_element_wrenches=self._wrenches('shell'),
            contact_measure='reference_line_length',assembly='sum_explicit_pair_potentials_on_shared_global_coordinates',
            descriptor='fixed_director_S3_rotations_and_rates_exact_zero',scope='DEVELOPMENT_NOT_QUALIFICATION')


class MultipleBeamS3TransientManager:
    def __init__(self,model,definition,settings,initial_displacement=None):
        plans=pair_definitions(model,definition,settings)
        self.model,self.definition,self.settings=model,definition,settings
        self.profile=PROFILE;self.swept_trial_cutbacks=0
        self.pairs=tuple((p.interactions[0].name,BeamS3TransientManager(model,p,settings,initial_displacement)) for p in plans)
        self.shell_dofs=np.array(sorted({tuple(row) for _,c in self.pairs for row in c.shell_dofs}),int)
        # Only a redundant summary threshold. Acceptance is the conjunction of
        # child endpoint flags and child path proofs with THEIR own tolerances.
        self.gap_tolerance=max(c.gap_tolerance for _,c in self.pairs)
        self._captured=(id(model),self.profile,repr(definition),repr(settings),tuple((n,id(c)) for n,c in self.pairs),_model_binding(model),self.shell_dofs.tobytes(),self.shell_dofs.shape,self.gap_tolerance)

    def guard(self):
        current=(id(self.model),self.profile,repr(self.definition),repr(self.settings),tuple((n,id(c)) for n,c in self.pairs),_model_binding(self.model),self.shell_dofs.tobytes(),self.shell_dofs.shape,self.gap_tolerance)
        if current!=self._captured:raise ValueError('multiple beam-S3 operator/routing mutated')
        for _,child in self.pairs:
            if child.model is not self.model:raise ValueError('multiple beam-S3 child model routing mutated')
            child.guard()

    def evaluate(self,displacement,*,tangent=True):
        self.guard();size=self.model.mesh.dof_manager.total_dofs
        residual=np.zeros(size);K=sparse.csr_matrix((size,size)) if tangent else None;results=[]
        for name,child in self.pairs:
            r,H,result=child.evaluate(displacement,tangent=tangent)
            residual+=r
            if tangent:K=K+H
            results.append((name,result))
        self.guard()
        return residual,K,MultipleBeamS3Results(PROFILE,tuple(results),
            tuple((name,child.gap_tolerance) for name,child in self.pairs))

    def restart_contract(self):
        self.guard()
        return dict(profile=PROFILE,revision='COMPLETE_EXPLICIT_PAIR_GLOBAL_WORK_V1',
            settings=_contact_settings_contract(self.settings),pair_order=[n for n,_ in self.pairs],
            actual_scalar_beam_operators=_beam_operator_binding(self.model),
            pairs=[{'name':n,'contact':c.restart_contract()} for n,c in self.pairs],
            shared_coordinate_policy='ONE_GLOBAL_STRUCTURAL_MASS_ORIGIN',
            path_budget_policy='SHARED_VISITED_INTERVAL_BUDGET_ALL_PAIRS')


def multiple_hermite_gap_bounds(manager,start,end,vstart,vend,dt,*,maximum_intervals=512,maximum_depth=20,check_cancel=lambda:None):
    """Conjunction of complete child proofs with one shared finite work cap.

    Child chart allowance is local, independent of global vector width. Each
    mapped position evaluates one reference plus <=4 CSR products; Hermite
    arithmetic evaluates each coordinate separately, without a global sum.
    The child 128*eps allowance therefore still covers the same operation
    chain when zero columns/global indices are added. Check that prerequisite
    explicitly. No complete-path acceptance uses an aggregate gap tolerance.
    """
    manager.guard()
    if type(maximum_intervals) is not int or maximum_intervals<1 or type(maximum_depth) is not int or maximum_depth<0:
        raise ValueError('multiple beam-S3 path budgets invalid')
    visited=0;bounds={name:-float('inf') for name,_ in manager.pairs};good=True
    for name,child in manager.pairs:
        if np.any(np.diff(child.work_map.indptr)>4):
            raise ValueError('multiple beam-S3 child chart arithmetic exceeds proof inventory')
        check_cancel();remaining=maximum_intervals-visited
        if remaining<=0:good=False;continue
        lower,admitted,used=hermite_gap_bounds(child,start,end,vstart,vend,dt,
            maximum_intervals=remaining,maximum_depth=maximum_depth,check_cancel=check_cancel)
        visited+=used;bounds[name]=lower;good=good and admitted
    manager.guard();check_cancel()
    return min(bounds.values()),bool(good and visited<=maximum_intervals),visited,bounds


@contextmanager
def multiple_beam_s3_scope(model,definition,settings,initial_displacement=None):
    manager=MultipleBeamS3TransientManager(model,definition,settings,initial_displacement)
    token=_ACTIVE_CONTACT.set(manager)
    try:yield manager
    finally:_ACTIVE_CONTACT.reset(token)
