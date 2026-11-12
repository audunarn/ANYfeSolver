"""Private opt-in, source-bound physical B2 assembly measurements.

Declared storage accounting is not allocator/RSS accounting or an upper bound.
No GC/global scan, dense conversion, source-equation change or public API.
Unknown retained storage refuses capture before numerical factorization.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from types import MappingProxyType
from time import perf_counter
import sys

import numpy as np
from scipy import sparse

_ACTIVE=ContextVar('physical_b2_assembly_capture',default=None)
ASSEMBLY=('shell_response_scatter','beam_response_scatter','K_CSR','attachment_rows','C_CSR')
CAPS=dict(objects=2_000_000,edges=8_000_000,buffers=500_000,depth=40)

# Explicit source classes and their permitted instance storage; optional caches
# are included when present. Any new field/type fails closed until reviewed.
SCHEMAS={
 ('anymaterial.isotropic','IsotropicMaterial'):('name','elastic_modulus','poisson_ratio','density','yield_stress','hardening_curve'),
 ('anyfesolver._b2_physical_flexibility_linear','OutputRepresentationCertificate'):('basic_energy_preserved','rigid_work_preserved','cross_work_preserved','rigid_wrench_error','rigid_work_error','cross_work_error','identity'),
 ('anyfesolver.fe_core','FEModel'):('name','mesh','materials','boundary_conditions','load_cases','current_material','constraint_equations'),
 ('anyfesolver.fe_core','FEMesh'):('nodes','elements','dof_manager','point_masses','element_activity','revisions','_qualified_direct_state_token','_sparsity_cache','_topology_signature_cache'),
 ('anyfesolver.fe_core','Node'):('id','x','y','z','dofs','_coordinate_revision','_qualified_direct_state_token','_qualified_direct_state_tokens'),
 ('anyfesolver.fe_core','DOFManager'):('_node_to_dof','_dof_to_node','_dof_to_local','_total_dofs','_constrained_dofs'),
 ('anyfesolver._b2_attachment_elimination_dev','AttachmentElimination'):('k','j','c','free','beam','beta','alpha','right','left','inverse_right','inverse_left','reduced','info','primal_count','constraint_count'),
 ('anyfesolver._b2_physical_linear_definition','CompiledPhysicalB2Member'):('definition','operator','_frame_bytes','_transform_bytes','production_qualified'),
 ('anyfesolver._b2_physical_flexibility_linear','LinearOperator'):('length','section','_stiffness','_compliance','_B','_modal','_basic_inverse','_trace_polynomials','_sealed','_tangent','_output_certificate','_uniform_load_map'),
 ('anyfesolver._b2_physical_linear_definition','PhysicalB2OperatorCache'):('_operators','hits','misses'),
 ('anyfesolver._stiffened_shell_b2_physical_rigid_footprint','PhysicalFootprintNode'):('footprint_id','shell_node','beam_node','row_start','shell_slots','beam_slots','shell_position','beam_position','director','tangent_basis','constraint','constraint_hessians','row_count'),
}
_ELEMENT_FIELDS=('id','node_ids','material_id','thickness','reference_normal','material_direction','director_polarity',
 'pl_stabilization','planar_tolerance','warped_formulation','legacy_warped_fallback','_qualified_plan_state_revision','reference_surface_offset',
 '_stiffness_matrix','_mass_matrix','_internal_forces','_nl_cache','_nl_cache_key','_qualified_component_guard',
 '_hourglass_stiffness_matrix','_qualified_components','_qualified_cache_key','_qualified_direct_state_token','_qualified_direct_state_tokens',
 '_is_3node','_is_4node','_is_6node','_is_8node','_is_quadrilateral','_is_triangular','drilling_stabilization',
 'element_id','hourglass_stabilization','material_angle_deg','material_name','reduced_integration','shell_section')
SCHEMAS[('anyfesolver.e4_pl_element','QualifiedE4PLShellElement')]=_ELEMENT_FIELDS
SCHEMAS[('anyfesolver.e4_pl_s3_v2d_element','E4PLS3V2DShellElement')]=_ELEMENT_FIELDS
SCHEMAS[('anyfesolver.e4_pl_s3_v2d_element','NativeParityE4PLS3V2DShellElement')]=_ELEMENT_FIELDS


class DiagnosticRefusal(RuntimeError):pass


class Inventory:
    """Bounded traversal of declared roots/adapters only; buffers deduplicated."""
    def __init__(self,borrowed=(),caps=None):
        self.caps=dict(CAPS if caps is None else caps)
        if set(self.caps)!=set(CAPS) or any(type(v) is not int or v<=0 for v in self.caps.values()):raise DiagnosticRefusal('invalid inventory caps')
        self.borrowed={id(x) for x in borrowed};self.seen=set();self.buffers=set();self.edges=0
        self.groups={};self.group=None
    def add(self,name,roots,*,declaration=False):
        self.group=name;self.groups.setdefault(name,dict(object_bytes=0,buffer_bytes=0,objects=0,buffers=0,input_references=0))
        if declaration:
            for value in roots.values():self.visit(value,0)
        else:self.visit(roots,0)
    def visit(self,value,depth):
        self.edges+=1
        if depth>self.caps['depth'] or self.edges>self.caps['edges']:raise DiagnosticRefusal('inventory traversal cap')
        if id(value) in self.borrowed:self.groups[self.group]['input_references']+=1;return
        if id(value) in self.seen:return
        if len(self.seen)>=self.caps['objects']:raise DiagnosticRefusal('inventory object cap')
        self.seen.add(id(value));g=self.groups[self.group];g['objects']+=1
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject or type(value) is not np.ndarray:raise DiagnosticRefusal('unsupported array storage')
            owner=value
            while isinstance(owner.base,np.ndarray):
                self.edges+=1
                if self.edges>self.caps['edges']:raise DiagnosticRefusal('array base-chain traversal cap')
                owner=owner.base
            base=owner.base
            if base is not None and type(base) is not bytes:raise DiagnosticRefusal('unknown array buffer owner')
            if base is not None:
                g['object_bytes']+=sys.getsizeof(value);self.visit(base,depth+1);return
            g['object_bytes']+=sys.getsizeof(value)-(value.nbytes if value.flags.owndata else 0)
            if owner is not value:self.visit(owner,depth+1)
            if id(owner) in self.borrowed:g['input_references']+=1;return
            if id(owner) not in self.buffers:
                if len(self.buffers)>=self.caps['buffers']:raise DiagnosticRefusal('inventory buffer cap')
                self.buffers.add(id(owner));g['buffers']+=1;g['buffer_bytes']+=owner.nbytes
            return
        g['object_bytes']+=sys.getsizeof(value)
        if value is None or type(value) in (bool,int,float,str,bytes):return
        if isinstance(value,np.generic):
            if value.dtype.hasobject:raise DiagnosticRefusal('unsupported scalar object storage')
            return
        if type(value) is Fraction:
            self.visit(value.numerator,depth+1);self.visit(value.denominator,depth+1);return
        if type(value) in (sparse.csr_matrix,sparse.csc_matrix):
            fields={'_shape','maxprint','_maxprint','indices','indptr','data','_has_sorted_indices','_has_canonical_format'}
            unknown=set(value.__dict__)-fields
            if unknown:raise DiagnosticRefusal('unknown sparse retained fields '+repr(sorted(unknown)))
            self.visit(value.__dict__,depth+1)
            return
        if type(value) in (tuple,list,set,frozenset) or type(value).__module__=='anyfesolver.fe_core' and type(value).__name__=='_QualifiedMutationEpoch':
            for x in value:self.visit(x,depth+1)
            return
        if type(value) in (dict,MappingProxyType) or type(value).__module__=='anyfesolver.fe_core' and type(value).__name__=='_QualifiedStateMapping':
            for k,v in value.items():self.visit(k,depth+1);self.visit(v,depth+1)
            if type(value).__name__=='_QualifiedStateMapping':self.visit(value._qualified_token,depth+1);self.visit(value._qualified_kind,depth+1)
            return
        key=(type(value).__module__,type(value).__name__);fields=SCHEMAS.get(key)
        if fields is None:raise DiagnosticRefusal('unknown retained storage type '+repr(key)+' fields='+repr(sorted(getattr(value,'__dict__',{}))))
        namespace=getattr(value,'__dict__',None)
        if namespace is not None:
            unknown=set(namespace)-set(fields)
            if unknown:raise DiagnosticRefusal('unknown retained fields '+repr(key)+': '+repr(sorted(unknown)))
            # Namespace is a declared, alias-deduplicated root, not a copy.
            self.visit(namespace,depth+1)
            return
        for name in fields:
            if hasattr(value,name):self.visit(getattr(value,name),depth+1)
    def result(self):
        return dict(groups=self.groups,unique_objects=len(self.seen),unique_buffers=len(self.buffers),edges=self.edges,
            total_bytes=sum(g['object_bytes']+g['buffer_bytes'] for g in self.groups.values()),caps=self.caps,
            interpretation='declared Python object plus unique ndarray backing storage; not RSS/allocator upper bound')


class Capture:
    def __init__(self,clock=perf_counter):self.clock=clock;self.phase=None;self.start=None;self.phases=[];self.storage=None;self.status='NEW'
    def mark(self,phase):
        now=self.clock()
        if self.phase is not None:self.phases.append(dict(phase=self.phase,seconds=now-self.start))
        self.phase=phase;self.start=now
    def begin(self):
        if self.status!='NEW':raise DiagnosticRefusal('capture admits one constructor only')
        self.status='RUNNING';self.mark('admission')
    def prefactor(self,persistent,transient,borrowed):
        self.mark('storage_inventory');inventory=Inventory(borrowed)
        inventory.add('persistent',persistent,declaration=True);inventory.add('transient',transient,declaration=True)
        self.storage=inventory.result();self.mark('factorization')
    def record_coo_release(self,phase,counts):
        if self.phase!=phase or phase not in ('K_CSR','C_CSR') or len(counts)!=3 or any(type(n) is not int or n<0 for n in counts) or len(set(counts))!=1:
            raise DiagnosticRefusal('invalid COO release accounting')
        if not hasattr(self,'coo_releases'):self.coo_releases=[]
        if any(r['phase']==phase for r in self.coo_releases):raise DiagnosticRefusal('duplicate COO release accounting')
        self.coo_releases.append(dict(phase=phase,preclear_entries=dict(rows=counts[0],columns=counts[1],values=counts[2]),
            remaining_entries=0,interpretation='actual scratch entries before release; not byte/peak/RSS bound'))
    def finish(self):
        self.mark(None);self.status='COMPLETE'
    def fail(self):
        if self.phase is not None:
            self.mark(None)
        self.status='FAILED'
    def report(self):
        return dict(status=self.status,phases=self.phases,storage=self.storage,
            coo_releases=getattr(self,'coo_releases',[]),
            assembly_seconds=sum(p['seconds'] for p in self.phases if p['phase'] in ASSEMBLY),
            timing_incomplete=getattr(self,'timing_incomplete',False),qualification=False,upper_bound=False)


def active_capture():return _ACTIVE.get()


@contextmanager
def capture_assembly(*,clock=perf_counter):
    if _ACTIVE.get() is not None:raise DiagnosticRefusal('nested capture refused')
    capture=Capture(clock);token=_ACTIVE.set(capture)
    try:yield capture
    except BaseException:
        # A diagnostic clock failure must not replace the original exception.
        try:capture.fail()
        except BaseException:
            capture.status='FAILED'
            capture.timing_incomplete=True
        raise
    finally:_ACTIVE.reset(token)
