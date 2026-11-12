"""Separate retained-native owner for explicit multiple physical S3 interiors.

All native internal unknowns and complete structural/contact blocks are retained.
This does not widen the original single-native/single-S3 profile.
"""
import copy
from dataclasses import dataclass, replace
from contextlib import nullcontext
from hashlib import sha256
from math import isfinite

import numpy as np
from scipy import sparse
from anymaterial import IsotropicMaterial
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_retained_nodal_loading import Context, Program
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import MAX_BYTES, _load
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.fe_core import FEModel
from .native_retained_line_s3_work import NativeRetainedLineS3Work
from .native_material_s3 import NativeRetainedMaterialS3Work
from .native_curved_s3_work import NativeCurvedMaterialS3Work
from .native_retained_s3_static import NativeRetainedRoundS3Analysis
from .native_retained_station import _readonly
from .s3_skin_work import S3SkinWorkAdapter
from .native_s3_swept import native_s3_swept_admissible
from .native_shell_static_driver import solve_native_shell
from .native_replay_diagnostics import component_digests, replay_difference_message

class _ShellArithmeticUnavailable(ValueError):
    """The alternate shell stiffness arithmetic does not apply to this shell."""


# Stiffness route of the exact V2D element, pinned at import. The fused
# arithmetic bypasses these entries, so any instance or class replacement
# must be refused rather than silently skipped.
_V2D_STIFFNESS_ROUTE = {name: vars(NativeParityE4PLS3V2DShellElement).get(name) for name in (
    'compute_stiffness_matrix', 'compute_stiffness_components', '_geometry', '_constitutive',
    '_operators', '_validate_configuration', '_is_v2c_elastic_overlap', '_coordinates',
    '_validate_model_scope', '_membrane_operator', '_pl_constraint')}


PROFILE = 'NATIVE_B3_GE_MULTIPLE_ELASTIC_S3_INTERIORS_STATIC_DEV_V1'
MATERIAL_PROFILE = 'NATIVE_B3_GE_TWO_MATERIAL_PATCHES_SHARED_S3_STATIC_DEV_V1'
MATERIAL_MULTIPATCH_PROFILE = 'NATIVE_B3_GE_MULTIPLE_MATERIAL_PATCHES_S3_STATIC_DEV_V1'
CURVED_MATERIAL_PROFILE = 'NATIVE_B3_GE_CURVED_MATERIAL_PATCHES_S3_STATIC_DEV_V1'
FIBRE_MATERIAL_PROFILE = 'NATIVE_B3_GE_PHYSICAL_FIBRE_MATERIAL_PATCHES_S3_STATIC_DEV_V1'
FIBRE_CURVED_MATERIAL_PROFILE = 'NATIVE_B3_GE_PHYSICAL_FIBRE_CURVED_MATERIAL_PATCHES_S3_STATIC_DEV_V1'
SCHEMA = 'NATIVE_B3_GE_MULTIPLE_S3_CHECKPOINT_DEV_V1'


def _candidate_work_signature(result):
    def matrix(value):
        if not sparse.isspmatrix_csr(value):
            raise ValueError('cached native/S3 contact requires CSR work')
        return value.shape, value.data, value.indices, value.indptr
    return sha(dict(profile=result.profile, residual=result.residual,
        tangent=matrix(result.tangent), pairs=[dict(name=p.name,
            residual=r.residual,tangent=matrix(r.tangent),profile=r.profile,
            physical=vars(r.contact)) for p,r in result.pairs]))


def _candidate_assembly_signature(assembled):
    """Integrity token for one complete cached candidate assembly."""
    residual, tangent, metrics, responses, contact = assembled
    return sha(dict(work=_candidate_work_signature(contact), residual=residual,
        tangent=tangent, metrics=metrics,
        responses=[dict(residual=r.residual, history=sha(r.history))
            for r in responses]))


@dataclass(frozen=True)
class NativeS3ContactPair:
    name: str
    work: NativeRetainedLineS3Work | NativeRetainedMaterialS3Work | NativeCurvedMaterialS3Work
    gap_tolerance_m: float
    maximum_swept_intervals: int

    def __post_init__(self):
        if type(self.name) is not str or not self.name or type(self.work) not in (NativeRetainedLineS3Work,NativeRetainedMaterialS3Work,NativeCurvedMaterialS3Work):
            raise ValueError('named native/S3 pair requires exact physical work')
        if (type(self.gap_tolerance_m) not in (int, float) or not isfinite(self.gap_tolerance_m)
                or self.gap_tolerance_m <= 0):
            raise ValueError('native/S3 pair requires positive dimensional gap tolerance')
        if type(self.maximum_swept_intervals) is not int or self.maximum_swept_intervals < 1:
            raise ValueError('native/S3 pair requires a finite swept interval budget')

    def descriptor(self):
        mapping = self.work.work_map
        if not sparse.isspmatrix_csr(mapping):
            raise ValueError('native/S3 combined work map requires CSR coordinates')
        return {'name': self.name, 'work': self.work._fingerprint(),
            'combined_work_map': {'shape': mapping.shape, 'data': mapping.data,
                'indices': mapping.indices, 'indptr': mapping.indptr},
            'kernel': self.work.contact._fingerprint(), 'gap_tolerance_m': self.gap_tolerance_m,
            'maximum_swept_intervals': self.maximum_swept_intervals}


def _physical_summary(pairs):
    fields = ('potential_j', 'active_intervals', 'integration_intervals', 'integration_evaluations',
              'energy_error_j', 'force_error_n', 'tangent_error_n_per_m')
    area=type(pairs[0][0].work) in (NativeRetainedMaterialS3Work,NativeCurvedMaterialS3Work)
    summed_fields=tuple(k for k in fields if k!='active_intervals') if area else fields
    result = {key: sum(getattr(r.contact, key) for _, r in pairs) for key in summed_fields}
    if area:
        result.update(active_intervals=None,measure='fixed_reference_material_area',
            minimum_gap_estimate_m=(None if any(r.contact.minimum_gap_estimate_m is None for _,r in pairs)
                                    else min(r.contact.minimum_gap_estimate_m for _,r in pairs)))
    for key in ('beam_force_n', 'shell_force_n', 'beam_moment_nm', 'shell_moment_nm'):
        result[key] = sum((getattr(r.contact, key) for _, r in pairs), np.zeros(3))
    result['minimum_gap_m'] = min(r.contact.minimum_gap_m for _, r in pairs)
    result['minimum_rim_clearance_m'] = min(r.contact.minimum_rim_clearance_m for _, r in pairs)
    result['pairs'] = [{'name': p.name, 'beam_element': p.work.contact.beam_element_id,
        'shell_body': p.work.shell.body_id, 'shell_element': p.work.shell.element_id,
        'shell_side': p.work.shell.side, 'gap_tolerance_m': p.gap_tolerance_m,
        'maximum_swept_intervals': p.maximum_swept_intervals,
        **{key: getattr(r.contact, key) for key in (*fields, 'beam_force_n', 'shell_force_n',
            'beam_moment_nm', 'shell_moment_nm', 'minimum_gap_m', 'minimum_rim_clearance_m')}}
        for p, r in pairs]
    if area:
        for row,(_,r) in zip(result['pairs'],pairs):
            row.update(measure='fixed_reference_material_area',minimum_gap_estimate_m=r.contact.minimum_gap_estimate_m)
    return result


@dataclass(frozen=True)
class NativeMultiS3Physical:
    pairs: tuple

    def __getattr__(self, name):
        values = _physical_summary(self.pairs)
        if name not in values: raise AttributeError(name)
        return values[name]

    def to_dict(self):
        return _physical_summary(self.pairs)


@dataclass(frozen=True)
class NativeMultiS3WorkResult:
    contact: NativeMultiS3Physical
    residual: np.ndarray
    tangent: sparse.csr_matrix
    pairs: tuple
    profile: str = PROFILE


@dataclass(frozen=True)
class NativeMultiS3Accepted:
    mechanical: object
    native_histories: tuple
    shell_displacement: np.ndarray
    completed_targets: int
    contact: NativeMultiS3WorkResult


@dataclass(frozen=True)
class NativeMultiS3Run:
    status: str
    accepted: NativeMultiS3Accepted
    failure: str | None
    checkpoint: bytes
    profile: str = PROFILE

    @property
    def checkpoint_sha256(self):
        return sha256(self.checkpoint).hexdigest()


class NativeRetainedMultiS3Analysis:
    """One or two retained native beams with explicit independent S3 pairs.

    The shell structural model is elastic reference-stiffness S3, with physical
    skin work and zero prescribed supports. Each selected beam must participate;
    all shell elements must be represented. Rims/patch unions remain unsupported.
    """
    profile = PROFILE
    schema = SCHEMA
    run_type = NativeMultiS3Run
    _assemble_structural = NativeRetainedRoundS3Analysis._assemble_structural
    _step = NativeRetainedRoundS3Analysis._step
    @staticmethod
    def _recoverable_trial_error(exc):
        if NativeRetainedRoundS3Analysis._recoverable_trial_error(exc):return True
        if type(exc) is MemoryError:
            return str(exc).startswith(('curved normal integration interval budget exhausted',
                                        'curved axial front interval budget exhausted'))
        return type(exc) is ValueError and str(exc).startswith((
            'curved material S3 patch approaches unadmitted physical rim',
            'curved material S3 has unresolved target geometry',
            'curved material whole-patch orientation unresolved',
            'curved axial front bracket work uncertainty exceeds integration budget',
            'curved normal integration combined error budget exhausted',
            'curved normal integration parameter resolution exhausted'))

    def __init__(self, definitions, boundaries, program, shell_model, pairs, *,
                 equilibrium_tolerance, maximum_pairs=8, maximum_coupled_coordinates=256, activation_settings=None):
        if (type(definitions) is not tuple or not 1 <= len(definitions) <= 2
                or any(type(d) is not NativeBeamDefinition for d in definitions)
                or type(program) is not Program or type(shell_model) is not FEModel):
            raise TypeError('multiple native/S3 owner requires one/two exact native definitions, program and shell model')
        curved_pairs=(type(pairs) is tuple and pairs and
                      all(type(p) is NativeS3ContactPair and type(p.work) is NativeCurvedMaterialS3Work for p in pairs))
        if (type(maximum_pairs) is not int or not 1 <= maximum_pairs <= 8
                or type(pairs) is not tuple or not (1 if curved_pairs else 2) <= len(pairs) <= maximum_pairs
                or any(type(p) is not NativeS3ContactPair for p in pairs)
                or len({p.name for p in pairs}) != len(pairs)):
            raise ValueError('multiple native/S3 owner requires two to eight unique explicit pairs within resource cap')
        if (type(equilibrium_tolerance) not in (int, float) or not isfinite(equilibrium_tolerance)
                or equilibrium_tolerance <= 0 or type(maximum_coupled_coordinates) is not int
                or not 1 <= maximum_coupled_coordinates <= 256):
            raise ValueError('multiple native/S3 equilibrium/resource controls invalid')
        kinds={type(p.work) for p in pairs}
        if len(kinds)!=1:
            raise ValueError('multiple native/S3 owner does not admit mixed line and area profiles')
        self._curved_mode=kinds=={NativeCurvedMaterialS3Work}
        self._material_mode=kinds in ({NativeRetainedMaterialS3Work},{NativeCurvedMaterialS3Work})
        if self._material_mode:
            if type(self) is not NativeRetainedMultiS3Analysis:
                raise ValueError('material-area multiple owner requires the exact owner without activation controller')
            self.profile=(MATERIAL_PROFILE if len(definitions)==2 and len(pairs)==2
                and shell_model.mesh.dof_manager.total_dofs==18
                and len({p.work.shell.element_id for p in pairs})==1 else MATERIAL_MULTIPATCH_PROFILE)
            if self._curved_mode:self.profile=CURVED_MATERIAL_PROFILE
        if activation_settings is not None:
            if self._curved_mode:
                raise ValueError('curved S3 candidate does not admit the straight material activation controller')
            if not self._material_mode:
                raise ValueError('base multiple activation settings require the material-area owner')
            from .native_material_activation import validate_settings
            validate_settings(activation_settings, tuple(p.work for p in pairs),
                              tuple(p.gap_tolerance_m for p in pairs))
        # The historical line subclass owns its own settings and hook.
        if self._material_mode or type(self) is NativeRetainedMultiS3Analysis:
            self.activation_settings = activation_settings
            self.activation_total_queries = 0
        self.native = NativeBeamAnalysis(definitions, boundaries)
        self._fibre_mode=self.native._family=='PHYSICAL_FIBRE_NODAL'
        if self._fibre_mode:
            from .native_fibre_context import NativeFibreContactContext,NativeRetainedFibreElement
            if not self._material_mode or activation_settings is not None:
                raise ValueError('physical-fibre contact requires explicit material-area pairs without generalized activation controller')
            self.context=NativeFibreContactContext(self.native,program)
            self.profile=FIBRE_CURVED_MATERIAL_PROFILE if self._curved_mode else FIBRE_MATERIAL_PROFILE
        else:
            self.context = Context(self.native.model, program)
        expected=NativeRetainedFibreElement if self._fibre_mode else NativeGeneralizedStaticElement
        if (len(self.context.elements) != len(definitions)
                or any(type(e) is not expected for _, e in self.context.elements)):
            raise ValueError('multiple native/S3 owner requires exact retained family elements')
        self.shell_model, self.program = shell_model, program
        self.pairs = tuple(sorted(pairs, key=lambda p: p.name))
        self.equilibrium_tolerance = float(equilibrium_tolerance)
        self.maximum_pairs, self.maximum_coupled_coordinates = maximum_pairs, maximum_coupled_coordinates
        n_shell = shell_model.mesh.dof_manager.total_dofs
        if self.context.count+n_shell > maximum_coupled_coordinates:
            raise MemoryError('multiple native/S3 coupled-coordinate resource cap exceeded')
        elements = {eid: (i, e) for i, (eid, e) in enumerate(self.context.elements)}
        indices = []; owned = set(); envelopes = {}; shell_bodies = {}
        for p in self.pairs:
            work, kernel = p.work, p.work.contact
            if kernel.beam_element_id not in elements:
                raise ValueError('native/S3 pair beam does not bind owner')
            i, element = elements[kernel.beam_element_id]
            if (tuple(element.node_ids) != kernel.beam_node_ids
                    or sha(element.operator.reference.canonical_data()) != sha(kernel.beam_reference.canonical_data())
                    or work.work_map.shape != (51, 42+n_shell)):
                raise ValueError('native/S3 pair work/reference does not bind authoritative coordinates')
            key = (kernel.beam_element_id, work.shell.element_id, work.shell.side)
            if key in owned: raise ValueError('duplicate physical native beam/S3 skin pair')
            owned.add(key); indices.append(i)
            eid = kernel.beam_element_id
            if eid in envelopes and envelopes[eid] != kernel.radius:
                raise ValueError('shared native beam has conflicting round radius')
            envelopes[eid] = kernel.radius
            sid = work.shell.element_id
            if sid in shell_bodies and shell_bodies[sid] != work.shell.body_id:
                raise ValueError('physical S3 feature has conflicting body identity')
            shell_bodies[sid] = work.shell.body_id
        if set(elements) != set(envelopes):
            raise ValueError('multiple native/S3 definition has an unreferenced beam')
        self.beam_indices = tuple(indices)
        self.shell_stiffness, self.shell_free, self.shell_fixed = self._shell_contract()
        self.shell_stiffness = _readonly(self.shell_stiffness)
        self.shell_free.setflags(write=False); self.shell_fixed.setflags(write=False)
        self.shell_contract_sha = self._shell_contract_sha()
        self.identity = self._identity()
        self.candidate_evaluations = self.pair_evaluations = 0
        self.contact_reuse_hits = 0
        self.assembly_reuse_hits = 0
        self._contact_reuse_enabled = False
        self._contact_reuse_entry = None
        self._assembly_reuse_enabled = False
        self._assembly_reuse_entry = None
        self._construction_contact_entry = None
        self.curved_trial_failures=[]
        # Identity of this owner as constructed (original arithmetic). It
        # authorises only the return from a switched arithmetic.
        self._constructed_identity = self.identity
        self.initial, self._construction_contact_entry = self._construct_initial()

    def _construct_initial(self):
        """The accepted initial state and its reusable construction evaluation
        under the CURRENT shell arithmetic (constructor and arithmetic switch)."""
        initial = self.context.initial
        u = np.zeros(self.shell_model.mesh.dof_manager.total_dofs)
        first = self._contact(initial.mechanical, u)
        for p, r in first.pairs:
            scale = max(1., np.max(np.abs(p.work.contact.beam_reference.coordinates)),
                        np.max(np.abs(p.work.shell.reference_positions)))
            if r.contact.minimum_gap_m < -256*np.finfo(float).eps*scale:
                raise ValueError(p.name+': unresolved initial native/S3 penetration')
        accepted = NativeMultiS3Accepted(initial.mechanical, initial.histories, _readonly(u), 0, first)
        entry = None
        if self._fibre_mode and self._curved_mode:
            # The construction preflight already evaluated this exact pose;
            # the first candidate assembly reuses it under the unchanged
            # key/intactness validation instead of duplicating that work.
            entry = (self._contact_cache_key(initial.mechanical, u), first, _candidate_work_signature(first))
        return accepted, entry

    def _rebind_initial(self):
        """Re-derive the initial state after a successful arithmetic switch.

        Curved contact work depends on the selected arithmetic (replay norms),
        so the initial evaluation made in the previous arithmetic is not valid.
        Evaluation counters and trial failures describe candidate work, not
        this re-derivation, and are left as they were."""
        if not self._curved_mode:
            return
        counters = (self.candidate_evaluations, self.pair_evaluations)
        failures = len(self.curved_trial_failures)
        try:
            self.initial, self._construction_contact_entry = self._construct_initial()
        finally:
            self.candidate_evaluations, self.pair_evaluations = counters
            del self.curved_trial_failures[failures:]

    # 'host_blas' is the established path (stiffness bits follow the host BLAS
    # kernel) and is the original dispatch of the element's own stiffness
    # method. 'contact_replay_fused' replaces ONLY the final 18x18 congruence
    # product of the V2C stiffness by a fixed fused order that reproduced the
    # archived matrices in the supplied traces; geometry, operators and local
    # products are still evaluated by the host libraries. It is therefore a
    # specification of one product, not a claim of host independence.
    # The mode is deliberately NOT part of the identity: the identity hashes
    # the stiffness bits themselves, and _restore tries the other mode only for
    # an authorised recorded identity and keeps it only after a full
    # recomputation reproduces that identity and every record replays.
    _shell_arithmetic = 'host_blas'
    _FUSED_ARITHMETIC = 'contact_replay_fused'
    # Original archived identities for which the fused arithmetic may be tried.
    # They authorise an attempt only; acceptance still needs exact recomputation.
    _FUSED_AUTHORISED_IDENTITIES = frozenset({
        'ecb95a0380bbb136c237060e3affcc8cb38591d00e0536fb3fe631b78707b5df',
        '00d6c0a0f66311a8791a42461ef78a4a285c8c0c7b4cd746ff4d2856b67b92be'})

    def _authorised_alternate_arithmetic(self, recorded_identity):
        """The other arithmetic to try for *recorded_identity*, or None."""
        if self._shell_arithmetic == 'host_blas':
            return self._FUSED_ARITHMETIC if recorded_identity in self._FUSED_AUTHORISED_IDENTITIES else None
        if recorded_identity == getattr(self, '_constructed_identity', None):
            return 'host_blas'
        return None

    def _element_stiffness(self, element, mesh, material):
        if self._shell_arithmetic == 'host_blas':
            return element.compute_stiffness_matrix(mesh, material)
        if self._shell_arithmetic != self._FUSED_ARITHMETIC:
            raise ValueError('multiple native/S3 unknown shell arithmetic')
        # The fused path bypasses the element's stiffness dispatch, so refuse
        # unless that dispatch is exactly the pinned canonical one.
        namespace = object.__getattribute__(element, '__dict__')
        if (type(element) is not NativeParityE4PLS3V2DShellElement
                or any(name in namespace for name in _V2D_STIFFNESS_ROUTE)
                or any(vars(NativeParityE4PLS3V2DShellElement).get(name) is not entry
                       for name, entry in _V2D_STIFFNESS_ROUTE.items())):
            raise ValueError('multiple native/S3 fused shell arithmetic refuses a shadowed or replaced stiffness route')
        element._validate_configuration()
        if (element.shell_section is not None or not element._is_v2c_elastic_overlap(material)
                or element.director_polarity != 1 or element.reference_surface_offset != 0.):
            raise _ShellArithmeticUnavailable('multiple native/S3 fused shell arithmetic requires the elastic V2C overlap element')
        from anyfesolver.e4_pl_s3_v2c_element import compute_contact_replay_stiffness_components
        return np.array(compute_contact_replay_stiffness_components(element, mesh, material)['total'], float)

    # Fields that define the stiffness/identity binding. They are replaced
    # together by a single dict update so no reader of a supported (GIL-
    # serialised) thread sees a mix of two arithmetics.
    _SHELL_BINDING_FIELDS = ('_shell_arithmetic', 'shell_stiffness', 'shell_free', 'shell_fixed',
                             'shell_contract_sha', 'identity')
    # Everything a restore replay may touch that can influence later operation.
    _RESTORE_STATE_FIELDS = _SHELL_BINDING_FIELDS + (
        'initial', '_contact_reuse_entry', '_assembly_reuse_entry', '_construction_contact_entry',
        'candidate_evaluations', 'pair_evaluations', 'contact_reuse_hits', 'assembly_reuse_hits',
        'activation_total_queries')

    def _probe_shell_arithmetic(self, mode):
        """Shell binding under *mode*, computed on a private shallow copy.

        The owner is never modified. Returns the field mapping, or None when
        the mode does not apply to this shell (only that case is swallowed)."""
        probe = copy.copy(self)
        probe._shell_arithmetic = mode
        try:
            K, free, fixed = probe._shell_contract()
        except _ShellArithmeticUnavailable:
            return None
        probe.shell_stiffness = _readonly(K)
        free.setflags(write=False); fixed.setflags(write=False)
        probe.shell_free, probe.shell_fixed = free, fixed
        probe.shell_contract_sha = probe._shell_contract_sha()
        probe.identity = probe._identity()
        return {name: probe.__dict__[name] for name in self._SHELL_BINDING_FIELDS}

    def _restore_snapshot(self):
        state = self.__dict__
        return ({name: state[name] for name in self._RESTORE_STATE_FIELDS if name in state},
                len(self.curved_trial_failures))

    def _restore_rollback(self, snapshot):
        fields, failures = snapshot
        for name in self._RESTORE_STATE_FIELDS:
            if name not in fields:
                self.__dict__.pop(name, None)
        self.__dict__.update(fields)
        del self.curved_trial_failures[failures:]

    def _shell_contract(self):
        model = self.shell_model
        selected = {p.work.shell.element_id for p in self.pairs}
        if (set(model.mesh.elements) != selected or model.constraint_equations
                or model.load_cases or model.mesh.element_activity is not None):
            raise ValueError('multiple native/S3 owner requires represented active S3 elements and simple supports')
        n = model.mesh.dof_manager.total_dofs
        if n+self.context.count > self.maximum_coupled_coordinates:
            raise MemoryError('multiple native/S3 shell coordinate resource cap exceeded')
        K = np.zeros((n, n)); nodes = set()
        for eid, element in sorted(model.mesh.elements.items()):
            if type(element) is not NativeParityE4PLS3V2DShellElement or element.shell_section is not None:
                raise ValueError('multiple native/S3 owner requires exact elastic S3 V2D')
            material = model.materials.get(element.material_name)
            if (type(material) is not IsotropicMaterial or material.yield_stress != 0.
                    or material.hardening_curve is not None):
                raise ValueError('multiple native/S3 owner requires isotropic elastic skin material')
            nodes.update(element.node_ids)
            slots = np.asarray(element.get_dof_mapping(model.mesh), int)
            local = self._element_stiffness(element, model.mesh, material)
            if local.shape != (18, 18) or not np.all(np.isfinite(local)):
                raise ValueError('multiple native/S3 structural block invalid')
            K[np.ix_(slots, slots)] += local
        if nodes != set(model.mesh.nodes) or not np.allclose(K, K.T, rtol=1e-10, atol=1e-10):
            raise ValueError('multiple native/S3 orphan coordinates or nonsymmetric shell structure')
        for p in self.pairs:
            shell = p.work.shell
            element = model.mesh.elements[shell.element_id]
            fresh = S3SkinWorkAdapter.from_model(model, shell.body_id, shell.element_id, shell.side)
            combined = sparse.block_diag((sparse.eye(42, format='csr'), fresh.work_map), format='csr')
            if (tuple(element.node_ids) != shell.node_ids
                    or not np.array_equal(fresh.reference_positions, shell.reference_positions)
                    or not np.array_equal(fresh.director, shell.director)
                    or fresh.physical_offset != shell.physical_offset
                    or (fresh.work_map != shell.work_map).nnz
                    or not sparse.isspmatrix_csr(p.work.work_map)
                    or p.work.work_map.shape != combined.shape
                    or (combined != p.work.work_map).nnz):
                raise ValueError('multiple native/S3 physical skin adapter does not bind model')
        fixed = set()
        for bc in model.boundary_conditions:
            for dof, value in bc.get_constrained_dofs(model.mesh.dof_manager):
                if not np.isfinite(value) or value != 0.:
                    raise ValueError('multiple native/S3 owner admits zero shell supports only')
                fixed.add(int(dof))
        free = np.array(sorted(set(range(n))-fixed), int)
        if len(free) == 0 or np.linalg.matrix_rank(K[np.ix_(free, free)]) < len(free):
            raise ValueError('multiple native/S3 shell has a static mechanism or no deformable coordinates')
        return K, free, np.array(sorted(fixed), int)

    def _shell_contract_sha(self):
        K, free, fixed = self._shell_contract()
        return sha({'K': K, 'free': free, 'fixed': fixed,
            'elements': [{'id': eid, 'nodes': e.node_ids, 'formulation': e.formulation_id,
                'thickness': e.thickness, 'offset': e.reference_surface_offset, 'polarity': e.director_polarity,
                'coordinates': [self.shell_model.mesh.nodes[n].coords() for n in e.node_ids],
                'material': {name: getattr(self.shell_model.materials[e.material_name], name)
                    for name in ('name', 'elastic_modulus', 'poisson_ratio', 'density', 'yield_stress')}}
                for eid, e in sorted(self.shell_model.mesh.elements.items())],
            'pairs': [p.descriptor() for p in self.pairs]})

    def _identity_components(self):
        descriptor = {'profile': self.profile if self._material_mode else PROFILE, 'revision': 'uncondensed_explicit_native_s3_work_v1',
            'native': self.context.identity, 'support': sha256(self.context.support_identity).hexdigest(),
            'shell': self.shell_contract_sha, 'cached_shell': [self.shell_stiffness, self.shell_free, self.shell_fixed],
            'pairs': [p.descriptor() for p in self.pairs], 'beam_indices': self.beam_indices,
            'native_slots': self.context.slots, 'equilibrium_tolerance': self.equilibrium_tolerance,
            'maximum_pairs': self.maximum_pairs, 'maximum_coupled_coordinates': self.maximum_coupled_coordinates}
        if self._curved_mode:
            descriptor['trial_policy']='authoritative_increment_bounded_gap_fraction_preflight_complete_curved_path_v2'
        if self._material_mode and self.activation_settings is not None:
            from .native_material_activation import REVISION
            descriptor['activation'] = (REVISION, self.activation_settings)
        return descriptor

    def _identity(self):
        return sha(self._identity_components())

    def _guard(self):
        self.context.guard(); self.native._guard()
        if self._identity() != self.identity or self._shell_contract_sha() != self.shell_contract_sha:
            raise ValueError('multiple native/S3 definition/settings/model changed')

    def _contact(self, mechanical, displacement):
        if self._identity() != self.identity:
            raise ValueError('multiple native/S3 consumed work/settings changed')
        key = None
        if self._contact_reuse_enabled:
            # Recheck all formulation guards even on a geometry-identical hit.
            # Native material/resultant state is never stored or reused here.
            for p, i in zip(self.pairs, self.beam_indices):
                nodes = self.context.nodes[i]
                p.work._configuration(mechanical.positions[nodes], mechanical.position_low[nodes],
                    mechanical.cell_rotations[i], displacement)
            key = self._contact_cache_key(mechanical, displacement)
            entry = self._contact_reuse_entry
            if entry is not None and key == entry[0]:
                try:
                    intact = _candidate_work_signature(entry[1]) == entry[2]
                except Exception:
                    self._contact_reuse_entry = None
                    raise
                if not intact:
                    self._contact_reuse_entry = None
                    raise ValueError('cached native/S3 contact work mutated')
                self.candidate_evaluations += 1
                self.contact_reuse_hits += 1
                return entry[1]
            # Keep at most one complete evaluation. Failed evaluations replace
            # nothing; they cannot expose an earlier pose as the new candidate.
            self._contact_reuse_entry = None
        total = self.context.count+len(displacement)
        residual = np.zeros(total); tangent = sparse.csr_matrix((total, total)); pairs = []
        for p, i in zip(self.pairs, self.beam_indices):
            nodes = self.context.nodes[i]
            try:
                arguments = (mechanical.positions[nodes], mechanical.position_low[nodes],
                             mechanical.cell_rotations[i], displacement)
                if self._curved_mode and self._shell_arithmetic == self._FUSED_ARITHMETIC:
                    result = p.work.evaluate(*arguments, replay_norms=True)
                else:
                    result = p.work.evaluate(*arguments)
            except (ValueError,MemoryError) as exc:
                if self._curved_mode:
                    self.curved_trial_failures.append(dict(pair=p.name,error=type(exc).__name__+': '+str(exc),
                        integration=getattr(exc,'integration_diagnostics',None)))
                raise
            slots = np.r_[self.context.slots[i], self.context.count+np.arange(len(displacement))]
            np.add.at(residual, slots, result.residual)
            P = sparse.coo_matrix((np.ones(len(slots)), (np.arange(len(slots)), slots)),
                                  shape=(len(slots), total)).tocsr()
            tangent += P.T@result.tangent@P
            pairs.append((p, result)); self.pair_evaluations += 1
        self.candidate_evaluations += 1
        result = NativeMultiS3WorkResult(NativeMultiS3Physical(tuple(pairs)), _readonly(residual), tangent.tocsr(), tuple(pairs),
            self.profile if self._material_mode else PROFILE)
        if key is not None:
            for matrix in (result.tangent, *(r.tangent for _, r in pairs)):
                for array in (matrix.data, matrix.indices, matrix.indptr):
                    array.setflags(write=False)
            self._contact_reuse_entry = key, result, _candidate_work_signature(result)
        return result

    def _contact_cache_key(self, mechanical, displacement):
        return sha(dict(positions=mechanical.positions, low=mechanical.position_low,
            rotations=mechanical.cell_rotations, shell=displacement))

    def _contact_admissible(self, result):
        return all(r.contact.minimum_gap_m >= -p.gap_tolerance_m for p, r in result.pairs)

    def _candidate_geometry_admissible(self,mechanical,displacement):
        if not self._curved_mode:return True
        for p,i in zip(self.pairs,self.beam_indices):
            nodes=self.context.nodes[i]
            gap,_=p.work.geometry_gap(mechanical.positions[nodes],mechanical.position_low[nodes],
                                      mechanical.cell_rotations[i],displacement)
            if gap < -p.gap_tolerance_m:return False
        return True

    def _assemble(self, mechanical, displacement, parameter, origins):
        key = None
        if self._assembly_reuse_enabled:
            # Recheck the complete binding even on an identical hit. The exact
            # candidate assembly and material trials are disposable: no physical
            # state is committed, and a failed assembly replaces nothing.
            self._guard()
            for p, i in zip(self.pairs, self.beam_indices):
                nodes = self.context.nodes[i]
                p.work._configuration(mechanical.positions[nodes], mechanical.position_low[nodes],
                    mechanical.cell_rotations[i], displacement)
            key = sha(dict(positions=mechanical.positions, low=mechanical.position_low,
                frames=mechanical.nodal_frames, rotations=mechanical.cell_rotations,
                resultants=mechanical.resultants, shell=displacement, parameter=parameter,
                origins=sha(origins)))
            entry = self._assembly_reuse_entry
            if entry is not None and key == entry[0]:
                try:
                    intact = _candidate_assembly_signature(entry[1]) == entry[2]
                except Exception:
                    self._assembly_reuse_entry = None
                    raise
                if not intact:
                    self._assembly_reuse_entry = None
                    raise ValueError('cached native/S3 candidate assembly mutated')
                self.assembly_reuse_hits += 1
                return entry[1]
            self._assembly_reuse_entry = None
        residual, tangent, responses = self._assemble_structural(mechanical, displacement, parameter, origins)
        contact = self._contact(mechanical, displacement)
        residual += contact.residual
        for (p, result), i in zip(contact.pairs, self.beam_indices):
            block = result.tangent.toarray()
            if type(p.work) not in (NativeRetainedMaterialS3Work,NativeCurvedMaterialS3Work):
                block[:42, :42] = spatial_jacobian(result.residual[:42], block[:42, :42])
            slots = np.r_[self.context.slots[i], self.context.count+np.arange(len(displacement))]
            rows, cols = np.meshgrid(slots, slots, indexing='ij')
            np.add.at(tangent, (rows, cols), block)
        n = self.context.count; scaled = residual[:n]/self.context.scale
        metrics = (float(np.linalg.norm(scaled[self.context.equilibrium])),
            float(np.linalg.norm(scaled[self.context.compatibility])),
            float(np.linalg.norm(residual[n+self.shell_free])))
        if key is not None:
            residual.setflags(write=False); tangent.setflags(write=False)
            assembled = (residual, tangent, metrics, responses, contact)
            self._assembly_reuse_entry = key, assembled, _candidate_assembly_signature(assembled)
            return assembled
        return residual, tangent, metrics, responses, contact

    def _activation_fraction(self, start, u_start, step, fraction, contact, cancellation_token):
        if self._curved_mode:
            from anyfesolver.control import cancellation_safe_point
            cancellation_safe_point(cancellation_token,'curved S3 initial gap fraction')
            end=self.context.advance(start,step[:self.context.count]*fraction)
            u_end=u_start+step[self.context.count:]*fraction
            factor=1.
            for (p,result),i in zip(contact.pairs,self.beam_indices):
                nodes=self.context.nodes[i]
                try:
                    gap,_=p.work.geometry_gap(end.positions[nodes],end.position_low[nodes],end.cell_rotations[i],u_end)
                except ValueError as exc:
                    if self._recoverable_trial_error(exc):return fraction
                    raise
                if gap < -p.gap_tolerance_m:
                    room=result.contact.minimum_gap_m+p.gap_tolerance_m
                    decrease=result.contact.minimum_gap_m-gap
                    if room<=0 or decrease<=0:
                        raise RuntimeError('curved S3 has no admissible gap step')
                    factor=min(factor,.8*room/decrease)
            return fraction*factor
        if self.activation_settings is None:
            return fraction
        from .native_material_activation import activation_fraction
        return activation_fraction(self, start, u_start, step, fraction, contact, cancellation_token)

    def _swept_admissible(self, start, u_start, increment, end, u_end, *_, cancellation_token=None):
        if self._curved_mode:
            # Curved certification uses the complete authoritative path, not
            # endpoint forces/tangents. Admission and equilibrium independently
            # evaluate endpoint work; do not repeat their quadrature here.
            self._guard()
            admitted = []
            for p, i in zip(self.pairs, self.beam_indices):
                nodes=self.context.nodes[i];work=p.work
                pose=work._configuration(start.positions[nodes],start.position_low[nodes],start.cell_rotations[i],u_start)
                translations=np.array([increment[6*index:6*index+3] for index in nodes])
                angular=np.array([increment[self.context.nodal_count+24*i+3*c:
                                            self.context.nodal_count+24*i+3*c+3] for c in (0,1)])
                control=work.certify_path(pose,translations,angular,increment[self.context.count:],
                    gap_tolerance_m=p.gap_tolerance_m,maximum_intervals=p.maximum_swept_intervals,
                    cancellation_token=cancellation_token)
                admitted.append(control.certified)
            return all(admitted)
        first = self._contact(start, u_start); last = self._contact(end, u_end)
        admitted = []
        for (p, r0), (_, r1), i in zip(first.pairs, last.pairs, self.beam_indices):
            a, b = r0.contact, r1.contact
            if type(p.work) is NativeRetainedMaterialS3Work:
                nodes=self.context.nodes[i];work=p.work
                pose=work._configuration(start.positions[nodes],start.position_low[nodes],
                    start.cell_rotations[i],u_start)
                translations=(end.positions[nodes]-start.positions[nodes]
                              +end.position_low[nodes]-start.position_low[nodes])
                angular=np.array([increment[self.context.nodal_count+24*i+3*c:
                                            self.context.nodal_count+24*i+3*c+3] for c in (0,1)])
                try:work.certify_orientation_path(pose,translations,angular,u_end-u_start,
                    p.maximum_swept_intervals)
                except ValueError as exc:
                    if self._recoverable_trial_error(exc):return False
                    raise
                def geometry_query(state,u):
                    return work.geometry_gap(state.positions[nodes],state.position_low[nodes],
                        state.cell_rotations[i],u)
                admitted.append(native_s3_swept_admissible(self.context,work,i,p.gap_tolerance_m,
                    p.maximum_swept_intervals,self._recoverable_trial_error,start,u_start,increment,
                    end,u_end,a.minimum_gap_m,b.minimum_gap_m,a.minimum_rim_clearance_m,
                    b.minimum_rim_clearance_m,geometry_only=True,geometry_query=geometry_query,
                    material_radius=work.contact.radius))
                continue
            admitted.append(native_s3_swept_admissible(self.context, p.work, i, p.gap_tolerance_m,
                p.maximum_swept_intervals, self._recoverable_trial_error, start, u_start, increment,
                end, u_end, a.minimum_gap_m, b.minimum_gap_m,
                a.minimum_rim_clearance_m, b.minimum_rim_clearance_m))
        return all(admitted)

    def _swept_cancellable(self,*args,cancellation_token=None):
        # Keep historical subclasses' dynamic dispatch and argument contracts.
        if self._curved_mode:
            return self._swept_admissible(*args,cancellation_token=cancellation_token)
        return self._swept_admissible(*args)

    def _record(self, target, mechanical, displacement, origins, previous_sha256):
        if type(target) is not int or not 1 <= target <= len(self.program.targets):
            raise ValueError('multiple native/S3 checkpoint target')
        mechanical = self.context.make(mechanical.descriptor())
        u = np.asarray(displacement, float).reshape(-1)
        if (u.size != self.shell_stiffness.shape[0] or not np.all(np.isfinite(u))
                or np.any(u[self.shell_fixed] != 0.)):
            raise ValueError('multiple native/S3 checkpoint shell displacement')
        r, K, metrics, responses, contact = self._assemble(mechanical, u, self.program.targets[target-1], origins)
        _, correction = self._step(r, K)
        if max(*metrics, correction) > self.equilibrium_tolerance or not self._contact_admissible(contact):
            raise ValueError('multiple native/S3 checkpoint requires complete equilibrium and every pair gap')
        histories = self.context.histories(tuple(r.history for r in responses))
        row = {'target': target, 'previous_sha256': previous_sha256,
            'mechanical': mechanical.descriptor(), 'histories': histories, 'shell_displacement': u,
            'contact': contact.contact.to_dict()}
        accepted = NativeMultiS3Accepted(mechanical, histories, _readonly(u), target, contact)
        return accepted, _load(canonical(row).decode('ascii'))

    def _checkpoint(self, records):
        self._guard()
        if type(records) is not tuple or len(records) > len(self.program.targets):
            raise ValueError('multiple native/S3 checkpoint extent')
        body = {'schema': self.schema, 'profile': self.profile, 'identity': self.identity,
                'completed_targets': len(records), 'records': records}
        raw = canonical({**body, 'checkpoint_sha256': sha(body)})
        if len(raw) > MAX_BYTES: raise MemoryError('multiple native/S3 checkpoint byte limit')
        return raw

    def _restore(self, raw, expected_sha256):
        self._guard()
        if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('multiple native/S3 external checkpoint hash mismatch')
        value = _load(raw.decode('ascii'))
        if type(value) is not dict or set(value) != {'schema', 'profile', 'identity', 'completed_targets', 'records', 'checkpoint_sha256'}:
            raise ValueError('multiple native/S3 checkpoint schema')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        def _binding_token(value):
            text = repr(value)
            return text if len(text) <= 96 else text[:93] + '...'
        body_sha256 = sha(body)
        # Legacy host-BLAS owners: the recorded identity may have been sealed
        # with the SkylakeX-observed stiffness bits. Adopt the fixed fused
        # arithmetic only if it reproduces that identity exactly; every other
        # component, the model and the settings remain exactly checked.
        #
        # Transaction: the owner is committed to an alternate arithmetic only
        # for the replay below, and ANY exception (refusal, replay mismatch,
        # cancellation, unexpected error) restores every binding field, cache
        # entry and counter the replay can touch, so a failed restore leaves
        # the owner exactly as it was.
        snapshot = self._restore_snapshot()
        try:
            return self._restore_bound(raw, value, body_sha256, _binding_token)
        except BaseException:
            self._restore_rollback(snapshot)
            raise

    def _restore_bound(self, raw, value, body_sha256, _binding_token):
        if (type(value['identity']) is str and value['identity'] != self.identity
                and value['schema'] == self.schema and value['profile'] == self.profile
                and body_sha256 == value['checkpoint_sha256']):
            # Only an authorised recorded identity triggers an alternate full
            # assembly; any other identity goes straight to the binding refusal.
            alternate = self._authorised_alternate_arithmetic(value['identity'])
            adopted = None if alternate is None else self._probe_shell_arithmetic(alternate)
            if adopted is not None and adopted['identity'] == value['identity']:
                self.__dict__.update(adopted)
                # Stiffness and curved contact error bounds use the selected arithmetic.
                self._assembly_reuse_entry = None
                self._contact_reuse_entry = None
                self._construction_contact_entry = None
                # The initial state was built in the previous arithmetic.
                self._rebind_initial()
        disagreements = []
        if value['schema'] != self.schema:
            disagreements.append('schema checkpoint=%s owner=%s' % (_binding_token(value['schema']), _binding_token(self.schema)))
        if value['profile'] != self.profile:
            disagreements.append('profile checkpoint=%s owner=%s' % (_binding_token(value['profile']), _binding_token(self.profile)))
        if value['identity'] != self.identity:
            disagreements.append('identity checkpoint=%s owner=%s owner components %s'
                % (_binding_token(value['identity']), _binding_token(self.identity),
                   component_digests(self._identity_components())))
        if body_sha256 != value['checkpoint_sha256']:
            disagreements.append('checkpoint_sha256 body=%s recorded=%s' % (_binding_token(body_sha256), _binding_token(value['checkpoint_sha256'])))
        if disagreements:
            raise ValueError('multiple native/S3 checkpoint definition/settings binding: ' + '; '.join(disagreements))
        count = value['completed_targets']
        if type(count) is not int or not 0 <= count <= len(self.program.targets) or type(value['records']) is not list or len(value['records']) != count:
            raise ValueError('multiple native/S3 checkpoint target extent')
        accepted = self.initial; records = (); previous = sha({'schema': self.schema, 'identity': self.identity})
        for index, row in enumerate(value['records'], 1):
            if (type(row) is not dict or set(row) != {'target', 'previous_sha256', 'mechanical', 'histories', 'shell_displacement', 'contact'}
                    or row['target'] != index or row['previous_sha256'] != previous):
                raise ValueError('multiple native/S3 checkpoint accepted chain')
            mechanical = self.context.make(row['mechanical'], decoded=True)
            accepted, replay = self._record(index, mechanical, row['shell_displacement'], accepted.native_histories, previous)
            if canonical(row) != canonical(replay): raise ValueError('multiple native/S3 checkpoint replay mismatch'
                + replay_difference_message(row, replay, 'record %d' % index))
            records += (replay,); previous = sha(replay)
        if self._checkpoint(records) != raw: raise ValueError('multiple native/S3 noncanonical checkpoint replay')
        return accepted, records

    def solve(self, *, checkpoint=None, expected_sha256=None, stop_after=None,
              cancellation_token=None, progress=None, initial_guess=None):
        operation=self.context.operation(cancellation_token) if self._fibre_mode else nullcontext()
        with operation:
            result=solve_native_shell(self, self.schema, self.run_type, checkpoint=checkpoint,
                expected_sha256=expected_sha256, stop_after=stop_after,
                cancellation_token=cancellation_token, progress=progress,initial_guess=initial_guess)
        return replace(result,profile=self.profile) if self._material_mode else result

    def make_initial_guess(self,source_analysis,checkpoint,expected_sha256):
        """Prepare coordinates-only initialization from a bound first target."""
        from .native_initial_guess import make_guess
        return make_guess(self,source_analysis,checkpoint,expected_sha256)

    def _initial_guess_trial(self,guess,cancellation_token):
        if not self._fibre_mode or not self._curved_mode:
            raise ValueError('first-target guess requires curved physical-fibre owner')
        from .native_initial_guess import prepare_trial
        return prepare_trial(self,guess,cancellation_token)

    def _begin_candidate_context(self,cancellation_token):
        if self._fibre_mode:self.context.activate_cancellation(cancellation_token)
        self._contact_reuse_enabled = self._fibre_mode and self._curved_mode
        self._assembly_reuse_enabled = self._contact_reuse_enabled
        # A candidate context starts from the construction evaluation of
        # the initial pose; every other entry was cleared at context end.
        self._contact_reuse_entry = self._construction_contact_entry
        self._assembly_reuse_entry = None

    def _end_candidate_context(self):
        self._contact_reuse_enabled = False
        self._contact_reuse_entry = None
        self._assembly_reuse_enabled = False
        self._assembly_reuse_entry = None

    def recover_material(self,checkpoint,expected_sha256,*,include_stations=False,cancellation_token=None):
        """Return bound committed fibre fields and separate energy summaries."""
        from .native_fibre_results import recover_native_material
        return recover_native_material(self,checkpoint,expected_sha256,
            include_stations=include_stations,cancellation_token=cancellation_token)


def create_native_contact_analysis(definitions, boundaries, program, shell_model, pairs, *,
        equilibrium_tolerance, maximum_pairs=8, maximum_coupled_coordinates=256,
        activation_settings=None):
    """Create the explicit retained multiple-S3 development owner.

    Native definitions stay under NativeBeamAnalysis, not generic FEModel.
    """
    owner = NativeRetainedMultiS3Analysis
    extra = {}
    if activation_settings is not None:
        if type(pairs) is tuple and any(type(p.work) is NativeCurvedMaterialS3Work for p in pairs):
            raise ValueError('curved S3 candidate does not admit the straight material activation controller')
        if (type(pairs) is tuple and pairs
                and all(type(p.work) is NativeRetainedMaterialS3Work for p in pairs)):
            extra['activation_settings'] = activation_settings
        else:
            from .native_activation_static import NativeActivationMultiS3Analysis
            owner = NativeActivationMultiS3Analysis
            extra['activation_settings'] = activation_settings
    return owner(definitions, boundaries, program, shell_model, pairs,
        equilibrium_tolerance=equilibrium_tolerance, maximum_pairs=maximum_pairs,
        maximum_coupled_coordinates=maximum_coupled_coordinates, **extra)
