"""Restricted coupled retained B3-GE and elastic S3 static contact owner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite

import numpy as np

from anymaterial import IsotropicMaterial
from anyfesolver._ge_beam3_generalized_static_boundary import spatial_jacobian
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_retained_nodal_loading import (
    Context as NativeNodalContext,
    Program as NativeNodalProgram,
)
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import MAX_BYTES, _load
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.fe_core import FEModel

from .native_retained_line_s3_work import (
    NativeRetainedLineS3Work,
    NativeRetainedLineS3WorkResult,
)
from .native_retained_station import _readonly
from .s3_skin_work import S3SkinWorkAdapter
from .native_material_s3 import NativeRetainedMaterialS3Work, PROFILE as MATERIAL_PROFILE


PROFILE = "NATIVE_B3_GE_ELASTIC_S3_INTERIOR_STATIC_DEV_V1"
CHECKPOINT_SCHEMA = "NATIVE_B3_GE_ELASTIC_S3_CHECKPOINT_DEV_V1"


@dataclass(frozen=True)
class NativeRetainedS3Accepted:
    mechanical: object
    native_histories: tuple
    shell_displacement: np.ndarray
    completed_targets: int
    contact: NativeRetainedLineS3WorkResult


@dataclass(frozen=True)
class NativeRetainedS3Run:
    status: str
    accepted: NativeRetainedS3Accepted
    failure: str | None
    checkpoint: bytes
    profile: str = PROFILE

    @property
    def checkpoint_sha256(self) -> str:
        return sha256(self.checkpoint).hexdigest()


class NativeRetainedRoundS3Analysis:
    """One native beam and one elastic S3, both active in one Newton system."""

    def __init__(self, definitions: tuple[NativeBeamDefinition, ...],
                 native_boundaries: tuple, program: NativeNodalProgram,
                 shell_model: FEModel, contact: NativeRetainedLineS3Work, *,
                 gap_tolerance_m: float, equilibrium_tolerance: float,
                 maximum_swept_intervals: int, activation_settings=None):
        if (type(definitions) is not tuple or len(definitions) != 1
                or type(definitions[0]) is not NativeBeamDefinition
                or type(program) is not NativeNodalProgram):
            raise TypeError("native/S3 owner needs one exact native beam and program")
        if type(shell_model) is not FEModel or type(contact) not in (NativeRetainedLineS3Work,NativeRetainedMaterialS3Work):
            raise TypeError("native/S3 owner needs exact S3 model and contact work")
        self.profile=MATERIAL_PROFILE if type(contact) is NativeRetainedMaterialS3Work else PROFILE
        if type(contact) is NativeRetainedMaterialS3Work and shell_model.mesh.dof_manager.total_dofs!=18:
            raise ValueError('single material-area owner requires one 18-DOF S3')
        if (type(gap_tolerance_m) not in (int, float)
                or not isfinite(gap_tolerance_m) or gap_tolerance_m <= 0.0
                or type(equilibrium_tolerance) not in (int, float)
                or not isfinite(equilibrium_tolerance) or equilibrium_tolerance <= 0.0
                or type(maximum_swept_intervals) is not int
                or maximum_swept_intervals < 1):
            raise ValueError("native/S3 owner needs positive finite solver controls")
        if activation_settings is not None:
            if type(contact) is not NativeRetainedMaterialS3Work:
                raise ValueError('single activation settings require the material-area owner')
            from .native_material_activation import validate_settings
            validate_settings(activation_settings, (contact,), (gap_tolerance_m,))
        self.activation_settings = activation_settings
        self.activation_total_queries = 0
        self.native = NativeBeamAnalysis(definitions, native_boundaries)
        self.context = NativeNodalContext(self.native.model, program)
        if (len(self.context.elements) != 1
                or type(self.context.elements[0][1]) is not NativeGeneralizedStaticElement):
            raise ValueError("native/S3 owner needs one native generalized element")
        element_id, element = self.context.elements[0]
        kernel = contact.contact
        if (element_id != kernel.beam_element_id
                or tuple(element.node_ids) != kernel.beam_node_ids
                or sha(element.operator.reference.canonical_data())
                   != sha(kernel.beam_reference.canonical_data())):
            raise ValueError("native/S3 contact does not bind native beam")
        self.shell_model = shell_model
        self.contact = contact
        self.program = program
        self.gap_tolerance = float(gap_tolerance_m)
        self.equilibrium_tolerance = float(equilibrium_tolerance)
        self.maximum_swept_intervals = maximum_swept_intervals
        self.shell_stiffness, self.shell_free, self.shell_fixed = self._shell_contract()
        self.shell_contract_sha = self._shell_contract_sha()
        self.identity = self._identity()
        displacement = np.zeros(self.shell_stiffness.shape[0])
        initial_native = self.context.initial
        first = self._contact(initial_native.mechanical, displacement)
        scale = max(1.0, float(np.max(np.abs(kernel.beam_reference.coordinates))),
                    float(np.max(np.abs(contact.shell.reference_positions))))
        if first.contact.minimum_gap_m < -256.0*np.finfo(float).eps*scale:
            raise ValueError("unresolved initial native/S3 contact penetration: "
                             f"{first.contact.minimum_gap_m} m")
        self.initial = NativeRetainedS3Accepted(
            initial_native.mechanical, initial_native.histories,
            _readonly(displacement), 0, first,
        )

    def _identity(self) -> str:
        descriptor = dict(
            profile=self.profile,
            native_model=self.context.identity,
            native_support_sha256=sha256(self.context.support_identity).hexdigest(),
            shell_contract=self.shell_contract_sha,
            contact=self.contact.identity,
            gap_tolerance_m=self.gap_tolerance,
            equilibrium_tolerance=self.equilibrium_tolerance,
            maximum_swept_intervals=self.maximum_swept_intervals,
        )
        if self.activation_settings is not None:
            from .native_material_activation import REVISION
            descriptor['activation'] = (REVISION, self.activation_settings)
        return sha(descriptor)

    def _shell_contract(self):
        model = self.shell_model
        if (len(model.mesh.elements) != 1 or model.constraint_equations
                or model.load_cases or model.mesh.element_activity is not None):
            raise ValueError("native/S3 owner needs one active S3 with simple supports")
        element_id = self.contact.shell.element_id
        element = model.mesh.elements.get(element_id)
        if (type(element) is not NativeParityE4PLS3V2DShellElement
                or element.shell_section is not None
                or tuple(element.node_ids) != self.contact.shell.node_ids):
            raise ValueError("native/S3 owner needs exact elastic S3 V2D")
        material = model.materials.get(element.material_name)
        if (type(material) is not IsotropicMaterial
                or material.yield_stress != 0.0
                or material.hardening_curve is not None):
            raise ValueError("native/S3 owner needs isotropic elastic S3 material")
        fresh = S3SkinWorkAdapter.from_model(
            model, self.contact.shell.body_id, element_id, self.contact.shell.side,
        )
        if (not np.array_equal(fresh.reference_positions,
                               self.contact.shell.reference_positions)
                or not np.array_equal(fresh.director, self.contact.shell.director)
                or fresh.physical_offset != self.contact.shell.physical_offset
                or (fresh.work_map != self.contact.shell.work_map).nnz):
            raise ValueError("native/S3 shell contact adapter does not bind model")
        n = model.mesh.dof_manager.total_dofs
        stiffness = np.zeros((n, n))
        slots = np.asarray(element.get_dof_mapping(model.mesh), dtype=int)
        local = element.compute_stiffness_matrix(model.mesh, material)
        if local.shape != (18, 18) or not np.all(np.isfinite(local)):
            raise ValueError("native/S3 shell stiffness is invalid")
        stiffness[np.ix_(slots, slots)] += local
        if not np.allclose(stiffness, stiffness.T, rtol=1e-10, atol=1e-10):
            raise ValueError("native/S3 shell stiffness is nonsymmetric")
        fixed = set()
        for boundary in model.boundary_conditions:
            for dof, value in boundary.get_constrained_dofs(model.mesh.dof_manager):
                if not np.isfinite(value) or value != 0.0:
                    raise ValueError("native/S3 owner admits only zero shell supports")
                fixed.add(int(dof))
        free = np.array(sorted(set(range(n))-fixed), dtype=int)
        if len(free) == 0 or np.linalg.matrix_rank(stiffness[np.ix_(free, free)]) < len(free):
            raise ValueError("native/S3 shell has a static rigid mechanism")
        return stiffness, free, np.array(sorted(fixed), dtype=int)

    def _shell_contract_sha(self):
        stiffness, free, fixed = self._shell_contract()
        element = self.shell_model.mesh.elements[self.contact.shell.element_id]
        material = self.shell_model.materials[element.material_name]
        return sha(dict(
            stiffness=stiffness, free=free, fixed=fixed,
            contact=self.contact._fingerprint(),
            kernel=self.contact.contact._fingerprint(),
            element=dict(identity=self.contact.shell.element_id,
                         nodes=element.node_ids,
                         formulation=element.formulation_id,
                         thickness=element.thickness,
                         reference_surface_offset=element.reference_surface_offset,
                         director_polarity=element.director_polarity),
            node_coordinates=[self.shell_model.mesh.nodes[node].coords()
                              for node in element.node_ids],
            material=dict(name=material.name,
                          elastic_modulus=material.elastic_modulus,
                          poisson_ratio=material.poisson_ratio,
                          density=material.density,
                          yield_stress=material.yield_stress),
        ))

    def _guard(self):
        self.context.guard()
        if self._identity() != self.identity:
            raise ValueError("native/S3 definition/settings changed")
        if self._shell_contract_sha() != self.shell_contract_sha:
            raise ValueError("native/S3 shell model/contact definition changed")
        self.native._guard()

    def _record(self, target: int, mechanical, displacement, origins,
                previous_sha256: str):
        if type(target) is not int or not 1 <= target <= len(self.program.targets):
            raise ValueError("native/S3 checkpoint target")
        mechanical = self.context.make(mechanical.descriptor())
        shell_u = np.asarray(displacement, dtype=float).reshape(-1)
        if (shell_u.size != self.shell_stiffness.shape[0]
                or not np.all(np.isfinite(shell_u))
                or np.any(shell_u[self.shell_fixed] != 0.0)):
            raise ValueError("native/S3 checkpoint shell displacement")
        residual, tangent, metrics, responses, contact = self._assemble(
            mechanical, shell_u, self.program.targets[target-1], origins,
        )
        _, correction = self._step(residual, tangent)
        if max(*metrics, correction) > self.equilibrium_tolerance:
            raise ValueError("native/S3 checkpoint requires coupled equilibrium")
        if contact.contact.minimum_gap_m < -self.gap_tolerance:
            raise ValueError("native/S3 checkpoint gap tolerance")
        histories = self.context.histories(
            tuple(response.history for response in responses)
        )
        physical = contact.contact
        summary = dict(
            potential_j=physical.potential_j,
            minimum_gap_m=physical.minimum_gap_m,
            minimum_rim_clearance_m=physical.minimum_rim_clearance_m,
            beam_force_n=physical.beam_force_n,
            shell_force_n=physical.shell_force_n,
            beam_moment_nm=physical.beam_moment_nm,
            shell_moment_nm=physical.shell_moment_nm,
            active_intervals=physical.active_intervals,
            integration_intervals=physical.integration_intervals,
            integration_evaluations=physical.integration_evaluations,
            energy_error_j=physical.energy_error_j,
            force_error_n=physical.force_error_n,
            tangent_error_n_per_m=physical.tangent_error_n_per_m,
        )
        if type(self.contact) is NativeRetainedMaterialS3Work:
            summary.update(measure='fixed_reference_material_area',
                minimum_gap_estimate_m=physical.minimum_gap_estimate_m)
        row = dict(
            target=target, previous_sha256=previous_sha256,
            mechanical=mechanical.descriptor(),
            histories=histories, shell_displacement=shell_u,
            contact=summary,
        )
        accepted = NativeRetainedS3Accepted(
            mechanical, histories, _readonly(shell_u), target, contact,
        )
        return accepted, _load(canonical(row).decode("ascii"))

    def _checkpoint(self, records: tuple[dict, ...]) -> bytes:
        self._guard()
        if type(records) is not tuple or len(records) > len(self.program.targets):
            raise ValueError("native/S3 checkpoint record extent")
        body = dict(schema=CHECKPOINT_SCHEMA, profile=self.profile,
                    identity=self.identity, completed_targets=len(records),
                    records=records)
        raw = canonical({**body, "checkpoint_sha256": sha(body)})
        if len(raw) > MAX_BYTES:
            raise ValueError("native/S3 checkpoint byte limit")
        return raw

    def _restore(self, raw: bytes, expected_sha256: str):
        self._guard()
        if (type(raw) is not bytes or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("native/S3 external checkpoint hash mismatch")
        value = _load(raw.decode("ascii"))
        keys = {"schema", "profile", "identity", "completed_targets",
                "records", "checkpoint_sha256"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("native/S3 checkpoint schema")
        body = {key: item for key, item in value.items()
                if key != "checkpoint_sha256"}
        if (value["schema"] != CHECKPOINT_SCHEMA
                or value["profile"] != self.profile
                or value["identity"] != self.identity
                or value["checkpoint_sha256"] != sha(body)):
            raise ValueError("native/S3 checkpoint definition/settings binding")
        count = value["completed_targets"]
        if (type(count) is not int or not 0 <= count <= len(self.program.targets)
                or type(value["records"]) is not list
                or len(value["records"]) != count):
            raise ValueError("native/S3 checkpoint target extent")
        accepted = self.initial
        previous = sha(dict(schema=CHECKPOINT_SCHEMA, identity=self.identity))
        records = ()
        for index, row in enumerate(value["records"], 1):
            if (type(row) is not dict or set(row) !=
                    {"target", "previous_sha256", "mechanical", "histories",
                     "shell_displacement", "contact"}
                    or row["target"] != index
                    or row["previous_sha256"] != previous):
                raise ValueError("native/S3 checkpoint accepted chain")
            mechanical = self.context.make(row["mechanical"], decoded=True)
            values = row["shell_displacement"]
            if (type(values) is not list
                    or len(values) != self.shell_stiffness.shape[0]
                    or any(type(item) is not float or not isfinite(item)
                           for item in values)):
                raise ValueError("native/S3 checkpoint shell displacement schema")
            proposed, replay = self._record(
                index, mechanical, np.array(values),
                accepted.native_histories, previous,
            )
            if canonical(row) != canonical(replay):
                raise ValueError("native/S3 checkpoint physical replay mismatch")
            accepted = proposed
            records = (*records, replay)
            previous = sha(replay)
        if self._checkpoint(records) != raw:
            raise ValueError("native/S3 noncanonical checkpoint replay")
        return accepted, records

    def _contact(self, mechanical, displacement):
        nodes = self.context.nodes[0]
        return self.contact.evaluate(
            mechanical.positions[nodes], mechanical.position_low[nodes],
            mechanical.cell_rotations[0], displacement,
        )

    def _assemble_structural(self, mechanical, displacement, parameter, origins):
        """Shared retained/native and S3 blocks, before any contact profile."""
        self._guard()
        native_r, native_j, _, responses, _ = self.context.assemble(
            mechanical, parameter, origins,
        )
        n_native = self.context.count
        n_shell = len(displacement)
        residual = np.r_[native_r, self.shell_stiffness@displacement]
        tangent = np.zeros((n_native+n_shell, n_native+n_shell))
        tangent[:n_native, :n_native] = native_j
        tangent[n_native:, n_native:] = self.shell_stiffness
        return residual, tangent, responses

    def _add_spatial_contact(self, residual, tangent, contact_residual, contact_tangent):
        """Add complete work blocks once; callers own chart conversion."""
        n_native = self.context.count
        n_shell = self.shell_stiffness.shape[0]
        slots = np.asarray((*self.context.slots[0],
                            *(n_native+i for i in range(n_shell))), dtype=int)
        np.add.at(residual, slots, contact_residual)
        rows, columns = np.meshgrid(slots, slots, indexing="ij")
        np.add.at(tangent, (rows, columns), contact_tangent)
        native_scaled = residual[:n_native]/self.context.scale
        metrics = (
            float(np.linalg.norm(native_scaled[self.context.equilibrium])),
            float(np.linalg.norm(native_scaled[self.context.compatibility])),
            float(np.linalg.norm(residual[n_native+self.shell_free])),
        )
        return metrics

    def _assemble(self, mechanical, displacement, parameter, origins):
        residual, tangent, responses = self._assemble_structural(
            mechanical, displacement, parameter, origins)
        contact = self._contact(mechanical, displacement)
        block = contact.tangent.toarray()
        if type(self.contact) is not NativeRetainedMaterialS3Work:
            block[:42, :42] = spatial_jacobian(contact.residual[:42], block[:42, :42])
        metrics = self._add_spatial_contact(residual, tangent, contact.residual, block)
        return residual, tangent, metrics, responses, contact

    def _step(self, residual, tangent):
        n_native = self.context.count
        free = np.r_[self.context.free, n_native+self.shell_free]
        step = np.zeros(len(residual))
        step[free] = np.linalg.solve(tangent[np.ix_(free, free)], -residual[free])
        if not np.all(np.isfinite(step)):
            raise ValueError("native/S3 Newton step is nonfinite")
        correction = max(
            float(np.max(np.abs(step[:n_native]/self.context.scale))),
            float(np.max(np.abs(step[n_native:]))/self.context.length),
        )
        return step, correction

    def solve_material_increment(self, transaction, parameter, *, cancellation_token=None):
        """Bounded opt-in material increment on this retained structural owner.

        Uses the same constraint solve, equilibrium scale and iteration limits.
        Total physical-origin chart coordinates use their declared tangent.
        Unresolved contact transitions require caller continuation/event cutback;
        no rejected Newton iterate can advance the transaction's physical state.
        """
        from anyfesolver.control import cancellation_safe_point
        if type(self.contact) is NativeRetainedMaterialS3Work:
            raise ValueError('normal material-patch owner does not admit the existing friction transaction')
        from .native_surface_transaction import NativeS3MaterialTransaction

        if type(transaction) is not NativeS3MaterialTransaction or transaction.contributor.owner is not self:
            raise ValueError('material transaction does not belong to native/S3 owner')
        n = self.context.count
        increment = np.zeros(n+self.shell_stiffness.shape[0])
        proposal = None
        for iteration in range(self.program.max_iterations+1):
            cancellation_safe_point(cancellation_token, 'native material equilibrium')
            if proposal is None:
                proposal = transaction.propose(increment[:n], increment[n:], parameter,
                                               cancellation_token=cancellation_token)
            else:
                # The accepted line-search candidate already has this exact
                # origin and total increment. Preserve every binding/deadline
                # guard; physical commit still independently reevaluates it.
                if proposal.origin_sha256 != transaction.origin_sha256:
                    raise ValueError('native material Newton candidate origin changed')
            assembly = proposal.assembly
            step, correction = self._step(assembly.residual, assembly.origin_chart_tangent)
            if max(*assembly.metrics, correction) <= self.equilibrium_tolerance:
                return transaction.commit(proposal, cancellation_token=cancellation_token)
            if iteration == self.program.max_iterations:
                raise RuntimeError('native material Newton iteration limit; physical state unchanged')
            accepted = False
            material_trial_reason = 'no candidate evaluated'
            for cut in range(self.program.max_backtracks+1):
                cancellation_safe_point(cancellation_token, 'native material backtrack')
                candidate_increment = increment+(0.5**cut)*step
                try:
                    # The complete continuous gap gate is independent of
                    # friction history. Apply the identical gate before
                    # expensive replay on candidates that cannot be accepted.
                    # The full proposal and post-assembly gate still run for
                    # every admitted candidate; no sampled-gap shortcut.
                    pair=transaction.contributor.pair
                    if getattr(pair,'gap_check',None)=='straight_patch':
                        from .native_surface_extrema import straight_patch_s3_gap
                        from .native_surface_s3_friction import NativeS3MaterialConfiguration
                        origin=transaction.accepted
                        posed=self.context.advance(origin.mechanical,candidate_increment[:n])
                        nodes=self.context.nodes[0]
                        geometry=NativeS3MaterialConfiguration(posed.positions[nodes],posed.position_low[nodes],
                            posed.cell_rotations[0],origin.shell_displacement+candidate_increment[n:])
                        if straight_patch_s3_gap(pair.rule,pair.target,geometry).lower_bound < -self.gap_tolerance:
                            material_trial_reason='candidate gap tolerance'
                            continue
                    candidate = transaction.propose(candidate_increment[:n], candidate_increment[n:], parameter,
                                                      cancellation_token=cancellation_token)
                except MemoryError as exc:
                    # A bounded support proof can be inconclusive for a large
                    # trial. Retry only this named geometric work limit, under
                    # the existing backtrack/deadline bounds; never actual OOM.
                    if str(exc)!='native friction aggregate support interval cap exhausted':
                        raise
                    material_trial_reason=str(exc)
                    continue
                except ValueError as exc:
                    material_trial_reason = str(exc)
                    if str(exc).startswith(('native/S3 material history path requires subdivision',
                            'native friction integration requires resolved historical contact envelope',
                            'native friction integration requires swept whole-patch target ownership',
                            'native material path exceeds admitted rotation chart',
                            'first-touch transverse monotone path unresolved',
                            'first-touch physical ownership unresolved',
                            'first-touch needs resolved open and closed endpoints',
                            'first-touch target area bound unresolved',
                            'first-touch root residual unresolved',
                            'first-touch derivative lost transversality',
                            'first-touch differentiated root mismatch',
                            'first-touch endpoint work geometry mismatch',
                            'retained nodal chart requires cutback', 'retained cell chart requires cutback')):
                        continue
                    raise
                if candidate.assembly.material.contribution.normal_gap_m < -self.gap_tolerance:
                    material_trial_reason = 'candidate gap tolerance'
                    continue
                if max(candidate.assembly.metrics) < max(assembly.metrics):
                    increment = candidate_increment
                    proposal = candidate
                    accepted = True
                    break
                material_trial_reason = ('equilibrium merit did not decrease: '
                    f'{max(candidate.assembly.metrics):.17g} >= {max(assembly.metrics):.17g}')
            if not accepted:
                raise RuntimeError('native material backtracking/event limit; physical state unchanged; '+material_trial_reason)

    @staticmethod
    def _recoverable_trial_error(exc):
        message = str(exc)
        if type(exc) is MemoryError:
            return message.startswith((
                "native line-S3 integration interval budget exhausted",
                "native line-S3 parameter resolution exhausted",
                "straight normal integration interval budget exhausted",
            ))
        return type(exc) is ValueError and message.startswith((
            "native line-S3 curve approaches unadmitted physical rim",
            "native line-S3 has unresolved or degenerate geometry",
            "native line-S3 activation interval is unresolved",
            "native line-S3 activation is ambiguous",
            "retained generalized step requires cutback",
            "native material S3 patch approaches unadmitted physical rim",
            "native material S3 has unresolved target geometry",
            "native/S3 material history path requires subdivision",
            "material plane angular endpoint/extremum gap unresolved",
            "S3 material skin is degenerate",
            "S3 material skin is inverted",
        ))

    def _activation_fraction(self, start, u_start, step, fraction, contact, cancellation_token):
        if self.activation_settings is None:
            return fraction
        from .native_material_activation import activation_fraction
        return activation_fraction(self, start, u_start, step, fraction, contact, cancellation_token)

    def _swept_admissible(self, start, u_start, increment, end, u_end,
                          start_gap, end_gap, start_rim, end_rim):
        from .native_s3_swept import native_s3_swept_admissible
        if type(self.contact) is NativeRetainedMaterialS3Work:
            nodes=self.context.nodes[0]
            pose=self.contact._configuration(start.positions[nodes],start.position_low[nodes],
                start.cell_rotations[0],u_start)
            translations=(end.positions[nodes]-start.positions[nodes]
                          +end.position_low[nodes]-start.position_low[nodes])
            angular=np.array([increment[self.context.nodal_count+3*c:
                                        self.context.nodal_count+3*c+3] for c in (0,1)])
            try:self.contact.certify_orientation_path(pose,translations,angular,u_end-u_start,
                self.maximum_swept_intervals)
            except ValueError as exc:
                if self._recoverable_trial_error(exc):return False
                raise
            def geometry_query(state,u):
                return self.contact.geometry_gap(state.positions[nodes],state.position_low[nodes],
                    state.cell_rotations[0],u)
            return native_s3_swept_admissible(self.context,self.contact,0,self.gap_tolerance,
                self.maximum_swept_intervals,self._recoverable_trial_error,start,u_start,
                increment,end,u_end,start_gap,end_gap,start_rim,end_rim,
                geometry_only=True,geometry_query=geometry_query,material_radius=self.contact.contact.radius)
        return native_s3_swept_admissible(self.context, self.contact, 0,
            self.gap_tolerance, self.maximum_swept_intervals, self._recoverable_trial_error,
            start, u_start, increment, end, u_end, start_gap, end_gap, start_rim, end_rim)

    def solve(self, *, checkpoint: bytes | None = None,
              expected_sha256: str | None = None,
              stop_after: int | None = None,cancellation_token=None) -> NativeRetainedS3Run:
        from .native_shell_static_driver import solve_native_shell
        result=solve_native_shell(self, CHECKPOINT_SCHEMA, NativeRetainedS3Run,
            checkpoint=checkpoint, expected_sha256=expected_sha256, stop_after=stop_after,
            cancellation_token=cancellation_token)
        return replace(result,profile=self.profile) if type(self.contact) is NativeRetainedMaterialS3Work else result
