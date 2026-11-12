"""Private frozen M_Q4 reference problem; no finite mixed or restart admission.

This separate development slice does not alter the preceding beam interface,
qualified Q4 operators, or public graph/element ownership.
"""
from dataclasses import asdict
from fractions import Fraction
import json
import threading
import numpy as np
from .e4_pl_element import QualifiedE4PLShellElement
from .fe_core import FEModel, Material, _QualifiedMutationEpoch
from ._ge_beam3_g1_element import ElasticElement
from ._ge_beam3_g1_elastic import canonical, owned, sha, solve
from ._ge_beam3_g1_operator import schur
from ._ge_beam3_g3b_reference import invariant, progress

POLICY = "G3B_M_Q4_REFERENCE_LINEAR_TRANSLATIONS_ONLY_DEVELOPMENT_V1"


class Q4TranslationReferenceProblem:
    """One reference-native beam and one explicit qualified planar Q4.

    Both sides are reference-linear. Only translations are tied, at coincident
    distinct nodes. No finite poses, material history, or accepted state exists.
    """
    def __init__(self, native, shell, material, *, nodes, fixed_nodes, tie,
                 reference_normal, policy=POLICY, adapter_allowlist=(), rotation_targets=None):
        if (type(self) is not Q4TranslationReferenceProblem or policy != POLICY or
                type(adapter_allowlist) is not tuple or adapter_allowlist or rotation_targets is not None):
            raise ValueError("M_Q4 translation-only reference policy required")
        if type(native) is not ElasticElement or type(shell) is not QualifiedE4PLShellElement or type(material) is not Material:
            raise ValueError("exact native/qualified Q4/isotropic material required")
        if native._mesh is not None or native._owner_guard is not None:
            raise ValueError("fresh unowned native element required")
        if (type(shell.element_id) is not int or shell.element_id <= 0 or native.element_id == shell.element_id or
                len(shell.node_ids) != 4 or any(type(n) is not int or n <= 0 for n in shell.node_ids) or
                len(set(shell.node_ids)) != 4 or set(native.node_ids) & set(shell.node_ids)):
            raise ValueError("disjoint exact element/node identities required")
        if (type(nodes) is not dict or len(nodes) != 7 or
                any(type(n) is not int or n <= 0 for n in nodes) or
                set(nodes) != set(native.node_ids) | set(shell.node_ids)):
            raise ValueError("exact seven-node M_Q4 inventory required")
        normal = owned(reference_normal, (3,))
        if (not np.array_equal(normal, [0., 0., 1.]) or shell.reference_normal is None or
                not np.array_equal(shell.reference_normal, normal) or shell.director_polarity != 1):
            raise ValueError("frozen physical +z owner normal required independently of numbering")
        if (shell.shell_section is not None or shell.material_direction is not None or shell.material_angle_deg != 0 or
                shell.pl_stabilization != 1 or shell.reduced_integration or
                shell.drilling_stabilization != .001 or shell.hourglass_stabilization != .001 or
                shell.planar_tolerance != 1e-10 or shell.warped_formulation != 'varying_frame' or
                shell.thickness != .2 or material.elastic_modulus != 1000 or material.poisson_ratio != .3 or
                material.hardening_curve is not None or material.yield_stress != 0):
            raise ValueError("frozen homogeneous reference-elastic M_Q4 inputs required")
        if (shell._qualified_components is not None or shell.__dict__.get('_nl_cache') is not None or
                shell.__dict__.get('_qualified_direct_state_token') is not None or
                shell.__dict__.get('_qualified_direct_state_tokens')):
            raise ValueError("fresh reference Q4 without prior caches/state required")
        ids = tuple(sorted(nodes)); positions = owned([nodes[n] for n in ids], (7, 3))
        x = native.operator.reference.coordinates
        if (not np.array_equal(positions[[ids.index(n) for n in native.node_ids]], x) or
                not np.array_equal(x[1], (x[0]+x[2])/2)):
            raise ValueError("native exact reference midpoint mismatch")
        sx = positions[[ids.index(n) for n in shell.node_ids]]
        # This first slice is the frozen planar square, not a geometry envelope.
        if not np.array_equal(sx, [[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]]):
            raise ValueError("frozen M_Q4 ordered square required")
        if (type(fixed_nodes) is not tuple or len(fixed_nodes) != 3 or
                any(type(n) is not int for n in fixed_nodes) or
                set(fixed_nodes) != {native.node_ids[0], shell.node_ids[1], shell.node_ids[2]}):
            raise ValueError("M_Q4 native root and opposite shell edge supports required")
        expected = dict(components=[0,1,2], masters=[[shell.node_ids[0], '1']],
                        offset=[0,0,0], slave=native.node_ids[-1])
        if type(tie) is not dict or canonical(tie) != canonical(expected):
            raise ValueError("exact translation-only unit tie required")
        if not np.array_equal(x[-1], sx[0]):
            raise ValueError("coincident traces required; no offset arm")
        self.native, self.shell, self.material = native, shell, material
        self.ids, self.positions, self.size, self.normal = ids, positions, 42, normal
        self.fixed_nodes, self._tie = tuple(sorted(fixed_nodes)), canonical(expected)
        self._lock = threading.Lock()
        self.model = FEModel('G3b M_Q4 reference development')
        for n, point in zip(ids, positions): self.model.add_node(n, *point)
        for e in sorted((native, shell), key=lambda e: e.element_id): self.model.add_element(e.element_id, e)
        self._dofs = tuple(tuple(6*ids.index(n)+j for n in e.node_ids for j in range(6)) for e in (native, shell))
        fixed = tuple(6*ids.index(n)+j for n in self.fixed_nodes for j in range(6))
        slave = tuple(6*ids.index(native.node_ids[-1])+j for j in range(3))
        master = tuple(6*ids.index(shell.node_ids[0])+j for j in range(3))
        free = tuple(i for i in range(42) if i not in fixed+slave)
        rows = [[Fraction(0) for _ in free] for _ in range(42)]
        for j, i in enumerate(free): rows[i][j] = Fraction(1)
        for a, b in zip(slave, master): rows[a] = list(rows[b])
        T = np.array(rows, dtype=float); J = np.zeros((21,42))
        for i, d in enumerate(fixed): J[i,d] = 1
        for i, (a,b) in enumerate(zip(slave,master)): J[18+i,a], J[18+i,b] = 1,-1
        assert np.array_equal(J @ T, np.zeros((21,21)))
        self.T, self.J = owned(T), owned(J)
        self._definition = self._describe()
        progress('capture')
        native.operator.guard(); shell.validate_quadrature_authority()
        full = native.operator.evaluate(x, np.zeros((3,3)), native.operator.reference.nodal_triads,
                                        np.tile(np.eye(3),(2,1,1)), np.zeros(18))
        invariant(full['residual'], np.zeros(42))
        _, nk, lift, _ = schur(full['residual'], full['jacobian'])
        self.native_full, self.native_lift = owned(full['jacobian']), owned(lift)
        progress('local solve')
        sk = shell.compute_stiffness_matrix(self.model.mesh, material)
        K = np.zeros((42,42))
        for dofs, block in zip(self._dofs,(nk,sk)): K[np.ix_(dofs,dofs)] += block
        invariant(K,K.T)
        self.K, self.reduced = owned(K), owned(T.T @ K @ T)
        np.linalg.cholesky(self.reduced)
        self._operators = sha(self._operator_data())
        # Mesh registration issues a mutation-epoch token even for reference
        # elasticity. Capture that token; it is not nonlinear material history.
        self._shell_token = self.model.mesh._qualified_direct_state_token
        self._shell_epoch = tuple(self._shell_token)
        self._guard(); progress('assembly')

    def _describe(self):
        return canonical(dict(policy=POLICY, native=self.native.to_dict(), shell=self.shell.to_dict(),
            material=asdict(self.material), normal=self.normal, positions=self.positions, size=self.size,
            nodes=[(n,self.model.mesh.nodes[n].coords(),list(self.model.mesh.dof_manager.get_node_dofs(n))) for n in self.ids],
            fixed=self.fixed_nodes, tie=json.loads(self._tie), dofs=self._dofs,T=self.T,J=self.J))

    def _operator_data(self):
        return dict(K=self.K,reduced=self.reduced,full=self.native_full,lift=self.native_lift)

    def _guard(self):
        if (type(self) is not Q4TranslationReferenceProblem or type(self.native) is not ElasticElement or
                type(self.shell) is not QualifiedE4PLShellElement or type(self.material) is not Material or
                self.native._mesh is not None or self.native._owner_guard is not None):
            raise ValueError('M_Q4 reference-only ownership changed')
        subscriptions = self.shell.__dict__.get('_qualified_direct_state_tokens')
        if (self.shell.__dict__.get('_nl_cache') is not None or
                self.shell.__dict__.get('_qualified_direct_state_token') is not self._shell_token or
                self.model.mesh._qualified_direct_state_token is not self._shell_token or
                type(self._shell_token) is not _QualifiedMutationEpoch or tuple(self._shell_token) != self._shell_epoch or
                type(subscriptions) is not list or len(subscriptions) != 1 or subscriptions[0] is not self._shell_token):
            raise ValueError('M_Q4 reference cache ownership/epoch changed or nonlinear cache present')
        if (self.model.constraint_equations or self.model.boundary_conditions or self.model.load_cases or
                self.model.mesh.dof_manager._constrained_dofs or self.model.mesh.point_masses or
                self.model.mesh.element_activity is not None or self.size != 42 or
                self.model.mesh.dof_manager.total_dofs != 42):
            raise ValueError('unsupported M_Q4 graph feature')
        if (tuple(self.model.mesh.nodes) != self.ids or len(self.model.mesh.elements) != 2 or
                self.model.mesh.elements.get(self.native.element_id) is not self.native or
                self.model.mesh.elements.get(self.shell.element_id) is not self.shell or
                self._describe() != self._definition or sha(self._operator_data()) != self._operators):
            raise ValueError('frozen M_Q4 graph/operator changed')
        self.native.operator.guard(); self.shell.validate_quadrature_authority()

    def solve(self, nodal, *, mode='REFERENCE_LINEAR', rotation_targets=None, cancel=lambda:False):
        if type(self) is not Q4TranslationReferenceProblem or mode != 'REFERENCE_LINEAR' or rotation_targets is not None:
            raise ValueError('finite mixed/rotational coupling remains unsupported')
        if not self._lock.acquire(blocking=False): raise RuntimeError('M_Q4 problem already in use')
        try:
            self._guard(); f = owned(nodal,(42,))
            if cancel(): raise InterruptedError('M_Q4 cancelled')
            self._guard()
            u = self.T @ solve(self.reduced,self.T.T @ f,assume_a='pos')
            balance = self.K @ u-f
            mu = solve(self.J @ self.J.T,-self.J @ balance,assume_a='pos')
            invariant(balance+self.J.T @ mu,np.zeros(42)); invariant(self.J @ u,np.zeros(21))
            internal = self.native_lift @ u[list(self._dofs[0])]
            invariant(self.native_full[18:,:18] @ u[list(self._dofs[0])]+self.native_full[18:,18:] @ internal,np.zeros(24))
            recovered = self.shell.compute_stresses(self.model.mesh,u[list(self._dofs[1])],self.material,return_global=True)
            if (recovered['numerical_fields_excluded'] is not True or
                    recovered['physical_director_authoritative'] is not True or
                    not np.array_equal(recovered['physical_director'],self.normal)):
                raise ValueError('M_Q4 physical recovery authority mismatch')
            result = dict(policy=POLICY,qualification=False,node_ids=self.ids,u=owned(u),multipliers=owned(mu),
                support_reactions=owned(-self.J[:18].T @ mu[:18]),tie_forces=owned(-self.J[18:].T @ mu[18:]),
                internal=owned(internal),residual=owned(balance+self.J.T @ mu),energy=float(u @ self.K @ u/2),
                native_stations=self.native.operator.cell.recover(internal[6:]),shell_recovery=recovered,
                shell_formulation_id=self.shell.to_dict()['formulation_id'],reference_normal=self.normal,
                native_station_policy='REFERENCE_LINEAR_STATIONARY_RECOVERY_NOT_FINITE_FRAMES')
            progress('recovery')
            if cancel(): raise InterruptedError('M_Q4 cancelled before return')
            self._guard(); progress('output')
            return result
        finally:
            self._lock.release()

    def checkpoint(self):
        raise ValueError('M_Q4 development has no restart/state authority; full G3b owner is later')
