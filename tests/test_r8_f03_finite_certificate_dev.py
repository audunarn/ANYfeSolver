"""Bounded exact/source/permutation/arithmetic controls, not an F03 campaign."""
import math
from fractions import Fraction as F
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scripts import r8_f03_finite_local_dev as local
from scripts.r8_f03_finite_inventory_dev import literal_inventory
from scripts import r8_f03_finite_sparse_dev as cert

MEASUREMENTS = {}
PROVENANCE = {key: 'synthetic exact binary64 endpoint; no assembly/scatter/scaling ambiguity'
              for key in ('assembly', 'scatter', 'scaling', 'shift', 'physical_reference')}


def floating(a):
    return np.array(a, dtype=float)


def diagonal(values):
    return [[F(v) if i == j else F() for j, v in enumerate(values)] for i in range(len(values))]


class LocalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.budget = local.ExactBudget()
        cls.c, cls.drill = local.isotropic_shell(210_000_000_000, F(3, 10), F(7, 200))
        cls.a, cls.b, cls.ell = F(1, 4), F(1, 10), F(2, 5)
        g = F(210_000_000_000)/F(26, 10)
        # Preserve original section coefficient literals, including rounded moments.
        cls.cn = diagonal([F(float(210e9*.01)), F(float(float(g)*.008)), F(float(float(g)*.008))])
        cls.cm = diagonal([F(float(float(g)*1.6666666666666667e-5)),
                           F(float(210e9*8.333333333333333e-6)), F(float(210e9*8.333333333333333e-6))])
        cls.inventory = literal_inventory(cls.a, cls.b, cls.ell, 2, cls.c, cls.drill, cls.cn, cls.cm,
            arm_z=F(-1, 20), width=F(1, 5), shell_top=F(7, 400), budget=cls.budget)
        cls.work = cls.inventory['shell_work']

    def test_01_source_stationary_and_completion(self):
        from anyfesolver import e4_pl_element as source
        xyz = floating([[-self.a, -self.b], [self.a, -self.b], [self.a, self.b], [-self.a, self.b]])
        coefficients = source._coefficients(xyz)
        station, coupling, gram = source._stationary_blocks(xyz, coefficients, floating(self.c))
        q = coupling[:, :14].T
        comparisons = [('Q', q, floating(self.work['Q'])),
                       ('F', station[14:, :14], floating(self.work['F'])),
                       ('E', station[14:, 14:], floating(self.work['E']))]
        for label, made, exact in comparisons:
            rel = np.max(np.abs(made-exact))/max(np.max(np.abs(exact)), 1.)
            MEASUREMENTS['source_'+label+'_max_relative_remainder'] = float(rel)
            np.testing.assert_allclose(made, exact, rtol=2e-14, atol=2e-14)
        self.assertEqual(source._centre_taylor(coefficients).shape, (3, 24))
        centre = source._centre_taylor(coefficients)
        gamma = np.zeros(24); gamma[5::6] = source._residual_mode(xyz, coefficients)
        diag = np.ones(24); diag[np.arange(24) % 6 >= 3] /= float(self.ell)
        expected = float(self.ell)*np.vstack((centre, gamma))*diag
        np.testing.assert_allclose(floating(self.work['rows_z'][14:]), expected, rtol=2e-15, atol=2e-15)
        z = np.arange(24)/31.
        qphysical = diag*z
        solution = np.linalg.solve(station, coupling.T@qphysical)
        corework = -float(qphysical@coupling@solution)
        y = floating(self.work['rows_z'])@z
        splitcore = float(y[:14]@floating(self.work['H_N_per_m'])[:14, :14]@y[:14])
        completed = (float(self.drill)*(centre@qphysical)@gram@(centre@qphysical)+
                     .002*float(self.drill)*4*float(self.a*self.b)*float(gamma@qphysical)**2)
        splitcompletion = float(y[14:]@floating(self.work['H_N_per_m'])[14:, 14:]@y[14:])
        MEASUREMENTS['source_core_work_relative_remainder'] = abs(corework-splitcore)/abs(splitcore)
        MEASUREMENTS['source_completion_relative_remainder'] = abs(completed-splitcompletion)/abs(splitcompletion)
        self.assertAlmostEqual(corework/splitcore, 1., places=12)
        self.assertAlmostEqual(completed/splitcompletion, 1., places=12)

    def test_02_exact_rigid_kernel_and_complete_rows(self):
        positions = [(-self.a, -self.b, F()), (self.a, -self.b, F()),
                     (self.a, self.b, F()), (-self.a, self.b, F())]
        for axis in range(6):
            z = [F()]*24
            for n, (x, y, _) in enumerate(positions):
                if axis < 3:
                    z[6*n+axis] = 1
                else:
                    if axis == 3: z[6*n+2] = y
                    if axis == 4: z[6*n+2] = -x
                    if axis == 5: z[6*n], z[6*n+1] = -y, x
                    z[6*n+axis] = self.ell
            self.assertEqual(local.multiply(self.work['rows_z'], [[v] for v in z], self.budget), local.zero(18, 1))
        self.assertEqual(self.work['completion_rows'], (14, 15, 16, 17))
        self.assertGreater(self.work['bound']['k0_N_per_m'], 0)

    def test_03_equivalent_length_rescaling(self):
        # Rescale geometry by2 and constitutive row families by their units:
        # membrane/shear N/m scale1/2; bending Nm scales2; drilling N/m scales1/2.
        c2 = [[x*(2 if i in (3, 4, 5) else F(1, 2)) for x in row] for i, row in enumerate(self.c)]
        w2 = local.rectangle_work(2*self.a, 2*self.b, 2*self.ell, c2, self.drill/2, self.budget)
        self.assertEqual(w2['rows_z'], self.work['rows_z'])
        self.assertEqual(w2['H_N_per_m'], [[x/2 for x in row] for row in self.work['H_N_per_m']])
        # raw source moments dimensions0..2/6..7 m2,3..5 m,8..9 m4,10..13 m3.
        exponents = (2, 2, 2, 1, 1, 1, 2, 2, 4, 4, 3, 3, 3, 3)
        for i, power in enumerate(exponents):
            # physical rotation components dimensionless, translations scale2.
            self.assertEqual([v*(2 if j % 6 < 3 else 1) for j, v in enumerate(w2['Q'][i])],
                             [v*2**power for v in self.work['Q'][i]])

    def test_04_exact_local_minor_and_strip_bound(self):
        strip = local.rectangular_strip_bounds(self.work, 2, self.budget)
        rec = strip['local_minor']; rows = self.work['rows_z']
        minor = [[rows[i][j] for j in rec['new']] for i in rec['selected_rows']]
        self.assertEqual(local.multiply(rec['inverse'], minor, self.budget), local.identity(12))
        # A known exact nonrigid field must obey propagated component bounds.
        left = [F()]*12; fields, allwork = [], []
        for cell in range(2):
            right = [F((-1)**(i+cell)*(i+1), 100) for i in range(12)]
            z = left[:6]+right+left[6:]
            y = local.multiply(rows, [[v] for v in z], self.budget)
            fields.append(right); allwork.extend(v[0] for v in y)
            left = right
        global_y_squared = sum((v*v for v in allwork), F())
        for cell, right in enumerate(fields):
            for value, bound in zip(right, strip['node_pair_bounds'][cell+1]):
                self.assertLessEqual(value**2, bound**2*global_y_squared)

    def test_05_completion_removal_refuses_material_proof(self):
        h = [row[:] for row in self.work['H_N_per_m']]
        h[17][17] = F()
        with self.assertRaises(local.Unresolved): local.spd_inverse_bound(h, self.budget)
        with self.assertRaises(local.Unresolved): local.rectangle_work(self.a, self.b, self.ell, self.c, self.drill, self.budget, pl=F())

    def test_06_native_actual_kinematics_and_LOCAL18(self):
        from anyfesolver._ge_beam3_mixed_ad import Jet2, constant_matrix, so3_exp
        from anyfesolver._ge_beam3_p5.compensated import compensated_strain
        from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
        from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
        from anyfesolver._ge_beam3_retained_generalized import RetainedGeneralizedOperator
        length = F(1); rows = local.native_member_rows(length, self.ell, self.budget)
        xyz = np.array(((0., 0., .0675), (.5, 0., .0675), (1., 0., .0675)))
        variables = [Jet2.variable(0., i, 24) for i in range(24)]
        for cell in range(2):
            u = so3_exp(variables[18+3*cell:21+3*cell])
            made = compensated_strain(u, xyz, xyz, np.zeros_like(xyz), variables, cell, cell+1)
            jac = np.array([v.gradient for v in made]); jac[:, np.arange(24)%6>=3] /= float(self.ell)
            # Cell rotations indices18..23 all scale byell; mask above already all axes?
            jac[:, 18:24] = np.array([v.gradient for v in made])[:, 18:24]/float(self.ell)
            np.testing.assert_array_equal(jac, floating(rows[9*cell:9*cell+3]))
        reference = CenteredCurvedBeam3ReferenceGeometry(xyz, np.tile(np.eye(3), (3, 1, 1)))
        section = EllipsoidalGeneralizedSection(np.block([[floating(self.cn), np.zeros((3, 3))],
                                                          [np.zeros((3, 3)), floating(self.cm)]]), np.eye(6), 1e20, 1e9)
        op = RetainedGeneralizedOperator(reference, section, order=4)
        result = op.cell.response([0.]*18)
        dactual = np.array([[float(hi)+float(lo) for hi, lo in zip(row[0], row[1])] for row in result['hessian']])
        d = local.zero(18, 18)
        for cell in range(2):
            for axis in range(3):
                d[3*cell+axis][3*cell+axis] = F(1, 2)/self.cn[axis][axis]
                for i in range(2):
                    for j in range(2):
                        d[6+6*cell+3*i+axis][6+6*cell+3*j+axis] = F(1 if i == j else 1, 6 if i == j else 12)/self.cm[axis][axis]
        np.testing.assert_allclose(dactual, floating(d), rtol=2e-14, atol=2e-14*np.max(abs(floating(d))))
        MEASUREMENTS['native_LOCAL18_relative_remainder'] = float(np.max(abs(dactual-floating(d)))/np.max(abs(floating(d))))
        # Original logs k=(-thetaL+phi,thetaR-phi), no rotation elimination.
        z = [F(i-8, 37) for i in range(24)]
        k = [F()]*18
        for cell in range(2):
            y = local.multiply(rows[9*cell:9*cell+9], [[v] for v in z], self.budget)
            for axis in range(3):
                k[3*cell+axis] = y[axis][0]
                k[6+6*cell+axis] = z[18+3*cell+axis]/self.ell-z[6*cell+3+axis]/self.ell
                k[9+6*cell+axis] = z[6*(cell+1)+3+axis]/self.ell-z[18+3*cell+axis]/self.ell
        dk = local.solve(d, [[v] for v in k], self.budget)
        signed_stationary_work = sum((x*y[0] for x, y in zip(k, dk)), F())
        y = local.multiply(rows, [[v] for v in z], self.budget)
        h = self.inventory['native_work']['H_N_per_m']
        localwork = sum((y[9*cell+i][0]*h[i][j]*y[9*cell+j][0] for cell in range(2) for i in range(9) for j in range(9)), F())
        self.assertEqual(signed_stationary_work, localwork)
        MEASUREMENTS['native_source_negative_complementary_block'] = '-D retained; exact resultant-only stationary identity'

    def test_07_actual_BL5_P0_yaw_row_with_reference_lifts(self):
        from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
        from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator
        from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
        from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
        xyz = np.array(((0., 0., .0675), (.5, 0., .0675), (1., 0., .0675)))
        triads = np.tile(np.eye(3), (3, 1, 1))
        ref = CenteredCurvedBeam3ReferenceGeometry(xyz, triads)
        state = Mechanical(xyz, np.zeros_like(xyz), triads, np.tile(np.eye(3), (1, 2, 1, 1)), np.zeros((1, 18)))
        actual = np.zeros((5, 60))
        for cell in range(2):
            edge = np.array(((cell*.5, 0., 0.), ((cell+1)*.5, 0., 0.)))
            for alpha in (.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)):
                port = NativeHalfCellPort.from_reference(ref, (0, 1, 2), 0, cell, alpha, 18, 42)
                op = BL5LocalOperator.at_reference(edge, np.zeros((2, 6)), .035, port, state, station=alpha)
                value = op.evaluate(np.zeros((2, 6)), state, np.zeros(5))
                shell_slots = np.r_[np.arange(12*cell, 12*cell+6), np.arange(12*(cell+1), 12*(cell+1)+6)]
                physical_to_z = np.ones(len(value.beam_slots)); physical_to_z[np.asarray(value.beam_slots)>=18] /= float(self.ell)
                actual[:, shell_slots] += .25*value.gap_jacobian[:, :12]/np.tile((1, 1, 1, float(self.ell), float(self.ell), float(self.ell)), 2)
                # Source beam slots include nodal18 and retained24; source local
                # gap reaches only translations and independent cell rotations.
                for slot, v, scale in zip(value.beam_slots, value.gap_jacobian[:, 12:].T, physical_to_z):
                    if slot < 18: index = 36+slot
                    elif slot < 24: index = 36+slot
                    else:
                        np.testing.assert_array_equal(v, np.zeros(5)); continue
                    actual[:, index] += .25*v*scale
                np.testing.assert_allclose(op.reference_arm, (0., 0., -.05), rtol=0., atol=2e-17)
        expected = floating(self.inventory['P0_five_channel_actions_z'])
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
        MEASUREMENTS['actual_BL5_P0_five_rows_max_abs_remainder'] = float(np.max(abs(actual-expected)))
        self.assertEqual(self.inventory['P0_yaw_action'], -F(1, 2))
        self.assertTrue(self.inventory['full_C_row_rank_certified'])
        self.assertFalse(self.inventory['growing_C_row_rank_certified'])
        rec = self.inventory['P0_row_rank_certificate']
        self.assertEqual(local.multiply(rec['minor'], rec['inverse'], self.budget), local.identity(5))

    def test_08_yaw_blind_and_caps_fail_closed(self):
        with self.assertRaises(local.Unresolved): local.close_yaw([F(1)], [F(1)], [F()], self.budget)
        with self.assertRaises(local.Unresolved): local.solve([[F(1)]], [[F(1)]], local.ExactBudget(operation_cap=1))
        with self.assertRaises(local.Unresolved): local.solve([[F(1, 2**80)]], [[F(1)]], local.ExactBudget(bit_cap=16))
        with self.assertRaises(local.Unresolved): local.rectangular_strip_bounds(self.work, 3, self.budget)

    def test_17_fixed_GX_sparse_source_norm_parity(self):
        from scripts.r8_bl5_mixed_shell_h1_dev import scalar_mixed_h1
        from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm
        from anyfesolver import e4_pl_element as source
        nodes = np.array([(x, y) for x in (0., .5, 1.) for y in (0., .2)])
        cells = ((0, 2, 3, 1), (2, 4, 5, 3))
        scalar, area = scalar_mixed_h1(nodes, cells, float(self.ell))
        shell = sparse.kron(scalar, sparse.diags((1., 1., 1., 1., 1., 0.)), format='csc')
        # Independent actual drilling discrepancy at source Gauss stations.
        for ids in cells:
            xyz = nodes[list(ids)]; c = source._coefficients(xyz)
            for r, s in source._GAUSS:
                nr, ns = source._shape_derivatives(r, s)
                xr, xs, yr, ys, det = source._jacobian(c, r, s)
                nx, ny = (ys*nr-yr*ns)/det, (-xs*nr+xr*ns)/det
                values, slots = [], []
                for i, node in enumerate(ids):
                    slots.extend((6*node, 6*node+1, 6*node+5))
                    values.extend((float(self.ell)*ny[i]/2, -float(self.ell)*nx[i]/2, source._shape(r, s)[i]))
                column = sparse.coo_matrix((values, (slots, np.zeros(len(slots), dtype=int))), shape=(36, 1)).tocsc()
                shell += det/area*(column@column.T)
        _, hbeam, active, _ = beam_norm(1., float(self.ell), 2, 1, np.array((0., 0., -.05)), .2)
        full = [6*node+axis for node in range(3) for axis in range(3)]+list(range(18, 24))
        rows = [full[i] for i in active]
        scales = [1./float(self.ell) if i >= 9 else 1. for i in active]
        emb = sparse.coo_matrix((scales, (rows, np.arange(len(active)))), shape=(24, len(active))).tocsc()
        native = emb@hbeam@emb.T
        # Independent physical nodal rotation augmentation with P1 interpolation.
        for cell in range(2):
            for i in range(2):
                for j in range(2):
                    value = .5*(2. if i == j else 1.)/6.+float(self.ell)**2*(1. if i == j else -1.)/.5
                    rr = [6*(cell+i)+3+axis for axis in range(3)]
                    cc = [6*(cell+j)+3+axis for axis in range(3)]
                    native += sparse.coo_matrix(([value]*3, (rr, cc)), shape=(24, 24)).tocsc()
        expected = sparse.block_diag((shell, native), format='csc')
        entries = self.inventory['Gz_sparse_exact']
        actual = sparse.coo_matrix(([float(v) for _, _, v in entries],
            ([i for i, _, _ in entries], [j for _, j, _ in entries])), shape=(60, 60)).tocsc()
        supported = list(range(12, 36))+list(range(41, 60))
        difference = actual[supported][:, supported]-expected[supported][:, supported]
        error = np.max(np.abs(difference.data), initial=0.)
        reference = np.max(np.abs(expected.data), initial=0.)
        MEASUREMENTS['fixed_GX_sparse_source_norm_max_relative_remainder'] = float(error/reference)
        self.assertLess(error/reference, 2e-14)

    @classmethod
    def tearDownClass(cls):
        MEASUREMENTS['exact_operations_total'] = cls.budget.operations
        MEASUREMENTS['local_order_H'] = (18, 9)
        MEASUREMENTS['ideal_literal_k0_N_per_m'] = str(cls.inventory['k0_N_per_m'])
        MEASUREMENTS['ideal_literal_Cgeom'] = str(cls.inventory['Cgeom'])
        MEASUREMENTS['ideal_literal_kappa_N_per_m'] = str(cls.inventory['candidate_ideal_kappa_N_per_m'])


class SparseControls(unittest.TestCase):
    def certificate(self, a, half=None, factor=None, radius=None, caps=None):
        half = a if half is None else half
        radius = sparse.csc_matrix(a.shape) if radius is None else radius
        return cert.sparse_lu_homotopy(a, half, radius, radius, splu(a) if factor is None else factor,
                                     uncertainty_provenance=PROVENANCE, caps=caps)

    def test_09_asymmetric_actual_SuperLU_permutations(self):
        a = sparse.csc_matrix(np.array([[0., 0., 2., 0.], [3., 0., 0., 1.],
                                        [0., 4., 0., 0.], [0., 0., 1., 5.]]))
        lu = splu(a)
        self.assertFalse(np.array_equal(lu.perm_r, lu.perm_c))
        result = self.certificate(a, factor=lu)
        self.assertEqual(result['status'], 'CERTIFIED_NONSINGULAR_DECLARED_INTERVAL')
        MEASUREMENTS['asymmetric_permutation_control'] = {'perm_r': lu.perm_r.tolist(), 'perm_c': lu.perm_c.tolist(), **result}
        wrong = SimpleNamespace(L=lu.L, U=lu.U, perm_r=np.argsort(lu.perm_r), perm_c=np.argsort(lu.perm_c))
        bad = self.certificate(a, factor=wrong)
        self.assertEqual(bad['status'], 'UNRESOLVED')

    def test_10_signed_pencil_homotopy_and_crossing(self):
        a = sparse.csc_matrix(np.array([[4., 1., 1.], [1., -2., 0.], [1., 0., 0.]]))
        half = a-sparse.diags([.125, 0., 0.])
        good = self.certificate(a, half)
        self.assertEqual(good['status'], 'CERTIFIED_NONSINGULAR_DECLARED_INTERVAL')
        self.assertIsNone(good['inertia']); self.assertIsNone(good['coercivity'])
        MEASUREMENTS['synthetic_signed_pencil'] = good
        crossing = self.certificate(sparse.eye(2, format='csc'), sparse.diags([-1., 1.], format='csc'))
        self.assertEqual(crossing['status'], 'UNRESOLVED')

    def test_11_sparse_row_cancellation_enclosure(self):
        l = sparse.csc_matrix([[1., 0., 0.], [1., 1., 0.], [1., -1., 1.]])
        u = sparse.csc_matrix([[1., 1e16, 1.], [0., 1e16, 1.], [0., 0., 2.]])
        factor = SimpleNamespace(L=l, U=u, perm_r=np.arange(3), perm_c=np.arange(3))
        a = l@u
        result = self.certificate(a, factor=factor)
        self.assertLess(result['residual_row_bound'][2], 1.)
        self.assertEqual(result['max_product_row_entries'], 3)

    def test_12_outward_primitives_exact_oracle(self):
        cap = cert.Caps()
        a, b = cert.Interval.point(.1), cert.Interval.point(.2)
        c = a.add(b, cap); exact = F(.1)+F(.2)
        self.assertLessEqual(F(c.lo), exact); self.assertGreaterEqual(F(c.hi), exact)
        p = c.multiply(cert.Interval(-3., 2.), cap)
        self.assertLessEqual(F(p.lo), -3*exact); self.assertGreaterEqual(F(p.hi), 2*exact)

    def test_13_missing_uncertainty_and_subnormal_overflow(self):
        a = sparse.eye(1, format='csc')
        with self.assertRaises(local.Unresolved): cert.sparse_lu_homotopy(a, a, a*0, a*0, splu(a), uncertainty_provenance={})
        with self.assertRaises(local.Unresolved): cert.Interval.point(5e-324)
        with self.assertRaises(local.Unresolved): cert.Interval.point(1e308).multiply(cert.Interval.point(1e308), cert.Caps())
        with self.assertRaises(local.Unresolved): cert.Interval.point(1e-300).multiply(cert.Interval.point(1e-300), cert.Caps())

    def test_14_zero_diagonal_fill_bit_operation_caps(self):
        a = sparse.eye(2, format='csc')
        factor = SimpleNamespace(L=a, U=sparse.diags([1., 0.]), perm_r=np.arange(2), perm_c=np.arange(2))
        with self.assertRaises(local.Unresolved): self.certificate(a, factor=factor)
        with self.assertRaises(local.Unresolved): self.certificate(a, caps=cert.Caps(fill=1))
        with self.assertRaises(local.Unresolved): self.certificate(a, caps=cert.Caps(operations=1))
        with self.assertRaises(local.Unresolved): cert.Interval.point(.1).add(cert.Interval.point(.2), cert.Caps(bits=4))

    def test_15_physical_skew_uncertainty_can_refuse(self):
        a = sparse.eye(2, format='csc')
        radius = sparse.csc_matrix([[0., 2.], [0., 0.]])
        self.assertEqual(self.certificate(a, radius=radius)['status'], 'UNRESOLVED')

    def test_16_original_scalar_and_empty_dimension_refusal(self):
        with self.assertRaises(local.Unresolved): cert.Interval.point(np.int64(2**53+1))
        with self.assertRaises(local.Unresolved): cert.Interval.point(np.float32(.1))
        with self.assertRaises(local.Unresolved): cert.row_map(sparse.csc_matrix([[2**53+1]], dtype=np.int64), cert.Caps())
        huge = sparse.coo_matrix((1_000_000, 1_000_000), dtype=float)
        factor = SimpleNamespace(L=huge, U=huge, perm_r=(), perm_c=())
        with self.assertRaises(local.Unresolved): cert.sparse_lu_homotopy(huge, huge, huge, huge, factor, uncertainty_provenance=PROVENANCE)


if __name__ == '__main__':
    unittest.main()
