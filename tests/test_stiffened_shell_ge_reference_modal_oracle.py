"""Independent bounded kinetic-volume and complete descriptor-QZ controls.

Dense matrices are test-only and capped at160 rows. The mass oracle integrates
point particles and linear half-cell velocities rather than calling the product
kinetic-factor or shell-mass routines. Numeric-unit covariance is a mechanics
check, not admission of non-SI public input.
"""
from dataclasses import replace
from fractions import Fraction
from math import fsum
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.transform import Rotation

from anyfesolver import b3_ge, stiffened_shell
from anyfesolver._stiffened_shell_ge_reference_modal import _assemble_mass
from test_stiffened_shell_ge_experimental import _panel


def _cross(p):
    x, y, z = p
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


PARTICLES = np.array(((0., .065, .01), (0., -.015, .01),
                      (0., .065, -.03), (0., -.015, -.03)))
WEIGHTS = np.full(4, .75)  # kg per reference metre, independently prescribed.


def _particle_inertia(points=PARTICLES, weights=WEIGHTS):
    result = np.zeros((6, 6))
    velocities = [np.c_[np.eye(3), -_cross(p)] for p in points]
    # Round each exact particle Gram entry once, using its mathematical
    # symmetry by construction. This is not projection of an input tensor.
    for i in range(6):
        for j in range(i+1):
            entry = sum(Fraction(float(w))*Fraction(float(v[k,i]))*Fraction(float(v[k,j]))
                        for w,v in zip(weights,velocities) for k in range(3))
            result[i,j] = result[j,i] = float(entry)
    return result


def _definition(*, angle=0., length_unit=1., mass_unit=1., stiffness_unit=1.):
    base = _panel()
    rotation = Rotation.from_rotvec((0., 0., angle)).as_matrix()
    shell = replace(base.shell,
        coordinates_m=tuple(map(tuple, length_unit*np.asarray(base.shell.coordinates_m)@rotation.T)),
        thickness_m=length_unit*base.shell.thickness_m,
        young_pa=stiffness_unit*mass_unit/length_unit*base.shell.young_pa,
        density_kg_m3=mass_unit/length_unit**3*base.shell.density_kg_m3)
    length_roles = np.diag((1.,)*3+(length_unit,)*3)
    beams = []
    for definition in base.beams:
        element, _ = definition.instantiate()
        section = element.section
        elastic = stiffness_unit*mass_unit*length_unit*length_roles@section.elastic@length_roles
        law = b3_ge.EllipsoidalGeneralizedSection(
            elastic, section.metric, section.yield_force, section.hardening)
        # Build the same physical unit conversion from the particle geometry.
        # A multiply-on-both-sides expression can lose bitwise symmetry through
        # differing operation orders before the legacy exact-symmetry decoder.
        inertia = _particle_inertia(length_unit*PARTICLES, mass_unit/length_unit*WEIGHTS)
        beams.append(b3_ge.define_beam('b3-ge', element.element_id, element.node_ids,
            length_unit*element.operator.reference.coordinates@rotation.T,
            np.tile(rotation, (3, 1, 1)), law, inertia))
    return replace(base, shell=shell, beams=tuple(beams))


def _independent_mass(owner, definition):
    """Rectangle volume and straight-half interpolation, independent scatter."""
    size = owner._K.shape[0]
    assert size <= 160
    mass = np.zeros((size, size))
    shell = definition.shell
    index = {node: i for i, node in enumerate(shell.node_ids)}
    xyz = np.asarray(shell.coordinates_m)
    shape_gram = np.array(((4.,2.,1.,2.), (2.,4.,2.,1.),
                           (1.,2.,4.,2.), (2.,1.,2.,4.))) / 36.
    for nodes in shell.q4_elements:
        p = xyz[[index[n] for n in nodes]]
        assert abs((p[1]-p[0])@(p[3]-p[0])) <= 1e-12*np.linalg.norm(p[1]-p[0])*np.linalg.norm(p[3]-p[0])
        area = np.linalg.norm(np.cross(p[1]-p[0], p[3]-p[0]))
        section = np.diag((shell.density_kg_m3*shell.thickness_m,)*3+
            (shell.density_kg_m3*shell.thickness_m**3/12.,)*2+(0.,))
        slots = [6*index[n]+j for n in nodes for j in range(6)]
        mass[np.ix_(slots, slots)] += np.kron(area*shape_gram, section)
    beam_ids = sorted({n for d in definition.beams for n in d.instantiate()[0].node_ids})
    beam_index = {n:i for i,n in enumerate(beam_ids)}
    shell_count = 6*len(index)
    retained_start = shell_count + 6*len(beam_ids)
    for ordinal, definition in enumerate(definition.beams):
        element, inertia = definition.instantiate()
        ref = element.operator.reference
        frame = ref.nodal_triads[0]
        np.testing.assert_array_equal(ref.nodal_triads, np.tile(frame, (3,1,1)))
        transform = linalg.block_diag(frame, frame)
        global_inertia = transform@inertia@transform.T
        for cell, (left, right) in enumerate(((0,1), (1,2))):
            length = np.linalg.norm(ref.coordinates[right]-ref.coordinates[left])
            a = [shell_count+6*beam_index[element.node_ids[left]]+j for j in range(3)]
            b = [shell_count+6*beam_index[element.node_ids[right]]+j for j in range(3)]
            theta = [retained_start+24*ordinal+3*cell+j for j in range(3)]
            mass[np.ix_(a,a)] += length/3.*global_inertia[:3,:3]
            mass[np.ix_(b,b)] += length/3.*global_inertia[:3,:3]
            mass[np.ix_(a,b)] += length/6.*global_inertia[:3,:3]
            mass[np.ix_(b,a)] += length/6.*global_inertia[:3,:3]
            for node in (a,b):
                mass[np.ix_(node,theta)] += length/2.*global_inertia[:3,3:]
                mass[np.ix_(theta,node)] += length/2.*global_inertia[3:,:3]
            mass[np.ix_(theta,theta)] += length*global_inertia[3:,3:]
    return mass


@pytest.mark.parametrize('angle', (0., .43))
def test_particle_and_plate_kinetic_work_in_authoritative_native_layout(angle):
    definition = _definition(angle=angle)
    owner = stiffened_shell.create_analysis(definition=definition)
    try:
        actual, algebraic, _ = _assemble_mass(owner)
        expected = _independent_mass(owner, definition)
        np.testing.assert_allclose(actual.toarray(), expected, rtol=3e-14, atol=3e-14)
        velocity = np.random.default_rng(224).normal(size=expected.shape[0])
        assert float(velocity@actual@velocity) == pytest.approx(float(velocity@expected@velocity), rel=3e-14)
        np.testing.assert_array_equal(actual[:,algebraic].toarray(), 0.)
        # These six per-member coordinates have strictly physical rotary inertia;
        # omitting them cannot be hidden by a nodal-only mode result.
        start = owner._shell_count+6*len(owner._beam_ids)
        for i in range(len(definition.beams)):
            assert np.all(actual.diagonal()[start+24*i:start+24*i+6] > 0.)
            np.testing.assert_array_equal(actual[start+24*i+6:start+24*(i+1),:].toarray(), 0.)
    finally:
        owner.close()


def _qz_reference(owner, mass):
    """Full singular pencil; independent bounded congruence for QZ precision."""
    free = owner._free
    A = linalg.block_diag(owner._K[free,:][:,free].toarray(),
                          np.zeros((owner._C.shape[0], owner._C.shape[0])))
    n = len(free)
    A[:n,n:] = owner._C[:,free].T.toarray()
    A[n:,:n] = owner._C[:,free].toarray()
    B = np.zeros_like(A); B[:n,:n] = mass[np.ix_(free,free)]
    assert A.shape[0] <= 160
    # Ruiz row equilibration acts on the complete pencil, including indefinite
    # resultant and multiplier blocks. No algebraic elimination or positive K.
    scale = np.ones(len(A))
    aa, bb = A.copy(), B.copy()
    for _ in range(24):
        magnitude = np.maximum(np.max(abs(aa),axis=1), np.max(abs(bb),axis=1))
        assert np.all(magnitude > 0.)
        d = 1./np.sqrt(magnitude)
        aa = d[:,None]*aa*d[None,:]; bb = d[:,None]*bb*d[None,:]; scale *= d
    # QZ balances both pencil operands. A time-unit change makes their global
    # norms comparable; it does not alter coordinates or finite/infinite modes.
    # Without this, beta cancellation loses precision in the singular pencil.
    frequency_scale = np.max(abs(aa))/np.max(abs(bb))
    values, vectors = linalg.eig(aa, frequency_scale*bb, homogeneous_eigvals=True)
    assert np.count_nonzero(values[1] == 0.) >= 18*len(owner._definition.beams)
    pairs = []
    balanced_mass = frequency_scale*bb
    for (alpha,beta), vector in zip(values.T, vectors.T):
        if beta == 0.: continue
        lam = frequency_scale*(alpha/beta)
        if abs(lam.imag) > 1e-8*abs(lam.real) or lam.real <= 0.: continue
        # QZ classifies the complete singular pencil. Its alpha/beta division
        # alone loses relative precision for this stiff mixed representation.
        # Evaluate conservative work on its vector with compensated summation;
        # the Rayleigh error is second order in the vector error.
        # Independently refine the complete QZ vector by dense inverse iteration
        # near its classified finite root. The singular B stays intact; only the
        # full shifted A-zeta B is factored. No retained/algebraic condensation.
        shift = float(lam.real)/frequency_scale*(1.+1e-4)
        factor = linalg.lu_factor(aa-shift*balanced_mass)
        direction = vector.real
        for _ in range(3):
            direction = linalg.lu_solve(factor, balanced_mass@direction)
            direction /= np.linalg.norm(direction)
        vv = np.outer(direction, direction)
        numerator = fsum((vv*aa).ravel())
        denominator = fsum((vv*bb).ravel())
        if denominator <= 0.: continue
        lam = numerator/denominator
        x = scale*direction
        norm = float(x@B@x)
        if norm <= 0. or not np.isfinite(norm): continue
        x /= np.sqrt(norm)
        residual = A@x-lam*(B@x)
        bound = abs(A)@abs(x)+abs(lam)*(abs(B)@abs(x))
        ratio = np.linalg.norm(scale*residual)/np.linalg.norm(scale*bound)
        if ratio > 1e-8: continue
        pairs.append((float(lam), x))
    assert len(pairs) >= 3
    return A,B,pairs,values


@pytest.mark.parametrize('shift_hz', (1., 80.))
def test_sparse_modes_match_complete_descriptor_qz_and_full_work(shift_hz, tmp_path):
    owner = stiffened_shell.create_analysis(definition=_definition())
    try:
        mass = _independent_mass(owner, owner._definition)
        A,B,pairs,raw_qz = _qz_reference(owner,mass)
        request = stiffened_shell.GEReferenceModesRequest(count=3,shift_hz=shift_hz)
        result = owner.modes(request)
        assert result.status == 'completed', result.failure_reason
        sigma = (2*np.pi*shift_hz)**2
        selected = sorted(sorted(pairs,key=lambda p:abs(p[0]-sigma))[:3],key=lambda p:p[0])
        np.savez_compressed(tmp_path/'descriptor-oracle.npz', A=A, B=B,
            raw_qz_alpha_beta=raw_qz,
            infinite_qz_count=np.count_nonzero(raw_qz[1]==0.),
            reference_eigenvalues=np.array([p[0] for p in pairs]),
            selected_reference_eigenvalues=np.array([p[0] for p in selected]),
            sparse_eigenvalues=np.array(result.eigenvalues_rad2_s2),
            sparse_coordinates=np.array(result.full_coordinates),
            sparse_multipliers=np.array(result.multiplier_modes))
        np.testing.assert_allclose(result.eigenvalues_rad2_s2,[p[0] for p in selected],rtol=1e-10,atol=1e-10)
        vectors = []
        for q,lam,value in zip(result.full_coordinates,result.multiplier_modes,result.eigenvalues_rad2_s2):
            x = np.r_[q[owner._free],lam]
            residual = A@x-value*(B@x)
            bound = abs(A)@abs(x)+abs(value)*(abs(B)@abs(x))
            assert np.linalg.norm(residual)/np.linalg.norm(bound) <= 1e-8
            np.testing.assert_allclose(owner._C@q,0.,atol=1e-9)
            vectors.append(q)
        V = np.column_stack(vectors)
        np.testing.assert_allclose(V.T@mass@V,np.eye(3),rtol=1e-7,atol=1e-7)
        assert owner._generation == 0 and owner._accepted is None and owner._finite_driver is None
    finally:
        owner.close()


def test_modes_preserve_frequency_under_numeric_unit_covariance():
    values=[]
    for length,mass in ((1.,1.),(1000.,.001)):
        owner=stiffened_shell.create_analysis(definition=_definition(length_unit=length,mass_unit=mass))
        try:
            result=owner.modes(stiffened_shell.GEReferenceModesRequest(count=3))
            assert result.status=='completed',result.failure_reason
            values.append(result.frequencies_hz)
            assert max(result.equation_residuals) <= 1e-8
            assert max(result.constraint_residuals) <= 1e-9
        finally:
            owner.close()
    np.testing.assert_allclose(values[0],values[1],rtol=1e-10,atol=1e-10)
