"""Public native virgin descriptors, with independent small dense QZ checks."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.linalg import eig
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken
from anyfesolver import _stiffened_shell_ge_reference_modal as modal
from examples.stiffened_shell_ge_mixed_reference_modes import definition


CASES = [(shell, family, beams, attachment) for shell in ("q4", "s3", "mixed")
    for family, beams, attachment in (("biaxial", "generalized", "point"),
        ("axial", "fibre", "point"), ("biaxial", "mixed", "bl5"),
        ("axial", "fibre", "bl5"))]


@pytest.mark.parametrize("shell,family,beams,attachment", CASES)
def test_actual_public_families_and_immutable_cache(shell,family,beams,attachment):
    owner = ss.create_analysis(definition=definition(shell,family,beams,attachment))
    try:
        assert owner.mixed_reference_modal_capabilities["eligible"]
        before = (owner._identity, owner._generation, owner._K.copy(), owner._C.copy())
        result = owner.modes(ss.GEMixedReferenceModesRequest(count=1))
        assert type(result) is ss.GEMixedReferenceModesResult
        assert result.status == "completed", result.failure_reason
        assert max(result.equation_residuals+result.physical_equation_residuals) <= 1e-8
        assert max(result.constraint_residuals) <= 1e-9
        mass, zero, identity = owner._mixed_reference_modal_mass
        assert mass[:,zero].nnz == mass[zero,:].nnz == 0
        assert owner._reference_modal_mass is None
        assert not mass.data.flags.writeable
        for q in result.full_coordinates:
            assert not q.flags.writeable
            np.testing.assert_allclose(q@mass@q, 1., atol=1e-12, rtol=1e-12)
        again = owner.modes(ss.GEMixedReferenceModesRequest(count=1))
        assert again.status == "completed", again.failure_reason
        assert again.mass_identity == identity == result.mass_identity
        assert owner._identity == before[0] and owner._generation == before[1]
        assert owner._accepted is owner._finite_driver is None
        assert (owner._K != before[2]).nnz == (owner._C != before[3]).nnz == 0
        assert not owner._writing
        if shell != "q4" or beams != "generalized" or attachment != "point":
            assert not owner.reference_modal_capabilities["eligible"]
            with pytest.raises(ValueError): owner.modes(ss.GEReferenceModesRequest())
    finally: owner.close()


@pytest.mark.parametrize("shell,attachment", [("s3","point"),("mixed","bl5")])
def test_independent_dense_qz_on_actual_full_mixed_pencil(shell,attachment):
    owner = ss.create_analysis(definition=definition(shell,"biaxial","mixed",attachment))
    try:
        result = owner.modes(ss.GEMixedReferenceModesRequest(count=2))
        assert result.status == "completed", result.failure_reason
        mass = owner._mixed_reference_modal_mass[0]
        n, r = len(owner._free), owner._C.shape[0]
        B = np.zeros((n+r,n+r)); B[:n,:n] = mass[owner._free,:][:,owner._free].toarray()
        # Independent binary equilibration of the entire singular KKT pencil.
        # beta changes frequency units only; cumulative diagonal maps QZ vectors
        # back to the original physical equations. No elimination or averaging.
        A=owner._A.toarray()
        beta=np.exp2(np.rint(np.log2(np.max(abs(A))/np.max(abs(B)))))
        assert np.isfinite(beta) and beta > 0.
        As=A.copy();Bs=beta*B;scale=np.ones(n+r)
        for _ in range(12):
            row=np.max(abs(As)+abs(Bs),axis=1)
            assert np.all(row > 0.)
            d=np.exp2(np.rint(-.5*np.log2(row)))
            assert np.isfinite(d).all() and np.all(d > 0.)
            As=d[:,None]*As*d[None,:];Bs=d[:,None]*Bs*d[None,:]
            scale*=d
            assert np.isfinite(As).all() and np.isfinite(Bs).all()
            assert np.isfinite(scale).all() and np.all(scale > 0.)
            assert np.array_equal(As != 0.,A != 0.)
            assert np.array_equal(Bs != 0.,B != 0.)
        values,vectors=eig(As,Bs)
        values*=beta
        keep=np.flatnonzero(np.isfinite(values) & (values.real > 0.) &
            (abs(values.imag) <= 1e-8*abs(values.real)))
        selected=keep[np.argsort(abs(values[keep].real-(2*np.pi)**2))[:2]]
        expected=values[selected].real
        for index in selected:
            value=values[index].real
            x=scale*vectors[:,index].real
            q=np.zeros(owner._K.shape[0]);q[owner._free]=x[:n]
            lam=x[n:]
            physical=owner._K@q+owner._C.T@lam-value*(mass@q)
            bound=abs(owner._K)@abs(q)+abs(owner._C.T)@abs(lam)+value*(abs(mass)@abs(q))
            equation=np.max(abs(physical[owner._free])*scale[:n])/np.max(bound[owner._free]*scale[:n])
            gap=owner._C@q;gap_bound=abs(owner._C)@abs(q)
            constraint=np.max(abs(gap)*scale[n:])/np.max(gap_bound*scale[n:])
            assert equation <= 1e-8 and constraint <= 1e-9
            assert q@mass@q > 0.
        np.testing.assert_allclose(result.eigenvalues_rad2_s2, sorted(expected), rtol=1e-8, atol=0.)
    finally: owner.close()


@pytest.mark.parametrize("shell,beams", [("q4","generalized"),("s3","fibre"),("mixed","mixed")])
def test_new_free_boundary_six_rigid_fields(shell,beams):
    model = definition(shell,"biaxial",beams,"point",free=True)
    owner = ss.create_analysis(definition=model)
    try:
        capabilities=owner.capabilities
        assert capabilities['eligible']
        assert capabilities['policy']==ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        assert capabilities['modal_boundary_policy']==ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        assert capabilities['scope']==owner.mixed_reference_modal_capabilities['scope']
        assert capabilities['accepted_state_writes'] is False
        preflight=owner.preflight()
        assert preflight['modal_boundary_policy']==ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        assert preflight['identity']==owner._identity
        if shell=='q4' and beams=='generalized':
            legacy=ss.create_analysis(definition=replace(model,
                modal_boundary_policy=ss.GE_FREE_FREE_MODAL_ONLY_V1))
            try:
                assert legacy.capabilities['eligible']
                assert legacy.capabilities['policy']==ss.GE_FREE_FREE_MODAL_ONLY_V1
                assert legacy.capabilities['scope']==legacy.reference_modal_capabilities['scope']
                assert legacy.capabilities['accepted_state_writes'] is False
                assert legacy.preflight()['modal_boundary_policy']==ss.GE_FREE_FREE_MODAL_ONLY_V1
            finally:legacy.close()
        result = owner.modes(ss.GEMixedReferenceModesRequest(count=1))
        assert result.status == "completed", result.failure_reason
        assert result.policy == ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        assert result.flexible.policy == ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        assert owner.mixed_reference_modal_capabilities["policy"] == ss.GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        with pytest.raises(ValueError,match="requires GEMixedReferenceModesRequest"):
            owner.modes(ss.GEReferenceModesRequest())
        assert len(result.rigid_coordinates) == 6
        assert max(result.rigid_equation_residuals) <= 1e-8
        assert max(result.rigid_constraint_residuals) <= 1e-9
        assert result.rigid_flexible_mass_overlap <= 1e-7
        mass = owner._mixed_reference_modal_mass[0]
        R = np.column_stack(result.rigid_coordinates)
        np.testing.assert_allclose(R.T@mass@R, np.eye(6), atol=1e-12)
        with pytest.raises(ValueError,match="modal-only"): owner.solve_linear(())
        if shell != "q4" or beams != "generalized":
            with pytest.raises(ValueError): ss.create_analysis(definition=replace(model,
                modal_boundary_policy=ss.GE_FREE_FREE_MODAL_ONLY_V1))
    finally: owner.close()


def test_independent_curved_centered_lift_kinetic_work():
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver._stiffened_shell_ge_mixed_modal_mass import native_physical_mass
    from test_stiffened_shell_ge_reference_modal import _definition as eccentric_definition
    inertia=eccentric_definition().beams[0].instantiate()[1]
    nodes=np.array(((0.,0.,0.),(.5,.2,.1),(1.,0.,0.)))
    triads=[]
    for xi in (-1.,0.,1.):
        tangent=.5*(nodes[2]-nodes[0])+xi*(nodes[0]-2*nodes[1]+nodes[2])
        tangent/=np.linalg.norm(tangent)
        third=np.cross(tangent,(0.,1.,0.));third/=np.linalg.norm(third)
        second=np.cross(third,tangent)
        triads.append(np.column_stack((tangent,second,third)))
    ref=CenteredCurvedBeam3ReferenceGeometry(nodes,np.array(triads))
    mass=native_physical_mass(ref,inertia)
    q=np.arange(24,dtype=float)/13.
    points,weights=np.polynomial.legendre.leggauss(24)
    work=0.
    for cell,(left,right) in enumerate(((0,1),(1,2))):
        for point,weight in zip(points,weights):
            t=(point+1)/2;xi=cell-1+t
            shape=np.array((xi*(xi-1)/2,1-xi*xi,xi*(xi+1)/2))
            offset=shape@nodes-((1-t)*nodes[left]+t*nodes[right])
            omega=q[18+3*cell:21+3*cell]
            v=(1-t)*q[6*left:6*left+3]+t*q[6*right:6*right+3]+np.cross(omega,offset)
            frame=ref.frame(xi)
            velocity=np.r_[frame.T@v,frame.T@omega]
            derivative=.5*(nodes[2]-nodes[0])+xi*(nodes[0]-2*nodes[1]+nodes[2])
            work+=weight*np.linalg.norm(derivative)/2*(velocity@inertia@velocity)
    np.testing.assert_allclose(q@mass@q,work,rtol=1e-14,atol=0.)


@pytest.mark.parametrize("stage", ["start","shell_mass","beam_mass","mass_publish","iteration","publish"])
def test_cancelled_public_owner_no_state_or_partial_mass_cache(stage):
    owner = ss.create_analysis(definition=definition("s3","axial","fibre","point"))
    class Stop(CancellationToken):
        def raise_if_cancelled(self,current=""):
            if current == "ge_reference_modes."+stage: self.cancel("boundary")
            super().raise_if_cancelled(current)
    try:
        result = owner.modes(ss.GEMixedReferenceModesRequest(count=1),cancellation_token=Stop())
        assert result.status == "cancelled" and not result.full_coordinates
        if stage in ("start","shell_mass","beam_mass","mass_publish"):
            assert owner._mixed_reference_modal_mass is None
        assert owner._accepted is owner._finite_driver is None and owner._generation == 0
        assert not owner._writing
    finally: owner.close()


@pytest.mark.parametrize("kwargs", [dict(count=True),dict(count=0),dict(shift_hz=0.),
    dict(shift_hz=float("nan")),dict(max_iterations=False)])
def test_malformed_new_request(kwargs):
    with pytest.raises(ValueError): ss.GEMixedReferenceModesRequest(**kwargs)


def test_refusal_of_unsupported_line_mechanism_bubble_family_and_nonvirgin(monkeypatch):
    owner = ss.create_analysis(definition=definition("s3","biaxial","fibre","point"))
    try:
        with pytest.raises(ValueError): owner.modes(object())
        owner._unsupported_modes = ("single-support BL5 leaves a free native beam rigid mode",)
        with pytest.raises(ValueError,match="free native beam rigid mode"):
            owner.modes(ss.GEMixedReferenceModesRequest())
        owner._unsupported_modes = ()
        element = owner._shell_parts[0][1]
        monkeypatch.setattr(type(element),"formulation_id","UNADMITTED_BUBBLE_FAMILY")
        with pytest.raises(ValueError,match="bubble families unsupported"):
            owner.modes(ss.GEMixedReferenceModesRequest())
        monkeypatch.undo()
        owner._generation = 1
        assert not owner.mixed_reference_modal_capabilities["eligible"]
        with pytest.raises(ValueError,match="fresh virgin"):
            owner.modes(ss.GEMixedReferenceModesRequest())
    finally: owner.close()


def test_actual_single_p0_bl5_support_retains_existing_mechanism_refusal():
    model = definition("q4","biaxial","generalized","bl5")
    line = replace(model.lines[0],multiplier_span=4)
    owner = ss.create_analysis(definition=replace(model,lines=(line,)))
    try:
        assert not owner.mixed_reference_modal_capabilities["eligible"]
        with pytest.raises(ValueError,match="free native beam rigid mode"):
            owner.modes(ss.GEMixedReferenceModesRequest())
        assert owner._mixed_reference_modal_mass is None
    finally: owner.close()


def test_native_kinetic_only_matches_preserved_evaluator_and_straight_work():
    from anyfesolver._ge_beam3_p5_centered.mass import reference_kinetic_factors
    from anyfesolver._stiffened_shell_ge_mixed_modal_mass import native_physical_mass
    from test_stiffened_shell_ge_reference_modal import _definition as eccentric_definition
    owner = ss.create_analysis(definition=eccentric_definition())
    try:
        for inertia,(element,_,_) in zip(owner._reference_modal_inertias,owner._beam_parts):
            ref = element.operator.reference
            mass = native_physical_mass(ref,inertia)
            factors = reference_kinetic_factors(ref,element.section.elastic,inertia,order=24)
            np.testing.assert_array_equal(mass,factors.full.T@factors.full)
            q = np.arange(24,dtype=float)/11.
            work = 0.
            nodes,weights = np.polynomial.legendre.leggauss(3)
            for cell,(left,right) in enumerate(((0,1),(1,2))):
                length = np.linalg.norm(ref.coordinates[right]-ref.coordinates[left])
                omega = q[18+3*cell:21+3*cell]
                for node,weight in zip(nodes,weights):
                    t=(node+1)/2
                    v=(1-t)*q[6*left:6*left+3]+t*q[6*right:6*right+3]
                    velocity=np.r_[v,omega]
                    work += weight*length/2*(velocity@inertia@velocity)
            np.testing.assert_allclose(q@mass@q,work,rtol=1e-14)
            assert not np.any(mass[:,[3,4,5,9,10,11,15,16,17]])
    finally: owner.close()
