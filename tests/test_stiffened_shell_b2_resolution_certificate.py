"""Bounded residual certificates preserve explicit physical loads, not floors."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell as api
from anyfesolver import _stiffened_shell_b2_q4_panel_experimental as panel
from anyfesolver._b2_physical_linear_batch_dev import B2NumericalSolveFailure


def _rigid_tip_definition():
    """The original failed 40-Q4 free-stub slice, without changing stiffness."""
    nx, ny = 10, 4
    nodes = {1+j*(nx+1)+i: (i/nx, -.1+.2*j/ny, 0.)
             for j in range(ny+1) for i in range(nx+1)}
    cells = tuple((a, a+1, a+nx+2, a+nx+1)
                  for j in range(ny) for i in range(nx)
                  for a in (1+j*(nx+1)+i,))
    patch = tuple(cell for cell in cells if nodes[cell[0]][0] >= .8)
    selected = tuple(sorted({n for cell in patch for n in cell}))
    shells = tuple(api.Q4PanelShell(f"cell-{i}", cell, 210e9, .3, .02)
                   for i, cell in enumerate(cells))
    section = api.LinearSection.isotropic(E=210e9, nu=.3, area=.0004,
        Iy=1.3333333333333335e-8, Iz=1.3333333333333335e-8,
        J=2.249e-8, ky=5/6, kz=5/6)
    members = tuple(api.PhysicalB2Member(f"port-stub-{i}", f"source-{i}",
        (100001+i, 100002+i), ((.9+.05*i, 0., .05),
                             (.9+.05*(i+1), 0., .05)),
        (0., 1., 0.), "port-only", section) for i in range(2))
    count = 6*(len(nodes)+3)
    root = tuple(1+j*(nx+1) for j in range(ny+1))
    fixed = {6*(n-1)+d for n in root for d in range(6)}
    free = [i for i in range(count) if i not in fixed]
    return api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, (), sparse.eye(count, format="csr")[:, free],
        np.zeros(count), (api.B2NodalRigidFootprint(
            "rigid-tip", selected, 100001, patch),)), root


def _small_definition():
    """Existing two-panel point topology used for micro/large-load controls."""
    nodes = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
             4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
    shells = (api.Q4PanelShell("left", (1, 2, 5, 4), 1000., .3, .15),
              api.Q4PanelShell("right", (2, 3, 6, 5), 1000., .3, .15))
    section = api.LinearSection.isotropic(E=2000., nu=.25, area=.04,
        Iy=.00015, Iz=.0003, J=.0004, ky=.8, kz=.7)
    members = tuple(api.PhysicalB2Member(f"beam-{label}", f"source-{label}",
        (10+i, 11+i), ((.2+i, .5, .2), (1.2+i, .5, .2)),
        (0., 1., 0.), "section", section)
        for i, label in enumerate(("left", "right")))
    points = tuple(api.Q4B2Point(f"joint-{label}", label, f"beam-{label}",
                               .5, (0., .08, -.03))
                   for label in ("left", "right"))
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    fixed |= {6*(n-1)+5 for n in (2, 3, 5, 6)}
    free = np.array([i for i in range(54) if i not in fixed])
    return api.B2Q4PanelDefinition(api.B2_E4_PL_FAST_LINEAR_V1,
        nodes, shells, members, points,
        sparse.eye(54, format="csr")[:, free], np.zeros(54)), free


def _owner(request, definition):
    owner = api.create_analysis(definition=definition)
    request.addfinalizer(owner.close)
    return owner


def _fraction_residual(matrix, solution, rhs):
    """Exact arithmetic on stored binary64 values, independent of the helper."""
    matrix = matrix.tocsr()
    result = []
    for row in range(matrix.shape[0]):
        total = -Fraction.from_float(float(rhs[row]))
        for p in range(matrix.indptr[row], matrix.indptr[row+1]):
            total += (Fraction.from_float(float(matrix.data[p])) *
                      Fraction.from_float(float(solution[matrix.indices[p]])))
        result.append(total)
    return result


@pytest.mark.parametrize("coefficients,solution,rhs", (
    ([1., 1., 1.], [1e16, 1., -1e16], 1.),
    ([1.]*514, [1e16]+[2.**-30]*512+[-1e16], 2.**-21),
    ([1.0000000000000002], [1.0000000000000002],
     1.0000000000000004),
    ([1e-300], [1e-300], 0.),
    ([.3, -.3], [np.nextafter(0., 1.), np.nextafter(0., 1.)], 0.),
))
def test_residual_upper_bound_contains_exact_fraction_oracle(
        coefficients, solution, rhs):
    matrix = sparse.csr_matrix([coefficients])
    values, load = np.array(solution), np.array([rhs])
    defect, bound = panel._bounded_accurate_residual(matrix, values[:, None], load[:, None])
    exact, = _fraction_residual(matrix, values, load)
    assert np.isfinite(defect).all() and np.isfinite(bound).all()
    assert bound.shape == defect.shape == (1, 1)
    assert abs(exact) <= Fraction.from_float(float(bound[0, 0]))
    # A global gamma*sum(abs(products)) bound would miss this useful certificate.
    assert bound[0, 0] < 1e-9


def test_nonzero_cancellation_defect_is_not_mistaken_for_a_zero_residual():
    matrix = sparse.csr_matrix([[1., 1., 1.]])
    values = np.array([1e16, 1., -1e16])
    defect, bound = panel._bounded_accurate_residual(matrix, values[:, None], np.zeros((1, 1)))
    assert defect[0, 0] == 1.
    assert Fraction.from_float(float(bound[0, 0])) >= Fraction(1)


@pytest.mark.parametrize("coefficient,value,rhs", (
    (1e308, 2., 0.), (float("nan"), 1., 0.),
    (1., float("inf"), 0.), (1., 1., float("inf")),
))
def test_nonfinite_products_and_inputs_fail_closed(coefficient, value, rhs):
    with pytest.raises(B2NumericalSolveFailure):
        panel._bounded_accurate_residual(sparse.csr_matrix([[coefficient]]),
                                        np.array([[value]]), np.array([[rhs]]))


def test_original_rigid_tip_three_physical_loads_and_cached_certificate(request):
    definition, root = _rigid_tip_definition()
    owner = _owner(request, definition)
    assert owner._resolution_kkt is None
    shell_count = 6*len(definition.shell_nodes_m)
    selection = api.B2NodalRigidFootprintSelection(("rigid-tip",))
    origin = np.array((.9, 0., .05))
    cache = None
    for case_index, (dof, value) in enumerate(((2, 10.), (3, 1.), (4, 1.)), 1):
        load = np.zeros(definition.transformation.shape[0])
        load[shell_count+dof] = value
        result = owner.solve_linear(load, selected_footprints=selection)
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == 1
        assert result.solve_count == case_index
        assert owner._handle.factorization_count == 1
        assert owner._handle.solve_count == case_index
        assert owner._resolution_kkt is not None
        if cache is None:
            cache = owner._resolution_kkt
        assert owner._resolution_kkt is cache
        displacement = np.array(result.displacements)
        port = displacement[shell_count:shell_count+6]
        pairing = float(load@displacement)
        assert pairing > 0.
        assert abs(2*result.internal_strain_energy-pairing) <= 1e-8*pairing
        footprint, = result.footprint_wrenches
        force = load[shell_count:shell_count+3]
        moment = load[shell_count+3:shell_count+6]
        # Physical reference length is 1 m; no fixed 1 N/Nm residual floor.
        force_scale = np.sum(abs(force))+np.sum(abs(moment))/1.
        moment_scale = np.sum(abs(moment))+1.*np.sum(abs(force))
        np.testing.assert_allclose(footprint.shell_force_N, force,
                                   rtol=0., atol=1e-8*force_scale)
        np.testing.assert_allclose(footprint.shell_moment_Nm, moment,
                                   rtol=0., atol=1e-8*moment_scale)
        reactions = np.array(result.affine_reactions).reshape((-1, 6))
        support_force = sum((reactions[n-1, :3] for n in root), np.zeros(3))
        support_moment = sum((reactions[n-1, 3:]+np.cross(
            definition.shell_nodes_m[n], reactions[n-1, :3]) for n in root),
            np.zeros(3))
        np.testing.assert_allclose(support_force, -force,
                                   rtol=0., atol=1e-8*force_scale)
        np.testing.assert_allclose(support_moment, -moment-np.cross(origin, force),
                                   rtol=0., atol=1e-8*moment_scale)
        for n in definition.footprints[0].shell_nodes:
            shell = displacement[6*(n-1):6*n]
            np.testing.assert_allclose(shell[:3], port[:3]+np.cross(
                port[3:], np.array(definition.shell_nodes_m[n])-origin),
                rtol=0., atol=1e-12)
            np.testing.assert_allclose(shell[3:], port[3:], rtol=0., atol=1e-12)
    owner.close()
    assert owner._resolution_kkt is None
    assert owner._handle is None


def _gap_owner(arm, length, *, point_rows=False):
    """Explicit physical lever-arm rows; no factorization or assembled shell."""
    x, y, z = arm
    constraint = np.c_[-np.eye(6), np.eye(6)]
    constraint[:3, 3:6] = ((0., -z, y), (z, 0., -x), (-y, x, 0.))
    if point_rows:
        # Even enormous point coefficients must not gain a quiet-channel floor.
        constraint = np.vstack((1e12*constraint, constraint, constraint))
        incidences = (SimpleNamespace(row_start=6), SimpleNamespace(row_start=12))
    else:
        incidences = (SimpleNamespace(row_start=0),)
    return SimpleNamespace(_abs_C=abs(sparse.csr_matrix(constraint)),
        _footprint_incidence=incidences, _compatibility_length=length)


def test_gap_allowance_caps_each_footprint_row_and_preserves_point_rows():
    owner = _gap_owner((.3, -.2, .4), 2., point_rows=True)
    u = np.array((0.,)*6+(2e4, -1e4, 0., 3e5, 0., -2e5))
    scale = np.r_[np.arange(1., 7.), np.tile((0., 1e12, 1e-12), 4)]
    unchanged = scale.copy()
    tolerance = panel.ExperimentalB2Q4PanelAnalysis._footprint_gap_tolerance(
        owner, u, scale, .1)
    np.testing.assert_array_equal(scale, unchanged)
    np.testing.assert_array_equal(tolerance[:6], 1e-9*scale[:6])
    cap = np.array((2e-9,)*3+(1e-9,)*3)
    np.testing.assert_array_equal(tolerance[6:12], cap)
    np.testing.assert_array_equal(tolerance[12:18], cap)


@pytest.mark.parametrize("translation,rotation", ((2e-6, 3e-4), (2e3, 3e-8)))
def test_gap_allowance_has_physical_lever_arms_and_separate_units(translation, rotation):
    arm = np.array((.3, -.2, .4))
    fraction = 1e-14
    u = np.array((0.,)*6+(translation, 0., 0., rotation, 0., 0.))
    owner = _gap_owner(arm, 2.)
    tolerance = panel.ExperimentalB2Q4PanelAnalysis._footprint_gap_tolerance(
        owner, u, np.zeros(6), fraction)
    # Each translation gap has two translation terms and two physical arms;
    # rotation gaps contain two dimensionless rotation coordinates.
    expected = fraction*np.array((2*translation+.6*rotation,
        2*translation+.7*rotation, 2*translation+.5*rotation,
        2*rotation, 2*rotation, 2*rotation))
    np.testing.assert_allclose(tolerance, expected, rtol=1e-15, atol=0.)
    unit_scale = 1000.
    changed_u = u.copy().reshape((-1, 6))
    changed_u[:, :3] *= unit_scale
    converted = panel.ExperimentalB2Q4PanelAnalysis._footprint_gap_tolerance(
        _gap_owner(arm*unit_scale, 2.*unit_scale), changed_u.ravel(),
        np.zeros(6), fraction)
    np.testing.assert_allclose(converted[:3], unit_scale*tolerance[:3],
                               rtol=1e-15, atol=0.)
    np.testing.assert_allclose(converted[3:], tolerance[3:], rtol=1e-15, atol=0.)
    opposite_arm = panel.ExperimentalB2Q4PanelAnalysis._footprint_gap_tolerance(
        _gap_owner(-arm, 2.), u, np.zeros(6), fraction)
    np.testing.assert_array_equal(opposite_arm, tolerance)


@pytest.mark.parametrize("amplitude", (1e308, float("inf"), float("nan")))
def test_nonfinite_gap_allowance_is_not_hidden_by_the_finite_cap(amplitude):
    owner = _gap_owner((3., -2., 4.), 2.)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(panel._NumericalFailure,
                           match="compatibility allowance is not representable"):
            panel.ExperimentalB2Q4PanelAnalysis._footprint_gap_tolerance(
                owner, np.full(12, amplitude), np.zeros(6), .1)


@pytest.mark.parametrize("rotate,shift", ((True, False), (False, True), (True, True)))
def test_reference_compatibility_length_is_invariant_to_rigid_frame_changes(
        request, rotate, shift):
    definition, _ = _small_definition()
    fixed = set(range(6)) | set(range(18, 24)) | set(range(36, 42))
    free = [i for i in range(54) if i not in fixed]
    group = api.B2NodalRigidFootprint("right-patch", (2, 3, 6, 5), 11,
                                     ((2, 3, 6, 5),))
    definition = replace(definition, points=(), footprints=(group,),
        transformation=sparse.eye(54, format="csr")[:, free])
    baseline = _owner(request, definition)
    rotation = np.array(((.6, -.8, 0.), (.8, .6, 0.), (0., 0., 1.))) if rotate else np.eye(3)
    offset = np.array((16., -8., 4.)) if shift else np.zeros(3)
    nodes = {n: tuple(rotation@xyz+offset)
             for n, xyz in definition.shell_nodes_m.items()}
    shells = tuple(replace(shell, reference_normal=tuple(rotation@shell.reference_normal))
                   for shell in definition.shells)
    members = tuple(replace(member,
        reference_points_m=tuple(tuple(rotation@xyz+offset) for xyz in member.reference_points_m),
        local_y_global=tuple(rotation@member.local_y_global)) for member in definition.members)
    moved = _owner(request, replace(definition, shell_nodes_m=nodes,
                                    shells=shells, members=members))
    assert moved._compatibility_length == pytest.approx(
        baseline._compatibility_length, rel=1e-13, abs=0.)


def test_historical_point_load_guard_does_not_enter_the_new_certificate(request, monkeypatch):
    definition, _ = _small_definition()
    owner = _owner(request, definition)
    assert owner._compatibility_length is None
    def forbidden(*args):
        raise AssertionError("historical point guard entered footprint certificate")
    monkeypatch.setattr(owner, "_explicit_load_resolved", forbidden)
    load = np.zeros(54)
    load[14], load[49] = 1000., 1e-12
    result = owner.solve_linear(load)
    assert result.status == "failed"
    assert result.failure_reason == "panel mixed-load resolution insufficient"
    assert owner._resolution_kkt is None


@pytest.mark.parametrize('affine', (False, True))
def test_certified_tiny_load_does_not_accept_a_real_footprint_gap(request, monkeypatch, affine):
    definition, _ = _rigid_tip_definition()
    if affine:
        transform=definition.transformation.tolil()
        slot=6*(definition.footprints[0].shell_nodes[0]-1)
        transform[slot,transform.getrow(slot).nonzero()[1][0]]=.25
        definition=replace(definition,transformation=transform.tocsr())
    owner = _owner(request, definition)
    original = owner._handle
    shell_count = 6*len(definition.shell_nodes_m)
    node = definition.footprints[0].shell_nodes[0]
    physical_slot = 6*(node-1)  # Unloaded shell ux, independent of loaded beam uz.
    reduced_slot = int(owner._T.getrow(physical_slot).indices[0])
    defect = 100.*1e-9*owner._compatibility_length
    def incompatible_backend(rhs):
        answer = original.solve(rhs)
        answer[reduced_slot] += defect/owner._factor_scale[reduced_slot]
        return answer
    class Backend:
        solve = staticmethod(incompatible_backend)
        @property
        def factorization_count(self):
            return original.factorization_count
        @property
        def solve_count(self):
            return original.solve_count
    owner._handle = Backend()
    certificates, incompatible = [], []
    certify = owner._explicit_load_resolved
    tolerance = owner._footprint_gap_tolerance
    def record_certificate(*args, **kwargs):
        accepted = certify(*args, **kwargs)
        certificates.append(accepted)
        return accepted
    def record_gap(u, scale, fraction):
        allowed = tolerance(u, scale, fraction)
        incompatible.append(bool(np.any(abs(owner._C@u) > allowed)))
        return allowed
    monkeypatch.setattr(owner, "_explicit_load_resolved", record_certificate)
    monkeypatch.setattr(owner, "_footprint_gap_tolerance", record_gap)
    load = np.zeros(definition.transformation.shape[0])
    load[shell_count+2] = 1e-12
    result = owner.solve_linear(load)
    assert certificates and all(certificates)
    assert incompatible and all(incompatible)
    assert result.status == "failed"
    assert result.failure_reason == "panel point gap or equilibrium failed"
    assert owner._handle.factorization_count == owner._handle.solve_count == 1


@pytest.mark.parametrize("dof", (1, 3))
def test_zero_backend_cannot_erase_a_micro_force_or_couple(request, dof):
    definition, _ = _small_definition()
    owner = _owner(request, definition)
    load = np.zeros(54)
    load[48+dof] = 1e-12
    reference = owner.solve_linear(load*1e12)
    small = owner.solve_linear(load)
    assert reference.status == small.status == "completed"
    np.testing.assert_allclose(small.displacements,
        np.array(reference.displacements)*1e-12, rtol=1e-8, atol=1e-24)
    factor = owner._handle
    owner._handle = SimpleNamespace(solve=lambda rhs: np.zeros_like(rhs),
        factorization_count=factor.factorization_count, solve_count=factor.solve_count)
    failed = owner.solve_linear(load)
    assert failed.status == "failed"
    assert "equilibrium" in failed.failure_reason


@pytest.mark.parametrize("omit_component", (False, True))
def test_large_load_with_tiny_component_remains_non_success(request, omit_component):
    definition, free = _small_definition()
    owner = _owner(request, definition)
    factor = owner._handle
    tiny_slot = int(np.flatnonzero(free == 49)[0])
    def omit_tiny(rhs):
        altered = rhs.copy()
        altered[tiny_slot] = 0.
        return factor.solve(altered)
    if omit_component:
        owner._handle = SimpleNamespace(solve=omit_tiny,
            factorization_count=factor.factorization_count, solve_count=factor.solve_count)
    load = np.zeros(54)
    load[14], load[49] = 1000., 1e-12
    result = owner.solve_linear(load)
    assert result.status == "failed"
    assert result.failure_reason == "panel mixed-load resolution insufficient"


@pytest.mark.parametrize("change", ("scaled", "negative", "mixed", "offset"))
def test_historical_point_only_affine_coordinates_remain_ineligible(
        request, monkeypatch, change):
    definition, _ = _small_definition()
    transform = definition.transformation.tolil()
    offset = definition.prescribed_offset.copy()
    row = transform[:, 0].nonzero()[0][0]
    if change == "scaled":
        transform[row, 0] = 2.
    elif change == "negative":
        transform[row, 0] = -1.
    elif change == "mixed":
        second = transform[:, 1].nonzero()[0][0]
        transform[second, 0] = .17
    else:
        offset[0] = 1e-5
    owner = _owner(request, replace(definition,
        transformation=transform.tocsr(), prescribed_offset=offset))
    def forbidden(*args):
        raise AssertionError("ineligible affine certificate evaluated a residual")
    monkeypatch.setattr(panel, "_bounded_accurate_residual", forbidden)
    count = owner._T.shape[1]+owner._C.shape[0]
    assert not owner._explicit_load_resolved(np.zeros(count), np.zeros(count),
                                            np.ones(owner._T.shape[1]))
    assert owner._resolution_kkt is None


def test_underflowed_relative_cap_cannot_certify_a_load(request, monkeypatch):
    definition, _ = _small_definition()
    owner = _owner(request, definition)
    explicit = np.zeros(owner._T.shape[1])
    explicit[0] = np.nextafter(0., 1.)
    count = owner._T.shape[1]+owner._C.shape[0]
    rhs = np.zeros(count)
    rhs[0] = explicit[0]
    def forbidden(*args):
        raise AssertionError("underflowed explicit cap evaluated a residual")
    monkeypatch.setattr(panel, "_bounded_accurate_residual", forbidden)
    assert not owner._explicit_load_resolved(np.zeros(count), rhs, explicit)
    assert owner._resolution_kkt is None


@pytest.mark.parametrize("failure", (B2NumericalSolveFailure, TypeError))
def test_certificate_handles_only_the_typed_numerical_failure(request, monkeypatch, failure):
    definition, _ = _small_definition()
    owner = _owner(request, definition)
    count = owner._T.shape[1]+owner._C.shape[0]
    def broken(*args):
        raise failure("injected residual failure")
    monkeypatch.setattr(panel, "_bounded_accurate_residual", broken)
    args = (np.zeros(count), np.zeros(count), np.ones(owner._T.shape[1]))
    if failure is B2NumericalSolveFailure:
        assert not owner._explicit_load_resolved(*args)
    else:
        with pytest.raises(TypeError, match="injected residual failure"):
            owner._explicit_load_resolved(*args)


def test_ordinary_accepted_load_does_not_allocate_certificate_cache(request):
    definition, _ = _small_definition()
    owner = _owner(request, definition)
    load = np.zeros(54)
    load[49] = 2.
    result = owner.solve_linear(load)
    assert result.status == "completed", result.failure_reason
    assert owner._resolution_kkt is None
