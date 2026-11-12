"""Bounded Q4 distributing derivative/work controls; no solver campaign."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from anyfesolver import _stiffened_shell_distributing_footprint as footprint
from anyfesolver._stiffened_shell_b2_nodal_footprint import reference_shell_matrix


def fixture():
    xyz = [(0., 0., 0.), (1., 0., 0.), (2., 0., 0.),
           (0., 1., 0.), (1., 1., 0.), (2., 1., 0.)]
    nodes = tuple(footprint.FlatQ4Node(i+1, p, 6*i) for i, p in enumerate(xyz))
    hosts = (footprint.CartesianQ4Host(21, (1, 2, 5, 4)),
             footprint.CartesianQ4Host(22, (2, 3, 6, 5)))
    beam = footprint.PhysicalBeamNodePort(101, (1.3, .3, .5), 36, footprint.IDENTITY_AXES)
    definition = footprint.DistributingFootprintDefinition("patch", (21, 22), 101, (.5, 1.5, .25, .75))
    return definition, nodes, hosts, beam


def compiled():
    return footprint.compile_footprint(*fixture())


def test_literal_historical_two_host_identity_and_array_bytes():
    # Installed historical baseline frozen before this extension. No runtime
    # evidence-file dependency and no regenerated expected production values.
    c = compiled()
    assert c.identity == "da9c58fe046a51ece66b20ec902f911ed173e8ea56a4c8c50e251e79df667d21"
    assert c.area_m2 == .5
    expected = {
        "shell_positions_m": ((6, 3), "float64", "c7bfb7ab6b2fb6ba422b8743cbab2746e2dd473b775ef75ae678deac6c8b21c7"),
        "beam_position_m": ((3,), "float64", "5ce5d036867014a3dcf577c6d6f279252df230030be20026f1c0a9f6ea86d42a"),
        "centroid_m": ((3,), "float64", "0de1d739840a889ceab625181a5cd6096c41a8a5b6be42544b96ea2d4a623871"),
        "eccentricity_m": ((3,), "float64", "7726e2faaec98a16e62bf1f2be38b012398ed248ec4cd07e274ecd2118402d29"),
        "area_mean": ((6,), "float64", "f4b60cc7708dbd309eafb7a4a0aee15db62ca3fce90c704bdb5d63fb7970cb34"),
        "planar_gradients": ((2, 6), "float64", "3ca45774061dcbe4ff10aaeb656ef3b3acca52d5704330db6199bbd3275f0b08"),
        "shell_trace": ((6, 36), "float64", "5bbd934624632d08a5d70d643a039ac5b9e4ffa14b319c5b5909f59a138f951d"),
        "constraint_matrix": ((6, 42), "float64", "2348cdfad3135c1abd77ea045a007704d31ec7504e77a73d596d4311688071da"),
        "gather_slots": ((42,), "int64", "9eb9d21600526db7daaff8e22251b571faa3eb4c2515d5f45afba0e9fe77c14d"),
    }
    for name, (shape, dtype, digest) in expected.items():
        array = getattr(c, name)
        assert array.shape == shape and str(array.dtype) == dtype
        assert sha256(array.tobytes()).hexdigest() == digest, name
    np.testing.assert_array_equal(c.area_mean, (.0625, .375, .0625, .0625, .375, .0625))
    np.testing.assert_array_equal(c.planar_gradients,
                                  ((-.25, 0., .25, -.25, 0., .25),
                                   (-.125, -.75, -.125, .125, .75, .125)))


def test_two_actual_hosts_integrate_frozen_partition_and_linear_fields():
    c = compiled()
    np.testing.assert_allclose(c.area_mean, [1/16, 3/8, 1/16, 1/16, 3/8, 1/16], rtol=0., atol=0.)
    assert c.area_m2 == .5
    np.testing.assert_array_equal(c.centroid_m, [1., .5, 0.])
    np.testing.assert_allclose(c.area_mean@c.shell_positions_m, c.centroid_m, atol=1e-15)
    np.testing.assert_allclose(c.planar_gradients@np.ones(6), 0., atol=1e-15)
    np.testing.assert_allclose(c.planar_gradients@c.shell_positions_m[:, :2], np.eye(2), atol=1e-15)
    assert np.linalg.matrix_rank(c.constraint_matrix) == 6
    assert c.constraint_matrix.shape == (6, 42)


def test_asymmetric_clips_reproduce_exact_bilinear_area_mean_and_gradients():
    definition, nodes, hosts, beam = fixture()
    definition = replace(definition, bounds_xy_m=(.25, 1.25, .125, .625))
    c = footprint.compile_footprint(definition, nodes, hosts, beam)
    xy = c.shell_positions_m[:, 0]*c.shell_positions_m[:, 1]
    # x*y is Q4-exact on both hosts; rectangular integrals factor independently.
    assert c.area_mean@xy == pytest.approx(.75*.375, abs=1e-15)
    np.testing.assert_allclose(c.planar_gradients@xy, [.375, .75], atol=1e-15)
    np.testing.assert_allclose(c.area_mean@c.shell_positions_m, [.75, .375, 0.], atol=1e-15)


def test_vertical_adjacency_with_asymmetric_clips_has_independent_axis_parity():
    definition, nodes, hosts, beam = fixture()
    nodes = tuple(replace(n, coordinates_m=(n.coordinates_m[1], n.coordinates_m[0], 0.)) for n in nodes)
    hosts = tuple(replace(h, node_ids=tuple(reversed(h.node_ids))) for h in hosts)
    definition = replace(definition, bounds_xy_m=(.125, .625, .25, 1.25))
    beam = replace(beam, coordinates_m=(.3, 1.3, .5))
    c = footprint.compile_footprint(definition, nodes, hosts, beam)
    xy = c.shell_positions_m[:, 0]*c.shell_positions_m[:, 1]
    assert c.area_mean@xy == pytest.approx(.375*.75, abs=1e-15)
    np.testing.assert_allclose(c.planar_gradients@xy, [.75, .375], atol=1e-15)
    np.testing.assert_allclose(c.area_mean@c.shell_positions_m, [.375, .75, 0.], atol=1e-15)


@pytest.mark.parametrize("axis", (3, 4))
def test_director_only_roll_pitch_remains_an_independent_port_channel(axis):
    c = compiled()
    shell = np.zeros((6, 6))
    shell[:, axis] = .7
    expected = np.zeros(6)
    expected[axis] = .7
    np.testing.assert_allclose(c.shell_trace@shell.ravel(), expected, atol=1e-15)
    # Keep translations fixed: director changes must reach eccentric translation rows.
    local = np.r_[shell.ravel(), np.zeros(6)]
    np.testing.assert_allclose(c.gap(local)[:3], np.cross(c.eccentricity_m, expected[3:]), atol=1e-15)
    np.testing.assert_allclose(c.gap(local)[3:], -expected[3:], atol=1e-15)


@pytest.mark.parametrize("kind,yaw", (("simple_shear", .5), ("symmetric_shear", 0.),
                                      ("isotropic_extension", 0.)))
def test_polar_yaw_is_half_curl_and_not_symmetric_strain(kind, yaw):
    c = compiled()
    shell = np.zeros((6, 6))
    x, y = c.shell_positions_m[:, :2].T
    if kind == "isotropic_extension":
        shell[:, 0], shell[:, 1] = x, y
    else:
        shell[:, 1] = x
        if kind == "symmetric_shear":
            shell[:, 0] = y
    trace = c.shell_trace@shell.ravel()
    np.testing.assert_array_equal(trace[3:5], [0., 0.])
    assert trace[5] == pytest.approx(yaw, abs=1e-15)


@pytest.mark.parametrize("axis", range(6))
def test_six_common_rigid_fields_reproduce_reference_spatial_port(axis):
    c = compiled()
    t, omega = np.zeros(3), np.zeros(3)
    (t if axis < 3 else omega)[axis % 3] = 1.
    local = np.zeros((7, 6))
    local[:6, :3] = t+np.cross(omega, c.shell_positions_m)
    local[:6, 3:] = omega
    local[6, :3] = t+np.cross(omega, c.beam_position_m)
    local[6, 3:] = omega
    np.testing.assert_allclose(c.gap(local.ravel()), 0., rtol=0., atol=2e-15)
    np.testing.assert_allclose(c.shell_trace@local[:6].ravel(),
                               np.r_[t+np.cross(omega, c.centroid_m), omega], atol=2e-15)


def test_transpose_force_moment_and_work_include_eccentric_signs():
    c = compiled()
    multiplier = np.array([2., -3., 5., 7., -11., 13.])
    covectors = c.transpose_work(multiplier).reshape(7, 6)
    xyz = np.vstack((c.shell_positions_m, c.beam_position_m))
    np.testing.assert_allclose(covectors[:, :3].sum(axis=0), 0., atol=1e-14)
    moment = (np.cross(xyz, covectors[:, :3])+covectors[:, 3:]).sum(axis=0)
    np.testing.assert_allclose(moment, 0., atol=2e-14)
    np.testing.assert_array_equal(covectors[6], multiplier)
    np.testing.assert_allclose(covectors[:6, :3].sum(axis=0), -multiplier[:3], atol=1e-14)
    np.testing.assert_allclose((np.cross(c.shell_positions_m-c.centroid_m, covectors[:6, :3])+
                               covectors[:6, 3:]).sum(axis=0),
                              -np.cross(c.eccentricity_m, multiplier[:3])-multiplier[3:], atol=2e-14)
    q = np.linspace(-.4, .6, 42)
    assert float(q@c.transpose_work(multiplier)) == pytest.approx(float(multiplier@c.gap(q)), abs=2e-14)


def test_nonrigid_transverse_shear_mode_is_admitted_but_rigid_node_law_rejects():
    c = compiled()
    q = np.zeros((7, 6))
    q[:6, 2] = c.shell_positions_m[:, 0]-c.centroid_m[0]
    np.testing.assert_allclose(c.gap(q.ravel()), 0., atol=1e-15)
    np.testing.assert_allclose(c.planar_gradients@q[:6, 2], [1., 0.], atol=1e-15)
    # A transverse slope changes no director: the law must not manufacture pitch.
    np.testing.assert_array_equal((c.shell_trace@q[:6].ravel())[3:], np.zeros(3))
    gaps = [reference_shell_matrix(x, c.beam_position_m)@row+q[6]
            for x, row in zip(c.shell_positions_m, q[:6])]
    assert np.linalg.norm(gaps) > 1.


def test_shell_drilling_coordinates_have_no_port_work():
    c = compiled()
    np.testing.assert_array_equal(c.constraint_matrix[:, 5:36:6], np.zeros((6, 6)))
    q = np.zeros(42)
    q[5:36:6] = np.arange(1., 7.)
    np.testing.assert_array_equal(c.gap(q), np.zeros(6))
    lam = np.array([0., 0., 0., 0., 0., 1.])
    actions = c.transpose_work(lam).reshape(7, 6)
    np.testing.assert_array_equal(actions[:6, 3:], np.zeros((6, 3)))
    assert np.linalg.norm(actions[:6, :3]) > 0.
    # Independent rigid boundary field u=(-y,x,0) gives unit yaw. Its work is
    # the force-couple moment about the centroid, without a shell rz coordinate.
    x, y = (c.shell_positions_m-c.centroid_m)[:, :2].T
    shell_force = actions[:6, :3]
    assert x@shell_force[:, 1]-y@shell_force[:, 0] == pytest.approx(-1., abs=1e-15)
    boundary_displacement = np.column_stack((-y, x, np.zeros(6)))
    assert np.sum(boundary_displacement*shell_force) == pytest.approx(-1., abs=1e-15)


def test_inventory_reordering_and_reidentified_elements_preserve_physical_map():
    definition, nodes, hosts, beam = fixture()
    first = footprint.compile_footprint(definition, nodes, hosts, beam)
    reordered = footprint.compile_footprint(definition, tuple(reversed(nodes)), tuple(reversed(hosts)), beam)
    assert first.identity == reordered.identity
    np.testing.assert_array_equal(first.constraint_matrix, reordered.constraint_matrix)
    renamed = {n.node_id: n.node_id+200 for n in nodes}
    nodes = tuple(replace(n, node_id=renamed[n.node_id]) for n in reversed(nodes))
    hosts = tuple(footprint.CartesianQ4Host(h.element_id+100, tuple(renamed[n] for n in h.node_ids)) for h in reversed(hosts))
    definition = replace(definition, host_element_ids=(121, 122))
    changed = footprint.compile_footprint(definition, nodes, hosts, beam)
    assert first.identity != changed.identity
    np.testing.assert_array_equal(first.gather_slots, changed.gather_slots)
    np.testing.assert_array_equal(first.constraint_matrix, changed.constraint_matrix)


def test_owned_definition_and_compiled_arrays_cannot_alias_or_reenable_writes():
    xyz = [0., 0., 0.]
    node = footprint.FlatQ4Node(1, xyz, 0)
    xyz[0] = 99.
    assert node.coordinates_m == (0., 0., 0.)
    c = compiled()
    for array in (c.gather_slots, c.shell_positions_m, c.beam_position_m, c.centroid_m,
                  c.eccentricity_m, c.area_mean, c.planar_gradients, c.shell_trace, c.constraint_matrix):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    q = np.arange(42., dtype=float)
    gathered = c.gather(q)
    q[:] = -1.
    np.testing.assert_array_equal(gathered, np.arange(42.))
    descriptor = c.descriptor()
    descriptor["host_element_ids"][0] = -1
    assert c.host_element_ids == (21, 22)


@pytest.mark.parametrize("change", (
    lambda d, n, h, b: (replace(d, bounds_xy_m=(-.1, 1.5, .25, .75)), n, h, b),
    lambda d, n, h, b: (replace(d, bounds_xy_m=(.1, .9, .25, .75)), n, h, b),
    lambda d, n, h, b: (d, (replace(n[0], q_offset=6),)+n[1:], h, b),
    lambda d, n, h, b: (d, (replace(n[0], coordinates_m=(.1, 0., 0.)),)+n[1:], h, b),
    lambda d, n, h, b: (d, (replace(n[0], coordinates_m=(0., 0., .01)),)+n[1:], h, b),
    lambda d, n, h, b: (d, tuple(replace(v, coordinates_m=(v.coordinates_m[0], v.coordinates_m[1], .1*v.coordinates_m[0])) for v in n), h, b),
    lambda d, n, h, b: (d, n, (h[0], h[0]), b),
    lambda d, n, h, b: (d, n, h, replace(b, q_offset=0)),
    lambda d, n, h, b: (d, n, h, replace(b, node_id=102)),
    lambda d, n, h, b: (d, n, (replace(h[0], node_ids=(1, 4, 5, 2)), h[1]), b),
    lambda d, n, h, b: (d, n, (replace(h[0], node_ids=(1, 3, 6, 4)), h[1]), b),
))
def test_unsupported_geometry_coverage_or_mapping_rejects(change):
    with pytest.raises(ValueError):
        footprint.compile_footprint(*change(*fixture()))


def test_axes_nonfinite_and_degenerate_declarations_reject_before_compilation():
    with pytest.raises(ValueError, match="identity"):
        footprint.PhysicalBeamNodePort(101, (1., .5, .2), 36,
                                       ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match="finite"):
        footprint.FlatQ4Node(1, (np.nan, 0., 0.), 0)
    with pytest.raises(ValueError, match="positive rectangular"):
        footprint.DistributingFootprintDefinition("bad", (21, 22), 101, (1., 1., 0., 1.))
    with pytest.raises(ValueError, match="distinct"):
        footprint.DistributingFootprintDefinition("bad", (21, 21), 101, (0., 1., 0., 1.))


def test_finite_real_gather_gap_and_work_inputs_are_required():
    c = compiled()
    for call, bad in ((c.gather, np.zeros(41)), (c.gap, np.full(42, np.inf)),
                      (c.transpose_work, np.ones(6, dtype=complex))):
        with pytest.raises(ValueError):
            call(bad)
    assert c.descriptor()["public_admission"] is False
    assert c.descriptor()["production_qualified"] is False


def test_compiled_model_binding_and_local_shapes_cannot_be_replaced_inconsistently():
    c = compiled()
    with pytest.raises(ValueError, match="binding"):
        replace(c, beam_node_id=102)
    with pytest.raises(ValueError, match="compiled local map"):
        replace(c, shell_trace=np.zeros((6, 35)))
    with pytest.raises(ValueError, match="gather"):
        replace(c, gather_slots=np.zeros(42, dtype=np.int64))


def test_finite_inputs_with_unrepresentable_gap_or_work_results_reject():
    c = compiled()
    maximum = np.finfo(float).max
    q = np.zeros((7, 6))
    q[:6, 0], q[6, 0] = -maximum, maximum
    with pytest.raises(ValueError, match="representable local gap"):
        c.gap(q.ravel())
    definition, nodes, hosts, beam = fixture()
    eccentric = footprint.compile_footprint(definition, nodes, hosts,
                                            replace(beam, coordinates_m=(4., .3, .5)))
    with pytest.raises(ValueError, match="representable local work"):
        eccentric.transpose_work([0., maximum, 0., 0., 0., 0.])


def _rectangular_fixture(nx, ny, bounds=(.125, 1.875, .0625, .9375)):
    positions = tuple((2*i/nx, j/ny, 0.) for j in range(ny+1) for i in range(nx+1))
    nodes = tuple(footprint.FlatQ4Node(i+1, point, 6*i) for i, point in enumerate(positions))
    hosts = tuple(footprint.CartesianQ4Host(21+j*nx+i,
        (j*(nx+1)+i+1, j*(nx+1)+i+2, (j+1)*(nx+1)+i+2, (j+1)*(nx+1)+i+1))
        for j in range(ny) for i in range(nx))
    beam = footprint.PhysicalBeamNodePort(1001, (1.3, .3, .5), 6*len(nodes), footprint.IDENTITY_AXES)
    definition = footprint.DistributingFootprintDefinition("fixed-pad", tuple(h.element_id for h in hosts),
                                                          beam.node_id, bounds)
    return definition, nodes, hosts, beam


@pytest.fixture(scope="module", params=((2, 1), (4, 2), (8, 4)), ids=("2-host", "8-host", "32-host"))
def refined_pad(request):
    return footprint.compile_footprint(*_rectangular_fixture(*request.param))


@pytest.mark.parametrize("bilinear", (False, True))
def test_one_fixed_physical_pad_reproduces_affine_and_bilinear_trace(refined_pad, bilinear):
    c = refined_pad
    x, y = c.shell_positions_m[:, :2].T
    b = int(bilinear)
    shell = np.column_stack((.4+.2*x-.3*y+b*.7*x*y,
                             -.2+.9*x+.1*y-b*.4*x*y,
                             1.-.5*x+.3*y+b*.2*x*y,
                             .7+.5*x-.2*y+b*.3*x*y,
                             -.3+.4*x+.6*y-b*.2*x*y,
                             np.full(len(x), 99.)))
    # Independent exact rectangular polynomial integrals: each x*y mean is
    # xbar*ybar. Physical directors remain independent of transverse slope.
    cx, cy = 1., .5
    expected = np.array((.4+.2*cx-.3*cy+b*.7*cx*cy,
                         -.2+.9*cx+.1*cy-b*.4*cx*cy,
                         1.-.5*cx+.3*cy+b*.2*cx*cy,
                         .7+.5*cx-.2*cy+b*.3*cx*cy,
                         -.3+.4*cx+.6*cy-b*.2*cx*cy,
                         .6-b*(.2*cy+.35*cx)))
    np.testing.assert_allclose(c.shell_trace@shell.ravel(), expected, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(c.centroid_m, (cx, cy, 0.))
    assert c.area_m2 == 1.75*.875
    port = np.r_[expected[:3]+np.cross(expected[3:], c.eccentricity_m), expected[3:]]
    np.testing.assert_allclose(c.gap(np.r_[shell.ravel(), port]), np.zeros(6), atol=2e-15, rtol=0)
    np.testing.assert_array_equal(c.shell_trace[:, 5::6], np.zeros((6, len(x))))


@pytest.mark.parametrize("axis", range(6))
def test_refined_fixed_pad_preserves_six_rigid_fields(refined_pad, axis):
    c = refined_pad
    t, omega = np.zeros(3), np.zeros(3)
    (t if axis < 3 else omega)[axis % 3] = 1.
    shell = np.column_stack((t+np.cross(omega, c.shell_positions_m),
                             np.tile(omega, (len(c.shell_node_ids), 1))))
    beam = np.r_[t+np.cross(omega, c.beam_position_m), omega]
    np.testing.assert_allclose(c.gap(np.r_[shell.ravel(), beam]), np.zeros(6), atol=2e-15, rtol=0)


@pytest.mark.parametrize("axis", range(6))
def test_refined_fixed_pad_preserves_all_action_work_and_wrenches(refined_pad, axis):
    c = refined_pad
    multiplier = np.eye(6)[axis]
    local = c.transpose_work(multiplier).reshape(-1, 6)
    xyz = np.vstack((c.shell_positions_m, c.beam_position_m))
    np.testing.assert_allclose(local[:, :3].sum(axis=0), np.zeros(3), atol=2e-14, rtol=0)
    np.testing.assert_allclose((np.cross(xyz, local[:, :3])+local[:, 3:]).sum(axis=0),
                               np.zeros(3), atol=2e-14, rtol=0)
    np.testing.assert_array_equal(local[-1], multiplier)
    q = np.sin(np.arange(len(c.gather_slots))+.2)
    assert q@local.ravel() == pytest.approx(multiplier@c.gap(q), abs=2e-14)
    np.testing.assert_array_equal(c.gather(q), q)


def _selected_inventory(definition, nodes, hosts, beam):
    selected = set().union(*(set(h.node_ids) for h in hosts))
    return (replace(definition, host_element_ids=tuple(h.element_id for h in hosts)),
            tuple(n for n in nodes if n.node_id in selected), tuple(hosts), beam)


@pytest.mark.parametrize("defect", ("hole", "overlap", "disconnected", "coarse_fine", "coincident", "plane", "unused"))
def test_multi_host_geometry_and_positive_selected_coverage_fail_closed(defect):
    nx, ny = (3, 3) if defect == "hole" else (3, 1) if defect == "disconnected" else (2, 2)
    inputs = _rectangular_fixture(nx, ny, (0., 2., 0., 1.))
    definition, nodes, hosts, beam = inputs
    if defect == "hole":
        hosts = hosts[:4]+hosts[5:]
    elif defect == "disconnected":
        hosts = (hosts[0], hosts[2])
    elif defect == "overlap":
        hosts += (replace(hosts[0], element_id=99),)
    elif defect == "coarse_fine":
        hosts = (footprint.CartesianQ4Host(99, (1, 2, 8, 7)), hosts[1], hosts[3])
    elif defect in ("coincident", "plane"):
        renamed = {n: n+200 for n in hosts[0].node_ids}
        extra = tuple(replace(n, node_id=renamed[n.node_id], q_offset=6*(len(nodes)+i))
                      for i, n in enumerate(n for n in nodes if n.node_id in renamed))
        if defect == "plane":
            extra = tuple(replace(n, coordinates_m=(n.coordinates_m[0], n.coordinates_m[1], .1)) for n in extra)
        nodes += extra
        hosts = (replace(hosts[0], node_ids=tuple(renamed[n] for n in hosts[0].node_ids)),)+hosts[1:]
        beam = replace(beam, q_offset=6*len(nodes))
    else:
        definition = replace(definition, bounds_xy_m=(1., 2., 0., 1.))
    message = {"hole": "complete rectangular", "disconnected": "complete rectangular",
               "overlap": "overlapping Cartesian", "coarse_fine": "coarse/fine",
               "coincident": "coincident", "plane": "common flat XY", "unused": "every selected host"}[defect]
    with pytest.raises(ValueError, match=message):
        footprint.compile_footprint(*_selected_inventory(definition, nodes, hosts, beam))


def test_bounded_host_and_unique_node_counts():
    for ids in ((21,), tuple(range(1, 130))):
        with pytest.raises(ValueError, match="two to 128 distinct"):
            footprint.DistributingFootprintDefinition("too-many", ids, 1001, (0., 2., 0., 1.))
    with pytest.raises(ValueError, match="256"):
        footprint.compile_footprint(*_rectangular_fixture(128, 1, (0., 2., 0., 1.)))
    c = footprint.compile_footprint(*_rectangular_fixture(16, 8, (0., 2., 0., 1.)))
    assert len(c.host_element_ids) == 128 and len(c.shell_node_ids) == 153
    assert c.constraint_matrix.shape == (6, 6*154)
    assert c.area_mean.sum() == 1.
    for array in (c.constraint_matrix, c.gather_slots, c.shell_trace):
        with pytest.raises(ValueError):
            array.setflags(write=True)
