"""Small reference/audit units; no nonlinear beam solve or mesh campaign."""
import ast
from hashlib import sha256
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from docs.reference_cases import ge_beam3_fibre_arch_comparison as comparison


def symmetric():
    x = np.linspace(-1., 1., 5)
    positions = np.column_stack((x, .1*(1-x*x), np.zeros(5)))
    frames = []
    for value in x:
        t = np.array([1., -.2*value, 0.]); t /= np.linalg.norm(t)
        frames.append(np.column_stack((t, [0., 0., 1.], np.cross(t, [0., 0., 1.]))))
    return dict(positions=positions.tolist(), position_low=np.zeros((5, 3)).tolist(),
        nodal_frames=np.asarray(frames).tolist(), cell_rotations=np.tile(np.eye(3), (2, 2, 1, 1)).tolist())


def test_reflection_preserves_material_director_and_pair_low_parts():
    value = symmetric()
    assert max(comparison.geometry_diagnostics(value).values()) == 0.
    value['position_low'][0][0] = 1e-20
    assert comparison.geometry_diagnostics(value)['position_reflection_error'] == 1e-20


@pytest.mark.parametrize('incident', ['position', 'plane', 'nodal', 'cell'])
def test_asymmetry_is_not_hidden_by_small_equilibrium_residual(incident):
    value = symmetric()
    if incident == 'position': value['positions'][1][1] += .001
    elif incident == 'plane': value['positions'][2][2] = .001
    else:
        a = .01; turn = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        if incident == 'nodal': value['nodal_frames'][1] = (turn@value['nodal_frames'][1]).tolist()
        else: value['cell_rotations'][0][0] = turn.tolist()
    assert max(comparison.geometry_diagnostics(value).values()) >= .001


@pytest.mark.parametrize('incident', ['boolean', 'nonfinite', 'frame', 'shape'])
def test_malformed_geometry_fails_closed(incident):
    value = symmetric()
    if incident == 'boolean': value['positions'][0][0] = True
    elif incident == 'nonfinite': value['positions'][0][0] = float('nan')
    elif incident == 'frame': value['cell_rotations'][0][0][0][0] = 2.
    else: value['position_low'].pop()
    with pytest.raises(ValueError): comparison.geometry_diagnostics(value)


def packet():
    def seal(body, key): return {**body, key: sha256(comparison.canonical(body)).hexdigest()}
    initial = seal(dict(target=0, previous_sha256='model'), 'record_sha256')
    row = seal(dict(target=1, previous_sha256=initial['record_sha256']), 'record_sha256')
    return seal(dict(schema='GE_BEAM3_PHYSICAL_FIBRE_TRANSLATION_CONTROL_CHAIN_V1',
        program=dict(schema='GE_BEAM3_KINEMATIC_SEEDED_SPATIAL_NEWTON_FIBRE_CONTROL_V1'),
        completed_targets=1, initial=initial, records=[row]), 'checkpoint_sha256')


def test_packet_byte_chain_validation_is_not_mechanical_replay():
    value = packet(); raw = comparison.canonical(value)
    assert comparison.preserved(raw, sha256(raw).hexdigest()) == value
    with pytest.raises(ValueError): comparison.preserved(raw, '0'*64)
    value['records'][0]['target'] = 2
    raw = comparison.canonical(value)
    with pytest.raises(ValueError): comparison.preserved(raw, sha256(raw).hexdigest())


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b' {"x":1}\n'])
def test_strict_packet_encoding(raw):
    with pytest.raises(ValueError): comparison.preserved(raw, sha256(raw).hexdigest())


def test_force_unit_change_preserves_continuum_equations_and_jacobians():
    ref = comparison.continuum
    t = np.array([-1., -.5, 0.]); y = np.array([t, .1*(1-t*t), [.2, .1, 0.], [.001, -.002, .003]])
    p = np.array([-.02, -.01]); c = 2.**20
    scaled = y.copy(); scaled[3] *= c
    expected = ref.equations(t, y, p, .1, 1., .4, .0001); expected[3] *= c
    np.testing.assert_array_equal(ref.equations(t, scaled, c*p, .1, c, c*.4, c*.0001), expected)
    a, b = ref.derivatives(t, y, p, .1, 1., .4, .0001)
    aa, bb = ref.derivatives(t, scaled, c*p, .1, c, c*.4, c*.0001)
    diagonal = np.array([1., 1., 1., c])
    np.testing.assert_array_equal(aa, a*diagonal[:, None, None]/diagonal[None, :, None])
    np.testing.assert_array_equal(bb, b*diagonal[:, None, None]/c)


def test_sampling_requires_existing_reference_nodes_and_proper_frames():
    parameter = np.linspace(-1., 0., 129)
    ref = SimpleNamespace(parameter=parameter,
        fields=np.array([parameter, .1*(1-parameter*parameter), np.arctan(-.2*parameter), 0.*parameter]))
    points, frames = comparison.sampled_geometry(ref, np.linspace(-1., 1., 9))
    np.testing.assert_allclose(points[:, 1], .1*(1-points[:, 0]**2), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(frames), 1., atol=1e-15)
    with pytest.raises(ValueError): comparison.sampled_geometry(ref, [-.123])


def test_normalized_small_reference_keeps_physical_units_separate(tmp_path):
    scale, ref = comparison.normalized_reference(.01, axial=1.e6, shear=4.e5, bending=100.)
    assert scale == 1.e6 and ref.axial == 1. and ref.bending == .0001
    assert ref.load > 0 and ref.load_slope > 0 and ref.work_error < 1e-9
    points, frames = comparison.sampled_geometry(ref, np.linspace(-1., 1., 5))
    value = symmetric(); value['positions'] = points.tolist(); value['nodal_frames'] = frames.tolist()
    row = dict(displacement_target=.01, parameter=scale*ref.load, mechanical=value)
    result = comparison.compare(row, np.linspace(-1., 1., 5), scale, ref)
    assert result['load_relative_error'] == result['nodal_position_error'] == result['nodal_frame_error'] == 0.
    assert not result['production_qualified'] and not result['same_equilibrium_branch_proved']
    (tmp_path/'normalized-reference.json').write_bytes(comparison.canonical(result))
    row['displacement_target'] = .02
    with pytest.raises(ValueError): comparison.compare(row, [-1., -.5, 0., .5, 1.], scale, ref)


def test_no_production_or_discrete_mechanics_imports():
    for module in (comparison, comparison.continuum):
        imports = [node for node in ast.walk(ast.parse(inspect.getsource(module)))
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            assert not any(name.startswith('anyfesolver') for name in names)
