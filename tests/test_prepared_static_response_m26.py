"""M26: the certified vectorized build in one pipelined pass.

Each owned scope probes the chunk whose storage memory_check accepted last and
reads the zero records of the next chunk; safe points and memory checks run
between scopes. The maps must be identical to a build with any other stride.
"""
import numpy as np
import pytest

from anyfesolver import FEModel, FixedSupport, LoadCase, RecoveryConfig, prepare_static_response_evaluator
from anyfesolver import prepared_static_recovery as recovery_module
from anyfesolver import prepared_static_vectorized as vector
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.elements import BeamElement, create_shell_element

STRESS = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)


def _grid_model(nx=5, ny=4):
    model = FEModel('m26 grid')
    model.add_material('steel', 210e9, .3, density=7850.)
    nid = lambda i, j: 1 + i + j * nx
    for j in range(ny):
        for i in range(nx):
            model.add_node(nid(i, j), .5 * i, .4 * j, .002 * ((i * 7 + j * 3) % 5))
    eid = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            if (i + j) % 7 == 3:
                for tri in ((corners[0], corners[1], corners[2]), (corners[0], corners[2], corners[3])):
                    eid += 1
                    model.add_element(eid, QualifiedE4PLS3ShellElement(eid, list(tri), 'steel', thickness=.01,
                                                                       reference_normal=(0., 0., 1.)))
            else:
                eid += 1
                model.add_element(eid, create_shell_element(eid, corners, 'steel', thickness=.01, formulation='e4-pl'))
    for i in range(nx - 1):
        eid += 1
        model.add_element(eid, BeamElement(eid, [nid(i, ny - 1), nid(i + 1, ny - 1)], 'steel',
                                           {'area': .003, 'Iy': 2e-5, 'Iz': 2e-5, 'J': 1e-5}))
    model.add_boundary_condition(FixedSupport('fixed', [nid(0, j) for j in range(ny)]))
    pressure = LoadCase('pressure')
    for k, e in enumerate(e for e, element in model.mesh.elements.items() if type(element) is not BeamElement):
        pressure.pressure_loads[e] = 1000. * (1 + k % 3)
    return model, {'pressure': pressure}


@pytest.fixture(scope='module')
def prepared():
    model, patterns = _grid_model()
    owner = prepare_static_response_evaluator(model, patterns, recovery_config=STRESS, envelope_recovery_config=STRESS,
                                              stress_execution='certified_vectorized')
    yield owner
    owner.close()


def _build(owner, *, safe_point=lambda stage: None, memory_check=None):
    return vector.build_vector_recovery(owner._library_elements(owner._snapshot), owner._snapshot.envelope_plan,
                                        STRESS.selected_components(), safe_point, memory_check=memory_check)


def _identical(a, b, path='root'):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        if a.dtype.hasobject:
            assert a.tolist() == b.tolist(), path
        else:
            assert a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            _identical(a[key], b[key], f'{path}[{key!r}]')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b)):
            _identical(x, y, f'{path}[{k}]')
    elif isinstance(a, float):
        assert a.hex() == b.hex(), path
    else:
        assert a == b, path


def _same_maps(left, right):
    for name in ('groups', 'locations', 'element_info', 'probes', 'worst', 'envelope_sources'):
        _identical(getattr(left, name), getattr(right, name), name)


@pytest.mark.parametrize('stride', [1, 4, 7, 128])
def test_maps_identical_for_every_stride(prepared, stride, monkeypatch):
    monkeypatch.setattr(vector, 'SAFE_POINT_STRIDE', stride)
    built = _build(prepared)
    _same_maps(built, prepared._vector)


def test_schedule_one_safe_point_per_chunk_plus_one(prepared, monkeypatch):
    monkeypatch.setattr(vector, 'SAFE_POINT_STRIDE', 4)
    stages, checks = [], []
    _build(prepared, safe_point=stages.append, memory_check=checks.append)
    count = len(prepared._library_elements(prepared._snapshot))
    chunks = -(-count // 4)
    assert stages == ['prepared_response.vector_schema'] + ['prepared_response.vector_recovery'] * chunks
    assert len(checks) == chunks and checks == sorted(checks) and len(set(checks)) == chunks
    monkeypatch.setattr(vector, 'SAFE_POINT_STRIDE', 128)
    full = []
    _build(prepared, memory_check=full.append)
    assert full == [checks[-1]]  # the last check sees the full (pre-M26) estimate


def test_elements_probed_only_after_a_check_that_includes_them(prepared, monkeypatch):
    monkeypatch.setattr(vector, 'SAFE_POINT_STRIDE', 4)
    order = list(prepared._library_elements(prepared._snapshot))
    events = []
    probe = vector._probe_element
    monkeypatch.setattr(vector, '_probe_element', lambda eid, *args: (events.append(('probe', eid)), probe(eid, *args))[1])
    _build(prepared, memory_check=lambda extra: events.append(('check', extra)))
    checks_seen = 0
    for kind, value in events:
        if kind == 'check':
            checks_seen += 1
        else:
            assert checks_seen >= order.index(value) // 4 + 1
    assert [value for kind, value in events if kind == 'probe'] == order


def test_refusal_stops_before_probing_the_refused_chunk(prepared, monkeypatch):
    monkeypatch.setattr(vector, 'SAFE_POINT_STRIDE', 4)
    order = list(prepared._library_elements(prepared._snapshot))
    probed = []
    probe = vector._probe_element
    monkeypatch.setattr(vector, '_probe_element', lambda eid, *args: (probed.append(eid), probe(eid, *args))[1])

    class Refused(Exception):
        pass

    calls = []

    def check(extra):
        calls.append(extra)
        if len(calls) == 2:
            raise Refused()

    with pytest.raises(Refused):
        _build(prepared, memory_check=check)
    assert probed == order[:4]


def test_scope_never_spans_a_safe_point_or_memory_check(prepared, monkeypatch):
    monkeypatch.setattr(vector, 'SAFE_POINT_STRIDE', 4)
    seen = []
    _build(prepared, safe_point=lambda stage: seen.append(recovery_module._OWNED_RECOVERY_SCOPE.get()),
           memory_check=lambda extra: seen.append(recovery_module._OWNED_RECOVERY_SCOPE.get()))
    assert seen and all(value is None for value in seen)


def test_empty_union_checks_zero_once_without_safe_points():
    stages, checks = [], []
    built = vector.build_vector_recovery({}, None, None, stages.append, memory_check=checks.append)
    assert stages == [] and checks == [0] and built.groups == [] and built.locations == {}
