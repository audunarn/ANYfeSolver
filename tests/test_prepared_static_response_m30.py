"""M30: per-case levers are bitwise identical to the code they replace.

Each changed function is compared with a copy of its pre-M30 form (or with the
shared implementation it reproduces) on realistic and edge-case inputs.
"""
import warnings

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import RecoveryConfig, prepare_static_response_evaluator, prepare_static_evaluator
from anyfesolver import prepared_static_vectorized as vector_module
from anyfesolver.analysis_session import _immutable_array, _immutable_csr
from anyfesolver.assembly import _backward_error
from anyfesolver.prepared_static import _block_norm_data, _certified_backward_error, _operator_norm_data
from anyfesolver.prepared_static_recovery import _planned_compact
from test_prepared_static_response_m26 import _grid_model

STRESS = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a.hex() == b.hex() if a == a else b != b
    return a == b


def _outcome(function, *args):
    """('value', result, warnings) or ('raise', type, message, warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            made = ('value', function(*args))
        except Exception as error:  # noqa: BLE001 - compared, not handled
            made = ('raise', type(error), str(error))
    return made + ([(item.category, str(item.message)) for item in caught],)


def _same_outcome(a, b):
    """The same kind of outcome, the same value or exception, and the same warnings."""
    if a[0] != b[0] or a[-1] != b[-1]:
        return False
    return _same(a[1], b[1]) if a[0] == 'value' else a[1:-1] == b[1:-1]


def test_outcome_comparison_includes_warnings():
    # M31c (review round 2): warnings are compared on both branches.
    warned = [(RuntimeWarning, 'overflow encountered in matmul')]
    assert not _same_outcome(('value', 1.0, []), ('value', 1.0, warned))
    assert not _same_outcome(('raise', ValueError, 'x', []), ('raise', ValueError, 'x', warned))
    assert not _same_outcome(('value', 1.0, warned), ('value', 2.0, warned))
    assert _same_outcome(('value', 1.0, warned), ('value', 1.0, list(warned)))
    assert _same_outcome(('raise', ValueError, 'x', warned), ('raise', ValueError, 'x', list(warned)))


ERROR_STATES = [None, {'all': 'raise'}, {'all': 'warn'}, {'over': 'raise', 'invalid': 'ignore'},
                {'under': 'raise'}, {'all': 'ignore'}]


def _in_state(state, function, *args):
    if state is None:
        return _outcome(function, *args)
    with np.errstate(**state):
        return _outcome(function, *args)


# --------------------------------------------------------------------------- reaction maps

def _reference_compact(order, made):  # pre-M30
    return {node: made[i].copy() for i, node in enumerate(order) if np.any(made[i] != 0.0)}


@pytest.mark.parametrize('rows', [0, 1, 7, 400])
def test_compact_maps_identical(rows):
    rng = np.random.default_rng(rows)
    made = rng.normal(size=(rows, 6))
    if rows:
        made[::3] = 0.0
        made[1::5, 2] = np.nan
        made[2::7] = -0.0
    order = tuple(range(100, 100 + rows))
    assert _same(_planned_compact(order, made), _reference_compact(order, made))


# --------------------------------------------------------------------------- backward error

def _operators():
    # M31c: immutable CSR operators, as the evaluator holds them (_immutable_csr).
    rng = np.random.default_rng(31)
    a = sparse.random(60, 60, density=0.1, random_state=1, format='csr') + sparse.eye(60, format='csr') * 4.0
    yield 'random', _immutable_csr(sparse.csr_matrix(a)), rng.normal(size=60)
    yield 'zero_matrix', _immutable_csr(sparse.csr_matrix((5, 5))), rng.normal(size=5)
    yield 'empty', _immutable_csr(sparse.csr_matrix((0, 0))), np.zeros(0)
    big = sparse.csr_matrix(np.diag([1e308, 1e308]))
    with np.errstate(over='ignore'):
        yield 'infinite_entries', _immutable_csr(big + big), np.ones(2)
    yield 'overflowing_row_sums', _immutable_csr(sparse.csr_matrix(np.array([[1e308, 1e308], [0.0, 1.0]]))), np.ones(2)
    yield 'nan_entry', _immutable_csr(sparse.csr_matrix(np.array([[np.nan, 0.0], [0.0, 1.0]]))), np.ones(2)
    yield 'tiny_entries', _immutable_csr(sparse.csr_matrix(np.array([[5e-324, 1e-310], [1e-310, 2e-308]]))), \
        np.array([1e-300, 2.0])
    yield 'large_solution', _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]]))), \
        np.array([1e308, -1e308])


@pytest.mark.parametrize('state', ERROR_STATES, ids=str)
@pytest.mark.parametrize('name, matrix, x', list(_operators()))
def test_certified_backward_error_identical(name, matrix, x, state):
    # M31b (review): values, exceptions and warnings, under several NumPy error
    # states; the block's norm data is computed first, as in the evaluator.
    assert _block_norm_data(matrix, _backward_error) is not None      # the shortcut is taken
    rng = np.random.default_rng(32)
    with np.errstate(over='ignore', invalid='ignore'):
        exact = np.asarray(matrix @ x).reshape(-1)
    rhs_cases = [exact + rng.normal(scale=1e-12, size=x.shape), np.zeros_like(x)]
    solutions = [x, np.zeros_like(x)]
    if x.size:
        bad = x.copy()
        bad[0] = np.nan
        solutions.append(bad)
    for solution in solutions:
        for rhs in rhs_cases:
            expected = _in_state(state, _backward_error, matrix, solution, rhs)

            def certified():
                norm = _block_norm_data(matrix, _backward_error)
                return _certified_backward_error(matrix, solution, rhs, norm, _backward_error)
            got = _in_state(state, certified)
            assert _same_outcome(got, expected), (name, got, expected)


def test_norm_data_is_used_only_for_unchanged_immutable_matrices():
    # M31c (review round 2, round-1 finding 4): norm data never outlives a change of
    # the matrix it describes.
    x, b = np.array([1.0]), np.array([1.0])
    mutable = sparse.csr_matrix(np.array([[1.0]]))
    assert _block_norm_data(mutable, _backward_error) is None
    norm = _operator_norm_data(mutable)                                # handed over directly
    mutable.data[0] = 2.0                                              # changed in place
    assert _certified_backward_error(mutable, x, b, norm, _backward_error) == _backward_error(mutable, x, b) == 1 / 3
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[1.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    assert norm is not None and _certified_backward_error(frozen, x, b, norm, _backward_error) == 0.0
    with pytest.raises(ValueError):
        frozen.data.setflags(write=True)                               # cannot be written in place
    frozen.data = _immutable_array(np.array([2.0]))                    # replaced
    assert _certified_backward_error(frozen, x, b, norm, _backward_error) == _backward_error(frozen, x, b) == 1 / 3
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[1.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    frozen.indptr = np.array(frozen.indptr)                             # a writable copy of a part
    assert _block_norm_data(frozen, _backward_error) is None
    assert _same(_certified_backward_error(frozen, x, b, norm, _backward_error), _backward_error(frozen, x, b))


def test_certified_backward_error_falls_back_for_foreign_matrix_and_blocks():
    a = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    other = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    x, b = np.array([1.0, -2.0]), np.array([0.1, -5.0])
    norm = _operator_norm_data(other)
    assert _same(_certified_backward_error(a, x, b, norm, _backward_error), _backward_error(a, x, b))
    assert _same(_certified_backward_error(a, x, b, None, _backward_error), _backward_error(a, x, b))
    X, B = np.column_stack([x, 2 * x]), np.column_stack([b, b])
    assert _same(_certified_backward_error(a, X, B, _operator_norm_data(a), _backward_error), _backward_error(a, X, B))
    bad = _immutable_csr(sparse.csr_matrix(np.array([[np.inf, 0.0], [0.0, 1.0]])))
    assert _same(_certified_backward_error(bad, x, b, _operator_norm_data(bad), _backward_error),
                 _backward_error(bad, x, b))


def test_replaced_backward_error_function_is_honoured():
    a = sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]]))
    calls = []
    def injected(*args):
        calls.append(args)
        return 1.0
    assert _certified_backward_error(a, np.ones(2), np.ones(2), _operator_norm_data(a), injected) == 1.0
    assert len(calls) == 1
    assert _block_norm_data(a, injected) is None
    assert _certified_backward_error(a, np.ones(2), np.ones(2), None, injected) == 1.0 and len(calls) == 2


def test_rebound_assembly_function_is_called_as_is(monkeypatch):
    # M31b (review): the original is the function captured at import, so a rebound
    # module attribute is an injected implementation like any other.
    import anyfesolver.assembly as assembly_module
    a = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    monkeypatch.setattr(assembly_module, '_backward_error', lambda *args: 7.0)
    assert _certified_backward_error(a, np.ones(2), np.ones(2), _operator_norm_data(a), assembly_module._backward_error) == 7.0
    assert _block_norm_data(a, assembly_module._backward_error) is None
    assert _same(_block_norm_data(a, _backward_error)[1:3], _operator_norm_data(a)[1:3])


# --------------------------------------------------------------------------- vector maps

def _reference_evaluate_groups(vector, full):  # pre-M30
    from anyfesolver.prepared_static_vectorized import PreparedStaticError, _split, derived_fields
    columns = full.shape[1]
    outputs = []
    for group in vector.groups:
        matrix, base, dofs = group['matrix'], group['base'], group['dofs']
        values = np.empty((columns,) + base.shape)
        for column in range(columns):
            local = np.ascontiguousarray(full[dofs, column])[:, :, None]
            values[column] = np.matmul(matrix, local)[:, :, 0] + base
        if not np.all(np.isfinite(values)):
            raise PreparedStaticError('nonfinite vectorized recovery values')
        elements = values.shape[1]
        fields = _split(values, group['linear'], (columns, elements))
        fields.update(derived_fields(group['derived'], fields))
        outputs.append(fields)
    return outputs


@pytest.fixture(scope='module')
def grid_owner():
    model, patterns = _grid_model(nx=8, ny=6)
    owner = prepare_static_response_evaluator(model, patterns, recovery_config=STRESS, envelope_recovery_config=STRESS,
                                              stress_execution='certified_vectorized')
    yield owner
    owner.close()


@pytest.mark.parametrize('chunk', [1, 3, 7, 512])
@pytest.mark.parametrize('columns', [1, 5, 32])
def test_chunked_vector_maps_identical(grid_owner, chunk, columns, monkeypatch):
    monkeypatch.setattr(vector_module, 'GROUP_ELEMENT_CHUNK', chunk)
    rng = np.random.default_rng(chunk * 100 + columns)
    full = rng.normal(scale=1e-3, size=(grid_owner._snapshot.total_dofs, columns))
    assert _same(vector_module.evaluate_groups(grid_owner._vector, full),
                 _reference_evaluate_groups(grid_owner._vector, full))


def test_non_default_error_state_takes_the_column_order(grid_owner, monkeypatch):
    # M31b (review): the chunked order runs only under NumPy's default error state.
    calls = []
    original = vector_module._chunked_values
    monkeypatch.setattr(vector_module, '_chunked_values', lambda *args: (calls.append(1), original(*args))[1])
    rng = np.random.default_rng(5)
    full = rng.normal(scale=1e-3, size=(grid_owner._snapshot.total_dofs, 4))
    for state in ({'over': 'raise'}, {'all': 'ignore'}, {'under': 'warn'}):
        with np.errstate(**state):
            assert _same(vector_module.evaluate_groups(grid_owner._vector, full),
                         _reference_evaluate_groups(grid_owner._vector, full))
    assert calls == []
    # M31c (review round 2): nor outside round to nearest without flush-to-zero.
    assert vector_module._standard_float_environment()
    monkeypatch.setattr(vector_module, '_standard_float_environment', lambda: False)
    assert _same(vector_module.evaluate_groups(grid_owner._vector, full),
                 _reference_evaluate_groups(grid_owner._vector, full))
    assert calls == []
    monkeypatch.setattr(vector_module, '_standard_float_environment', lambda: True)
    vector_module.evaluate_groups(grid_owner._vector, full)
    assert calls


@pytest.mark.parametrize('state', ERROR_STATES, ids=str)
@pytest.mark.parametrize('kind', ['overflow', 'nan', 'inf', 'finite'])
def test_vector_maps_raise_and_warn_as_before(grid_owner, state, kind, monkeypatch):
    # M31b (review): non-finite results (overflow, NaN or inf input) give the same
    # values, warnings and exceptions as the pre-M30 column order.
    monkeypatch.setattr(vector_module, 'GROUP_ELEMENT_CHUNK', 7)
    rng = np.random.default_rng(7)
    full = rng.normal(scale=1e-3, size=(grid_owner._snapshot.total_dofs, 3))
    if kind == 'overflow':
        full[::5, 1] = 1e306
    elif kind == 'nan':
        full[3, 2] = np.nan
    elif kind == 'inf':
        full[4, 0] = np.inf
    got = _in_state(state, vector_module.evaluate_groups, grid_owner._vector, full)
    expected = _in_state(state, _reference_evaluate_groups, grid_owner._vector, full)
    assert _same_outcome(got, expected), (got[0], expected[0], got[-1], expected[-1])


def test_certification_in_the_evaluator_equals_the_original(monkeypatch):
    # M31b (review): every certification equals assembly._backward_error at the time
    # of the call; the norm is computed per block (again after a canonical solve).
    from anyfesolver import PatternCase
    from anyfesolver import prepared_static as static_module
    from anyfesolver import prepared_static_response as response_module
    agreed, norms, fail_once, used = [], {'static': 0, 'response': 0}, [True], []
    for label, module in (('static', static_module), ('response', response_module)):
        certify, norm = module._certified_backward_error, module._block_norm_data

        def spy(matrix, solution, rhs, norm_data, implementation, _certify=certify, _label=label):
            got = _certify(matrix, solution, rhs, norm_data, implementation)
            agreed.append(_same(got, _backward_error(matrix, solution, rhs)))
            if _label == 'response' and fail_once[0]:
                fail_once[0] = False
                return float('inf')                       # one canonical fallback in the first block
            return got

        def count(matrix, implementation, _norm=norm, _label=label):
            norms[_label] += 1
            made = _norm(matrix, implementation)
            used.append(made is not None)                 # the evaluator's operators qualify
            return made
        monkeypatch.setattr(module, '_certified_backward_error', spy)
        monkeypatch.setattr(module, '_block_norm_data', count)
    # The response authority refuses rebound helpers; admit these spies for this test.
    monkeypatch.setattr(response_module, '_RESPONSE_FUNCTIONS', response_module._capture_response_functions())
    model, patterns = _grid_model(nx=6, ny=5)
    with prepare_static_response_evaluator(model, patterns, recovery_config=STRESS, envelope_recovery_config=STRESS,
                                           stress_execution='certified_vectorized') as owner:
        cases = [PatternCase(f'c{k}', {'pressure': 1.0 + k}) for k in range(10)]
        blocks = list(owner.iter_batches(cases, batch_size=4))
    assert len(blocks) == 3 and agreed and all(agreed) and used and all(used)
    assert norms['static'] == 3                       # once per block
    assert norms['response'] == 3 + 1                 # once per block, again after the fallback


# --------------------------------------------------------------------------- M31d (review round 3)

def _poisoned(norm):
    """Norm data with a wrong row-sum maximum: a result that used it would differ."""
    return norm[:2] + (norm[2] * 3.0 + 1.0,) + norm[3:]


def test_reinterpreted_parts_take_the_original():
    x, b = np.array([1.0]), np.array([1.0])
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[1.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    frozen.data.dtype = np.int64                                       # the same bytes read as an integer
    assert _same(_certified_backward_error(frozen, x, b, norm, _backward_error), _backward_error(frozen, x, b))
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    frozen.data.shape = (2, 1)                                         # read with another shape
    x2, b2 = np.ones(2), np.ones(2)
    got = _outcome(_certified_backward_error, frozen, x2, b2, _poisoned(norm), _backward_error)
    assert _same_outcome(got, _outcome(_backward_error, frozen, x2, b2)) and got[0] == 'raise'


def test_another_floating_point_environment_takes_the_original(monkeypatch):
    from anyfesolver import prepared_static as static_module
    x, b = np.array([1.0, 2.0]), np.array([0.5, 7.0])
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    assert not _same(_certified_backward_error(frozen, x, b, _poisoned(norm), _backward_error),
                     _backward_error(frozen, x, b))                    # the poison is visible ...
    monkeypatch.setattr(static_module, '_standard_float_environment', lambda: False)
    assert _block_norm_data(frozen, _backward_error) is None
    assert _same(_certified_backward_error(frozen, x, b, _poisoned(norm), _backward_error),
                 _backward_error(frozen, x, b))                        # ... unless the environment changed


@pytest.mark.parametrize('name', ['max', 'abs', 'asarray', 'errstate'])
def test_replaced_numpy_functions_take_the_original(monkeypatch, name):
    x, b = np.array([1.0, 2.0]), np.array([0.5, 7.0])
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    original = getattr(np, name)
    monkeypatch.setattr(np, name, lambda *args, **kwargs: original(*args, **kwargs))
    assert _block_norm_data(frozen, _backward_error) is None
    assert _same(_certified_backward_error(frozen, x, b, _poisoned(norm), _backward_error),
                 _backward_error(frozen, x, b))


def test_replaced_matmul_is_called_as_by_the_column_order(grid_owner, monkeypatch):
    # M31d (review round 3): a replaced np.matmul that warns per call sees exactly
    # the calls of the pre-M30 order.
    import warnings
    monkeypatch.setattr(vector_module, 'GROUP_ELEMENT_CHUNK', 7)
    original = np.matmul

    def warning_matmul(*args, **kwargs):
        warnings.warn('matmul called', UserWarning)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'matmul', warning_matmul)
    full = np.random.default_rng(9).normal(scale=1e-3, size=(grid_owner._snapshot.total_dofs, 3))
    got = _outcome(vector_module.evaluate_groups, grid_owner._vector, full)
    expected = _outcome(_reference_evaluate_groups, grid_owner._vector, full)
    assert _same_outcome(got, expected) and len(got[-1]) == 3 * len(grid_owner._vector.groups)


# --------------------------------------------------------------------------- M31e (review round 4)

def test_the_certification_shortcut_calls_no_np_ndim(monkeypatch):
    # M31e: the shortcut makes no np.ndim call (the original reads the converted
    # solution's ndim attribute).
    x, b = np.array([1.0, 2.0]), np.array([0.5, 7.0])
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    norm = _block_norm_data(frozen, _backward_error)
    calls = []
    original = np.ndim
    monkeypatch.setattr(np, 'ndim', lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1])
    got = _certified_backward_error(frozen, x, b, norm, _backward_error)
    assert calls == []
    assert _same(got, _backward_error(frozen, x, b))
    assert _same(_certified_backward_error(frozen, [1.0, 2.0], b, norm, _backward_error),
                 _backward_error(frozen, [1.0, 2.0], b))                # not an ndarray: the original


# --------------------------------------------------------------------------- M31f (review round 5)

def test_matrices_whose_conversion_could_warn_take_the_original():
    # M31f: complex data warns when converted to float; the block path must convert
    # (and warn) exactly as often as the original.
    x, b = np.array([1.0]), np.array([1.0])
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[1.0]])))
    frozen.data = _immutable_array(np.array([1.0 + 1.0j]))

    def block_path():
        return _certified_backward_error(frozen, x, b, _block_norm_data(frozen, _backward_error), _backward_error)
    got = _outcome(block_path)
    assert _block_norm_data(frozen, _backward_error) is None
    assert _same_outcome(got, _outcome(_backward_error, frozen, x, b)) and len(got[-1]) == 1
    narrow = _immutable_csr(sparse.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])))
    narrow.indices = _immutable_array(narrow.indices.astype(np.int16))
    assert _block_norm_data(narrow, _backward_error) is None


# --------------------------------------------------------------------------- M31i (review round 8)

def _noncanonical(data):
    matrix = sparse.csr_matrix((np.array(data, float), np.array([1, 0, 0]), np.array([0, 2, 3])), shape=(2, 2))
    matrix.data, matrix.indices, matrix.indptr = (_immutable_array(matrix.data), _immutable_array(matrix.indices),
                                                  _immutable_array(matrix.indptr))
    return matrix


@pytest.mark.parametrize('data', [[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0]])
def test_noncanonical_matrices_take_the_original(data):
    # M31i: the row sums of a non-canonical matrix would rewrite its read-only parts;
    # with non-finite data the original never sums them.
    matrix = _noncanonical(data)
    x, b = np.ones(2), np.array([3.0, 3.0])
    assert _block_norm_data(matrix, _backward_error) is None

    def block_path():
        return _certified_backward_error(matrix, x, b, _block_norm_data(matrix, _backward_error), _backward_error)
    assert _same_outcome(_outcome(block_path), _outcome(_backward_error, matrix, x, b))


def test_a_stale_canonical_flag_is_not_trusted():
    # M31i: SciPy caches has_canonical_format; replacing a part keeps the stale flag.
    frozen = _immutable_csr(sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]])))
    assert frozen.has_canonical_format and _block_norm_data(frozen, _backward_error) is not None
    frozen.indices = _immutable_array(np.array([1, 0, 1], dtype=frozen.indices.dtype))     # row 0 unsorted
    assert frozen.has_canonical_format                                                      # still cached
    assert _block_norm_data(frozen, _backward_error) is None
