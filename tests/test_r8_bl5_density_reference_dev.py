"""Pure preparation controls for the fixed supplementary density reference.

The reference script refuses to run in any process that already imported
product modules, so every operation that executes script code runs in a fresh
short-lived interpreter. The fresh driver enforces the no-product-import
invariant; the mathematical assertions below are unchanged.
"""
from fractions import Fraction as F
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts import r8_bl5_density_reference_dev as reference


ROOT = Path(__file__).resolve().parents[1]

FRESH_DRIVER = r'''
import copy
import importlib.abc
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
from scripts import r8_bl5_density_reference_dev as reference

FORBIDDEN_ROOTS = ('anyfesolver', 'openseespy', 'openseespywin')


class ForbiddenImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN_ROOTS:
            raise AssertionError(
                'pure preparation attempted native/product import: ' + fullname)


def product_modules():
    return [name for name in sys.modules
            if name.split('.')[0] in FORBIDDEN_ROOTS]


op = sys.argv[2]
with open(sys.argv[3], 'rb') as stream:
    payload = pickle.load(stream)

if op == 'prepare_pair':
    sys.meta_path.insert(0, ForbiddenImport())
    fine, coarse = reference.prepare(4), reference.prepare(2)
    assert not product_modules(), 'pure preparation imported product modules'
    result = (fine, coarse)
elif op == 'aggregate_refusal':
    model, key, value, with_affine = payload
    if with_affine:
        affine, _, _, _, _, ref = reference.services()
        model = copy.deepcopy(model)
        model[key] = value
        try:
            reference.aggregate_rows(model, ref, affine)
            result = ('no-refusal', None)
        except ValueError as error:
            result = (str(error), type(error).__name__)
    else:
        _, _, _, _, _, ref = reference.services()
        model = copy.deepcopy(model)
        model[key] = value
        try:
            reference.aggregate_rows(model, ref)
            result = ('no-refusal', None)
        except ValueError as error:
            result = (str(error), type(error).__name__)
elif op == 'geometry_refusal':
    key, value = payload
    _, refinement, _, _, _, ref = reference.services()
    changed = SimpleNamespace(**{name: getattr(ref, name) for name in
        (*reference.SECTIONS, 'WIDTH_M', 'BEAM_AXIS_Z_M', 'BEAM_BOTTOM_ARM_M')})
    setattr(changed, key, value)
    try:
        reference.geometry(changed, 4, refinement)
        result = ('no-refusal', None)
    except ValueError as error:
        result = (str(error), type(error).__name__)
elif op == 'changed_source_refusal':
    original = reference.sha

    def changed(path):
        return '0'*64 if Path(path).name == reference.AFFINE_SOURCE else original(path)

    loaded = []

    def forbidden(*args, **kwargs):
        loaded.append(True)
        raise AssertionError('changed source must be refused before loading services')

    reference.sha = changed
    reference.importlib.util.spec_from_file_location = forbidden
    try:
        try:
            reference.services()
            message, kind = 'no-refusal', None
        except ValueError as error:
            message, kind = str(error), type(error).__name__
    finally:
        reference.sha = original
    result = (message, kind, loaded)
elif op == 'prepare_owns':
    original = reference.prepare(4)
    prepared = reference.prepare(4)
    prepared['model']['nodes'][100001] = (9., 9., 9.)
    prepared['physical_rows'][0][2] = 17.
    prepared['supports'][0]['beam_shape_integrals_m'][0][100001] = 99.
    prepared['sections']['E_PA'] = 1.
    result = (original, prepared)
else:
    raise AssertionError('unknown fresh-process operation: ' + op)

with open(sys.argv[4], 'wb') as stream:
    pickle.dump(result, stream)
'''


def fresh_result(op, payload=None):
    """Run one reference operation in a fresh interpreter and unpickle its result."""
    with tempfile.TemporaryDirectory() as folder:
        inp, out = Path(folder)/'payload.pickle', Path(folder)/'result.pickle'
        inp.write_bytes(pickle.dumps(payload))
        run = subprocess.run(
            [sys.executable, '-c', FRESH_DRIVER, str(ROOT), op, str(inp), str(out)],
            capture_output=True, text=True, timeout=120)
        if run.returncode != 0:
            raise AssertionError(
                f'fresh-process {op} failed:\n{run.stdout}\n{run.stderr}')
        return pickle.loads(out.read_bytes())


def rational_maps(prepared):
    return [{(n, d): F(value) for n, d, value in row}
            for row in prepared['exact_physical_row_coefficients']]


@pytest.fixture(scope='module')
def prepared_pair():
    return fresh_result('prepare_pair')


def test_preparation_never_imports_product_or_opensees(prepared_pair):
    # The fresh driver installs a meta-path finder before prepare() and
    # asserts no product/native module is ever imported; it fails the whole
    # run otherwise. Here the returned preparation is verified as produced by
    # that guarded pure process.
    prepared = prepared_pair[0]
    assert prepared['model']['nx'] == 64


def test_all_channel_exact_density_nesting_including_endpoint_actions(prepared_pair):
    fine, coarse = map(rational_maps, prepared_pair)
    for support in range(2):
        for channel in range(5):
            left, right = 20*support+2*channel, 20*support+10+2*channel
            base = 10*support+2*channel
            keys = set(fine[left]) | set(fine[left+1]) | set(fine[right]) | set(fine[right+1]) | set(coarse[base]) | set(coarse[base+1])
            for key in keys:
                assert coarse[base].get(key, F(0)) == fine[left].get(key, F(0))+fine[right].get(key, F(0))
                assert coarse[base+1].get(key, F(0)) == (
                    -fine[left].get(key, F(0))+fine[left+1].get(key, F(0))+
                    fine[right].get(key, F(0))+fine[right+1].get(key, F(0)))/2
    # Original h16 cell means, not continuous affine endpoints or a new grid.
    for prepared, span in zip(prepared_pair, (4, 8)):
        for support in prepared['supports']:
            for cell, piece in enumerate(support['beam_pieces']):
                assert F(piece['xi_halfcell_mean_rational']) == F(2*cell+1-span, span)
        assert not any(d == 6 for row in rational_maps(prepared) for _, d in row)
    assert fine[2][100001, 2] == coarse[2][100001, 2] == F(1, 32)
    assert fine[3][100001, 2] == F(-3, 128)
    assert coarse[3][100001, 2] == F(-7, 256)


def test_required_inventory_load_sections_and_physical_ground(prepared_pair):
    for prepared, count in zip(prepared_pair, (4, 2)):
        model, expected = prepared['model'], prepared['expected_inventory']
        assert (model['nx'], model['ny'], model['beam_intervals'], model['support_count']) == (64, 16, 16, count)
        assert len(model['nodes']) == 1122 and len(model['shell_cells']) == 1024 and len(model['beam_cells']) == 16
        assert len(prepared['physical_rows']) == len(prepared['emitted_rows']) == 10*count
        assert len(expected['elements']) == 1040
        assert sum(sum(flags) for flags in expected['fixes'].values()) == 107
        assert expected['loads'] == {100017: [0., 0., 0., 0., 0., 10.]}
        assert model['fixes'][100001] == (1, 1, 1, 1, 1, 0)
        assert model['nodes'][100001] == (0., 0., .0675)
        assert model['nodes'][100017] == (1., 0., .0675)
        assert model['nodes'][1105] == (1., .2, 0.)
        assert prepared['sections'] == dict(E_PA=210e9, NU=.3, G_PA=80769230769.23077,
            SHELL_THICKNESS_M=.035, BEAM_AREA_M2=.01, BEAM_TORSION_M4=1.6666666666666667e-5,
            BEAM_BENDING_M4=8.333333333333334e-6, BEAM_SHEAR_AREA_M2=.008)
        assert [support['interval_m'] for support in prepared['supports']] == [[i/count, (i+1)/count] for i in range(count)]
        assert prepared['external_channels'] == ['translation_x', 'translation_y', 'translation_z', 'rotation_x', 'rotation_y']
        assert prepared['external_channel_major_from_support_major'] == [10*s+2*c+m for c in range(5) for s in range(count) for m in range(2)]
        assert 'native_channel_major_from_reference' not in prepared
        assert 'no direct native-row identification' in prepared['row_correspondence']
    for key in ('nodes', 'fixes', 'shell_cells', 'beam_cells', 'beam_element_tags'):
        assert prepared_pair[0]['model'][key] == prepared_pair[1]['model'][key]


def test_external_director_labels_require_rotation_permutation_and_gap_sign(prepared_pair):
    # Uniform shell ry, stationary beam: Rodrigues derivative of R_s ez is ex.
    # Native projected director components are (shellry-beamry, beamrx-shellrx),
    # whereas external physical channels are (beamrx-shellrx, beamry-shellry).
    for prepared in prepared_pair:
        rows = rational_maps(prepared)
        shell_nodes = set(prepared['model']['shell_node_ids'])
        width = F(1, prepared['model']['support_count'])
        external_rx = sum(value for (node, dof), value in rows[6].items()
                          if node in shell_nodes and dof == 5)
        external_ry = sum(value for (node, dof), value in rows[8].items()
                          if node in shell_nodes and dof == 5)
        native_director = (width, F(0))
        assert (external_rx, external_ry) == (F(0), -width)
        assert (external_rx, external_ry) != native_director
        assert native_director == (-external_ry, external_rx)
        # Uniform shell lateral translation separately exposes opposite gap sign.
        external_y = sum(value for (node, dof), value in rows[2].items()
                         if node in shell_nodes and dof == 2)
        assert external_y == -width
        assert external_y != width  # Native shell-minus-beam translation.


def test_exact_emitted_dependents_and_foreign_support_separation(prepared_pair):
    for prepared, stride in zip(prepared_pair, (4, 8)):
        dependent = []
        physical = rational_maps(prepared)
        emitted = [{(n, d): F(float(c)) for n, d, c in zip(row[::3], row[1::3], row[2::3])}
                   for row in prepared['emitted_rows']]
        for support in prepared['supports']:
            index = support['support_id']
            assert support['recovery_beam_nodes'] == [100001+stride*index+stride//4,
                                                     100001+stride*index+3*stride//4]
            selected = list(map(tuple, support['dependent_node_dofs']))
            assert len(set(selected)) == 10
            for i, row in enumerate(support['row_indices']):
                assert all(emitted[row].get(key, F(0)) == F(int(i == j)) for j, key in enumerate(selected))
            dependent.extend(selected)
        assert len(set(dependent)) == len(dependent)
        for support in prepared['supports']:
            local = set(map(tuple, support['dependent_node_dofs']))
            foreign = set(dependent)-local
            assert all(physical[row].get(key, F(0)) == 0 for row in support['row_indices'] for key in foreign)


@pytest.mark.parametrize('value', (True, False, 2., 4., 1, 3, 8, '4', None))
def test_fixed_tuple_admission_requires_plain_support_count(value):
    with pytest.raises(ValueError, match='plain integer'):
        reference.prepare(value)


@pytest.mark.parametrize('key,value', (('nx', 32), ('ny', 8), ('beam_intervals', 128),
                                      ('support_count', 8), ('nx', True)))
def test_model_tuple_cannot_expand_historical_admissions(prepared_pair, key, value):
    message, kind = fresh_result('aggregate_refusal',
                                 (prepared_pair[0]['model'], key, value, False))
    assert kind == 'ValueError'
    assert 'physical tuple' in message


@pytest.mark.parametrize('key,value', (
    ('nodes[100003]', (.125, .001, .0675)),
    ('fixes[100001]', (1, 1, 1, 1, 1, 1)),
    ('beam_tip', 100016),
    ('shell_cells', None),
    ('beam_element_tags', tuple(range(16))),
))
def test_actual_physical_inventory_mutations_are_rejected(prepared_pair, key, value):
    model = prepared_pair[0]['model']
    if key == 'nodes[100003]':
        mutation = ('nodes', {**model['nodes'], 100003: value})
    elif key == 'fixes[100001]':
        mutation = ('fixes', {**model['fixes'], 100001: value})
    elif key == 'shell_cells':
        mutation = (key, (*model['shell_cells'][:-1], (1, 2, 3, 4)))
    else:
        mutation = (key, value)
    message, kind = fresh_result('aggregate_refusal', (model, *mutation, True))
    assert kind == 'ValueError'
    assert 'physical' in message or 'inventory' in message


@pytest.mark.parametrize('key,value', (('E_PA', 200e9), ('WIDTH_M', .3),
                                      ('BEAM_AXIS_Z_M', .05), ('BEAM_BOTTOM_ARM_M', .05)))
def test_reference_section_or_offset_mismatch_is_not_admitted(key, value):
    message, kind = fresh_result('geometry_refusal', (key, value))
    assert kind == 'ValueError'
    assert 'physical sections' in message


def test_changed_source_is_refused_before_any_service_load():
    message, kind, loaded = fresh_result('changed_source_refusal')
    assert kind == 'ValueError'
    assert 'source identity' in message
    assert not loaded


def test_preparation_owns_its_nested_outputs(prepared_pair):
    original, prepared = fresh_result('prepare_owns')
    assert prepared['model']['nodes'][100001] == (9., 9., 9.)
    assert prepared['physical_rows'][0][2] == 17.
    assert prepared['supports'][0]['beam_shape_integrals_m'][0][100001] == 99.
    assert prepared['sections']['E_PA'] == 1.
    assert original['model']['nodes'][100001] == (0., 0., .0675)
    assert original['physical_rows'][0][2] != 17.
    assert original['supports'][0]['beam_shape_integrals_m'][0][100001] == 1/32
    assert reference.SECTIONS['E_PA'] == original['sections']['E_PA'] == 210e9


def test_non_mz_and_existing_output_refused_without_execution(tmp_path):
    with pytest.raises(ValueError, match='Mz10'):
        reference.main(tmp_path, tmp_path/'bad', 4, 'fy')
    assert not (tmp_path/'bad').exists()
    output = tmp_path/'existing'
    output.mkdir()
    (output/'marker').write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError):
        reference.main(tmp_path, output, 4)
    assert (output/'marker').read_text(encoding='utf-8') == 'preserve'


def test_finalization_keeps_primary_error_and_records_cleanup_failure(tmp_path):
    class BrokenFinalize:
        def finalize(self, *args, **kwargs):
            raise MemoryError('injected cleanup allocation failure')
    summary = {'status': 'failed', 'error': {'type': 'KeyboardInterrupt', 'message': 'primary'}}
    reference.finalize(None, None, BrokenFinalize(), tmp_path, summary)
    saved = json.loads((tmp_path/'summary.json').read_text(encoding='utf-8'))
    assert saved['error']['type'] == 'KeyboardInterrupt'
    assert 'injected cleanup allocation failure' in saved['finalization_error']
    with pytest.raises(MemoryError):
        reference.finalize(None, None, BrokenFinalize(), tmp_path, {'status': 'completed'})
