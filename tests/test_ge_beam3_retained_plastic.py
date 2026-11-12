"""Native directed-plastic cell/beam potential checks, not qualification."""
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_station_resultant_cell import StationResultantCell
from docs.reference_cases.ge_beam3_station_resultant_cell import StationResultantCell as ResearchCell
from anyfesolver._ge_beam3_retained_plastic import RetainedPlasticOperator
from anyfesolver._ge_beam3_retained_elastic import RetainedElasticOperator
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection
from anyfesolver._ge_beam3_p5.compensated_coordinates import split_sum
from test_ge_beam3_station_resultant_cell import data_for
from test_ge_beam3_complementary_section import section
from test_ge_beam3_curved_contrast_probe import make_curved


def element(contrast, *, elastic_only=False):
    model, _ = make_curved(100.); old = model.mesh.elements[1]
    law = section(contrast)
    if elastic_only: law = DirectedHardeningSection(law._elastic, law._direction, 1e12, law._hardening)
    return NativeP5BeamElement(1, old.node_ids, old.core.reference, law, line_force=np.zeros(3))


def args(probe):
    x = probe.reference.coordinates.copy(); x += np.sin(np.arange(9)).reshape(3, 3)*.01
    q = np.array([rotation(np.sin(np.arange(3)+i)*.12)@r for i, r in enumerate(probe.reference.nodal_triads)])
    u = np.array([rotation(np.cos(np.arange(3)+i)*.1) for i in range(2)])
    return x, np.zeros((3, 3)), q, u, np.linspace(-.3, .5, 18)


@pytest.mark.parametrize('contrast', [1., 10000., 1000000000000.])
@pytest.mark.parametrize('history', [False, True])
def test_native_cell_port_exactly_preserves_research(contrast, history, tmp_path):
    data = data_for(contrast, history); native = StationResultantCell(data); research = ResearchCell(data)
    p = np.linspace(-.3, .5, 18).tolist()
    first = native.response(p); second = research.response(p)
    assert canonical(first) == canonical(second)
    for scale in (2., 0., -1.):
        values = (scale*np.array(p)).tolist(); origins = first['history']
        first = native.response(values, origins=origins); second = research.response(values, origins=origins)
        assert canonical(first) == canonical(second)
    with pytest.raises(AttributeError): native.g = ()
    with pytest.raises(TypeError): native.c0[0][0] = 99
    (tmp_path/'native-port.json').write_bytes(canonical(first))


@pytest.mark.parametrize('contrast', [1., 1000000000000.])
def test_full_plastic_potential_first_second_variations(contrast, tmp_path):
    probe = RetainedPlasticOperator(element(contrast)); state = args(probe)
    origins = [[.01*(-1 if i % 2 else 1), 0., .02, 0.] for i in range(16)]
    saved = canonical(origins); d = .1*np.cos(np.arange(42)); centre = .01*np.sin(np.arange(42)); eps = 1e-5
    v = probe.evaluate(*state, origins=origins, increment=centre)
    plus = probe.evaluate(*state, origins=origins, increment=centre+eps*d)
    minus = probe.evaluate(*state, origins=origins, increment=centre-eps*d)
    h = v.hessian+v.hessian_low
    error = float(np.linalg.norm((plus.residual-minus.residual)/(2*eps)-h@d)/max(1., np.linalg.norm(h@d)))
    first = abs((plus.potential-minus.potential)/(2*eps)-v.residual@d)
    symmetry = float(np.linalg.norm(h-h.T)/max(1., np.linalg.norm(h)))
    assert error <= 1e-7 and first <= 1e-7 and symmetry <= 1e-11
    assert canonical(origins) == saved
    assert np.any(v.history[:, 2] > .02)
    for value in (v.residual, v.hessian, v.hessian_low, v.kinematics, v.history):
        with pytest.raises(ValueError): value.setflags(write=True)
    assert probe.evaluate(*state, origins=origins, increment=centre).material == v.material
    (tmp_path/'variations.json').write_bytes(canonical(dict(contrast=contrast, tangent=error, first=first,
        symmetry=symmetry, material=json.loads(v.material), production_qualified=False)))


@pytest.mark.parametrize('contrast', [1., 1000000000000.])
def test_plastic_objectivity_and_recovery(contrast, tmp_path):
    probe = RetainedPlasticOperator(element(contrast)); x, low, q, u, p = args(probe)
    origins = [[.01, 0., .02, 0.] for _ in range(16)]
    common = rotation([2.6, .8, -.3]); shift = [1000., -2000., 3000.]
    moved = np.zeros_like(x); moved_low = np.zeros_like(low)
    for i in range(3):
        for axis in range(3):
            moved[i, axis], moved_low[i, axis] = split_sum([shift[axis],
                *(float(common[axis,j]*x[i,j]) for j in range(3))])
    first = probe.evaluate(x, low, q, u, p, origins=origins)
    second = probe.evaluate(moved, moved_low, common@q, common@u, p, origins=origins)
    transform = np.eye(42); transform[:24, :24] = np.kron(np.eye(8), common)
    errors = dict(energy=abs(first.potential-second.potential),
        kinematics=float(np.linalg.norm(first.kinematics-second.kinematics)),
        residual=float(np.linalg.norm(transform@first.residual-second.residual)),
        hessian=float(np.linalg.norm(transform@first.hessian@transform.T-second.hessian)/max(1., np.linalg.norm(first.hessian))))
    assert max(errors.values()) <= 1e-11
    assert first.material == second.material
    before = probe.recover(u, p, origins=origins); after = probe.recover(common@u, p, origins=origins)
    for a, b in zip(before, after):
        for key in ('strain', 'strain_low', 'elastic_strain', 'elastic_strain_low', 'resultants', 'resultants_low', 'history'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(common@a['current_frame'], b['current_frame'], rtol=1e-11, atol=1e-11)
        assert 'fibre_stress' not in a
    (tmp_path/'objectivity.json').write_bytes(canonical(dict(contrast=contrast, errors=errors,
        recovery=before, production_qualified=False)))


@pytest.mark.parametrize('contrast', [1., 10000., 1000000000000.])
def test_virgin_elastic_limit_matches_preserved_operator(contrast):
    e = element(contrast, elastic_only=True); probe = RetainedPlasticOperator(e); old = RetainedElasticOperator(e)
    x, low, q, u, p = args(probe); p *= .01
    new = probe.evaluate(x, low, q, u, p); baseline = old.evaluate(x, low, q, u, p)
    assert np.all(new.history == 0)
    assert abs(new.potential-baseline.potential) <= 1e-11
    for key in ('residual', 'hessian', 'kinematics'):
        np.testing.assert_allclose(getattr(new, key), getattr(baseline, key), rtol=1e-11, atol=1e-11)


def test_high_contrast_small_mixed_resultants_can_yield(tmp_path):
    e = element(1e12); probe = RetainedPlasticOperator(e); state = list(args(probe)); state[-1] *= .01
    value = probe.evaluate(*state)
    assert np.count_nonzero(value.history[:, 2]) == 15
    with pytest.raises(ValueError, match='nonlinear section branch'): RetainedElasticOperator(e).evaluate(*state)
    (tmp_path/'genuine-plastic-state.json').write_bytes(value.material)


def test_invalid_origins_and_charts_cannot_produce_response():
    probe = RetainedPlasticOperator(element(1.)); state = args(probe)
    before = probe.evaluate(*state)
    with pytest.raises(ValueError): probe.evaluate(*state, origins=[[0., 0., 0., 0.]])
    delta = np.zeros(42); delta[3] = np.pi
    with pytest.raises(ValueError): probe.evaluate(*state, increment=delta)
    assert probe.evaluate(*state).material == before.material
    with pytest.raises(AttributeError): probe.cell = None
