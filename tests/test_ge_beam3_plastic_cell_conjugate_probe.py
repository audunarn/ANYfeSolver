"""Integrated nonlinear cell conjugacy and history checks, not qualification."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement
from anyfesolver._ge_beam3_p5.section import SectionHistory
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_plastic_cell_conjugate_probe import PlasticCellConjugate
from test_ge_beam3_complementary_section import section
from test_ge_beam3_curved_contrast_probe import make_curved


def build(contrast, origins=None):
    model, _ = make_curved(100.); old = model.mesh.elements[1]
    element = NativeP5BeamElement(1, old.node_ids, old.core.reference, section(contrast), line_force=np.zeros(3))
    return PlasticCellConjugate(element, origins)


@pytest.mark.parametrize('contrast', [1., 10000., 1000000000000.])
def test_shared_cell_conjugate_matches_partial_stationary_potential(contrast, tmp_path):
    probe = build(contrast); p = np.linspace(-.3, .5, 18)
    trace = []
    with (tmp_path/'cell-inputs.npz').open('xb') as stream:
        np.savez(stream, q=probe.q, k=probe.k, g=probe.g, c0=probe.c0,
            z0=probe.z0, radius=probe.radius, retained=p)
    try:
        response = probe.response(p, progress=trace.append)
    except Exception as exc:
        with (tmp_path/'failure.json').open('xb') as stream:
            stream.write(canonical(dict(contrast=contrast, error=type(exc).__name__, message=str(exc))))
        raise
    finally:
        with (tmp_path/'optimality-progress.json').open('xb') as stream: stream.write(canonical(trace))
    z = response.gradient[:6]; integrated = np.zeros(18); partial = 0.; history_error = 0.
    for cell in (0, 1):
        for index, t, _, w, v, _, _ in probe.source._stations[cell]:
            n = np.zeros((3, 12)); n[:, 6*cell:6*cell+3] = (1-t)*np.eye(3)
            n[:, 6*cell+3:6*cell+6] = t*np.eye(3)
            density = probe.source.section.mixed_response(v@z[3*cell:3*cell+3], n@p[6:],
                probe.origins[cell*probe.source.order+index])
            integrated[3*cell:3*cell+3] += w*v.T@density.gradient[:3]
            integrated[6:] -= w*n.T@density.gradient[3:]
            partial += w*density.potential
            expected = response.histories[cell*probe.source.order+index]
            history_error = max(history_error, abs(expected.plastic_coordinate-density.section.history.plastic_coordinate))
    record = dict(contrast=contrast, kkt=response.kkt_residual, iterations=response.iterations,
        force_error=float(np.linalg.norm(integrated[:6]-p[:6])),
        curvature_error=float(np.linalg.norm(integrated[6:]-response.gradient[6:])),
        fenchel_error=abs(float(p[:6]@z-partial-response.potential)),
        history_error=history_error, histories=response.histories, production_qualified=False)
    with (tmp_path/'cell.json').open('xb') as stream: stream.write(canonical(record))
    print(canonical(record).decode(), flush=True)
    assert max(record[k] for k in ('force_error', 'curvature_error', 'fenchel_error', 'history_error', 'kkt')) <= 1e-11


def test_nonlinear_cell_conjugate_derivatives_with_fixed_history():
    origins = tuple(SectionHistory(.01, .02) for _ in range(16))
    probe = build(1., origins); p = np.linspace(-.3, .5, 18)
    value = probe.response(p); direction = np.cos(np.arange(18))*.1; eps = 1e-5
    plus = probe.response(p+eps*direction); minus = probe.response(p-eps*direction)
    assert np.any(value.increments)
    np.testing.assert_allclose((plus.gradient-minus.gradient)/(2*eps), value.hessian@direction, rtol=1e-7, atol=1e-7)
    assert abs((plus.potential-minus.potential)/(2*eps)-value.gradient@direction) < 1e-7
    assert probe.origins == origins


@pytest.mark.parametrize('resultants', [[0.]*17, [float('nan')]*18, [float('inf')]*18])
def test_bad_cell_resultants_fail_closed(resultants):
    with pytest.raises(ValueError): build(1.).response(resultants)
