"""Model-owned generalized reference factor routing, no new mechanics."""
import numpy as np
import pytest

from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_native_generalized_paired_modal import solve_modes
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_native_generalized_modal import make


@pytest.mark.parametrize('curved', (False, True))
@pytest.mark.parametrize('supported', (False, True))
def test_reference_api_matches_original_factor_solver(curved, supported, tmp_path):
    model, _, inertias = make(curved, macros=2, clamped=supported)
    owner = analysis(model, inertias)
    before = owner.identity
    count = 3 if supported else 9
    packet, modes = owner.reference_modes(bounds=(-100., 1e6), num_modes=count)
    zero = np.zeros(owner.model.mesh.dof_manager.total_dofs)
    direct, expected = solve_modes(owner.model, owner._initial(), zero, owner._inertias, zero,
        bounds=(-100., 1e6), num_modes=count)
    assert canonical(packet) == canonical(direct)
    assert canonical(modes) == canonical(expected)
    if not supported:
        assert np.max(np.abs(modes.eigenvalues[:6])) < 1e-11
        assert np.all(modes.eigenvalues[6:] > 0.)
    else:
        assert np.all(modes.eigenvalues > 0.)
    assert owner.identity == before and not owner._lock.locked()
    assert not hasattr(packet, 'checkpoint_sha256')
    (tmp_path/'reference.json').write_bytes(canonical(dict(packet=packet, modes=modes,
        definition_sha256=owner.identity, production_qualified=False)))


@pytest.mark.parametrize('kind', ('cancel', 'invalid-bounds', 'invalid-count', 'dirty', 'concurrent'))
def test_reference_controls_fail_before_initialization(kind, monkeypatch):
    owner = analysis(make(False, clamped=True)[0])
    kwargs = dict(bounds=(-100., 1e6), num_modes=3)
    expected = ValueError
    if kind == 'cancel':
        token = CancellationToken(); token.cancel(); kwargs['cancellation_token'] = token; expected = SolveCancelled
    elif kind == 'invalid-bounds': kwargs['bounds'] = (1., 0.)
    elif kind == 'invalid-count': kwargs['num_modes'] = True
    elif kind == 'dirty': owner.model.mesh.nodes[1].x += .1
    else: owner._lock.acquire()
    def forbidden(*args, **kwargs): raise AssertionError('entered mechanics before preflight')
    monkeypatch.setattr(owner, '_initial', forbidden)
    try:
        with pytest.raises(expected): owner.reference_modes(**kwargs)
    finally:
        if owner._lock.locked(): owner._lock.release()
