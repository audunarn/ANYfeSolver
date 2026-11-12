"""Accepted-control modes use physical inertia and the original factor pencil."""
from hashlib import sha256
import numpy as np
import pytest

from anyfesolver import _ge_beam3_analysis_translation_modal as route
from anyfesolver import _ge_beam3_retained_translation_modal as ordinary_modal
from anyfesolver import _ge_beam3_elastic_seed_modal as seeded_modal
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_elastic_seed_continuation import seed, model, programme
from test_ge_beam3_analysis_fibre_translation import analysis as fibre_analysis, program as fibre_program


def digest(raw):
    return sha256(raw).hexdigest()


@pytest.mark.parametrize('seeded', (False,True))
def test_actual_modes_equal_original_native_capture_and_solver(seed, seeded, tmp_path):
    made = analysis(model()); p = programme((.0002,))
    kw = dict(seed=seed, expected_seed_sha256=digest(seed)) if seeded else {}
    state = made.solve_translation(p, **kw)
    assert state.status == 'completed', state.backend_result.failure
    raw = state.backend_result.checkpoint
    result = made.translation_modes(p, state.checkpoint, expected_sha256=state.checkpoint_sha256,
        bounds=(-100.,1e6), num_modes=6, **kw)
    if seeded:
        packet, guard = seeded_modal.prepare(made.model, p, seed, raw, made._inertias,
            expected_sha256=digest(raw), expected_seed_sha256=digest(seed))
    else:
        packet, guard = ordinary_modal.prepare(made.model, p, raw, made._inertias, expected_sha256=digest(raw))
    direct = route.solve_paired_factor_chain_modes(packet.left, packet.right, packet.geometric, packet.kinetic,
        packet.free_dofs, packet.algebraic_dofs, bounds=(-100.,1e6), num_modes=6)
    guard()
    assert canonical(result.packet) == canonical(packet)
    assert canonical(result.modes) == canonical(direct)
    assert result.definition_graph_sha256 == made.identity
    assert result.checkpoint_sha256 == state.checkpoint_sha256
    assert 12 in packet.free_dofs and not packet.control_constraint_in_physical_stiffness
    assert not np.any(packet.mass[:,list(packet.algebraic_dofs)])
    assert not result.production_qualified
    # The accepted history remains byte-identical after analysis.
    replay = made.solve_translation(p, checkpoint=state.checkpoint, expected_sha256=state.checkpoint_sha256, **kw)
    assert replay.checkpoint == state.checkpoint
    (tmp_path/'modes.json').write_bytes(canonical(result))


@pytest.mark.parametrize('kwargs', (dict(bounds=(False,100.)), dict(bounds=(1.,0.)), dict(num_modes=True),
    dict(root_width=0.), dict(relative_width=1.), dict(bounds=(float('nan'),100.))))
def test_controls_rejected_before_modal_capture(kwargs, monkeypatch):
    made = analysis(model()); options = dict(bounds=(-100.,1e6)); options.update(kwargs)
    monkeypatch.setattr(ordinary_modal, 'prepare', lambda *a,**kw: pytest.fail('capture on invalid controls'))
    with pytest.raises(ValueError):
        made.translation_modes(programme(), b'invalid', expected_sha256='0'*64, **options)


def test_foreign_envelope_rejected_before_modal_capture(monkeypatch):
    made = analysis(model())
    raw = made._envelope(b'force-owner')
    monkeypatch.setattr(ordinary_modal, 'prepare', lambda *a,**kw: pytest.fail('capture on foreign state'))
    with pytest.raises(ValueError):
        made.translation_modes(programme(), raw, expected_sha256=digest(raw), bounds=(-100.,1e6))


def test_physical_fibre_modes_never_use_generalized_material_owner():
    with pytest.raises(ValueError, match='external translation checkpoint hash'):
        fibre_analysis().translation_modes(fibre_program(), b'invalid', expected_sha256='0'*64, bounds=(-1.,100.))


def test_cancellation_and_size_checks_precede_capture(monkeypatch):
    made = analysis(model()); p = programme((.0002,)); state = made.solve_translation(p)
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        made.translation_modes(p, state.checkpoint, expected_sha256=state.checkpoint_sha256,
            bounds=(-100.,1e6), cancellation_token=token)
    # A declared model size outside the preserved solver domain must not
    # trigger expensive reconstruction or select a different arithmetic path.
    class TooLarge:
        _elements = tuple(range(32))
        model = type('Model', (), {'mesh': type('Mesh', (), {'dof_manager': type('Dofs', (), {'total_dofs':390})()})()})()
    with pytest.raises(ValueError, match='coordinate bound'):
        route._controls(TooLarge(), (-100.,1e6), 6, 1e-10, 1e-12)
