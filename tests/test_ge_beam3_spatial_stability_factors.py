"""New-owner custody, actual full Schur, and bounded sparse congruence."""
from decimal import Decimal as D, localcontext
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver import _ge_beam3_elastic_seed_modal as modal
from anyfesolver import _ge_beam3_elastic_seed_continuation as owner
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_elastic_seed_continuation import seed, model, programme
from docs.reference_cases.ge_beam3_sparse_inertia_audit import inertia
from docs.reference_cases.ge_beam3_decimal_inertia_audit import inertia as dense


def capture(seed):
    made = model(); p = programme((.0002,))
    c = owner.Context(made, p, seed, expected_seed_sha256=sha256(seed).hexdigest())
    raw = c.checkpoint(())
    masses = {i: np.diag([1., 1., 1., 3e-5, 1e-5, 2e-5]) for i in made.mesh.elements}
    packet, check = modal.prepare(made, p, seed, raw, masses,
        expected_seed_sha256=sha256(seed).hexdigest(), expected_sha256=sha256(raw).hexdigest())
    return made, p, c, raw, masses, packet, check


def test_actual_seed_schur_and_control_free(seed):
    _, _, c, raw, _, packet, check = capture(seed)
    state = c.initial; p = c.physical
    _, j, metrics, _, _ = c.assemble(state.mechanical, state.parameter, state.histories, c.seed_target)
    coordinates = list(range(p.nodal_count))
    resultants = []
    for i in range(len(p.elements)):
        coordinates.extend(range(p.nodal_count+24*i, p.nodal_count+24*i+6))
        resultants.extend(range(p.nodal_count+24*i+6, p.nodal_count+24*(i+1)))
    expected = j[np.ix_(coordinates, coordinates)] - j[np.ix_(coordinates, resultants)] @ np.linalg.solve(
        j[np.ix_(resultants, resultants)], j[np.ix_(resultants, coordinates)])
    assert np.linalg.norm(expected-packet.stiffness)/max(1., np.linalg.norm(expected)) < 1e-11
    assert 12 in packet.free_dofs and packet.displacement_target == .0001
    assert not packet.control_constraint_in_physical_stiffness
    assert not packet.physical_loading_path_from_rest and not packet.production_qualified
    assert c.checkpoint(()) == raw and max(metrics) <= 1e-11
    from docs.reference_cases.ge_beam3_spatial_stability_worker import serialize_packet
    from docs.reference_cases.ge_beam3_retained_prestress_protocol import canonical as plain_json
    encoded = serialize_packet(packet)
    assert plain_json(encoded) == canonical(packet)
    assert encoded['seed_sha256'] == sha256(seed).hexdigest()
    assert len(encoded['geometric']) == len(packet.geometric)
    check()


@pytest.mark.parametrize('mutation', ('seed_hash', 'checkpoint_hash', 'foreign_schema', 'mass', 'packet', 'program'))
def test_capture_mutations(seed, mutation):
    made, p, c, raw, masses, packet, check = capture(seed)
    seed_sha = sha256(seed).hexdigest(); raw_sha = sha256(raw).hexdigest()
    if mutation == 'mass':
        masses[1][0, 0] *= 2
    elif mutation == 'packet':
        object.__setattr__(packet, 'control_constraint_in_physical_stiffness', True)
    elif mutation == 'program':
        object.__setattr__(p, 'direction', (0., 1., 0.))
    else:
        if mutation == 'seed_hash': seed_sha = '0'*64
        elif mutation == 'checkpoint_hash': raw_sha = '0'*64
        else:
            raw = raw.replace(owner.SCHEMA.encode(), b'FOREIGN_SCHEMA')
            raw_sha = sha256(raw).hexdigest()
        with pytest.raises(ValueError):
            modal.prepare(made, p, seed, raw, masses, expected_seed_sha256=seed_sha, expected_sha256=raw_sha)
        return
    with pytest.raises(ValueError): check()


@pytest.mark.parametrize('digits', (80, 100))
@pytest.mark.parametrize('n', (8, 30, 438))
def test_known_signed_chain_congruence(digits, n):
    with localcontext() as context:
        context.prec = digits
        signs = [D(-2) if i % 7 == 0 else D(3) for i in range(n)]
        a = [[D(0)]*n for _ in range(n)]
        for i in range(n):
            a[i][i] = signs[i] + (signs[i-1]/16 if i else D(0))
            if i: a[i][i-1] = a[i-1][i] = signs[i-1]/4
        result = inertia(a)
        assert result['negative'] == sum(x < 0 for x in signs)
        assert result['max_front'] <= 2 and D(result['reconstruction_relative']) <= D('1e-60')
        if n <= 30: assert dense(a)['negative'] == result['negative']


def test_two_by_two_pivot_and_exact_nonzero_coupling():
    with localcontext() as context:
        context.prec = 80
        a = [[D(0), D(2), D(0)], [D(2), D(0), D('1e-30')], [D(0), D('1e-30'), D(3)]]
        result = inertia(a)
        assert result['negative'] == dense(a)['negative'] == 1
        assert 2 in result['pivot_sizes']


@pytest.mark.parametrize('kind', ('asymmetric', 'nonfinite', 'singular', 'fill', 'dimension', 'cancel'))
def test_sparse_fail_closed(kind):
    with localcontext() as context:
        context.prec = 80
        a = [[D(2), D(1)], [D(1), D(2)]]
        if kind == 'asymmetric': a[1][0] = D(0)
        elif kind == 'nonfinite': a[0][0] = D('NaN')
        elif kind == 'singular': a = [[D(0)]]
        elif kind == 'fill': a = [[D(2) if i == j else D(1) for j in range(100)] for i in range(100)]
        elif kind == 'dimension': a = [[D(1)]] * 513
        def check():
            if kind == 'cancel': raise ValueError('cancelled audit')
        with pytest.raises(ValueError): inertia(a, check)
