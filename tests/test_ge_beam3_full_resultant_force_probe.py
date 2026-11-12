"""Small actual force solves of the elastic retained-resultant experiment."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_full_resultant_force_probe import solve_two_macro_probe
from test_ge_beam3_curved_contrast_probe import make_curved
from anyfesolver._ge_beam3_p5.algebra import rotation


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_full_resultant_two_macro_force_equilibrium(slenderness, tmp_path):
    model, _ = make_curved(slenderness)
    nodes = np.array([model.mesh.nodes[i].coords() for i in range(1, 6)])
    with (tmp_path/'progress.jsonl').open('xb') as stream:
        def checkpoint(row): stream.write(canonical(row)); stream.flush()
        try:
            state, records = solve_two_macro_probe(model, [.05, -.001, 0.], checkpoint)
        except Exception as exc:
            with (tmp_path/'failure.json').open('xb') as failed:
                failed.write(canonical(dict(slenderness=slenderness, error=type(exc).__name__,
                    message=str(exc), production_qualified=False)))
            raise
    with (tmp_path/'state.npz').open('xb') as stream:
        np.savez(stream, positions=state[0], low=state[1], nodal_frames=state[2],
                 cell_rotations=state[3], resultants=state[4])
    record = dict(slenderness=slenderness, records=records, tip_displacement=state[0][-1]-nodes[-1],
        production_qualified=False, native_state_committed=False, diagnostic_only=True)
    with (tmp_path/'equilibrium.json').open('xb') as stream: stream.write(canonical(record))
    print(canonical(record).decode(), flush=True)
    assert len(records) == 2
    assert max(max(r['equilibrium'], r['compatibility']) for r in records) <= 1e-11
    np.testing.assert_array_equal(nodes, [model.mesh.nodes[i].coords() for i in range(1, 6)])


@pytest.mark.parametrize('slenderness', [100., 1000000.])
def test_retained_resultant_covariance(slenderness, tmp_path):
    q = rotation([.4, -.3, .2]); f = np.array([.05, -.001, 0.])
    model, _ = make_curved(slenderness)
    rotated, _ = make_curved(slenderness, q)
    first, records = solve_two_macro_probe(model, f)
    second, other_records = solve_two_macro_probe(rotated, q@f)
    np.testing.assert_allclose(second[0]+second[1], (first[0]+first[1])@q.T, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(second[2], q@first[2], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(second[3], q@first[3]@q.T, rtol=1e-11, atol=1e-11)
    # Cell force is conjugate to the global-reference common chord strain;
    # endpoint moment components use the physical material triad.
    transformed = first[4].copy()
    transformed[:, :6] = (first[4][:, :6].reshape(2, 2, 3)@q.T).reshape(2, 6)
    np.testing.assert_allclose(second[4], transformed, rtol=1e-11, atol=1e-11)
    with (tmp_path/'covariance.json').open('xb') as stream:
        stream.write(canonical(dict(slenderness=slenderness, baseline=records,
            rotated=other_records, first_state=first, rotated_state=second,
            production_qualified=False)))


def test_probe_rejects_missing_root_and_distributed_work():
    model, _ = make_curved(100.)
    model.boundary_conditions.clear()
    with pytest.raises(ValueError, match='root-only'):
        solve_two_macro_probe(model, [.05, -.001, 0.])
    model, _ = make_curved(100.)
    e = model.mesh.elements[1]
    model.mesh.elements[1] = type(e)(1, e.node_ids, e.core.reference, e.core.section,
        line_force=np.array([0., 1., 0.]))
    with pytest.raises(ValueError, match='distributed load'):
        solve_two_macro_probe(model, [.05, -.001, 0.])
