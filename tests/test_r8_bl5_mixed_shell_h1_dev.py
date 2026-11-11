"""Independent flat Q4/S3 shell-side F03 auxiliary norm controls."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.b3_e4_bl5_riesz_controls_dev import shell_scalar
from scripts.r8_bl5_mixed_shell_h1_dev import (
    flat_shell_embedding, scalar_mixed_h1, supported_shell_h1,
)


def _mesh():
    nodes = np.array(((0., 0.), (.5, 0.), (1., 0.),
                      (0., .2), (.5, .2), (1., .2)))
    mixed = ((0, 1, 4, 3), (1, 2, 5), (1, 5, 4))
    quads = ((0, 1, 4, 3), (1, 2, 5, 4))
    return nodes, mixed, quads


def test_q4_restriction_matches_existing_independent_rectangular_norm():
    nodes, _, quads = _mesh()
    ell = .4
    root = tuple((node, component) for node in (0, 3)
                 for component in range(6))
    h5, active, embedding = supported_shell_h1(nodes, quads, ell, root)
    _, _, independent, scalar_active = shell_scalar(1., .2, ell, 2, 1)
    assert active.tolist() == [5*node+field for node in scalar_active for field in range(5)]
    translation_x = np.arange(0, len(active), 5)
    director_x = translation_x+3
    np.testing.assert_allclose(h5[translation_x][:, translation_x].toarray(),
                               independent.toarray(), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(h5[director_x][:, director_x].toarray(),
                               ell**2*independent.toarray(), rtol=2e-14, atol=2e-14)
    assert embedding.shape == (6*len(nodes), len(active))
    assert embedding.nnz == len(active)
    # The y=width shell nodes are present; the root drilling supports add no
    # auxiliary coordinate or spurious stiffness.
    assert {5*node for node in (4, 5)}.issubset(set(active))


def test_mixed_s3_q4_affine_patch_physical_director_and_equivalent_units():
    nodes, mixed, quads = _mesh()
    ell = .4
    field = np.array((
        (.2, -.1, .3, .04, -.03),
        (.3, .2, -.4, -.02, .06),
        (-.2, .1, .5, .03, .01),
    ))  # value = first row + x*second row + y*third row
    q = field[0]+nodes[:, 0, None]*field[1]+nodes[:, 1, None]*field[2]
    mean = field[0]+.5*field[1]+.1*field[2]
    covariance = (field[1]**2)/12+(.2**2*field[2]**2)/12
    mean_square = mean**2+covariance
    gradient_square = field[1]**2+field[2]**2
    expected = float(np.sum(mean_square[:3]+ell**2*gradient_square[:3])+
                     np.sum(ell**2*mean_square[3:]+ell**4*gradient_square[3:]))
    for elements in (mixed, quads):
        h, active, embedding = supported_shell_h1(nodes, elements, ell, ())
        assert active.tolist() == list(range(5*len(nodes)))
        energy = float(q.reshape(-1)@(h@q.reshape(-1)))
        np.testing.assert_allclose(energy, expected, rtol=1e-13, atol=1e-14)
        assert np.linalg.eigvalsh(h.toarray())[0] > 0.
        actual = (embedding@q.reshape(-1)).reshape(-1, 6)
        np.testing.assert_allclose(actual[:, 3], -q[:, 4])
        np.testing.assert_allclose(actual[:, 4], q[:, 3])
        np.testing.assert_array_equal(actual[:, 5], 0.)
        # omega cross e_z = eta; no artificial normal spin is introduced.
        np.testing.assert_allclose(np.cross(actual[:, 3:], (0., 0., 1.))[:, :2], q[:, 3:])

        scale = 1000.
        h_mm, _, _ = supported_shell_h1(nodes*scale, elements, ell*scale, ())
        q_mm = q.copy()
        q_mm[:, :3] *= scale
        energy_mm = float(q_mm.reshape(-1)@(h_mm@q_mm.reshape(-1)))
        np.testing.assert_allclose(energy_mm, scale**2*energy, rtol=1e-13)


def test_support_map_and_owned_matrix_reject_bad_reference_geometry():
    nodes, mixed, _ = _mesh()
    h, active, embedding = supported_shell_h1(
        nodes, mixed, .4, ((0, 3), (0, 4), (0, 5), (3, 3), (3, 4)))
    assert 5*0+3 not in active and 5*0+4 not in active
    assert 5*3+3 not in active and 5*3+4 not in active
    assert len(active) == 26
    before = h.toarray().copy()
    nodes[2, 0] += .01
    np.testing.assert_array_equal(h.toarray(), before)
    assert embedding.shape == (36, 26)
    with pytest.raises(ValueError, match="oriented"):
        scalar_mixed_h1(nodes, ((1, 5, 2),), .4)
    with pytest.raises(ValueError, match="valid S3/Q4"):
        scalar_mixed_h1(nodes, ((1, 1, 2),), .4)
    with pytest.raises(ValueError, match="positive frozen"):
        scalar_mixed_h1(nodes, mixed, 0.)
    with pytest.raises(ValueError, match="duplicate"):
        scalar_mixed_h1(nodes, (*mixed, mixed[0]), .4)
    with pytest.raises(ValueError, match="every declared node"):
        scalar_mixed_h1(nodes, (mixed[0],), .4)


def test_flat_embedding_has_only_five_physical_fields_per_node():
    embedding = flat_shell_embedding(1).toarray()
    np.testing.assert_array_equal(embedding[3], (0., 0., 0., 0., -1.))
    np.testing.assert_array_equal(embedding[4], (0., 0., 0., 1., 0.))
    np.testing.assert_array_equal(embedding[5], 0.)


def test_actual_mixed_bl5_owner_keeps_supported_offline_shell_fields():
    """The physical trace has zero off-line columns, but its H1 norm does not."""
    from anyfesolver import stiffened_shell as api
    from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition

    base = _bl5_joint_mesh_definition(4, 4, 1)
    quads = base.shell.q4_elements
    triangles = tuple(tri for a, b, c, d in quads[2:]
                      for tri in ((a, b, c), (a, c, d)))
    definition = replace(base, shell=replace(
        base.shell, q4_elements=quads[:2], s3_elements=triangles))
    owner = api.create_analysis(definition=definition)
    try:
        shell = definition.shell
        index = {node: i for i, node in enumerate(shell.node_ids)}
        xy = np.asarray(shell.coordinates_m, dtype=float)[:, :2]
        elements = tuple(tuple(index[node] for node in element)
                         for element in (*shell.q4_elements, *shell.s3_elements))
        fixed = tuple((index[support.node], support.component)
                      for support in definition.fixed if support.domain == "shell")
        h, active, embedding = supported_shell_h1(xy, elements, .4, fixed)
        assert embedding.shape == (owner._shell_count, len(active))
        shell_work = (owner._C[:, :owner._shell_count]@embedding).tocsr()
        assert shell_work.nnz > 0
        offline = np.array([i for i, index5 in enumerate(active)
                            if xy[index5//5, 1] > .1], dtype=int)
        online = np.array([i for i, index5 in enumerate(active)
                           if xy[index5//5, 1] < .1], dtype=int)
        assert len(offline) > 0 and len(online) > 0
        assert shell_work[:, offline].nnz == 0
        assert shell_work[:, online].nnz > 0
        # Removing all off-line shell fields as if their zero work columns
        # permitted it would replace the actual trace norm by a harder one.
        coupling = h[online][:, offline]
        assert coupling.nnz > 0
        from scipy.sparse.linalg import spsolve
        change = coupling@spsolve(h[offline][:, offline], coupling.T.toarray())
        assert np.linalg.norm(change) > 0.
        rng = np.random.default_rng(606)
        vector = rng.normal(size=len(active))
        np.testing.assert_allclose(shell_work@vector,
                                   owner._C[:, :owner._shell_count]@(embedding@vector),
                                   rtol=0., atol=2e-14)
    finally:
        owner.close()
