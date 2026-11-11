"""Bounded actual-owner bridge for the flat F03 auxiliary norm; no quotient."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as api
from scripts.r8_bl5_actual_primal_work_dev import actual_flat_primal_work
from test_stiffened_shell_bl5_affine_supports import candidate
from test_stiffened_shell_l1_assembly_experimental import _bl5_joint_mesh_definition


def _definition(topology):
    base = candidate(4, 4)
    if topology == "q4":
        return base
    quads = base.shell.q4_elements
    if topology == "mixed":
        kept = quads[:2]
        split = quads[2:]
    else:
        kept = ()
        split = quads
    triangles = tuple(triangle for a, b, c, d in split
                      for triangle in ((a, b, c), (a, c, d)))
    return replace(base, shell=replace(
        base.shell, q4_elements=kept, s3_elements=triangles))


@pytest.mark.parametrize("topology", ("q4", "mixed", "s3"))
def test_actual_supported_primal_preserves_all_bl5_virtual_work(topology):
    owner = api.create_analysis(definition=_definition(topology))
    try:
        h, embedding, work = actual_flat_primal_work(owner, .4, .1)
        assert embedding.shape == (owner._K.shape[0], h.shape[0])
        assert work.shape == (owner._C.shape[0], h.shape[0])
        assert np.linalg.eigvalsh(h.toarray())[0] > 0.
        assert work.nnz > 0
        assert embedding[owner._fixed].nnz == 0
        rng = np.random.default_rng(947)
        auxiliary = rng.normal(size=h.shape[0])
        multiplier = rng.normal(size=owner._C.shape[0])
        full = embedding@auxiliary
        np.testing.assert_allclose(work@auxiliary, owner._C@full,
                                   rtol=0., atol=3e-14)
        np.testing.assert_allclose(multiplier@(owner._C@full),
                                   (embedding.T@(owner._C.T@multiplier))@auxiliary,
                                   rtol=0., atol=3e-13)
        assert float(auxiliary@(h@auxiliary)) > 0.
        # Root axis translation is supported, whereas all native cell
        # directors are retained as independent auxiliary coordinates.
        root = owner._shell_count+6*owner._beam_index[101]
        np.testing.assert_array_equal(embedding[root:root+3].toarray(), 0.)
        for member in range(len(owner._beam_parts)):
            for cell in (0, 1):
                row = (owner._shell_count+6*len(owner._beam_ids)+
                       24*member+3*cell)
                assert embedding[row:row+3].nnz == 3
    finally:
        owner.close()


def test_actual_primal_refuses_unmapped_root_support_and_invalid_width():
    owner = api.create_analysis(definition=_bl5_joint_mesh_definition(4, 4, 1))
    try:
        with pytest.raises(ValueError, match="omitted native"):
            actual_flat_primal_work(owner, .4, .1)
        with pytest.raises(ValueError, match="positive auxiliary"):
            actual_flat_primal_work(owner, .4, -.1)
    finally:
        owner.close()


def test_shell_input_order_does_not_change_compiled_primal_work():
    definition = _definition("mixed")
    shell = definition.shell
    reordered = replace(definition, shell=replace(
        shell, node_ids=tuple(reversed(shell.node_ids)),
        coordinates_m=tuple(reversed(shell.coordinates_m))))
    first = api.create_analysis(definition=definition)
    second = api.create_analysis(definition=reordered)
    try:
        h1, e1, b1 = actual_flat_primal_work(first, .4, .1)
        h2, e2, b2 = actual_flat_primal_work(second, .4, .1)
        def keys(owner, embedding):
            result = []
            for col in range(embedding.shape[1]):
                rows = embedding.indices[
                    embedding.indptr[col]:embedding.indptr[col+1]]
                assert len(rows) == 1
                row = int(rows[0])
                result.append(("shell", owner._shell_ids[row//6], row%6)
                              if row < owner._shell_count else
                              ("beam", row-owner._shell_count))
            return result
        old_keys, new_keys = keys(first, e1), keys(second, e2)
        assert set(old_keys) == set(new_keys)
        old_index = {key: i for i, key in enumerate(old_keys)}
        permutation = np.zeros((len(new_keys), len(old_keys)))
        for row, key in enumerate(new_keys):
            permutation[row, old_index[key]] = 1.
        np.testing.assert_allclose(h1.toarray(),
                                   permutation.T@h2.toarray()@permutation,
                                   rtol=0., atol=2e-14)
        np.testing.assert_allclose(b1.toarray(), b2.toarray()@permutation,
                                   rtol=0., atol=2e-14)
    finally:
        first.close()
        second.close()
