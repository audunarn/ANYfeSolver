"""Independent flat P0 physical action and dual against actual BL5 rows."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell as api
from scripts.b3_e4_bl5_coherent_aggregation_dev import model as coherent_model
from scripts.b3_e4_bl5_riesz_controls_dev import beam_norm, shell_scalar
from scripts.r8_bl5_actual_mref_dev import (
    _complete_rectangular_partition, flat_p0_reference_bracket,
    flat_p0_reference_dual,
)
from scripts.r8_bl5_actual_primal_work_dev import actual_flat_primal_work
from test_r8_bl5_actual_primal_work_dev import _definition


def _owner(topology, basis="p0"):
    definition = _definition(topology)
    definition = replace(definition, lines=(replace(
        definition.lines[0], basis=basis),))
    return api.create_analysis(definition=definition)


def test_independent_all_row_density_action_and_refined_dual():
    matrices = []
    for topology in ("q4", "mixed", "s3"):
        owner = _owner(topology)
        try:
            levels = (1, 2, 4) if topology == "mixed" else (1,)
            error, records = flat_p0_reference_dual(
                owner, refinements=levels)
            assert error < 2e-12
            assert all(row["minimum_eigenvalue"] > 0. for row in records)
            assert all(row["inverse_residual"] < 1e-10 for row in records)
            matrices.append(records[0]["matrix"])
            if topology == "mixed":
                assert len(records) == 3
                assert np.linalg.eigvalsh(records[2]["matrix"]-
                                          records[0]["matrix"])[0] > -1e-10
        finally:
            owner.close()
    np.testing.assert_allclose(matrices[0], matrices[1], rtol=0., atol=2e-12)
    np.testing.assert_allclose(matrices[1], matrices[2], rtol=0., atol=2e-12)


def test_non_p0_line_cannot_use_this_reference_action():
    owner = _owner("mixed", basis="discontinuous_p1")
    try:
        with pytest.raises(ValueError, match="P0 density"):
            flat_p0_reference_dual(owner, refinements=(1,))
    finally:
        owner.close()


def test_extra_trace_restraint_cannot_reuse_root_only_reference_dual():
    definition = _definition("mixed")
    definition = replace(definition, lines=(replace(
        definition.lines[0], basis="p0"),),
        fixed=definition.fixed+(api.FixedCoordinate("shell", 7, 0),))
    owner = api.create_analysis(definition=definition)
    try:
        with pytest.raises(ValueError, match="root-only physical trace"):
            flat_p0_reference_dual(owner, refinements=(1,))
    finally:
        owner.close()


def test_independent_continuous_dual_brackets_refined_discrete_action(
        record_property):
    owner = _owner("mixed")
    try:
        _, records = flat_p0_reference_dual(owner, refinements=(1, 2, 4))
        lower, upper, info = flat_p0_reference_bracket(
            owner, modes=1024, subdivisions=32)
        record_property("flat_p0_dual_bound", json.dumps(dict(
            relative_loewner_bound=info["relative_loewner_bound"],
            shell_tail=info["shell"]["relative_loewner_tail_bound"],
            beam_error=info["beam"]["relative_loewner_error_bound"],
            minimum_lower_eigenvalue=float(np.linalg.eigvalsh(lower)[0]),
            condition_lower=float(np.linalg.cond(lower)),
            refined_upper_margin=float(np.linalg.eigvalsh(
                upper-records[-1]["matrix"])[0])), sort_keys=True))
        assert info["relative_loewner_bound"] <= 1e-3
        assert np.linalg.eigvalsh(lower)[0] > 0.
        assert np.linalg.eigvalsh(upper-records[-1]["matrix"])[0] > -1e-10
    finally:
        owner.close()


def test_registered_q4_family_entry_has_actual_complete_row_work(record_property):
    definition, policy = coherent_model(16, 4, 16, 2)
    assert policy["span_halfcells"] == 2
    assert policy["shell_trace_segments"]/(policy["beam_halfcells"]/
            policy["span_halfcells"]) == 2.
    owner = api.create_analysis(definition=definition)
    try:
        error, records = flat_p0_reference_dual(owner, refinements=(1,))
        h, _, _ = actual_flat_primal_work(owner, .4, .1)
        _, _, scalar, scalar_active = shell_scalar(1., .2, .4, 16, 4)
        supported = [(node, field) for node, position in
                     enumerate(owner._shell_xyz) if position[0] != 0.
                     for field in range(5)]
        lookup = {key: row for row, key in enumerate(supported)}
        permutation = np.zeros((len(supported), 5*len(scalar_active)))
        for field in range(5):
            for column, node in enumerate(scalar_active):
                permutation[lookup[(int(node), field)],
                            field*len(scalar_active)+column] = 1.
        expected = sparse.block_diag((scalar, scalar, scalar,
                                      .4**2*scalar, .4**2*scalar)).toarray()
        actual_shell = h[:len(supported), :len(supported)].toarray()
        np.testing.assert_allclose(permutation.T@actual_shell@permutation,
                                   expected, rtol=2e-13, atol=2e-12)
        _, expected_beam, _, _ = beam_norm(
            1., .4, 16, 1, np.array((0., 0., -.05)), .1)
        np.testing.assert_allclose(
            h[len(supported):, len(supported):].toarray(),
            expected_beam.toarray(), rtol=0., atol=2e-12)
        assert h[:len(supported), len(supported):].nnz == 0
        assert owner._C.shape[0] == 40
        assert error < 2e-12
        assert records[0]["minimum_eigenvalue"] > 0.
        assert records[0]["inverse_residual"] < 1e-10
        lower, upper, info = flat_p0_reference_bracket(
            owner, modes=1024, subdivisions=32)
        ordered = .5*(upper-records[0]["matrix"]
                        +upper.T-records[0]["matrix"].T)
        minimum_margin = float(np.linalg.eigvalsh(ordered)[0])
        condition = float(np.linalg.cond(lower))
        record_property("registered_q4_reference_dual", json.dumps(dict(
            relative_loewner_bound=info["relative_loewner_bound"],
            shell_tail=info["shell"]["relative_loewner_tail_bound"],
            beam_error=info["beam"]["relative_loewner_error_bound"],
            minimum_lower_eigenvalue=float(np.linalg.eigvalsh(lower)[0]),
            condition_lower=condition,
            minimum_discrete_upper_margin=minimum_margin), sort_keys=True))
        assert info["relative_loewner_bound"] <= 1e-3
        assert minimum_margin >= -1e-10*np.linalg.norm(upper)
        assert condition <= 1e4
    finally:
        owner.close()


@pytest.mark.parametrize("defect", ("missing_interior_q4", "overlapping_s3"))
def test_reference_dual_rejects_incomplete_or_overlapping_partition(defect):
    definition, _ = coherent_model(16, 4, 16, 2)
    shell = definition.shell
    if defect == "missing_interior_q4":
        # Removing one interior cell leaves every grid node present.
        altered = replace(shell, q4_elements=shell.q4_elements[:17]+
                          shell.q4_elements[18:])
    else:
        a, b, c, d = shell.q4_elements[17]
        altered = replace(shell,
                          q4_elements=shell.q4_elements[:17]+
                          shell.q4_elements[18:],
                          s3_elements=((a, b, c), (a, b, d)))
    proxy = SimpleNamespace(
        _definition=replace(definition, shell=altered),
        _shell_ids=shell.node_ids,
        _shell_xyz=np.asarray(shell.coordinates_m))
    with pytest.raises(ValueError, match="complete nonoverlapping"):
        _complete_rectangular_partition(
            proxy, np.linspace(0., 1., 17), np.linspace(0., .2, 5))
