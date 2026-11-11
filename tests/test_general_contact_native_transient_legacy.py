"""Prospective changed-shared-service compatibility checks; UNEXECUTED."""
import json
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver.ge_beam3_native_connected_dynamics import create_analysis, SELECTOR
from examples.native_connected_dynamic_dev import build
from examples.general_contact_native_pair_impact_dev import definitions


def test_legacy_connected_graph_and_definition_capsule_identity():
    with pytest.raises(ValueError,match='graph is disconnected'):
        create_analysis(SELECTOR,definitions(),(),inertia_order=4)
    owner = build()
    # Exact pre-extraction definition schema: new contact metadata must never
    # leak into legacy identity even though the mechanics service is shared.
    names = ('analysis','elements','maps','nodes','node_index','n','e','size','fixed','free',
        'p','z','physical','algebraic','reference_positions','reference_frames','histories',
        'inertia','inertia_order','iterations','backtracks','coordinate_budget',
        'runtime_limit_seconds','length','scale')
    actual = owner._definition()
    expected = dict(profile='GE_BEAM3_CONNECTED_CENTERED_ELASTIC_MIDPOINT_DEV_V1',
                    **{name:actual[name] for name in names})
    assert set(actual) == set(expected) and owner.identity == sha(expected)
    owner.initialize(); body = json.loads(owner.checkpoint())['body']
    assert body['profile'] == expected['profile'] and body['identity'] == sha(expected)
    assert body['schema'] == 'GE_BEAM3_CONNECTED_NATIVE_DYNAMIC_CHECKPOINT_DEV_V1'
    assert set(body) == {'schema','profile','identity','state'}
