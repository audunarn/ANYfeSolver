"""Fibre S3 framing and direct-entry numerical-policy boundaries."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_displacement_checkpoint import MAGIC, decode_snapshot
from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode, encode
from anyfesolver._stiffened_shell_beam_history import shapes
from test_stiffened_shell_native_point_displacement import owner, ACTIONS, DIRECTION, CONTROL, OPTIONS
from test_stiffened_shell_native_fibre_s3_displacement import fibre_model


@pytest.fixture(scope='module')
def fibre_endpoint():
    definition = fibre_model()
    analysis = ss.create_analysis(definition=definition)
    try:
        analysis.solve(ACTIONS, (.5, 1.), **OPTIONS)
        dof = analysis._shell_count + 6*analysis._beam_index[105] + 2
        target = analysis._finite_driver._state.q[dof]*1.01
        analysis.solve(DIRECTION, (target,), displacement_control=CONTROL, **OPTIONS)
        return definition, analysis.checkpoint()
    finally:
        analysis.close()


@pytest.mark.parametrize('relabel', ('controller', 'mechanical'))
def test_fibre_s3_relabelled_generalized_identity_is_atomic(request, fibre_endpoint, relabel):
    definition, raw = fibre_endpoint
    analysis = owner(request, definition)
    metadata, arrays = decode_snapshot(analysis, raw)
    start = len(MAGIC)+4
    length = int.from_bytes(raw[len(MAGIC):start], 'little')
    inner = raw[start+length:-32]
    if relabel == 'controller':
        metadata = dict(metadata, policy='B3_GE_NATIVE_POINT_ELASTIC_S3_DISPLACEMENT_V1_DEV')
    else:
        inner = encode('finite_displacement_native_point_elastic_s3',
                       analysis._identity, metadata['generation'], tuple(arrays.items()))
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    content = MAGIC+len(header).to_bytes(4, 'little')+header+inner
    changed = content+sha256(content).digest()
    with pytest.raises(ValueError):
        analysis.restore(changed)
    assert analysis._generation == 0 and analysis._finite_driver is None
    assert analysis._accepted is None


def test_legacy_s3_direct_corrector_rejects_before_assembly(request, monkeypatch):
    definition = fibre_model()
    analysis = owner(request, definition)
    legacy = owner(request, definition)
    zero = (ss.NodalAction('beam', 105, (0., 0., 0., 0., 0., 0.)),)
    analysis.solve(zero, (1.,), **OPTIONS)
    def layout(kind, generation):
        n = analysis._K.shape[0]
        return (('q', (n,)), ('frames', (len(analysis._beam_ids), 3, 3)),
                ('cells', (len(analysis._beam_parts), 2, 3, 3)),
                ('resultants', (len(analysis._beam_parts), 18)),
                ('multipliers', (analysis._C.shape[0],)), ('pattern', (n,)),
                ('accepted_factors', (generation,))) + shapes(analysis._beam_parts, 'origin') + shapes(analysis._beam_parts, 'accepted')
    _, generation, values = decode(analysis.checkpoint(), analysis._identity, layout)
    historical = encode('finite_native_material', analysis._identity, generation, tuple(values.items()))
    legacy.restore(historical)
    before = legacy.recover()
    from anyfesolver._stiffened_shell_displacement_corrector import correct
    def forbidden(*args, **kwargs):
        raise AssertionError('incompatible numerical policy reached assembly')
    monkeypatch.setattr(legacy._finite_driver, '_assemble', forbidden)
    with pytest.raises(ValueError, match='seed numerical policy mismatch'):
        correct(legacy._finite_driver, base_load=np.zeros(legacy._K.shape[0]),
                load_direction=legacy._finite_driver._load(DIRECTION), amplitude=0.,
                control_dof=CONTROL.dof(legacy), target=.001)
    assert legacy.recover() is before and legacy._generation == generation
    assert legacy.checkpoint() == historical and not legacy._finite_driver._stable_s3
    # A native adaptive envelope cannot smuggle the same incompatible force seed.
    from anyfesolver._stiffened_shell_displacement_adaptive import Context,encode as envelope,direction_hash
    from anyfesolver._stiffened_shell_displacement_increment import DisplacementIncrementState
    context=Context(DisplacementIncrementState(ss.AdaptiveIncrementControl(),(.001,),0.),
        CONTROL.dof(legacy),direction_hash(legacy._finite_driver._load(DIRECTION)),0,20,12)
    fresh=owner(request,definition)
    with pytest.raises(ValueError,match='seed numerical policy mismatch'):
        fresh.restore(envelope(legacy._identity,context,historical,native=True))
    assert fresh._generation==0 and fresh._finite_driver is None
    with pytest.raises(ValueError,match='seed numerical policy mismatch'):
        legacy.solve(DIRECTION,(.001,),displacement_control=CONTROL,
                     increment_control=ss.AdaptiveIncrementControl(),**OPTIONS)
    assert legacy.recover() is before and legacy.checkpoint()==historical
