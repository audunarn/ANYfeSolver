"""Transfer a completed native capture into immutable numerical ownership.

The capture must finish under its original deadline. Its context is never
restarted, extended or used to advance history during the spectral operation.
The model operation lock remains held; inputs and packet are checked on return.
"""
from hashlib import sha256
from .control import cancellation_safe_point
from ._ge_beam3_native_analysis import _require
from ._ge_beam3_p5_seeded.core import canonical

POLICY='GE_BEAM3_MODEL_OWNED_MODAL_SNAPSHOT_V1'


def seal(analysis,program,packet,native_guard,cancellation_token=None):
    native_guard()  # An expired or invalid capture can never be detached.
    analysis._guard()
    program_bytes=canonical(program);packet_hash=sha256(canonical(packet)).hexdigest()
    identity=sha256(canonical(dict(policy=POLICY,model=analysis.identity,
        program_sha256=sha256(program_bytes).hexdigest(),packet_sha256=packet_hash))).hexdigest()
    def check():
        cancellation_safe_point(cancellation_token,'model-owned-modal.snapshot')
        analysis._guard()
        _require(canonical(program)==program_bytes,'captured modal program changed')
        _require(sha256(canonical(packet)).hexdigest()==packet_hash,'captured modal packet changed')
    check()
    return identity,check
