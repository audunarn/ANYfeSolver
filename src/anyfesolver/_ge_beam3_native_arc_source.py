"""Explicit authenticated translation history for a private native arc handoff."""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from ._ge_beam3_native_translation import TranslationProgram
from ._ge_beam3_native_history_profile import limit
from ._ge_beam3_p5_seeded.core import canonical

POLICY='GE_BEAM3_NATIVE_TRANSLATION_TO_ARC_SOURCE_V1'


@dataclass(frozen=True)
class TranslationArcSource:
    program: object
    checkpoint: bytes
    expected_sha256: str
    forward_sign: float

    def require(self,model):
        if type(self) is not TranslationArcSource or type(self.program) is not TranslationProgram:
            raise ValueError('exact native translation arc source required')
        if (type(self.checkpoint) is not bytes or not 0<len(self.checkpoint)<=limit(self.program.history_profile)
            or type(self.expected_sha256) is not str or sha256(self.checkpoint).hexdigest()!=self.expected_sha256):
            raise ValueError('translation arc source external SHA-256/size mismatch')
        if type(self.forward_sign) is not float or self.forward_sign not in (-1.,1.):
            raise ValueError('explicit source-trajectory continuation orientation required')
        self.program.require(model)

    def descriptor(self):
        return dict(policy=POLICY,program=self.program.descriptor(),checkpoint_sha256=self.expected_sha256,
                    forward_sign=self.forward_sign,orientation='ACCEPTED_SPATIAL_INCREMENT_SECANT_V1')


def require_compatible(model,arc):
    source=arc.source
    if type(source) is not TranslationArcSource:raise ValueError('exact native translation arc source required')
    source.require(model)
    if (arc.history_profile!=source.program.history_profile or arc.initial_sign!=1.
        or canonical((arc.distributed,arc.nodal_moments))!=canonical((source.program.distributed,source.program.nodal_moments))):
        raise ValueError('translation arc source load/profile/orientation mismatch')


def load_source(model,source):
    from ._ge_beam3_native_translation_restart import decode_checkpoint
    source.require(model);observed=source.checkpoint;identity=canonical(source.descriptor())
    chain,records=decode_checkpoint(model,source.program,observed,expected_sha256=source.expected_sha256)
    if not records:raise ValueError('translation arc source requires an accepted non-genesis state')
    source.require(model)
    if source.checkpoint!=observed or canonical(source.descriptor())!=identity:
        raise ValueError('translation arc source changed during validation')
    return chain,records[-1]['parameter'],(records[-2]['parameter'] if len(records)>1 else 0.)


def secant_orientation(chain,parameter,previous_parameter,metric,forward_sign):
    """Source rotation-coordinate differences are authenticated spatial increments."""
    value=np.r_[chain[-1]['displacements']-chain[-2]['displacements'],parameter-previous_parameter]
    if value.shape!=metric.shape or not np.isfinite(value).all():raise ValueError('source secant shape/range')
    norm=float(np.sum(metric*value*value))
    if not np.isfinite(norm) or norm<=0.:raise ValueError('source requires a nonzero last accepted increment')
    value*=forward_sign/np.sqrt(norm)
    if not np.isfinite(value).all():raise ValueError('source secant orientation range')
    return value
