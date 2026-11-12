"""Model-owned retained translation dispatch; no state conversion or reseeding."""
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
from math import isfinite
from ._ge_beam3_native_analysis import NativeBeamRun, NativeBeamAnalysisError, _require, MAX_CHECKPOINT_BYTES
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_operator_validation_scope import operator_validation_scope
from ._ge_beam3_refinement_capacity import n32_refinement_capacity
from . import _ge_beam3_retained_translation_control as ordinary
from . import _ge_beam3_elastic_seed_continuation as seeded

SCHEMA = 'GE_BEAM3_MODEL_OWNED_TRANSLATION_CHECKPOINT_V1'
KEYS = {'schema','definition_graph_sha256','workflow','program','seed_sha256',
        'backend','backend_sha256','production_qualified'}


def _strict(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_CHECKPOINT_BYTES, 'bounded translation checkpoint bytes')
    def pairs(rows):
        data={}
        for key,value in rows:
            _require(key not in data, 'duplicate translation checkpoint key')
            data[key]=value
        return data
    def number(text):
        value=float(text);_require(isfinite(value),'finite translation checkpoint numbers');return value
    def forbidden(text):raise NativeBeamAnalysisError('nonfinite translation checkpoint')
    try:
        data=json.loads(raw.decode('ascii'),object_pairs_hook=pairs,parse_float=number,parse_constant=forbidden)
        _require(canonical(data)==raw,'canonical translation checkpoint required')
    except (UnicodeError,RecursionError,json.JSONDecodeError) as error:
        raise NativeBeamAnalysisError('invalid translation checkpoint JSON') from error
    return data


def _owner(analysis,program,seed,digest):
    analysis._family_required('GENERALIZED_DISTRIBUTED')
    _require(type(program) is ordinary.Program,'exact retained translation program required')
    program.__post_init__()
    _require((seed is None)==(digest is None),'seed/hash pair required')
    if seed is None:return ordinary,'RETAINED_FROM_REFERENCE',None
    seeded._packet(seed,digest)
    return seeded,'ELASTIC_ACCEPTED_SEED',digest


def _envelope(analysis,program,backend,workflow,seed_digest):
    analysis._guard()
    value=dict(schema=SCHEMA,definition_graph_sha256=analysis.identity,workflow=workflow,
        program=program,seed_sha256=seed_digest,backend=backend.decode('ascii'),
        backend_sha256=sha256(backend).hexdigest(),production_qualified=False)
    raw=canonical(value);_strict(raw);return raw


def _backend(analysis,program,raw,expected,workflow,seed_digest):
    _require(type(raw) is bytes and len(raw)<=MAX_CHECKPOINT_BYTES and type(expected) is str
             and sha256(raw).hexdigest()==expected,'external translation checkpoint hash mismatch')
    data=_strict(raw)
    _require(type(data) is dict and set(data)==KEYS,'complete translation envelope required')
    _require(data['schema']==SCHEMA and data['definition_graph_sha256']==analysis.identity
        and data['workflow']==workflow and data['seed_sha256']==seed_digest
        and canonical(data['program'])==canonical(program) and data['production_qualified'] is False
        and type(data['backend']) is str,'translation definition/program/owner mismatch')
    backend=data['backend'].encode('ascii');digest=sha256(backend).hexdigest()
    _require(digest==data['backend_sha256'],'translation backend hash mismatch')
    return backend,digest


def _check_backend_header(raw,digest,program,owner,seed_digest):
    _require(type(raw) is bytes and type(digest) is str and sha256(raw).hexdigest()==digest,
             'external backend byte authority')
    data=_strict(raw)
    _require(type(data) is dict and data.get('schema')==owner.SCHEMA
        and canonical(data.get('program'))==canonical(program),'backend translation schema/program mismatch')
    if owner is seeded:_require(data.get('seed_sha256')==seed_digest,'backend seed mismatch')


@contextmanager
def _operation(analysis,cancellation_token=None):
    with analysis._operation():
        _require(bool(analysis._admit_boundaries()),'supported translation model required')
        with (n32_refinement_capacity() if analysis._retained_refinement else nullcontext()):
            with operator_validation_scope(cancellation_token=cancellation_token):yield


def _context(analysis,program,owner,seed,digest):
    if owner is seeded:return owner.Context(analysis.model,program,seed,expected_seed_sha256=digest)
    return owner.Context(analysis.model,program)


def solve(analysis,program,*,seed=None,expected_seed_sha256=None,checkpoint=None,expected_sha256=None,
          stop_after=None,cancellation_token=None,progress=None):
    owner,workflow,seed_digest=_owner(analysis,program,seed,expected_seed_sha256)
    _require((checkpoint is None)==(expected_sha256 is None),'checkpoint/hash pair required')
    _require(progress is None or callable(progress),'callable translation observer required')
    end=len(program.targets) if stop_after is None else stop_after
    _require(type(end) is int and 0<=end<=len(program.targets),'bounded translation stop target')
    with _operation(analysis,cancellation_token):
        backend=digest=None
        if checkpoint is not None:
            backend,digest=_backend(analysis,program,checkpoint,expected_sha256,workflow,seed_digest)
            _check_backend_header(backend,digest,program,owner,seed_digest)
        def observed(row):
            analysis._guard()
            if progress is not None:progress(row)
            analysis._guard()
        kw=dict(checkpoint=backend,expected_sha256=digest,stop_after=stop_after,
                cancellation_token=cancellation_token,progress=observed)
        if owner is seeded:
            result=owner.solve(analysis.model,program,seed,expected_seed_sha256=seed_digest,**kw)
        else:result=owner.solve(analysis.model,program,**kw)
        raw=_envelope(analysis,program,result.checkpoint,workflow,seed_digest)
        return NativeBeamRun(result.status,raw,result)


def adopt(analysis,program,backend,*,expected_sha256,seed=None,expected_seed_sha256=None):
    owner,workflow,seed_digest=_owner(analysis,program,seed,expected_seed_sha256)
    _check_backend_header(backend,expected_sha256,program,owner,seed_digest)
    with _operation(analysis):
        context=_context(analysis,program,owner,seed,seed_digest)
        state,records=context.restore(backend,expected_sha256=expected_sha256)
        _require(context.checkpoint(records)==backend,'native adoption changed checkpoint')
        return _envelope(analysis,program,backend,workflow,seed_digest)


def recover(analysis,program,checkpoint,*,expected_sha256,seed=None,expected_seed_sha256=None):
    owner,workflow,seed_digest=_owner(analysis,program,seed,expected_seed_sha256)
    with _operation(analysis):
        backend,digest=_backend(analysis,program,checkpoint,expected_sha256,workflow,seed_digest)
        _check_backend_header(backend,digest,program,owner,seed_digest)
        context=_context(analysis,program,owner,seed,seed_digest)
        state,records=context.restore(backend,expected_sha256=digest);before=canonical(state)
        result=context.recover(state)
        _require(canonical(state)==before and context.checkpoint(records)==backend,'recovery changed accepted state')
        return result


def prefix(analysis,program,checkpoint,accepted_steps,*,expected_sha256,seed=None,expected_seed_sha256=None):
    owner,workflow,seed_digest=_owner(analysis,program,seed,expected_seed_sha256)
    _require(type(accepted_steps) is int and accepted_steps>=0,'exact nonnegative prefix cursor')
    with _operation(analysis):
        backend,digest=_backend(analysis,program,checkpoint,expected_sha256,workflow,seed_digest)
        _check_backend_header(backend,digest,program,owner,seed_digest)
        context=_context(analysis,program,owner,seed,seed_digest)
        state,records=context.restore(backend,expected_sha256=digest)
        _require(accepted_steps<=len(records),'prefix exceeds accepted chain')
        return _envelope(analysis,program,context.checkpoint(records[:accepted_steps]),workflow,seed_digest)
