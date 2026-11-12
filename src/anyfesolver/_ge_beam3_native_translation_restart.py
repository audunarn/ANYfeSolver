"""Signed translation path envelope around the unchanged complete-state codec."""
from hashlib import sha256
from ._ge_beam3_native_generalized_combined_restart import encode_checkpoint as encode_states,decode_checkpoint as decode_states
from ._ge_beam3_native_fibre_restart import _keys
from . import _ge_beam3_native_history_profile as capacity
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._ge_beam3_native_translation import TranslationProgram,capture,load_point,scalar

SCHEMA='GE_BEAM3_NATIVE_GENERALIZED_TRANSLATION_RESTART_V1'


def encode_checkpoint(model,program,chain,records):
    _,_,_,identity,control,_=capture(model,program)
    profile=program.history_profile;bound=capacity.limit(profile)
    if type(records) is not tuple or type(chain) is not tuple or len(chain)!=len(records)+1 or len(records)>len(program.targets):
        raise ValueError('complete bounded translation path required')
    observed=canonical((program.descriptor(),chain,records))
    if len(observed)>bound:raise ValueError('translation checkpoint byte limit')
    for index,snapshot in enumerate(chain):
        if index:
            row=records[index-1];_keys(row,('index','target','parameter','iterations'))
            if type(row['index']) is not int or row['index']!=index or type(row['iterations']) is not int or not 0<=row['iterations']<=program.max_iterations:
                raise ValueError('translation ordered record/iteration mismatch')
            scalar(row['target']);parameter=scalar(row['parameter'])
            if row['target']!=program.targets[index-1] or snapshot['displacements'][control]!=row['target']:
                raise ValueError('translation target/accepted coordinate mismatch')
        else:parameter=0.
        if snapshot['load_point']!=load_point(program,parameter,genesis=index==0):
            raise ValueError('translation signed parameter/effective load mismatch')
    inner=encode_states(model,chain,history_profile=profile)
    if canonical((program.descriptor(),chain,records))!=observed:raise ValueError('translation checkpoint inputs changed')
    body=dict(schema=capacity.schema(SCHEMA,profile),model_sha256=identity,program=program.descriptor(),program_sha256=sha(program.descriptor()),
        records=records,accepted_chain=capacity.load(inner,profile),production_qualified=False,**capacity.binding(profile))
    raw=canonical({**body,'checkpoint_sha256':sha(body)})
    if len(raw)>bound:raise ValueError('translation checkpoint byte limit')
    return raw


def decode_checkpoint(model,program,raw,*,expected_sha256):
    if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('translation external checkpoint SHA-256 mismatch')
    if type(program) is not TranslationProgram:raise ValueError('explicit translation programme required')
    profile=program.history_profile;value=capacity.load(raw,profile)
    capacity.envelope(value,('schema','model_sha256','program','program_sha256','records','accepted_chain','production_qualified','checkpoint_sha256'),SCHEMA,profile)
    body={k:v for k,v in value.items() if k!='checkpoint_sha256'}
    if value['production_qualified'] is not False or value['checkpoint_sha256']!=sha(body):
        raise ValueError('translation checkpoint schema/hash mismatch')
    if value['model_sha256']!=capture(model,program)[3] or value['program_sha256']!=sha(program.descriptor()) or canonical(value['program'])!=canonical(program.descriptor()):
        raise ValueError('translation checkpoint programme/model mismatch')
    if type(value['records']) is not list:raise ValueError('translation path record list required')
    inner=canonical(value['accepted_chain']);chain=decode_states(model,inner,expected_sha256=sha256(inner).hexdigest(),history_profile=profile)
    records=tuple(value['records'])
    if encode_checkpoint(model,program,chain,records)!=raw:raise ValueError('translation typed canonical roundtrip mismatch')
    return chain,records
