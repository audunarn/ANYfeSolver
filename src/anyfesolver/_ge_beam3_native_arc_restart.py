"""Complete signed arc history with replayed objective predictor geometry."""
from hashlib import sha256
from time import monotonic
import numpy as np
from ._ge_beam3_native_generalized_combined_restart import encode_checkpoint as encode_states,decode_checkpoint as decode_states
from ._ge_beam3_native_fibre_restart import _keys,_array
from . import _ge_beam3_native_history_profile as capacity
from ._ge_beam3_p5_seeded.core import canonical,sha
from ._ge_beam3_native_translation import load_point,scalar
from ._ge_beam3_native_arc import ArcProgram,capture,arc_row,replay_direction

SCHEMA='GE_BEAM3_NATIVE_GENERALIZED_FRAME_CHORD_ARC_RESTART_V1'
SOURCE_SCHEMA='GE_BEAM3_NATIVE_GENERALIZED_FRAME_CHORD_ARC_TRANSLATION_SOURCE_V1'


def encode_checkpoint(model,program,chain,records):
    started=monotonic();_,n,free,identity,maps,metric,previous=capture(model,program)
    profile=program.history_profile;bound=capacity.limit(profile)
    origin=0;origin_parameter=0.;source_raw=None
    if program.source is not None:
        from ._ge_beam3_native_arc_source import load_source,secant_orientation
        source_raw=program.source.checkpoint;prefix,origin_parameter,previous_parameter=load_source(model,program.source);origin=len(prefix)-1
        if len(prefix)+len(program.steps)>65:raise ValueError('complete source plus arc history exceeds snapshot bound')
        previous=secant_orientation(prefix,origin_parameter,previous_parameter,metric,program.source.forward_sign)
        if type(chain) is not tuple or canonical(chain[:origin+1])!=canonical(prefix):raise ValueError('native arc source prefix mismatch')
    if type(records) is not tuple or type(chain) is not tuple or len(chain)!=len(records)+1+origin or len(records)>len(program.steps):
        raise ValueError('complete bounded native arc path required')
    observed=canonical((program.descriptor(),chain,records))
    if len(observed)>bound:raise ValueError('native arc checkpoint byte limit')
    inner=encode_states(model,chain,history_profile=profile);old_parameter=origin_parameter;fixed=np.setdiff1d(np.arange(n),free)
    for index,snapshot in enumerate(chain[origin:]):
        if monotonic()-started>60.:raise RuntimeError('native arc checkpoint validation deadline')
        if index:
            row=records[index-1];_keys(row,('index','step_size','parameter','iterations','direction','arc_residual'))
            if type(row['index']) is not int or row['index']!=index or type(row['iterations']) is not int or not 0<=row['iterations']<=program.max_iterations:
                raise ValueError('native arc ordered record/iteration mismatch')
            step=scalar(row['step_size']);parameter=scalar(row['parameter']);residual=scalar(row['arc_residual'])
            tangent=np.asarray(row['direction'])
            if step!=program.steps[index-1] or tangent.dtype!=np.float64 or tangent.shape!=(n+1,) or not np.isfinite(tangent).all():
                raise ValueError('native arc step/predictor schema mismatch')
            if np.any(tangent[fixed]) or abs(float(np.sum(metric*tangent*tangent))-1.)>1e-11 or float((metric*previous)@tangent)<=0.:
                raise ValueError('native arc predictor metric/orientation mismatch')
            expected=replay_direction(model,program,chain[origin+index-1],old_parameter,previous)
            if not np.array_equal(expected,tangent):raise ValueError('native arc predictor origin replay mismatch')
            gap,_=arc_row(snapshot['displacements'],chain[origin+index-1]['displacements'],tangent,metric,parameter,old_parameter,step,maps)
            if abs(gap)>1e-12 or abs(gap)!=residual:raise ValueError('native arc objective constraint replay mismatch')
            previous=tangent.copy()
        else:parameter=origin_parameter
        if snapshot['load_point']!=load_point(program,parameter,genesis=index==0 and origin==0):raise ValueError('native arc signed effective load mismatch')
        old_parameter=parameter
    if canonical((program.descriptor(),chain,records))!=observed:raise ValueError('native arc checkpoint inputs changed')
    program.require(model)
    if source_raw is not None and program.source.checkpoint!=source_raw:raise ValueError('native arc source changed during checkpoint')
    body=dict(schema=capacity.schema(SCHEMA if source_raw is None else SOURCE_SCHEMA,profile),model_sha256=identity,program=program.descriptor(),program_sha256=sha(program.descriptor()),
        metric=metric,records=records,accepted_chain=capacity.load(inner,profile),production_qualified=False,**capacity.binding(profile),
        **({} if source_raw is None else {'source_checkpoint':capacity.load(source_raw,profile)}))
    raw=canonical({**body,'checkpoint_sha256':sha(body)})
    if len(raw)>bound:raise ValueError('native arc checkpoint byte limit')
    return raw


def decode_checkpoint(model,program,raw,*,expected_sha256):
    if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('native arc external checkpoint SHA-256 mismatch')
    if type(program) is not ArcProgram:raise ValueError('explicit native arc programme required')
    if program.source is not None:
        from ._ge_beam3_native_arc_source import TranslationArcSource
        if type(program.source) is not TranslationArcSource:raise ValueError('exact native translation arc source required')
    profile=program.history_profile;value=capacity.load(raw,profile)
    keys=('schema','model_sha256','program','program_sha256','metric','records','accepted_chain','production_qualified','checkpoint_sha256')
    capacity.envelope(value,keys+(() if program.source is None else ('source_checkpoint',)),SCHEMA if program.source is None else SOURCE_SCHEMA,profile)
    if program.source is not None and canonical(value['source_checkpoint'])!=program.source.checkpoint:
        raise ValueError('native arc embedded source mismatch')
    body={k:v for k,v in value.items() if k!='checkpoint_sha256'}
    _,n,_,identity,_,metric,_=capture(model,program)
    if value['production_qualified'] is not False or value['checkpoint_sha256']!=sha(body):
        raise ValueError('native arc checkpoint schema/hash mismatch')
    if value['model_sha256']!=identity or value['program_sha256']!=sha(program.descriptor()) or canonical(value['program'])!=canonical(program.descriptor()):
        raise ValueError('native arc programme/model mismatch')
    if not np.array_equal(_array(value['metric'],(n+1,)),metric) or type(value['records']) is not list:raise ValueError('native arc metric/records mismatch')
    inner=canonical(value['accepted_chain']);chain=decode_states(model,inner,expected_sha256=sha256(inner).hexdigest(),history_profile=profile)
    records=[]
    for row in value['records']:
        _keys(row,('index','step_size','parameter','iterations','direction','arc_residual'))
        records.append({**row,'direction':_array(row['direction'],(n+1,))})
    records=tuple(records)
    if encode_checkpoint(model,program,chain,records)!=raw:raise ValueError('native arc typed canonical roundtrip mismatch')
    return chain,records
