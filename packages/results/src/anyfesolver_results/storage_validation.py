"""Structural admission of storage_prediction/1; no recovery or certification.

Predictors are byte-compression inputs. Their presence grants no engineering
availability, case eligibility or right to reconstruct missing stored rows.
"""
from collections.abc import Mapping
import numpy as np
from .errors import CapsuleValidationError

STORAGE_PREDICTION_API = 1
_INPUTS = {
 'beam_von_mises':('axial_stress','bending_stress_y','bending_stress_z','shear_stress_y','shear_stress_z','torsional_stress'),
 'shell_von_mises':('membrane_xx','membrane_yy','membrane_xy','bending_xx','bending_yy','bending_xy','shear_xz','shear_yz'),
 'equivalent_stress':('von_mises',),
}
def _require(value,message):
    if not value:raise CapsuleValidationError(message)
def _finite(value):
    # Keep finite admission bounded regardless of original array strides.
    iterator=np.nditer(value,flags=['external_loop','buffered','zerosize_ok'],op_flags=['readonly'],buffersize=32768)
    for block in iterator:_require(np.isfinite(block).all(),'nonfinite storage predictor data')
def _plain_json(value):
    if value is None or type(value) in (str,int,bool):return True
    if type(value) is float:return bool(np.isfinite(value))
    if isinstance(value,Mapping):return all(type(key) is str and _plain_json(item) for key,item in value.items())
    if isinstance(value,(list,tuple)):return all(_plain_json(item) for item in value)
    return False
def validate_storage_prediction(role, *, patterns, descriptors=None, associations=None):
    """Admit a separate optional role without calculating any prediction."""
    _require(isinstance(role,Mapping),'storage prediction must be a mapping')
    _require(set(role)<= {'version','runtime','combination','responses','derived','targets','associations','provenance'},'unknown storage prediction role fields')
    _require(type(role.get('version')) is int and role['version']==1,'unsupported storage prediction role version')
    _require(role.get('runtime')=='anysolver-results/2' and role.get('combination')=='ordered_float64/1','storage prediction runtime/combination mismatch')
    responses=role.get('responses');derived=role.get('derived');ids=role.get('associations');targets=role.get('targets')
    _require(isinstance(responses,Mapping) and isinstance(derived,Mapping) and isinstance(ids,Mapping) and isinstance(targets,(list,tuple)),'invalid storage prediction tables')
    _require(all(type(key) is str and key for key in responses) and all(type(key) is str and key for key in derived) and all(type(key) is str and key for key in targets),'invalid storage prediction keys')
    keys=set(responses)|set(derived)
    _require(bool(keys) and not set(responses)&set(derived) and set(ids)==keys,'overlapping/missing storage predictor associations')
    _require(len(set(targets))==len(targets) and set(targets)<=keys,'invalid storage prediction target list')
    _require(isinstance(role.get('provenance',{}),Mapping) and _plain_json(dict(role.get('provenance',{}))), 'storage prediction provenance must be plain finite JSON')
    for key,table in ids.items():
        _require(isinstance(table,np.ndarray) and table.ndim==1 and table.dtype.kind in 'iu','storage predictor associations must be integer arrays')
        if table.dtype.kind=='u' and table.size:_require(np.max(table)<=np.iinfo(np.int64).max,'storage association outside portable int64 range')
        if associations is not None:
            _require(key in associations and np.array_equal(table,np.asarray(associations[key])),'storage predictor/source association mismatch')
    for key,values in responses.items():
        _require(isinstance(values,np.ndarray) and values.dtype==np.dtype('float64') and values.shape==(len(ids[key]),len(patterns)+1),'storage primitive must preserve full float64 row/pattern shape')
        _finite(values)
    for key,spec in derived.items():
        _require(isinstance(spec,Mapping) and set(spec)=={'operation','inputs'} and spec.get('operation') in _INPUTS,'unsupported storage derived rule; principal predictors remain explicit')
        inputs=spec['inputs']
        _require(isinstance(inputs,Mapping) and set(inputs)==set(_INPUTS[spec['operation']]) and all(type(dep) is str and dep in keys for dep in inputs.values()),'storage derived rule has missing inputs')
        for dep in inputs.values():_require(np.array_equal(ids[key],ids[dep]),'storage derived predictor associations differ')
    state={}
    def visit(key):
        _require(state.get(key)!=1,'cyclic storage prediction dependencies')
        if state.get(key)==2:return
        state[key]=1
        for dep in derived.get(key,{}).get('inputs',{}).values():visit(dep)
        state[key]=2
    for key in keys:visit(key)
    if descriptors is not None:
        declared={item['key']:item for item in descriptors}
        _require(keys<=declared.keys(),'storage prediction refers to missing descriptor namespaces')
        for key in targets:_require(declared[key]['availability']=='stored','storage-only predictor cannot grant scientific reconstruction')
    return None
