"""Optional expected-fact accelerator. Every decline uses the literal guard."""
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import types

_MODULE_TYPE=types.ModuleType
_BUILTIN_FUNCTION_TYPE=types.BuiltinFunctionType
_MODULES=sys.modules
_DICT_TYPE=dict
_STATE=None
_COUNTERS=[0,0]  # Diagnostic values only; the owned container identity is fixed.

def initialise(classes,helpers,missing,literal_roots):
    global _STATE
    if _STATE is not None:
        raise RuntimeError('native guard already initialised')
    if (sys.platform!='win32' or sys.version_info[:2]!=(3,11)
            or platform.machine().upper() not in ('AMD64','X86_64')
            or os.environ.get('ANYSOLVER_NATIVE_GUARD')=='python'):
        _STATE=('fallback','unsupported_abi_or_explicit_python')
        return
    try:
        module=importlib.import_module('anyfesolver._native_guard')
    except ModuleNotFoundError as error:
        if error.name!='anyfesolver._native_guard':
            raise
        _STATE=('fallback','extension_absent')
        return
    except ImportError as error:
        _STATE=('refusal',str(error))
        return
    try:
        if type(module) is not _MODULE_TYPE:
            raise ValueError('native module type changed')
        if any(type(key) is not str for key in module.__dict__):
            _STATE=('fallback','unsupported_native_namespace')
            return
        names=('make_plan','compare','build_info','binary_identity','entry_identity','statistics','reset_statistics')
        entries=tuple((name,module.__dict__[name]) for name in names)
        # Do not call a module-supplied proof function until every entry is an
        # actual C builtin owned by this exact native module. Python functions
        # cannot manufacture an entry/image proof at cold admission.
        if any(type(entry) is not _BUILTIN_FUNCTION_TYPE or entry.__self__ is not module or entry.__name__!=name
               for name,entry in entries):
            raise ValueError('native entries are not canonical module-owned C functions')
        identity=module.entry_identity
        entry_facts=tuple((name,entry,identity(entry),entry.__module__) for name,entry in entries)
        if any(facts is NotImplemented or facts[0]!=name or facts[1] is not module for name,_,facts,_ in entry_facts):
            raise ValueError('native entry ownership mismatch')
        image=module.binary_identity()
        api,abi,source_sha=module.build_info()
        if api!=1 or abi!='cp311-win_amd64' or image is NotImplemented:
            raise ValueError('native build ABI mismatch')
        mapped=Path(image[0]).resolve(strict=True)
        if mapped.parent!=Path(__file__).resolve().parent or mapped.suffix.lower()!='.pyd':
            raise ValueError('mapped native image is outside the owning package')
        binary_sha=hashlib.sha256(mapped.read_bytes()).hexdigest()
        source=Path(__file__).with_name('_native_guard.cpp')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:
            raise ValueError('native compiled source identity mismatch')
        distribution=importlib.metadata.distribution('ANYfeSolver')
        record=distribution.read_text('RECORD')
        if record is None:
            _STATE=('fallback','unadmitted_no_binary_record')
            return
        matches=[]
        for member,digest,size in csv.reader(record.splitlines()):
            if Path(distribution.locate_file(member)).resolve()==mapped:
                matches.append((digest,size))
        expected='sha256='+base64.urlsafe_b64encode(bytes.fromhex(binary_sha)).decode().rstrip('=')
        if not matches:
            # Editable/source layouts may build a valid in-place extension but
            # do not record it in installed wheel metadata. Never fast-pass an
            # unrecorded binary; the complete literal guard remains available.
            _STATE=('fallback','unadmitted_no_binary_record')
            return
        if len(matches)!=1 or matches[0]!=(expected,str(mapped.stat().st_size)):
            raise ValueError('mapped native binary does not match installed wheel RECORD')
        plan=module.make_plan(classes,helpers,missing,literal_roots)
        if plan is NotImplemented:
            _STATE=('fallback','unsupported_expected_grammar')
            return
        attributes=tuple((name,module.__dict__.get(name)) for name in ('__name__','__spec__','__loader__','__package__','__file__'))
        _STATE=('native',module,plan,entry_facts,image,(api,abi,source_sha),binary_sha,str(mapped),attributes)
    except (KeyError,ValueError,OSError,importlib.metadata.PackageNotFoundError) as error:
        _STATE=('refusal',str(error))

def try_compare(classes,helpers):
    state=_STATE
    if state[0]=='fallback':
        _COUNTERS[0]+=1
        return NotImplemented
    if state[0]!='native':
        return 2
    _,module,plan,entries,image,build,binary_sha,path,attributes=state
    if (type(module) is not _MODULE_TYPE or type(sys) is not _MODULE_TYPE
            or type(_MODULES) is not _DICT_TYPE):
        return 2
    if any(type(key) is not str for key in sys.__dict__) or any(type(key) is not str for key in _MODULES):
        return NotImplemented
    if sys.__dict__.get('modules') is not _MODULES or _MODULES.get('anyfesolver._native_guard') is not module:
        return 2
    namespace=module.__dict__
    if any(type(key) is not str for key in namespace):
        return NotImplemented
    for name,value in attributes:
        if namespace.get(name) is not value:
            return 2
    # Exact native module dictionary: no custom getattr is called here.
    for name,entry,facts,label in entries:
        if namespace.get(name) is not entry or entry.__module__ is not label:
            return 2
    identity=entries[4][1]
    for name,entry,facts,label in entries:
        if identity(entry)!=facts:
            return 2
    if entries[2][1]()!=build:
        return 2
    result=entries[1][1](plan,classes,helpers)
    if result is True:
        _COUNTERS[1]+=1
    elif result is False or result is NotImplemented:
        _COUNTERS[0]+=1
    return result

def diagnostics():
    state=_STATE
    result={'backend':state[0] if state else 'uninitialised','python_fallbacks':_COUNTERS[0],'native_passes':_COUNTERS[1]}
    if state and state[0]=='native':
        result.update(binary_sha256=state[6],mapped_image=state[7],build=state[5],native=state[3][5][1]())
    elif state:
        result['reason']=state[1]
    return result
