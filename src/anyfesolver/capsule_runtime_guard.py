"""Pin producer replay operations without extending the mechanical trust span."""
import types
import numpy as np
from .boundary import _callable_authority, _callable_authority_matches
from .static_batch import _class_state
from .prepared_static_loads import PreparedStaticCapabilityError


def make_capsule_runtime_guard(namespace, functions, classes):
    source_name=namespace['__name__']
    records,modules,tables,class_records=[],[],[],[]
    seen_functions,seen_classes=set(),set()
    def visit_class(cls):
        if id(cls) in seen_classes:
            return
        seen_classes.add(id(cls))
        class_records.append((cls,_class_state(cls)))
        for value in vars(cls).values():
            if type(value) in (staticmethod,classmethod):
                value=value.__func__
            if type(value) is property:
                for accessor in (value.fget,value.fset,value.fdel):
                    visit(accessor)
            else:
                visit(value)
    def visit(function):
        if type(function) is not types.FunctionType or id(function) in seen_functions:
            return
        seen_functions.add(id(function))
        bindings=tuple((name,function.__globals__[name]) for name in function.__code__.co_names
                       if name in function.__globals__ and name != '_require_capsule_runtime')
        records.append((function,_callable_authority(function),bindings))
        for name,value in bindings:
            module_name=getattr(value,'__module__','') if type(value) in (types.FunctionType,type) else ''
            if type(value) is types.FunctionType and (module_name == source_name or module_name.startswith('anyfesolver_results')):
                visit(value)
            elif type(value) is type and (module_name == source_name or module_name.startswith('anyfesolver_results')):
                visit_class(value)
            elif type(value) is dict and name == 'OPERATIONS':
                tables.append((value,tuple(value.items())))
            elif type(value) is types.ModuleType and value.__name__.startswith('anyfesolver_results'):
                module_vars=vars(value)
                for member in function.__code__.co_names:
                    if member in module_vars and type(module_vars[member]) is types.FunctionType:
                        target=module_vars[member]
                        modules.append((module_vars,member,target));visit(target)
    for function in functions:
        visit(function)
    for cls in classes:
        visit_class(cls)
    numpy_namespace=vars(np)
    numpy_type=type(np)
    numpy_bindings=tuple((name,numpy_namespace[name]) for name in
        ('add','multiply','array','asarray','frombuffer','isfinite','all','zeros','empty',
         'float64','uint64','sqrt','maximum','broadcast_arrays','arange','intp','int64',
         'ndarray','errstate','matmul','power','concatenate','repeat','searchsorted','linalg'))
    numpy_linalg=np.linalg
    numpy_spectral=(numpy_linalg,vars(numpy_linalg)['eigvalsh'])
    numpy_linalg_type=type(numpy_linalg)
    spectral_authority=_callable_authority(numpy_spectral[1])
    # Module entrypoints are pinned as well as aliases used by the producer.
    import anyfesolver_results as runtime
    from anyfesolver_results import capsule, registry
    for module,names in ((runtime,('reconstruct_response','derive_field')),
                         (capsule,('reconstruct_response','_combine_response')),
                         (registry,('derive_field',))):
        modules.extend((vars(module),name,vars(module)[name]) for name in names)
    facts=[]
    def require(_matches=_callable_authority_matches,_state=_class_state):
        reason=None
        if namespace.get('_require_capsule_runtime') is not require or not _matches(require,facts[0]):
            reason='guard'
        for cls,state in class_records:
            if reason is None and _state(cls) != state:
                reason='class '+cls.__name__
        for function,authority,bindings in records:
            if reason is None and not _matches(function,authority):
                reason='function '+function.__qualname__
            for name,value in bindings:
                if reason is None and function.__globals__.get(name) is not value:
                    reason='binding '+function.__qualname__+'.'+name
        if reason is None and any(module.get(name) is not value for module,name,value in modules):
            reason='runtime entrypoint'
        if reason is None and any(tuple(table.items()) != state for table,state in tables):
            reason='operation registry'
        if reason is None and type(np) is not numpy_type:
            reason='NumPy module type'
        for name,value in numpy_bindings:
            if reason is None and numpy_namespace.get(name) is not value:
                reason='NumPy '+name
        if reason is None and (type(numpy_linalg) is not numpy_linalg_type
                or vars(numpy_linalg).get('eigvalsh') is not numpy_spectral[1]):
            reason='NumPy spectral entrypoint'
        if reason is None and not _matches(numpy_spectral[1],spectral_authority):
            reason='NumPy spectral executable'
        if reason is not None:
            raise PreparedStaticCapabilityError('capsule replay executable authority changed: '+reason)
    facts.append(_callable_authority(require))
    return require
