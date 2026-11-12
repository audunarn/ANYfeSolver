"""Disposable scalar observers for the development radial contact-path runner.

The public contact-path API has no live observer. These wrappers delegate the
unchanged functions, record scalar cost/branch data, and restore bindings even
when the solve throws. They cannot advance or commit solver state.
"""
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import time

import numpy as np


@contextmanager
def radial_solver_trace(path):
    from anyfesolver.general_contact import semismooth_path as driver
    from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
    import anyfesolver
    path = Path(path)
    originals, counts = [], {}
    started = time.perf_counter()
    with path.open('x', encoding='utf-8') as stream:
        def emit(event, **values):
            stream.write(json.dumps(dict(event=event,
                elapsed_seconds=time.perf_counter()-started, **values))+'\n')
            stream.flush()

        def hook(owner, name, phase):
            original = getattr(owner, name)
            @wraps(original)
            def observed(*args, **kwargs):
                counts[phase] = counts.get(phase, 0)+1
                call = counts[phase]
                details = {}
                if phase == 'contact':
                    z = np.asarray(args[2])
                    details = dict(positive_force_modes=int(np.count_nonzero(z > 0.)),
                        total_nodal_force_n=float(np.sum(z)), tangent=kwargs.get('tangent', True))
                elif phase == 'structural':
                    details = dict(tangent=kwargs.get('tangent', True))
                elif phase == 'factorization':
                    details = dict(shape=list(args[0].shape), nnz=getattr(args[0], 'nnz', None))
                emit(phase+'_begin', call=call, **details)
                before = time.perf_counter()
                try:
                    value = original(*args, **kwargs)
                except BaseException as exc:
                    emit(phase+'_exception', call=call,
                         seconds=time.perf_counter()-before, reason=type(exc).__name__+': '+str(exc))
                    raise
                if phase == 'swept':
                    details = dict(admissible=bool(value))
                elif phase == 'contact':
                    details = dict(contact_residual_norm_n=float(np.linalg.norm(value.force)),
                        minimum_normalized_gap_m=float(np.min(value.state.normalized_gap)),
                        candidate_evaluations=int(value.evaluations))
                elif phase == 'complementarity':
                    details = dict(complementarity_l1_n=float(np.sum(abs(value[0]))),
                                   active_modes=int(np.count_nonzero(value[1])))
                else:
                    details = {}
                emit(phase+'_end', call=call, seconds=time.perf_counter()-before, **details)
                return value
            originals.append((owner, name, original))
            setattr(owner, name, observed)

        try:
            emit('trace_begin', imported_source=anyfesolver.__file__,
                 scope='development observer only; no state or numerical setting changes')
            hook(GeneralS3PatchManager, '__init__', 'preparation')
            hook(GeneralS3PatchManager, '_swept_admissible', 'swept')
            hook(driver, '_assemble_nonlinear_system', 'structural')
            hook(driver, 'linearize', 'contact')
            hook(driver, 'complementarity', 'complementarity')
            hook(driver, 'factorize', 'factorization')
            yield
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)
            emit('trace_end', calls=counts)
