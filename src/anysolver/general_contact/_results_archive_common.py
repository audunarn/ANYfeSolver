"""Shared inert static/arc archive mechanics; producer admission stays local."""
from dataclasses import fields
import hashlib
import numpy as np
from . import transient_results_archive as codec


def _vector(value, width=None):
    if type(value) is not np.ndarray or value.ndim != 1 or value.size < 1 or \
            value.dtype.kind not in 'fi' or not np.all(np.isfinite(value)) or \
            (width is not None and value.size != width):
        codec._fail('invalid static generalized displacement vector')


def _state(value):
    if type(value) is not dict or any(type(k) is not int or k < 0 for k in value):
        codec._fail('material states require nonnegative integer element identities')


def _diagnostics(value):
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is str and key in ('integration_force','target_force','shell_force','rigid_force',
                    'beam_force','integration_moment','target_moment','shell_moment','rigid_moment',
                    'beam_moment','wrench_origin_m'):
                if type(item) not in (tuple,list) or len(item) != 3: codec._fail('invalid diagnostic wrench vector')
                for component in item: codec._number(component)
            elif key in ('minimum_gap','sampled_overlap_gap','minimum_physical_gap'):
                if item is not None: codec._number(item, gap=True)
            elif key == 'gap_tolerance_met':
                if type(item) not in (bool,np.bool_): codec._fail('invalid diagnostic flag')
            _diagnostics(item)
    elif type(value) in (tuple,list):
        for item in value: _diagnostics(item)


def _encode_committed_results(data, cap, records, *, schema, conventions, snapshot_error):
    """Preserve snapshot preflight, encoding order and the shared byte budget."""
    data = codec._prepare_material_data(data, cap)
    snapshots = data['snapshots']
    if type(snapshots) is not tuple:
        codec._fail(snapshot_error)
    budget = codec._Budget(cap)
    data['snapshots'] = ()
    payload = codec._encode(data, budget, records)
    nodes = []
    from anysolver.nonlinear_static import NonlinearIncrementSnapshot
    for snapshot in snapshots:
        if type(snapshot) is not NonlinearIncrementSnapshot:
            codec._fail('exact committed snapshot required')
        try: values = {f.name: getattr(snapshot, f.name) for f in fields(snapshot)}
        except AttributeError: codec._fail('incomplete committed snapshot')
        values = codec._prepare_material_data(values, max(1, cap-budget.bytes))
        # Encode fields before a bounded known-producer summary. No constructor,
        # deepcopy or executable type restoration is invoked by the reader.
        encoded = codec._encode(values, budget, records, 2)
        nodes.append({'tag': 'record', 'record_type': 'anysolver.nonlinear_static.NonlinearIncrementSnapshot',
                      'fields': encoded,
                      'summary': codec._encode(snapshot.to_dict(), budget, records, 2)})
    next(node for key, node in payload['items'] if key == 'snapshots')['items'] = nodes
    document = dict(schema=schema, version=1, conventions=dict(conventions), payload=payload)
    document['sha256'] = hashlib.sha256(codec._canonical(document)).hexdigest()
    raw = codec._canonical(document)
    if len(raw) > cap: codec._fail('archive encoded byte limit exceeded')
    return raw
