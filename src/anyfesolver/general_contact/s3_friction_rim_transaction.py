"""New rim profile invocation publication rollback on owned physical state."""
import copy
from contextvars import ContextVar
_ENTRY=ContextVar('s3_rim_publication_entry',default=None)

def register_entry(owner):
    entries=_ENTRY.get()
    if entries is None:raise RuntimeError('rim replay requires public invocation guard')
    entries.append((owner,copy.deepcopy((owner.frames,owner.committed_displacement,owner.cumulative_dissipation,
        owner.committed_storage,owner.last_committed_summary,owner.normal_multipliers))))

def guarded_invocation(operation):
    entries=[];token=_ENTRY.set(entries)
    try:return operation()
    except BaseException:
        for owner,state in reversed(entries):
            frames,pose,D,E,summary,multipliers=state
            owner.frames=frames;owner.committed_displacement=pose;owner.cumulative_dissipation=D
            owner.committed_storage=E;owner.last_committed_summary=summary
            if owner.normal.augmented:owner.normal.set_normal_multipliers(multipliers)
            owner._material_history_frames=None;owner._material_history_cache.clear()
        raise
    finally:_ENTRY.reset(token)
