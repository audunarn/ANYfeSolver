"""Exact live-input snapshot for the original canonical cell fingerprint.

Every guard still reads all fingerprint inputs and validates section/compiled
data. Snapshot mismatch uses the original fingerprint, without adopting it.
"""
import json


def _methods():
    from . import _ge_beam3_generalized_cell as source
    return source.canonical,source.sha256,json.loads


def _snapshot(cell):
    from . import _ge_beam3_generalized_cell as source
    if type(cell) is not source.GeneralizedCellConjugate:
        raise ValueError('exact generalized cell identity source')
    if any(type(v) is not str for v in (source.POLICY,cell.section.identity,cell.identity)) or type(cell._capture) is not bytes:
        raise ValueError('exact cell identity values')
    return source.POLICY,cell.section.identity,cell._capture,cell.identity


def _fingerprint(cell):
    from . import _ge_beam3_generalized_cell as source
    return source.sha256(source.canonical(dict(policy=source.POLICY,
        section=cell.section.identity,stations=json.loads(cell._capture)))).hexdigest()


class CapturedCellIdentity:
    __slots__=('_cell','_snapshot','_identity','_methods','_capture','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('captured cell identity is immutable')
        object.__setattr__(self,name,value)

    def __init__(self,cell):
        self._cell=cell;self._methods=_methods();self._snapshot=_snapshot(cell)
        self._identity=_fingerprint(cell)
        if self._identity!=cell.identity or self._snapshot!=_snapshot(cell) or self._methods!=_methods():
            raise ValueError('cell identity changed during capture')
        self._capture=(self._cell,self._snapshot,self._identity,self._methods);self._sealed=True

    def require(self,cell):
        if (cell is not self._cell or any(a is not b for a,b in zip(
                (self._cell,self._snapshot,self._identity,self._methods),self._capture))
                or _methods()!=self._methods or cell.identity!=self._identity):
            raise ValueError('captured cell authority changed')
        try:unchanged=_snapshot(cell)==self._snapshot
        except (TypeError,ValueError,AttributeError):unchanged=False
        if not unchanged and _fingerprint(cell)!=self._identity:
            raise ValueError('generalized cell authority changed')
        return self._identity
