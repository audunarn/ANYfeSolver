"""Live-byte and authority mutation checks for the guard optimization."""
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_generalized_cell as source
from anyfesolver import _ge_beam3_cell_identity as identity
from anyfesolver import _ge_beam3_retained_translation_control as control
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases import ge_beam3_refined_controlled_case as case

@pytest.fixture
def cell():return case.model(2).mesh.elements[1].operator.cell

def test_original_identity_and_no_reencoding(cell,monkeypatch):
    assert identity._fingerprint(cell)==cell.identity
    def forbidden(*a,**kw):raise AssertionError('unchanged cell must not re-encode')
    monkeypatch.setattr(identity,'_fingerprint',forbidden)
    for _ in range(20):cell.guard()

@pytest.mark.parametrize('kind',('capture','identity','section','stations','constraint','policy','canonical','binding','snapshot','methods'))
def test_mutation_rejected(cell,kind,monkeypatch):
    if kind=='capture':
        data=json.loads(cell._capture);data[0]['weight']*=2;object.__setattr__(cell,'_capture',source.canonical(data))
    elif kind=='identity':object.__setattr__(cell,'identity','0'*64)
    elif kind=='section':object.__setattr__(cell.section,'yield_force',cell.section.yield_force*2)
    elif kind in ('stations','constraint'):object.__setattr__(cell,kind,())
    elif kind=='policy':monkeypatch.setattr(source,'POLICY','changed')
    elif kind=='canonical':monkeypatch.setattr(source,'canonical',lambda x:b'{}')
    elif kind=='binding':object.__setattr__(cell._identity_binding,'_identity','0'*64)
    elif kind=='snapshot':object.__setattr__(cell._identity_binding,'_snapshot',())
    else:object.__setattr__(cell._identity_binding,'_methods',())
    with pytest.raises(ValueError):cell.guard()

def test_original_canonical_fallback_without_adoption(cell):
    binding=cell._identity_binding;old=binding._snapshot
    object.__setattr__(cell,'_capture',b' '+cell._capture+b' ')
    cell.guard();assert binding._snapshot is old and identity._fingerprint(cell)==cell.identity
    data=json.loads(cell._capture);data[0]['weight']*=2;object.__setattr__(cell,'_capture',source.canonical(data))
    with pytest.raises(ValueError):cell.guard()

def test_foreign_cell_and_sealed_binding(cell):
    with pytest.raises(ValueError):cell._identity_binding.require(case.model(2).mesh.elements[1].operator.cell)
    with pytest.raises(AttributeError):cell._identity_binding._identity='0'*64

def test_expected_snapshot_still_protects_mutated_owner_table():
    m=case.model(4);p=case.program(4,.045);c=control.Context(m,p);s=c.initial;expected=canonical(s)
    raw=c._require_issued(s,expected_snapshot=expected)
    object.__setattr__(s,'parameter',.1);entry=c._issued[id(s)]
    c._issued[id(s)]=(s,canonical(s),entry[2])
    with pytest.raises(ValueError):c._require_issued(s,expected_snapshot=expected)
    with pytest.raises(ValueError):c._require_issued(s,expected_snapshot={})
    assert raw==entry[2]
