"""Frozen generalized yielding: actual accepted flow and exact adaptive restart."""
from dataclasses import replace,fields
from pathlib import Path
from math import fsum
import numpy as np
import pytest
from anyfesolver import b3_ge,stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from test_stiffened_shell_native_point_displacement import model,owner,native_state,compare,OPTIONS,CONTROL,DIRECTION


def definition():
    base=model('generalized');beams=[]
    for native in base.beams:
        element,inertia=native.instantiate();reference=element.operator.reference
        section=b3_ge.EllipsoidalGeneralizedSection(element.section.elastic,element.section.metric,5.,1e6)
        beams.append(b3_ge.define_beam('b3-ge',element.element_id,element.node_ids,
            reference.coordinates,reference.nodal_triads,section,inertia,
            quadrature=element.operator.order))
    return replace(base,beams=tuple(beams))


def exact(a,b):
    compare(a,b)
    for left,right in zip(a.beam_generalized_material,b.beam_generalized_material,strict=True):
        assert left.element_id==right.element_id
        for ls,rs in zip(left.stations,right.stations,strict=True):
            for field in fields(ls):
                lv,rv=getattr(ls,field.name),getattr(rs,field.name)
                if isinstance(lv,np.ndarray):np.testing.assert_array_equal(lv,rv)
                else:assert lv==rv,field.name


def test_generalized_adaptive_signed_history_exact_restart(request):
    value=definition();first=owner(request,value);second=owner(request,value)
    # Actual recorded loading, strength5/hardening1e6; not a synthetic history.
    raw=(Path(__file__).parent/'data/native_generalized_adaptive_history/loaded-v1.bin').read_bytes()
    previous=first.restore(raw);other=second.restore(raw);exact(previous,other)
    assert first.checkpoint()==second.checkpoint()==raw
    nominal=previous.target
    kw=dict(OPTIONS,displacement_control=CONTROL,increment_control=ss.AdaptiveIncrementControl(6,24))
    token=CancellationToken();token.cancel('before generalized continuation')
    with pytest.raises(SolveCancelled):first.solve(DIRECTION,(.25*nominal,),cancellation_token=token,**kw)
    assert first.checkpoint()==raw
    for target in (.25*nominal,-nominal,1.1*nominal):
        before=native_state(first)
        current=first.solve(DIRECTION,(target,),**kw)
        repeated=second.solve(DIRECTION,(target,),**kw)
        exact(current,repeated);assert first.checkpoint()==second.checkpoint()
        after=native_state(first)
        for name in after:np.testing.assert_array_equal(after[name],native_state(second)[name])
        old=np.array([fsum(row) for row in before['beam_accumulated']])
        new=np.array([fsum(row) for row in after['beam_accumulated']])
        delta=np.array([fsum((*a,*(-b))) for a,b in zip(after['beam_accumulated'],before['beam_accumulated'],strict=True)])
        floor=64.*np.finfo(float).eps*np.maximum(abs(new),abs(old))
        assert np.all(delta>=-floor) and np.any(delta>floor)
        assert not np.array_equal(before['beam_plastic'],after['beam_plastic'])
        stations=[s for e in current.beam_generalized_material for s in e.stations]
        assert stations and not current.beam_fibre_material
        increments=[fsum(s.plastic_increment) for s in stations]
        density=[fsum(s.dissipation_per_length) for s in stations]
        assert any(i>0. for i in increments) and all(d>=0. for d in density)
        errors=[abs(d-5.*i)/max(abs(d),abs(5.*i),np.finfo(float).tiny) for d,i in zip(density,increments)]
        assert max(errors)<=1e-10
        assert current.free_equilibrium_ratio<=1e-8 and current.constraint_max<=1e-9
        assert not current.production_qualified
