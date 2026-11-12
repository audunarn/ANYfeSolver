"""Refactor parity; scalar-law/mechanical oracles remain in the phase corpus."""
from fractions import Fraction as Q
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver.general_contact import s3_pressure_hermite_bounds as module


class RecomputedPartition:
    """Previous exact list algorithm, isolated from production partition logic."""
    def __init__(self,cells,count,maximum_depth):
        self.cells=list(cells);self.count=count;self.maximum_depth=maximum_depth

    @property
    def totals(self):return module._aggregate(self.cells,self.count)

    def pop(self,violations=None):
        choices=[i for i,cell in enumerate(self.cells)
                 if cell[3][3] and cell[2]<self.maximum_depth
                 and (violations is None or any(
                     index in violations and
                     cell[3][2 if violations[index] else 1][local][0]
                         <cell[3][2 if violations[index] else 1][local][1]
                     for local,index in enumerate(cell[0].ids)))]
        if not choices:return None
        return self.cells.pop(max(choices,key=lambda i:module._fraction(self.cells[i][1])))

    def add(self,cell):self.cells.append(cell)


def synthetic(ids,vertices,depth,eligible):
    # Include cancellation, signed values, tiny positive support and shared IDs.
    area=module._fraction(vertices)
    values=tuple((area*Q(i+1,10**20),area*Q(i+2,10**20)) for i in range(3))
    opens=tuple((area*Q(-i-1,13),area*Q(i+1,17)) for i in range(3))
    closed=tuple((area*Q(-i-3,19),area*Q(i-2,23)) for i in range(3))
    return (SimpleNamespace(ids=ids),vertices,depth,(values,opens,closed,eligible))


def test_shared_modes_totals_and_stable_ties_match_after_every_replacement():
    cells=[synthetic(ids,module._ROOT,0,eligible)
           for ids,eligible in [((0,1,2),True),((2,3,0),True),((1,3,2),False)]]
    fast=module._SpatialPartition(cells,4,4);slow=RecomputedPartition(cells,4,4)
    for iteration in range(120):
        assert fast.totals==slow.totals
        a,b=fast.pop(),slow.pop()
        assert a==b
        if a is None:break
        for index,child in enumerate(module._children(a[1])):
            cell=synthetic(a[0].ids,child,a[2]+1,index!=2)
            fast.add(cell);slow.add(cell)
    assert fast.totals==slow.totals


def test_depth_exhaustion_keeps_all_nonrefinable_cell_contributions():
    cells=[synthetic((0,1,2),v,1,True) for v in module._children(module._ROOT)]
    fast=module._SpatialPartition(cells,3,1)
    assert fast.pop() is None
    assert fast.totals==module._aggregate(cells,3)
    assert all(v[0]>0 for v in fast.totals[0])


def test_pop_refuses_cells_that_cannot_tighten_the_violated_inequality():
    # Production defers eligible cells whose failed-side bound interval is
    # already empty; the reference must refuse them identically and keep
    # every deferred contribution in the totals.
    def cell(ids,vertices,opens,closed):
        return (SimpleNamespace(ids=ids),vertices,0,
                (((Q(1),Q(2)),)*3,opens,closed,True))
    live=((Q(-1),Q(1)),(Q(-2),Q(2)),(Q(-3),Q(3)))
    flat=tuple((Q(n),Q(n)) for n in (5,6,7))
    blocked=cell((0,1,2),module._ROOT,live,flat)
    usable=cell((0,1,2),module._children(module._ROOT)[0],live,live)
    fast=module._SpatialPartition([blocked,usable],3,4)
    slow=RecomputedPartition([blocked,usable],3,4)
    assert fast.totals==slow.totals
    a,b=fast.pop({0:True}),slow.pop({0:True})
    assert a==b and a is usable
    assert fast.totals==slow.totals==module._aggregate([blocked],3)
    assert fast.pop({0:True}) is None and slow.pop({0:True}) is None
    assert fast.totals==slow.totals==module._aggregate([blocked],3)
    blocked_opens=cell((0,1,2),module._ROOT,flat,live)
    fast2=module._SpatialPartition([blocked_opens],3,4)
    slow2=RecomputedPartition([blocked_opens],3,4)
    assert fast2.pop({0:False}) is None and slow2.pop({0:False}) is None
    assert fast2.totals==slow2.totals==module._aggregate([blocked_opens],3)


@pytest.mark.parametrize('case',('mixed','tiny','hidden','cancel'))
def test_complete_proof_matches_recomputed_algorithm(monkeypatch,case):
    source=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    target=np.array([[-2.,-2.,0.],[.5,-2.,0.],[.5,3.,0.]])
    target[:,2]=.0003-.001*target[:,0]
    zeros=np.zeros((3,3));v0=zeros.copy();v1=zeros.copy()
    phase=np.array([True,False,True]);tau=1.e-6
    if case in ('tiny','cancel'):
        target=np.array([[0.,0.,.001],[1.e-12,0.,.001],[0.,1.e-12,.001]])
        phase[:]=False;tau=1.e-10
    if case=='hidden':
        target=np.array([[-2.,-2.,-.001],[5.,-2.,-.001],[-2.,5.,-.001]])
        v0[:,2]=-.012;v1[:,2]=.012;phase[:]=False
    def run():
        queries=0
        def cancel():
            nonlocal queries
            queries+=1
            return case=='cancel' and queries==9
        result=module.s3_pressure_hermite_bounds(source,target,((0,1,2),),
            source,source,target,target,v0,v1,zeros,zeros,1.,active=phase,
            gap_tolerance_m=tau,maximum_spatial_cells=512,maximum_time_panels=16,
            maximum_depth=4,cancel=cancel)
        return result,queries
    fast=run()
    monkeypatch.setattr(module,'_SpatialPartition',RecomputedPartition)
    slow=run()
    assert fast==slow
    if case=='cancel':assert fast[0].status=='cancelled'
    elif case=='tiny':assert not fast[0].success
