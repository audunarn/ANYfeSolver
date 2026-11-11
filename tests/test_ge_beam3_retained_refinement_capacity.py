"""Real capacity and byte-exact old state/factor preservation, not qualification."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_translation_control as control
from anyfesolver import _ge_beam3_retained_translation_modal as modal
from anyfesolver._ge_beam3_retained_generalized_state import _retained_size
from anyfesolver._ge_beam3_native_generalized_program import model_identity,retained_model_identity
from anyfesolver._ge_beam3_native_generalized_restart import _model
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken,SolveCancelled
from docs.reference_cases import ge_beam3_refined_controlled_case as case
from docs.reference_cases import ge_beam3_retained_arch_case as old_case
from docs.reference_cases.ge_beam3_retained_prestress_protocol import strict_bytes
from docs.reference_cases.ge_beam3_refined_factor_partition import partition
from docs.reference_cases.ge_beam3_retained_arc_spectrum import planar_partition
from docs.reference_cases.ge_beam3_decimal_inertia_audit import audit

ARCHIVE=Path('C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-controlled-onset-bb40689-20260909')
MANIFEST='A7B049FCE2BD753DF1903678A2CA585319D449A6008DA5294007DE8B25DE2647'

def historical(name):
    raw=(ARCHIVE/'archive-manifest.json').read_bytes();assert sha256(raw).hexdigest().upper()==MANIFEST
    binding=strict_bytes(raw)[name];data=(ARCHIVE/name).read_bytes()
    assert [len(data),sha256(data).hexdigest().upper()]==binding;return data

@pytest.mark.parametrize('n',(2,4,8,12))
def test_old_geometry_identity(n):
    m=case.model(n);assert model_identity(m)==retained_model_identity(m)==model_identity(old_case.model(n))

@pytest.mark.parametrize('n',(16,20,24))
def test_explicit_capacity_route(n):
    m=case.model(n);parts=_model(m,retained=True)
    assert parts[-1]==retained_model_identity(m)
    assert _retained_size(parts[1],n)==36*n+6 and modal._factor_size(parts[1],n)==18*n+6
    if n>16:
        with pytest.raises(ValueError,match='bounded standalone'):_model(m)
    with pytest.raises(ValueError,match='exact retained'):_model(m,retained=1)
    m.materials.clear()
    with pytest.raises(ValueError,match='authority'):retained_model_identity(m)

@pytest.mark.parametrize('count',(0,25,100))
def test_oversize_before_operator_access(count):
    m=SimpleNamespace(mesh=SimpleNamespace(elements={i:object() for i in range(count)},nodes={},
        dof_manager=SimpleNamespace(total_dofs=0)))
    with pytest.raises(ValueError,match='bounded standalone'):retained_model_identity(m)

@pytest.mark.parametrize('nodal,elements',((True,2),(30,True),(0,1),(31,2),(516,1),(510,24),(30,25)))
def test_retained_size_rejects(nodal,elements):
    with pytest.raises(ValueError):_retained_size(nodal,elements)

@pytest.mark.parametrize('nodal,elements',((True,2),(30,True),(0,1),(31,2),(510,1),(30,25)))
def test_factor_size_rejects(nodal,elements):
    with pytest.raises(ValueError):modal._factor_size(nodal,elements)

def test_native_split_prefix_and_packet_equal_frozen_old_point(tmp_path):
    expected=historical('runs/rehearsal/n4/point-10/native/output/checkpoint.json')
    expected_packet=historical('runs/rehearsal/n4/point-10/native/output/packet.json')
    p=case.program(4,.03634765625);raw=None
    for k in range(1,5):
        print(dict(stage='same-programme-prefix',target=k),flush=True)
        kw={} if raw is None else dict(checkpoint=raw,expected_sha256=sha256(raw).hexdigest())
        result=control.solve(case.model(4),p,stop_after=k,**kw)
        assert result.status==('completed' if k==4 else 'paused') and result.completed_targets==k,result.failure
        raw=result.checkpoint;(tmp_path/f'prefix-{k}.json').write_bytes(raw)
    assert raw==expected
    m=case.model(4);packet,check=modal.prepare(m,p,raw,case.masses(m),expected_sha256=sha256(raw).hexdigest())
    check();assert canonical(packet)==expected_packet
    assert canonical(partition(packet,54,check))==canonical(planar_partition(packet,54,check))
    with pytest.raises(ValueError):control.solve(case.model(4),replace(p,targets=(.01,.02,.03,.04)),checkpoint=raw,expected_sha256=sha256(raw).hexdigest())
    (tmp_path/'packet.json').write_bytes(canonical(packet))

def test_upper_capacity_virgin_original_factors(tmp_path):
    print(dict(stage='n24-virgin-capture'),flush=True)
    m=case.model(24);p=case.program(24,.045);context=control.Context(m,p);raw=context.checkpoint(())
    assert context.physical.count==870
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):modal.prepare(m,p,raw,case.masses(m),expected_sha256=sha256(raw).hexdigest(),cancellation_token=token)
    packet,check=modal.prepare(m,p,raw,case.masses(m),expected_sha256=sha256(raw).hexdigest())
    check();assert packet.mass.shape==(438,438) and packet.completed_targets==0 and packet.parameter==0.
    groups=partition(packet,294,check);audits={}
    for name,f in groups.items():
        print(dict(stage='n24-independent-zero-audit',family=name),flush=True)
        rows=[audit(f['left'].tolist(),f['right'].tolist(),f['geometric'].tolist(),f['kinetic'].tolist(),f['free'],f['algebraic'],(0.,),digits=d) for d in (80,100)]
        assert all(a['rows'][0]['negative']==0 and a['trace_positive'] and a['mass_positive'] for a in rows)
        audits[name]=rows
    check();(tmp_path/'science.json').write_bytes(canonical(dict(packet=packet,audits=audits,checkpoint_sha256=sha256(raw).hexdigest(),production_qualified=False)))
    (tmp_path/'checkpoint.json').write_bytes(raw)
    m.mesh.nodes[1].x+=.01
    with pytest.raises(ValueError):check()

@pytest.mark.parametrize('which',('material','kinetic','geometric','layout','family-size'))
def test_partition_rejects_any_cross_factor(which):
    size=510 if which=='family-size' else 18;nodal=6 if which=='family-size' else 18
    left=np.eye(size);right=np.eye(size);g=np.zeros((size,size));b=np.eye(size)
    packet=SimpleNamespace(left=left,right=right,geometric=g,kinetic=b,free_dofs=tuple(range(size)),algebraic_dofs=())
    if which=='material':right[0,2]=1e-300
    elif which=='kinetic':b[0,2]=1e-300
    elif which=='geometric':g[0,2]=1e-300
    elif which=='layout':nodal=True
    with pytest.raises(ValueError):partition(packet,nodal)
