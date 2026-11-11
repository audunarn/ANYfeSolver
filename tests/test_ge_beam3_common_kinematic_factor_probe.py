"""Small diagnostic: preserve a common kinematic map before constitutive rows."""
import json
from decimal import Decimal as D
import numpy as np
from scipy.linalg import block_diag
from docs.reference_cases.ge_beam3_common_kinematic_factor_probe import reference_chain
from docs.reference_cases.ge_beam3_decimal_chain_audit import factor_chain_modes
from docs.reference_cases.ge_beam3_reassembly_runtime_diagnostic import contrast
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
import anyfesolver._ge_beam3_reassembled_signed_modes as adapter
from anyfesolver._native_factor_chain_modes import solve_factor_chain_modes


def test_decimal_chain_preserves_unit_term_lost_by_expanded_binary64():
    # L R = [[1,1],[1e16,1e16+1]] in exact supplied data.
    left=[[1.,0.],[1e16,1.]]; right=[[1.,1.],[0.,1.]]
    row=factor_chain_modes(left,right,[[1.,0.],[0.,1.]],[0,1],[],digits=90)
    values=[D(x) for x in row['eigenvalues']]
    assert abs(values[0]/D('5e-33')-1)<D('1e-15')
    assert values[0]>0


def test_reference_factor_chain_covariance_for_both_coordinate_constructions(tmp_path,monkeypatch):
    original=adapter.solve_reassembled_factor_modes; captured={}; records=[]
    def capture(f,g,b,free,algebraic,**kwargs):
        captured.update(factor=f,geometric=g,kinetic=b,free=free,algebraic=algebraic)
        return original(f,g,b,free,algebraic,**kwargs)
    monkeypatch.setattr(adapter,'solve_reassembled_factor_modes',capture)
    for batched in (False,True):
        results=[]; variant='batched' if batched else 'per-node'
        for case,q in (('E',np.eye(3)),('GENERAL',rotation([.4,-.3,.2]))):
            print(variant+' '+case+' initialized',flush=True)
            model,inertias=contrast(1e6,q,batched)
            states={i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1)
                for i,e in model.mesh.elements.items()}
            packet,native=adapter.solve_signed_loaded_modes(model,states,np.zeros(30),inertias,np.zeros(30),
                load_parameter=0.,bounds=(-10000.,100000000.))
            lefts=[]; rights=[]
            for eid,internal in packet.internal_layout:
                e=model.mesh.elements[eid]
                decoded=e.validate_model_bound_nonlinear_state(model.mesh,e.core.section,states[eid],1)
                left,right=reference_chain(e,decoded['material_state'])
                slots=tuple(int(i) for i in e.get_dof_mapping(model.mesh))+internal
                full=np.zeros((18,42));full[:,slots]=right
                lefts.append(left);rights.append(full)
            left=block_diag(*lefts);right=np.vstack(rights)
            expanded=left@right
            relative=np.linalg.norm(expanded-captured['factor'])/np.linalg.norm(captured['factor'])
            assert relative<1e-11 and np.count_nonzero(captured['geometric'])==0
            with (tmp_path/(variant+'-'+case+'.npz')).open('xb') as stream:
                np.savez(stream,left=left,right=right,kinetic=captured['kinetic'],mass=packet.mass,
                    free=np.array(captured['free']),algebraic=np.array(captured['algebraic']),rotation=q,
                    expanded_old=captured['factor'],native_modes=native.full_modes)
            print(variant+' '+case+' Decimal chain started',flush=True)
            audit=factor_chain_modes(left.tolist(),right.tolist(),captured['kinetic'].tolist(),
                list(captured['free']),list(captured['algebraic']),digits=90)
            with (tmp_path/(variant+'-'+case+'-audit.json')).open('xb') as stream:stream.write(canonical(audit))
            modes=np.array(audit['full_modes'],dtype=float)[:,:6]
            roots=np.array(audit['eigenvalues'],dtype=float)[:6]
            chain_native=solve_factor_chain_modes(left,right,np.zeros((42,42)),captured['kinetic'],
                captured['free'],captured['algebraic'],bounds=(-10000.,100000000.))
            with (tmp_path/(variant+'-'+case+'-native.json')).open('xb') as stream:
                stream.write(canonical(chain_native))
            np.testing.assert_allclose(chain_native.eigenvalues,roots,rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(np.abs(chain_native.full_modes.T@packet.mass@modes),np.eye(6),rtol=1e-11,atol=1e-11)
            results.append((modes,roots,packet.mass,q,relative))
        a,b=results;transform=np.kron(np.eye(14),b[3])
        cross=(transform@a[0]).T@b[2]@b[0]
        records.append(dict(variant=variant,correlation_error=(np.abs(cross)-np.eye(6)),
            relative_roots=(b[1]-a[1])/a[1],expanded_factor_relative_differences=[a[4],b[4]],
            production_qualified=False))
    with (tmp_path/'diagnostic.json').open('xb') as stream:stream.write(canonical(records))
    print(canonical(records).decode(),flush=True)
    for record in records:
        np.testing.assert_allclose(record['correlation_error'],0.,rtol=0.,atol=1e-11)
        np.testing.assert_allclose(record['relative_roots'],0.,rtol=0.,atol=1e-11)
