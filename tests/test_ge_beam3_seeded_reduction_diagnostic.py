"""Diagnostic-only separation of kinetic whitening and strain-factor errors."""
import numpy as np
from test_ge_beam3_curved_contrast_probe import make_curved, ROTATION
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_seeded_loaded_modal import prepare
from anyfesolver._ge_beam3_seeded_signed_modes import _split
from anyfesolver._native_signed_factor_modes import _reduce
from anyfesolver._native_relative_reference_factor_spectrum import solve_relative_reference_factor_spectrum
from anyfesolver._ge_beam3_p5_centered.mass import reference_kinetic_factors


def test_capture_reference_factor_reduction(tmp_path):
    for case, rotation in [('E', np.eye(3)), ('R90', ROTATION)]:
        model, inertia = make_curved(1e6, rotation)
        states = {i:e.init_model_bound_nonlinear_state(model.mesh,e.core.section,1)
            for i,e in model.mesh.elements.items()}
        packet, _ = prepare(model, states, np.zeros(30), inertia, load_parameter=0.)
        rows=[]; kinetic=[]; reference=[]; g=np.zeros((42,42))
        for i, internal in packet.internal_layout:
            e=model.mesh.elements[i]; slots=tuple(e.get_dof_mapping(model.mesh))+internal
            f, local_g = _split(e, states[i]['material_state'], lambda _:None)
            g[np.ix_(slots,slots)] += local_g
            factors=reference_kinetic_factors(e.core.reference,e.core.section._elastic,inertia[i],order=e.core.order)
            for target, local in [(rows,f),(kinetic,factors.full),(reference,factors.uncondensed_stiffness_factor)]:
                row=np.zeros((len(local),42)); row[:,slots]=local; target.append(row)
        f=np.vstack(rows); b=np.vstack(kinetic); ref=np.vstack(reference)
        assert np.count_nonzero(g)==0  # Reference-only experiment; no prestress discarded.
        results={}
        for name, factor in [('loaded_factor',f),('reference_factor',ref)]:
            d, _, _ = _reduce(factor,g,packet.mass,packet.free_dofs,packet.algebraic_dofs,lambda _:None)
            other=solve_relative_reference_factor_spectrum(factor,b,packet.free_dofs,packet.algebraic_dofs,num_modes=6)
            results[name]=dict(dense_mass=np.sort(d)[:6],kinetic_qr=other.eigenvalues)
        with (tmp_path/(case+'.json')).open('xb') as out:
            out.write(canonical(dict(results=results,production_qualified=False)))
        # Raw matrices are diagnostics, never canonical qualification evidence.
        np.savez(tmp_path/(case+'.npz'),factor=f,kinetic=b,reference=ref,mass=packet.mass,
            free=packet.free_dofs,algebraic=packet.algebraic_dofs)
