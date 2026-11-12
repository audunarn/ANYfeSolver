"""Experimental native BL5 fixed displacement caller and durable histories."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot, encode_snapshot, identities, MAGIC
from anyfesolver._stiffened_shell_displacement_control import publish
from anyfesolver._stiffened_shell_affine_endpoint import is_native_point_material, uses_native_material_history
from examples.stiffened_shell_native_fibre_bl5_displacement import ACTIONS, CONTROL, OPTIONS, TARGETS
from examples.stiffened_shell_native_fibre_bl5_material import definition


def owner(request, model=None):
    value = ss.create_analysis(definition=definition() if model is None else model)
    request.addfinalizer(value.close)
    return value


def maxalpha(rows):
    return max((max(row) for row in rows), default=0.)


@pytest.mark.parametrize('family,shell_material', [('biaxial', True), ('axial', False)])
def test_public_history_cancel_restore_unload(request, tmp_path, family, shell_material):
    model = definition(family=family, shell_material=shell_material)
    control = owner(request, model); fresh = owner(request, model)
    assert not is_native_point_material(control) and uses_native_material_history(control)
    seed = control.solve(ACTIONS, (.001,), **OPTIONS)
    token = CancellationToken(); observed = []
    def accepted(v):
        observed.append(v)
        if len(observed) == 2: token.cancel('preserve loaded prefix')
    with pytest.raises(SolveCancelled):
        control.solve(ACTIONS, TARGETS, displacement_control=CONTROL, cancellation_token=token,
                      on_accepted=accepted, **OPTIONS)
    assert [v.target for v in observed] == list(TARGETS[:2])
    assert maxalpha(observed[-1].beam_station_accumulated) > 0.
    if shell_material:
        assert maxalpha(observed[-1].shell_layer_alpha) > maxalpha(observed[0].shell_layer_alpha)
        assert maxalpha(observed[-1].beam_station_accumulated) > maxalpha(observed[0].beam_station_accumulated)
    assert observed[-1].beam_fibre_material and observed[-1].beam_generalized_material
    assert all(v.free_equilibrium_ratio <= 1e-8 and v.constraint_max <= 1e-9 for v in observed)
    saved = control.checkpoint(); (tmp_path/(family+'-loaded.bin')).write_bytes(saved)
    m,a = decode_snapshot(control, saved)
    assert identities(control) == ('finite_displacement_native_fibre_bl5', 'B3_GE_NATIVE_FIBRE_BL5_DISPLACEMENT_V1_DEV', None, None)
    assert m['control_dof'] == CONTROL.dof(control)
    assert all(not x.flags.writeable for x in a.values())
    restored = fresh.restore(saved)
    assert fresh.checkpoint() == saved
    assert restored.beam_station_accumulated == observed[-1].beam_station_accumulated
    assert restored.shell_layer_alpha == observed[-1].shell_layer_alpha
    final = control.solve(ACTIONS, TARGETS[2:], displacement_control=CONTROL, **OPTIONS)
    resumed = fresh.solve(ACTIONS, TARGETS[2:], displacement_control=CONTROL, **OPTIONS)
    assert control.checkpoint() == fresh.checkpoint()
    assert final.history == resumed.history and final.target == TARGETS[-1]
    np.testing.assert_array_equal(final.cell_resultants, resumed.cell_resultants)
    np.testing.assert_array_equal(final.support_reactions, resumed.support_reactions)
    assert control.recover().beam_fibre_material == final.beam_fibre_material


def test_refusals_and_restore_are_atomic(request):
    control = owner(request)
    with pytest.raises(ValueError, match='accepted force-controlled seed'):
        control.solve(ACTIONS, (TARGETS[0],), displacement_control=CONTROL, **OPTIONS)
    control.solve(ACTIONS, (.001,), **OPTIONS)
    seed = control.checkpoint()
    for kwargs in (dict(increment_control=ss.AdaptiveIncrementControl()),
                   dict(displacement_control=replace(CONTROL, trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL))):
        options = dict(displacement_control=CONTROL, **OPTIONS); options.update(kwargs)
        with pytest.raises(ValueError): control.solve(ACTIONS, (TARGETS[0],), **options)
        assert control.checkpoint() == seed
    with pytest.raises(ValueError):
        control.solve(ACTIONS, (TARGETS[0],), displacement_control=replace(CONTROL, component=3), **OPTIONS)
    control.solve(ACTIONS, (TARGETS[0],), displacement_control=CONTROL, **OPTIONS)
    saved = control.checkpoint(); generation = control.recover().generation
    token = CancellationToken(); token.cancel('before target')
    with pytest.raises(SolveCancelled):
        control.solve(ACTIONS, (TARGETS[1],), displacement_control=CONTROL, cancellation_token=token, **OPTIONS)
    assert control.checkpoint() == saved
    with pytest.raises(ValueError, match='match restart'):
        control.solve((ss.NodalAction('beam',105,(501.,0.,0.,0.,0.,0.)),), (TARGETS[1],), displacement_control=CONTROL, **OPTIONS)
    with pytest.raises(ValueError):
        control.solve(ACTIONS, (TARGETS[1],), displacement_control=replace(CONTROL, component=1), **OPTIONS)
    with pytest.raises(ValueError): control.rebase_rotations(generation)
    with pytest.raises(ValueError): control.rebase_rotations(generation-1)
    with pytest.raises(ValueError): publish(control,saved,expected_driver=control._finite_driver,expected_generation=generation-1)
    m,a=decode_snapshot(control,saved)
    meta={k:m[k] for k in ('generation','seed_generation','seed_sha256','control_dof')}
    for extra in (dict(chart_events=1),dict(trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL)):
        with pytest.raises(ValueError): encode_snapshot(control,arrays=a,**meta,**extra)
    for field in ('beam_fibres','beam_origin_fibres','applied_load'):
        changed={k:v.copy() for k,v in a.items()}; changed[field].flat[0] += 1e-5
        untouched=owner(request)
        with pytest.raises(ValueError): untouched.restore(encode_snapshot(control,arrays=changed,**meta))
        assert untouched._finite_driver is None and untouched._generation==0
        assert control.checkpoint() == saved
    start=len(MAGIC)+4; n=int.from_bytes(saved[len(MAGIC):start],'little')
    for patch in (dict(policy='B3_GE_NATIVE_POINT_COUPLED_J2_DISPLACEMENT_V1_DEV'),dict(chart_events=1)):
        header=json.loads(saved[start:start+n]);header.update(patch)
        data=json.dumps(header,sort_keys=True,separators=(',',':')).encode('ascii')
        body=MAGIC+len(data).to_bytes(4,'little')+data+saved[start+n:-32]
        untouched=owner(request)
        with pytest.raises(ValueError): untouched.restore(body+sha256(body).digest())
        assert untouched._finite_driver is None and untouched._generation==0
    from anyfesolver._stiffened_shell_displacement_adaptive import MAGICS
    for magic in MAGICS:
        untouched=owner(request)
        with pytest.raises(ValueError,match=r'^(?:bounded adaptive displacement checkpoint required|native material BL5 displacement excludes adaptive checkpoints)$'): untouched.restore(magic+saved)
        assert untouched._finite_driver is None and untouched._generation==0
    assert control.checkpoint() == saved


def test_pending_force_seed_refused_before_mutation(request):
    control=owner(request); token=CancellationToken()
    def accepted(v): token.cancel('pending force cursor')
    with pytest.raises(SolveCancelled):
        control.solve(ACTIONS,(.001,.002),increment_control=ss.AdaptiveIncrementControl(4,64),
                      on_accepted=accepted,cancellation_token=token,**OPTIONS)
    saved=control.checkpoint(); generation=control.recover().generation
    with pytest.raises(ValueError,match='ordinary unrebased force seed'):
        control.solve(ACTIONS,(TARGETS[0],),displacement_control=CONTROL,**OPTIONS)
    assert control.checkpoint()==saved and control.recover().generation==generation
    assert getattr(control,'_displacement_snapshot',None) is None


def test_force_seed_parity_and_fixed_origin_tangent(request):
    model=definition(shell_material=False)
    control=owner(request,model); force=owner(request,model)
    control.solve(ACTIONS,(.001,),**OPTIONS)
    force.solve(ACTIONS,(.001,),**OPTIONS)
    result=control.solve(ACTIONS,(TARGETS[0],),displacement_control=CONTROL,**OPTIONS)
    reference=force.solve(ACTIONS,(.001+result.amplitude,),**OPTIONS)
    assert np.linalg.norm(result.applied_dead_load-(.001+result.amplitude)*control._finite_driver._load(ACTIONS))==0.
    np.testing.assert_allclose(result.cell_resultants,reference.cell_resultants,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(result.beam_positions,reference.beam_positions,rtol=0.,atol=1e-10)
    driver=control._finite_driver; state=driver._state; shell=driver._shell_predecessors; beam=driver._beam_predecessors
    base=driver._assemble(state,0.,shell,beam,applied_load=result.applied_dead_load)
    size=base[1].shape[0]; direction=np.zeros(size)
    direction[control._free]=np.linspace(-1.,1.,len(control._free))*1e-6
    direction[len(state.q):]=np.linspace(-1.,1.,len(state.multipliers))
    # Native retained resultants are separate from additive q coordinates.
    for j in range(len(control._beam_parts)):
        offset=control._shell_count+6*len(control._beam_ids)+24*j
        direction[offset+6:offset+24]=np.linspace(-1.,1.,18)
    exact=base[1]@direction; errors=[]
    for h in (1e-5,1e-6):
        plus=driver._advance(state,direction,h); minus=driver._advance(state,direction,-h)
        rp=driver._assemble(plus,0.,shell,beam,applied_load=result.applied_dead_load)[0]
        rm=driver._assemble(minus,0.,shell,beam,applied_load=result.applied_dead_load)[0]
        errors.append(float(np.linalg.norm((rp-rm)/(2*h)-exact)/max(np.linalg.norm(exact),1.)))
    assert max(errors)<=1e-5,errors


def test_fixed_origin_material_tangent(request,tmp_path):
    control=owner(request,definition(shell_material=False))
    control.solve(ACTIONS,(.001,),**OPTIONS)
    result=control.solve(ACTIONS,(TARGETS[0],),displacement_control=CONTROL,**OPTIONS)
    driver=control._finite_driver; state=driver._state; shell=driver._shell_predecessors; beam=driver._beam_predecessors
    base=driver._assemble(state,0.,shell,beam,applied_load=result.applied_dead_load)
    size=base[1].shape[0]; direction=np.zeros(size)
    direction[control._free]=np.linspace(-1.,1.,len(control._free))*1e-6
    direction[len(state.q):]=np.linspace(-1.,1.,len(state.multipliers))
    # Native retained resultants are separate from additive q coordinates.
    for j in range(len(control._beam_parts)):
        offset=control._shell_count+6*len(control._beam_ids)+24*j
        direction[offset+6:offset+24]=np.linspace(-1.,1.,18)
    from anyfesolver._stiffened_shell_ge_finite_experimental import _fibre_recovery
    def branches(st,ev):
        records,_=_fibre_recovery(control,st,beam,ev[5])
        return tuple((f.branch,f.algorithmic_modulus) for row in records for f in row.fibres)
    labels=branches(state,base)
    exact=base[1]@direction; errors=[]; stencil=[]
    saved=control.checkpoint()

    for h in (1e-5,1e-6):
        plus=driver._advance(state,direction,h); minus=driver._advance(state,direction,-h)
        ep=driver._assemble(plus,0.,shell,beam,applied_load=result.applied_dead_load)
        lp=branches(plus,ep)
        em=driver._assemble(minus,0.,shell,beam,applied_load=result.applied_dead_load)
        lm=branches(minus,em)
        assert lp==labels and lm==labels
        stencil.append(dict(h=h,plus=lp,minus=lm))
        rp,rm=ep[0],em[0]
        errors.append(float(np.linalg.norm((rp-rm)/(2*h)-exact)/max(np.linalg.norm(exact),1.)))
    (tmp_path/'fixed-origin-tangent.json').write_text(json.dumps(dict(errors=errors,h=(1e-5,1e-6),criterion=1e-5,
        norm_floor=1.,norm='all-row unweighted relative L2',direction=direction.tolist(),labels=labels,stencil=stencil),indent=2))
    assert control.checkpoint()==saved
    assert max(errors)<=1e-5,errors


def test_legacy_elastic_native_bl5_remains_refused(request):
    from examples.stiffened_shell_native_fibre_bl5 import definition as elastic
    control=owner(request,elastic())
    assert not uses_native_material_history(control)
    assert 'displacement_control' not in control.capabilities
    with pytest.raises(ValueError):
        control.solve(ACTIONS,(TARGETS[0],),displacement_control=CONTROL,**OPTIONS)
    assert control._generation==0 and control._finite_driver is None
