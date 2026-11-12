"""New ordinary-domain fixtures/oracles; no closed quiet-case data access."""
from pathlib import Path
from dataclasses import replace
from decimal import Decimal,localcontext
import json,os,numpy as np

def coefficient_oracle(high,low=0.,*,decimal=False):
    # Independent defining-series solution of the scalar Log ODE.
    with localcontext() as ctx:
        ctx.prec=80;c=Decimal.from_float(float(high))+Decimal.from_float(float(low));d=1-c
        b=Decimal(1);power=Decimal(1);f=Decimal(1);first=Decimal(0);second=Decimal(0)
        for n in range(1,4097):
            b=b*Decimal(n)/Decimal(2*n+1);power*=d;term=b*power
            f+=term;first-=n*term/d;second+=n*(n-1)*term/(d*d)
        return (f,first,second) if decimal else tuple(float(v) for v in (f,first,second))

def definition():
    from anyfesolver._stiffened_shell_ge_experimental import PROFILE,LinearDefinition,ShellPanel,PointConnection,FixedCoordinate
    from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY
    from anyfesolver._stiffened_shell_q4_algorithmic_pullback import POLICY as Q4_POLICY
    section=EllipsoidalGeneralizedSection(np.diag((1e4,4e3,4e3,100.,100.,100.)),np.eye(6),2.,1000.)
    ref=CenteredCurvedBeam3ReferenceGeometry(np.array([[0.,-.5,.1],[.5,-.5,.1],[1.,-.5,.1]]),np.tile(np.eye(3),(3,1,1)))
    element=NativeGeneralizedStaticElement(1,(101,102,103),ref,section,order=4,arithmetic_policy=POLICY)
    beam=NativeBeamDefinition.capture(element,np.eye(6))
    shell=ShellPanel((1,2,3,4),((0.,-.5,0.),(1.,-.5,0.),(1.,.5,0.),(0.,.5,0.)),
        ((1,2,3,4),),.07,1e6,.3,1000.)
    return LinearDefinition(PROFILE,shell,(beam,),(PointConnection(1,101),PointConnection(2,103)),
        tuple(FixedCoordinate('shell',n,k) for n in (1,4) for k in range(6)),q4_pullback_policy=Q4_POLICY)

def actions():
    from anyfesolver._stiffened_shell_ge_experimental import NodalAction
    return (NodalAction('beam',103,(0.,0.,10.,0.,0.,0.)),)

def public_route(out):
    from anyfesolver import stiffened_shell as ss
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver._stiffened_shell_beam_history import arrays
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    import anyfesolver._stiffened_shell_f06_increment_arithmetic as arithmetic
    out=Path(out);owner=ss.create_analysis(definition=definition());calls=[];original=arithmetic.full_log_coefficients
    def capture(c):
        calls.append(c.scalar());return original(c)
    arithmetic.full_log_coefficients=capture
    try:
        reused=os.environ.get('F06_LOG_REUSE')
        result=(owner.restore(Path(reused).read_bytes()) if reused else
                owner.solve(actions(),(.4,.8),max_iterations=16,max_backtracks=8))
        driver=owner._finite_driver;state=driver._state
        assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
        assert calls and min(calls)<1.-1e-7,'ordinary Log never consumed'
        assert max(sum(st.accumulated) for h in driver._beam_origins for st in h.stations)>0.,'native material inactive'
        assert any(v[1]!=0. for h in driver._beam_origins for st in h.stations for v in st.plastic)
        accepted=canonical(driver._beam_origins);predecessor=canonical(driver._beam_predecessors)
        assert predecessor!=accepted
        recovery=owner.recover();assert recovery.beam_generalized_material
        immediate=driver._assemble(state,driver._load_factor,driver._shell_origins,driver._beam_origins)
        assert immediate[2]<=1e-8 and immediate[3]<=1e-9
        raw=owner.checkpoint();(out/'accepted-active.bin').write_bytes(raw)
        # New schema is binary-only; failed fresh restore must not publish state.
        from anyfesolver._stiffened_shell_ge_checkpoint_binary import SCHEMAS,decode_compensated,encode_compensated,decode,encode
        from anyfesolver._stiffened_shell_beam_history import shapes
        from anyfesolver._stiffened_shell_f06_increment_arithmetic import Pair
        corrections_shapes=tuple((n,getattr(state,n).shape) for n in ('q_low','frames_low','cells_low','resultants_low'))
        inner,corrections=decode_compensated(raw,owner._identity,corrections_shapes)
        base_shapes=tuple((n,v.shape) for n,v in (('q',state.q),('frames',state.frames),('cells',state.cells),
            ('resultants',state.resultants),('multipliers',state.multipliers),('pattern',driver._pattern),
            ('accepted_factors',np.asarray(driver._accepted_factors))))
        kind,generation,values=decode(inner,owner._identity,lambda k,g:base_shapes+shapes(owner._beam_parts,'origin')+shapes(owner._beam_parts,'accepted'))
        assert kind=='finite_compensated_generalized_material'
        for name in ('origin_accumulated','accepted_accumulated'):
            changed={k:v.copy() for k,v in values.items()}
            pair=Pair(*changed[name][0])+1e-4
            changed[name][0]=(pair.high,pair.low)
            tampered=encode_compensated(encode(kind,owner._identity,generation,tuple(changed.items())),owner._identity,corrections)
            refused=ss.create_analysis(definition=definition())
            try:
                with __import__('pytest').raises(ValueError):refused.restore(tampered)
                assert refused._finite_driver is None and refused._accepted is None
            finally:refused.close()
        refused=ss.create_analysis(definition=definition())
        try:
            with __import__('pytest').raises(ValueError,match='binary framing'):
                refused.restore(encode_compensated(json.dumps(dict(schema=SCHEMAS['finite_compensated_generalized_material'])).encode(),owner._identity,corrections))
            assert refused._finite_driver is None and refused._accepted is None
        finally:refused.close()
        fresh=ss.create_analysis(definition=definition())
        try:
            fresh.restore(raw);assert fresh.checkpoint()==raw
            fd=fresh._finite_driver
            assert canonical(fd._beam_origins)==accepted and canonical(fd._beam_predecessors)==predecessor
            for name in ('q','frames','cells','resultants','q_low','frames_low','cells_low','resultants_low'):
                np.testing.assert_array_equal(getattr(state,name),getattr(fd._state,name))
        finally:fresh.close()
        lows={n:getattr(state,n).copy() for n in ('q_low','frames_low','cells_low','resultants_low')}
        owner.rebase_rotations(owner._generation)
        assert canonical(driver._beam_origins)==accepted and canonical(driver._beam_predecessors)==predecessor
        for n,v in lows.items():np.testing.assert_array_equal(getattr(driver._state,n),v)
        rebased=owner.checkpoint();fresh=ss.create_analysis(definition=definition())
        try:fresh.restore(rebased);assert fresh.checkpoint()==rebased
        finally:fresh.close()
        # Next real increment starts from current accepted histories, not the rebase replay origin.
        committed=driver._beam_origins
        continued=owner.solve(actions(),(.81,),max_iterations=16,max_backtracks=8)
        assert driver._beam_predecessors is committed
        assert continued.accepted_factors[-1]==.81 and continued.free_equilibrium_ratio<=1e-8
        assert continued.beam_generalized_material
        after_update=owner.checkpoint();fresh=ss.create_analysis(definition=definition())
        try:
            fresh.restore(after_update);assert fresh.checkpoint()==after_update
            assert canonical(fresh._finite_driver._beam_origins)==canonical(driver._beam_origins)
            assert canonical(fresh._finite_driver._beam_predecessors)==canonical(committed)
        finally:fresh.close()
        class CancelCandidate(CancellationToken):
            visits=0
            def raise_if_cancelled(self,stage=''):
                if stage=='stiffened-shell.finite.trial':
                    self.visits+=1
                    if self.visits==2:self.cancel('ordinary active candidate cancellation')
                super().raise_if_cancelled(stage)
        cancel=CancelCandidate()
        with __import__('pytest').raises(SolveCancelled):owner.solve(actions(),(.82,),cancellation_token=cancel)
        assert cancel.visits==2 and owner.checkpoint()==after_update
        # A deliberately bounded rejected increment preserves the accepted prefix.
        from anyfesolver._stiffened_shell_increment_control import FiniteIncrementFailure
        with __import__('pytest').raises(FiniteIncrementFailure) as rejected:
            owner.solve(actions(),(1.2,),max_iterations=1,max_backtracks=0)
        assert rejected.value.attempt.status=='rejected' and rejected.value.attempt.accepted_from==.81
        assert rejected.value.attempt.target==1.2 and rejected.value.attempt.failure
        assert owner.checkpoint()==after_update
        metrics=dict(status='PASS',reused_new_fixture=reused,ordinary_coefficient_calls=len(calls),min_cosine=min(calls),
            max_accumulated=max(sum(s.accumulated) for h in driver._beam_origins for s in h.stations),
            equilibrium=result.free_equilibrium_ratio,gap=result.constraint_max,accepted_factors=list(result.accepted_factors),
            continued_equilibrium=continued.free_equilibrium_ratio,continued_gap=continued.constraint_max,
            continued_factors=list(continued.accepted_factors),history_tamper_refused=('origin','accepted'),binary_only_refused=True,
            exact_accepted_and_predecessor_restore=True,rebase_cancel_rejection=True)
        (out/'public-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        return metrics
    finally:arithmetic.full_log_coefficients=original;owner.close()
