"""Authentic full-amplitude public single-native-beam vibration development CLI."""
import argparse
import math
import os
import sys
import json
from pathlib import Path
from time import perf_counter
from hashlib import sha256
from fractions import Fraction
import numpy as np
from anyfesolver.ge_beam3_native import define_beam, SELECTOR as STATIC_SELECTOR
from anyfesolver.ge_beam3_native_dynamics import create_analysis, SELECTOR, PROFILE_BYTES
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from examples import native_vibration_physical_reference as physical

def fixture():
    x = np.array((-1., 0., 1.))
    points = np.column_stack((x, .2*(1-x*x), np.zeros(3)))
    # Fixed approved binary64 orientation INPUTS; public operators remain computed.
    frames = np.array((
        ((float.fromhex('0x1.db614bfce6a6bp-1'), float.fromhex('0x0.0p+0'), float.fromhex('0x1.7c4dd663ebb89p-2')),
         (float.fromhex('0x1.7c4dd663ebb89p-2'), float.fromhex('0x0.0p+0'), float.fromhex('-0x1.db614bfce6a6bp-1')),
         (float.fromhex('0x0.0p+0'), float.fromhex('0x1.0000000000000p+0'), float.fromhex('0x0.0p+0'))),
        ((float.fromhex('0x1.0000000000000p+0'), float.fromhex('0x0.0p+0'), float.fromhex('-0x0.0p+0')),
         (float.fromhex('-0x0.0p+0'), float.fromhex('0x0.0p+0'), float.fromhex('-0x1.0000000000000p+0')),
         (float.fromhex('0x0.0p+0'), float.fromhex('0x1.0000000000000p+0'), float.fromhex('0x0.0p+0'))),
        ((float.fromhex('0x1.db614bfce6a6bp-1'), float.fromhex('0x0.0p+0'), float.fromhex('-0x1.7c4dd663ebb89p-2')),
         (float.fromhex('-0x1.7c4dd663ebb89p-2'), float.fromhex('0x0.0p+0'), float.fromhex('-0x1.db614bfce6a6bp-1')),
         (float.fromhex('0x0.0p+0'), float.fromhex('0x1.0000000000000p+0'), float.fromhex('0x0.0p+0'))),
    ))
    rho = 100.
    ea = 3*rho*rho
    d = np.diag(np.sqrt((ea, ea/3, ea/3.5, 2., 1., 1.5)))
    b = np.eye(6)
    b[0, 3], b[1, 5] = .02, .03
    f = b@d
    section = EllipsoidalGeneralizedSection(f.T@f, np.eye(6), 1e6, 1.)
    mass = 2.7
    center = np.array((.035, -.025, .012))
    a, bb, c = center
    skew = np.array(((0., -c, bb), (c, 0., -a), (-bb, a, 0.)))
    shift = -mass*(skew@skew)
    shift = np.triu(shift)+np.triu(shift, 1).T
    inertia = np.block([[mass*np.eye(3), -mass*skew],
        [mass*skew, np.diag((.12, .17, .19))+shift]])
    definition = define_beam(STATIC_SELECTOR, 1, (1, 2, 3), points, np.array(frames),
        section, inertia, quadrature=4)
    return points, np.array(frames), section, inertia, definition


APPROVAL_SHA256='50b66f7349e6179082cdaa0b88816bfbb00d8d142f178b60b0a65e2812418d7f'
APPROVAL_PATH=Path(__file__).with_name('native_vibration_energy_acceptance_20261018.json')


class EnergyAcceptanceError(ValueError):
    """The named human exception is unavailable; no scientific gate is relaxed."""


def _approval_record(raw):
    if type(raw)is not bytes or sha256(raw).hexdigest()!=APPROVAL_SHA256:
        raise EnergyAcceptanceError('approval bytes do not match immutable human record')
    record=json.loads(raw)
    if record['schema']!='ANY_NATIVE_VIBRATION_MEASURED_EXCEPTION_V1' or record['status']!='HUMAN_APPROVED_MEASURED_RESULT_ONLY':
        raise EnergyAcceptanceError('wrong approval schema/status')
    return record


def _validate_acceptance_scope(actual,record):
    """Pure admission of measured runtime fields against the pinned capsule."""
    expected=record['observed_runtime_profile']
    def pool_rows(rows):return sorted(rows,key=lambda row:row['filepath'])
    checks=dict(platform=actual['platform']==expected['platform'],versions=actual['versions']==expected['versions'],
        pools=pool_rows(actual['pools'])==pool_rows(expected['pools']) and bool(actual['pools']) and all(p['num_threads']==1 for p in actual['pools']),
        settings=actual['settings']==expected['settings'],hash0=actual['hash0']is True,
        assertions=actual['assertions']is True,jit_off=actual['jit_off']is True,
        numba_threads=actual['numba_threads']is True,natural_deadline=actual['natural_deadline']is True,
        default_cpu=actual['default_cpu']is True,no_injected_plugins=actual['no_injected_plugins']is True,
        product=actual['product_python_manifest_sha256']==record['scope']['product_python_manifest_sha256'])
    if not all(checks.values()):raise EnergyAcceptanceError('observed source/host/runtime profile differs: '+','.join(k for k,v in checks.items()if not v))
    return actual


def _measured_acceptance_scope(record):
    """Measure actual package and runtime; no historical workspace file is needed."""
    import anyfesolver
    import numba
    import scipy
    import threadpoolctl
    parallel=sys.modules.get('numba.np.ufunc.parallel')
    active=parallel is not None and vars(parallel).get('_is_initialized')is True
    package=Path(anyfesolver.__file__).resolve().parent
    files={'anyfesolver/'+p.relative_to(package).as_posix():sha256(p.read_bytes()).hexdigest()
        for p in package.rglob('*.py')if '__pycache__'not in p.parts}
    digest=sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    actual=dict(platform=sys.platform,
        versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),
        pools=threadpoolctl.threadpool_info(),
        settings={k:os.environ.get(k)for k in record['observed_runtime_profile']['settings']},
        hash0=sys.flags.hash_randomization==0,assertions=not sys.flags.optimize,
        jit_off=bool(numba.config.DISABLE_JIT),
        numba_threads=numba.config.NUMBA_NUM_THREADS==1 and(not active or vars(parallel)['_get_num_threads']()==1),
        natural_deadline=os.environ.get('ANYSOLVER_GE_BEAM3_DEADLINE_SCALE')is None,
        default_cpu=all(os.environ.get(k)is None for k in ('NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES','OPENBLAS_CORETYPE','MKL_DEBUG_CPU_TYPE')),
        no_injected_plugins=os.environ.get('PYTEST_PLUGINS')is None and os.environ.get('PYTEST_ADDOPTS')is None,
        product_python_manifest_sha256=digest)
    return _validate_acceptance_scope(actual,record)


class _EnergyAcceptance:
    """One pinned measured exception per complete qualification run, never a limit."""
    def __init__(self):
        self._uses=0

    def check(self,error,facts):
        if type(error)is not float or not math.isfinite(error)or error<0.:
            raise EnergyAcceptanceError('finite nonnegative binary64 step energy required')
        if error<=1e-6:
            return dict(status='STRICT_ORIGINAL_STEP_ENERGY_PASS',exception_used=False)
        target=type(facts)is dict and type(facts.get('label'))is str and type(facts.get('epoch'))is int and facts['label']=='coarse' and facts['epoch']==2
        if not target:raise EnergyAcceptanceError('original step-energy gate failed outside named exception')
        if self._uses!=0:raise EnergyAcceptanceError('named exception already consumed')
        record=_approval_record(APPROVAL_PATH.read_bytes())
        expected=record['binding']
        if set(facts)!=set(expected)or any(type(facts[k])is not type(expected[k])or facts[k]!=expected[k]for k in expected):
            raise EnergyAcceptanceError('case/state/energy-scale binding differs from approval')
        if error.hex()!=record['accepted_measurement']['normalized_step_energy_hex']:
            raise EnergyAcceptanceError('step-energy float bits differ from measured acceptance')
        scope=_measured_acceptance_scope(record)
        self._uses=1
        return dict(status='HUMAN_APPROVED_MEASURED_EXCEPTION_USED',exception_used=True,
            approval_id=record['id'],approval_sha256=APPROVAL_SHA256,normalized_step_energy_hex=error.hex(),
            binding=dict(facts),scope=scope)

    def finish(self):
        if type(self._uses)is not int or self._uses not in (0,1):raise EnergyAcceptanceError('invalid exception usage ledger')
        if self._uses==0:
            return dict(exception_uses=0,strict_original_energy_pass=True,outcome='STRICT_ORIGINAL_GATES_PASS',scientific_acceptance=False,full_C7_acceptance=False)
        _approval_record(APPROVAL_PATH.read_bytes())
        return dict(exception_uses=1,approval_sha256=APPROVAL_SHA256,strict_original_energy_pass=False,
            outcome='PASS_WITH_NAMED_HUMAN_ENERGY_ACCEPTANCE_ONLY',scientific_acceptance=False,full_C7_acceptance=False)


CASE_APPROVAL_SHA256='8ab00c45c1d128b3292b3ab3e9c1e74493f4db2bf52bb33a41e9e6c642030de4'
CASE_APPROVAL_PATH=Path(__file__).with_name('native_vibration_case_acceptance_20261018.json')
GOVERNANCE_APPROVAL_SHA256='1353bdb40ffe8245fb40cb7e9c3228058aca33c565b5473c5e90e552867651a0'


def _case_approval_record(raw):
    if type(raw)is not bytes or sha256(raw).hexdigest()!=CASE_APPROVAL_SHA256:
        raise EnergyAcceptanceError('case approval bytes do not match immutable capsule')
    record=json.loads(raw)
    if (record['schema']!='ANY_NATIVE_VIBRATION_CASE_ACCEPTANCE_V1'
            or record['status']!='HUMAN_APPROVED_SCOPED_CONTRACT_AMENDMENT'
            or record['governance_amendment_sha256']!=GOVERNANCE_APPROVAL_SHA256):
        raise EnergyAcceptanceError('wrong case approval provenance')
    return record


def _finite_error(value,name):
    if type(value)is not float or not math.isfinite(value)or value<0.:
        raise EnergyAcceptanceError('finite nonnegative binary64 '+name+' required')


def _case_energy_binding(facts,record):
    """Case threshold admission; no unobserved future state hash is invented."""
    expected=record['measured_velocity_exception']['binding']
    if (type(facts)is not dict or set(facts)!=set(expected)
            or any(type(facts[k])is not type(expected[k])for k in expected)):
        raise EnergyAcceptanceError('case energy binding fields/types differ')
    if any(facts[k]!=record['scope'][k]for k in ('definition_sha256','profile_sha256')):
        raise EnergyAcceptanceError('case energy definition/profile differs')
    if not any(all(facts[k]==row[k]for k in ('label','amplitude','steps'))
               for row in record['prescribed_workload']):
        raise EnergyAcceptanceError('case energy workload differs')
    if not 1<=facts['epoch']<=facts['steps']:
        raise EnergyAcceptanceError('case energy endpoint outside prescribed workload')
    for key,value in facts.items():
        if key.endswith('_sha256')and (len(value)!=64 or any(c not in '0123456789abcdef'for c in value)):
            raise EnergyAcceptanceError('invalid case digest')
        if key.endswith('_hex'):
            try:decoded=float.fromhex(value)
            except (ValueError,OverflowError) as error:raise EnergyAcceptanceError('invalid case scale/time')from error
            if not math.isfinite(decoded)or decoded<=0. or decoded.hex()!=value:
                raise EnergyAcceptanceError('invalid case scale/time')


class _CaseAcceptance:
    """Case energy thresholds and one exact measured velocity; no general velocity envelope."""
    def __init__(self):
        self._energy_uses=0
        self._velocity_uses=0

    def check_energy(self,step,cumulative,facts):
        _finite_error(step,'step energy');_finite_error(cumulative,'cumulative energy')
        if step<=1e-6 and cumulative<=1e-6:
            return dict(status='STRICT_ORIGINAL_ENERGY_PASS',acceptance_used=False)
        record=_case_approval_record(CASE_APPROVAL_PATH.read_bytes())
        _case_energy_binding(facts,record)
        limits=record['energy_limits']
        if step>limits['approved_step']or cumulative>limits['approved_cumulative']:
            raise EnergyAcceptanceError('case step/cumulative energy exceeds approved limit')
        self._energy_uses+=1
        return dict(status='HUMAN_APPROVED_CASE_ENERGY_PASS',acceptance_used=True,
            approval_id=record['id'],approval_sha256=CASE_APPROVAL_SHA256,
            normalized_step_energy_hex=step.hex(),normalized_cumulative_energy_hex=cumulative.hex(),binding=dict(facts))

    def check_velocity(self,error,facts):
        _finite_error(error,'discrete midpoint velocity')
        if error<=.002:
            return dict(status='STRICT_ORIGINAL_VELOCITY_PASS',exception_used=False)
        record=_case_approval_record(CASE_APPROVAL_PATH.read_bytes())
        measured=record['measured_velocity_exception'];expected=measured['binding']
        if (type(facts)is not dict or set(facts)!=set(expected)
                or any(type(facts[k])is not type(expected[k])or facts[k]!=expected[k]for k in expected)):
            raise EnergyAcceptanceError('measured velocity case/state binding differs')
        if error.hex()!=measured['approved_observed_hex']:
            raise EnergyAcceptanceError('velocity float bits differ from measured acceptance')
        if self._velocity_uses!=0:
            raise EnergyAcceptanceError('measured velocity exception already consumed')
        scope=_measured_acceptance_scope(record)
        self._velocity_uses=1
        return dict(status='HUMAN_APPROVED_MEASURED_VELOCITY_USED',exception_used=True,
            approval_id=record['id'],approval_sha256=CASE_APPROVAL_SHA256,
            normalized_velocity_hex=error.hex(),binding=dict(facts),scope=scope)

    def finish(self):
        if type(self._energy_uses)is not int or self._energy_uses<0 or type(self._velocity_uses)is not int or self._velocity_uses not in (0,1):
            raise EnergyAcceptanceError('invalid case acceptance ledger')
        used=self._energy_uses>0 or self._velocity_uses>0
        if used:_case_approval_record(CASE_APPROVAL_PATH.read_bytes())
        return dict(case_energy_uses=self._energy_uses,velocity_exception_uses=self._velocity_uses,
            strict_original_gates_pass=not used,approval_sha256=CASE_APPROVAL_SHA256 if used else None,
            governance_amendment_sha256=GOVERNANCE_APPROVAL_SHA256 if used else None,
            outcome='PASS_WITH_SCOPED_HUMAN_CASE_ACCEPTANCE_ONLY'if used else'STRICT_ORIGINAL_GATES_PASS',
            scientific_acceptance=False,full_C7_acceptance=False)


ENVELOPE_APPROVAL_SHA256='f4d25dd480021609e087a1d0120b60cbd02783c89dae62170fb40e811a41a628'
ENVELOPE_APPROVAL_PATH=Path(__file__).with_name('native_vibration_canonical_case_envelope_acceptance_20261018.json')
ENVELOPE_GOVERNANCE_SHA256='571b7a4bf6ad4cc045097dce5e858345ba12bee7dfc4c89429e717685633e4af'


def _envelope_approval_record(raw):
    if type(raw)is not bytes or sha256(raw).hexdigest()!=ENVELOPE_APPROVAL_SHA256:
        raise EnergyAcceptanceError('case envelope approval bytes differ from immutable capsule')
    record=json.loads(raw)
    if (record['schema']!='ANY_NATIVE_VIBRATION_CASE_ENVELOPE_ACCEPTANCE_V3_CANONICAL_PRIMITIVES'
            or record['status']!='HUMAN_APPROVED_SCOPED_CONTRACT_AMENDMENT'
            or record['governance_amendment_sha256']!=ENVELOPE_GOVERNANCE_SHA256
            or record['technical_source_binding_sha256']!='2a9523155e54a4f15ce3399a0a7d17eb2bca90f97270dc5332ec7b4acf4386e1'):
        raise EnergyAcceptanceError('wrong case envelope approval provenance')
    return record


def _envelope_case_binding(facts,record):
    # Reuse the original typed case/workload validator, without its measured-value gate.
    _case_energy_binding(facts,dict(scope=record['scope'],prescribed_workload=record['prescribed_workload'],
        measured_velocity_exception=dict(binding=record['binding_template'])))


def _measure_envelope_source():
    import anyfesolver
    package=Path(anyfesolver.__file__).resolve().parent
    files={'anyfesolver/'+p.relative_to(package).as_posix():sha256(p.read_bytes()).hexdigest()
        for p in package.rglob('*.py')if '__pycache__'not in p.parts}
    return dict(product_python_count=len(files),
        product_python_manifest_sha256=sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        physical_reference_sha256=sha256(Path(physical.__file__).read_bytes()).hexdigest())


def _validate_envelope_source(actual,record):
    expected=dict(product_python_count=700,product_python_manifest_sha256=record['scope']['product_python_manifest_sha256'],
        physical_reference_sha256=record['scope']['physical_reference_sha256'])
    if (type(actual)is not dict or set(actual)!=set(expected)
            or any(type(actual[k])is not type(expected[k])or actual[k]!=expected[k]for k in expected)):
        raise EnergyAcceptanceError('actual product/reference source differs from case envelope')
    return actual


def _measured_envelope_scope(record):
    return _validate_envelope_source(_measure_envelope_source(),record)


class _CaseEnvelopeAcceptance:
    """Three human-approved thresholds for one bound case and eligible runtime."""
    def __init__(self):
        self._seen=dict(energy=set(),velocity=set())

    def _admit(self,channel,facts,record):
        _envelope_case_binding(facts,record)
        slot=(facts['label'],facts['epoch'])
        if slot in self._seen[channel]:raise EnergyAcceptanceError('case envelope endpoint already admitted')
        scope=_measured_envelope_scope(record)
        self._seen[channel].add(slot)
        return dict(approval_id=record['id'],approval_sha256=ENVELOPE_APPROVAL_SHA256,binding=dict(facts),scope=scope)

    def check_energy(self,step,cumulative,facts):
        _finite_error(step,'step energy');_finite_error(cumulative,'cumulative energy')
        if step<=1e-6 and cumulative<=1e-6:
            return dict(status='STRICT_ORIGINAL_ENERGY_PASS',acceptance_used=False)
        record=_envelope_approval_record(ENVELOPE_APPROVAL_PATH.read_bytes());limits=record['limits']
        if step>limits['step_energy']or cumulative>limits['cumulative_energy']:
            raise EnergyAcceptanceError('case envelope step/cumulative energy limit exceeded')
        verdict=self._admit('energy',facts,record)
        return dict(status='HUMAN_APPROVED_CASE_ENVELOPE_ENERGY_PASS',acceptance_used=True,
            normalized_step_energy_hex=step.hex(),normalized_cumulative_energy_hex=cumulative.hex(),**verdict)

    def check_velocity(self,error,facts):
        _finite_error(error,'discrete midpoint velocity')
        if error<=.002:return dict(status='STRICT_ORIGINAL_VELOCITY_PASS',acceptance_used=False)
        record=_envelope_approval_record(ENVELOPE_APPROVAL_PATH.read_bytes())
        if error>record['limits']['velocity']:raise EnergyAcceptanceError('case envelope velocity limit exceeded')
        verdict=self._admit('velocity',facts,record)
        return dict(status='HUMAN_APPROVED_CASE_ENVELOPE_VELOCITY_PASS',acceptance_used=True,
            normalized_velocity_hex=error.hex(),**verdict)

    def finish(self):
        # The ledger names original prescribed endpoint slots, independently for each channel.
        expected={(label,epoch)for label,count in (('coarse',4),('fine',8),('half_fine',8))for epoch in range(1,count+1)}
        if (type(self._seen)is not dict or set(self._seen)!={'energy','velocity'}
                or any(type(slots)is not set or not slots<=expected for slots in self._seen.values())):
            raise EnergyAcceptanceError('invalid case envelope usage ledger')
        energies=len(self._seen['energy']);velocities=len(self._seen['velocity']);used=bool(energies or velocities)
        if used:_envelope_approval_record(ENVELOPE_APPROVAL_PATH.read_bytes())
        return dict(case_energy_uses=energies,case_velocity_uses=velocities,strict_original_gates_pass=not used,
            approval_sha256=ENVELOPE_APPROVAL_SHA256 if used else None,
            governance_amendment_sha256=ENVELOPE_GOVERNANCE_SHA256 if used else None,
            outcome='PASS_WITH_SCOPED_HUMAN_CASE_ENVELOPE_ACCEPTANCE_ONLY'if used else'STRICT_ORIGINAL_GATES_PASS',
            scientific_acceptance=False,full_C7_acceptance=False)


CRITERIA=dict(amplitude=1e-4,stiffness_antisymmetry_limit=1e-10,eigenpair_residual_limit=1e-10,
    nonlinear_to_discrete_limit=.002,normalized_defect_limit=1e-6,elastic_witness_limit=1e-10,
    case_step_energy_limit=3.5e-6,case_cumulative_energy_limit=2.5e-6,case_velocity_limit=.005,
    case_energy_scope='fixed definition/profile and prescribed coarse/fine/half_fine workload only',
    measured_velocity_exception_policy='historical exact-value capsule preserved; live named-case envelope requires actual product/reference eligibility; registered runtime guards remain separate',
    case_approval_sha256=ENVELOPE_APPROVAL_SHA256,governance_amendment_sha256=ENVELOPE_GOVERNANCE_SHA256,
    trajectory_floor=1e-6,invariant_floor=1e-10,time_refinement_ratio=.4,amplitude_refinement_ratio=.75,
    internal_time_limit_seconds=120.,maximum_accepted_steps=20)
AUDIT_PARITY=1e-10  # Prospective independent audit gate; root adoption required.


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,bytes):return {'bytes_hex':value.hex()}
    if hasattr(value,'__dataclass_fields__'):return {k:plain(getattr(value,k))for k in value.__dataclass_fields__}
    if isinstance(value,(list,tuple)):return [plain(x)for x in value]
    if isinstance(value,dict):return {str(k):plain(v)for k,v in value.items()}
    if isinstance(value,np.generic):return value.item()
    return value


def encoded(value):
    return json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def mechanical_data(mechanical):
    return {name:np.array(getattr(mechanical,name),copy=True)for name in
        ('positions','position_low','nodal_frames','cell_rotations','resultants')}


def new_owner(definition):
    # No tolerance, runtime, iteration, backtrack or chart override.
    return create_analysis(SELECTOR,(definition,),())


def current_reference(owner):
    """Shared spatial construction only, transparently not an independent FE oracle.

    No trial or commit is used for the baseline. Public initialization below owns
    actual finite-state algebraic recovery. K/M are CURRENT output, not old arrays.
    """
    c=owner._owner();p=list(physical.PHYSICAL);z=list(physical.ALGEBRAIC)
    ref=c._reference_mechanical
    work=c._dynamic.evaluate(c._dynamic.reference_state())
    response=c._inertia(ref,np.zeros(24),np.zeros(24))
    K=np.asarray(work.spatial_jacobian);M=response.mixed_mass[np.ix_(p,p)]
    Kzz=K[np.ix_(z,z)];singular=np.linalg.svd(Kzz,compute_uv=False)
    rank_tolerance=64*np.finfo(float).eps*27*singular[0]
    assert np.count_nonzero(singular>rank_tolerance)==27
    assert np.linalg.norm(M-M.T)/np.linalg.norm(M)<=1e-10
    assert np.count_nonzero(response.mixed_mass[z,:])==0
    M=(M+M.T)/2.;L=np.linalg.cholesky(M)
    Z=-np.linalg.solve(Kzz,K[np.ix_(z,p)])
    Ke=K[np.ix_(p,p)]+K[np.ix_(p,z)]@Z
    asymmetry=float(np.linalg.norm(Ke-Ke.T)/np.linalg.norm(Ke));assert asymmetry<=1e-10
    Ke=(Ke+Ke.T)/2.;C=np.linalg.solve(L,np.linalg.solve(L,Ke).T).T
    eigenvalues,vectors=np.linalg.eigh((C+C.T)/2.)
    resolution=256*np.finfo(float).eps*15*float(np.max(abs(eigenvalues)))
    assert np.all(abs(eigenvalues[:6])<=resolution)and np.all(eigenvalues[6:]>resolution)
    mode=np.linalg.solve(L.T,vectors[:,6]);mode*=1. if mode[np.argmax(abs(mode))]>=0 else -1.
    full=np.zeros(42);full[p]=mode;length=2.
    normalization=max(*(np.linalg.norm(full[6*n:6*n+3])/length for n in range(3)),
                      *(np.linalg.norm(full[18+3*c:21+3*c])for c in (0,1)))
    mode/=normalization;full[p]=mode;full[z]=Z@mode
    eig_error=float(np.linalg.norm(Ke@mode-eigenvalues[6]*(M@mode))/(np.linalg.norm(Ke)*np.linalg.norm(mode)))
    assert eig_error<=1e-10
    omega=float(np.sqrt(eigenvalues[6]));duration=float(np.pi/(2*omega));modal_mass=float(mode@M@mode)
    T=np.zeros((42,3))
    for node in range(3):T[6*node:6*node+3]=np.eye(3)
    mass=float(np.trace(T.T@response.mixed_mass@T)/3.)
    assert modal_mass>0. and mass>0.
    return dict(K42=K,M42=response.mixed_mass,M15=M,Ke15=Ke,Z=Z,eigenvalues=eigenvalues,
        mode15=mode,full42=full,eigenpair_error=eig_error,omega=omega,duration=duration,
        modal_mass=modal_mass,mass=mass,length=length,reference=mechanical_data(ref),
        rank=27,relative_antisymmetry=asymmetry,normalization=normalization,
        provenance='CURRENT shared spatial operator, original construction; not independent spatial qualification')


def compare(coarse,fine,floor,ratio):
    if coarse<=floor:return dict(coarse=coarse,fine=fine,floor=floor,ratio=None,
        disposition='resolution_limited'if fine<=floor else 'unresolved_NONPASS',passes=bool(fine<=floor))
    return dict(coarse=coarse,fine=fine,floor=floor,ratio=fine/coarse,
        disposition='bounded_comparison',passes=bool(fine<=ratio*coarse))


def delta_account(now,before,name):
    return np.array([float(Fraction(float(a))+Fraction(float(al))-Fraction(float(b))-Fraction(float(bl)))
        for a,al,b,bl in zip(getattr(now,name+'_momentum'),getattr(now,name+'_momentum_low'),
                            getattr(before,name+'_momentum'),getattr(before,name+'_momentum_low'))])


def coordinates(state,reference):
    m=mechanical_data(state.mechanical);q=np.zeros(42)
    for n in range(3):
        q[6*n:6*n+3]=[float(Fraction(float(a))+Fraction(float(al))-Fraction(float(b))-Fraction(float(bl)))
            for a,al,b,bl in zip(m['positions'][n],m['position_low'][n],reference['positions'][n],reference['position_low'][n])]
    for c in (0,1):q[18+3*c:21+3*c]=physical.log_rotation(m['cell_rotations'][0,c]@reference['cell_rotations'][0,c].T)
    return q[list(physical.PHYSICAL)]


def seed(reference,full,amplitude):
    # Only physical fields enter the public initializer; no issued object forged.
    positions=reference['positions'].copy();low=reference['position_low'].copy();cells=reference['cell_rotations'][0].copy()
    for n in range(3):
        for axis in range(3):
            exact=Fraction(float(positions[n,axis]))+Fraction(float(low[n,axis]))+Fraction(float(amplitude*full[6*n+axis]))
            positions[n,axis]=float(exact);low[n,axis]=float(exact-Fraction(float(positions[n,axis])))
    for c in (0,1):cells[c]=physical.rotation(amplitude*full[18+3*c:21+3*c])@cells[c]
    return positions,low,cells


def independent_audit(state,account,audit,scales,previous=None):
    m=mechanical_data(state.mechanical);kin=physical.kinetic(m,state.physical_velocity24);elastic=physical.elastic(m)
    E=scales['energy_j'];P=scales['momentum'];H=scales['angular_momentum']
    rp=np.array(account.linear_momentum)+np.array(account.linear_momentum_low)
    rh=np.array(account.angular_momentum)+np.array(account.angular_momentum_low)
    errors=dict(kinetic=abs(kin['T']-account.kinetic_energy)/E,
        mixed=abs(elastic['Pi']-account.elastic_energy)/E,
        linear=float(np.linalg.norm(kin['P']-rp)/P),angular=float(np.linalg.norm(kin['H']-rh)/H),
        witness=abs(abs(elastic['witness'])-account.elastic_consistency_witness)/E)
    result=dict(endpoint=dict(kinetic=kin,elastic=elastic),parity=errors)
    if audit is not None:
        sm=mechanical_data(audit.stage.mechanical)
        sk=physical.kinetic(sm,audit.stage.physical_velocity24,audit.stage.physical_acceleration24)
        se=physical.elastic(sm);R=np.asarray(audit.stage.structural_residual);I=np.asarray(audit.stage.inertia.mixed_inertia)
        errors['stage_structural']=float(np.linalg.norm((se['residual']-R)/audit.residual_scales42))
        errors['stage_inertial']=float(np.linalg.norm((sk['inertia']-I)/audit.residual_scales42))
        errors['stage_residual_sum']=float(np.linalg.norm((audit.stage.residual-R-I)/audit.residual_scales42))
        # Physical rigid virtual maps include every nodal frame/cell moment block.
        T=np.zeros((42,3));U=np.zeros((42,3))
        for n in range(3):
            T[6*n:6*n+3]=np.eye(3);x=sm['positions'][n]+sm['position_low'][n]
            U[6*n:6*n+3]=-physical.skew(x);U[6*n+3:6*n+6]=np.eye(3)
        for c in (0,1):U[18+3*c:21+3*c]=np.eye(3)
        signed={key:dict(force=T.T@v,moment=U.T@v,work=float(v@audit.increment42))
            for key,v in [('independent_structural',se['residual']),('independent_inertial',sk['inertia']),
                          ('reported_structural',R),('reported_inertial',I),('reported_solve_residual',audit.stage.residual)]}
        result['stage']=dict(authenticated_audit=plain(audit),independent_kinetic=sk,independent_elastic=se,
            rigid_translation=T,rigid_rotation=U,signed_projections=signed,
            signed_impulse_remainders=dict(linear=delta_account(account,previous,'linear')-audit.dt*(T.T@I),
                angular=delta_account(account,previous,'angular')-audit.dt*(U.T@I)))
    assert all(np.isfinite(v)and v>=0. for v in errors.values())
    return result


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=perf_counter()
    evidence=dict(status='RUNNING',criteria=CRITERIA,audit_parity=AUDIT_PARITY,
        scientific_endpoints=0,replay_increments=0,maximum_scientific_endpoints=20,
        maximum_replay_increments=2,scope='CURRENT public authentic single native vibration; full C7 OPEN',resolutions={})
    def save(stage):
        evidence['last_stage']=stage;evidence['elapsed_seconds']=perf_counter()-started
        (output/'ASSESSMENT.json').write_bytes(encoded(evidence)+b'\n')
        print(json.dumps(dict(stage=stage,scientific_endpoints=evidence['scientific_endpoints'],replay_increments=evidence['replay_increments'])),flush=True)
        assert evidence['elapsed_seconds']<120.,'original internal120s bound'
    try:
        case_acceptance=_CaseEnvelopeAcceptance()
        points,frames,section,inertia,definition=fixture()
        ip,ifr,ic,im=physical.primitive_inputs()
        assert np.array_equal(points,ip)and np.array_equal(frames,ifr)
        assert np.array_equal(section.elastic,ic)and np.array_equal(inertia,im)
        owner=new_owner(definition);ref=current_reference(owner);evidence['reference']=ref
        evidence['definition_raw_hex']=definition.raw.hex();evidence['profile_hex']=PROFILE_BYTES.hex();save('current_reference')
        checkpoints={};coarse_states=[];p=list(physical.PHYSICAL);z=list(physical.ALGEBRAIC)
        for label,amplitude,count in [('coarse',1e-4,4),('fine',1e-4,8),('half_fine',5e-5,8)]:
            if label!='coarse':owner=new_owner(definition)
            assert encoded(mechanical_data(owner._owner()._reference_mechanical))==encoded(ref['reference'])
            positions,low,cells=seed(ref['reference'],ref['full42'],amplitude)
            state=owner.initialize(positions,cells,np.zeros(15),position_low=low)
            assert np.array_equal(state.mechanical.positions,positions)
            assert np.array_equal(state.mechanical.position_low,low)
            assert np.array_equal(state.mechanical.cell_rotations[0],cells)
            initial=owner.endpoint_accounting(state);E=initial.total_energy;assert E>0. and np.isfinite(E)
            assert initial.elastic_consistency_witness/E<=1e-10
            speed=float(np.sqrt(2*E/ref['mass']));h=ref['duration']/count;theta=2*np.arctan(ref['omega']*h/2)
            scales=dict(energy_j=E,momentum=ref['mass']*speed,angular_momentum=ref['mass']*2.*speed,
                configuration=amplitude*np.sqrt(ref['modal_mass']),velocity=amplitude*ref['omega']*np.sqrt(ref['modal_mass']))
            row=evidence['resolutions'][label]=dict(amplitude=amplitude,steps=count,dt=h,scales=scales,initial=plain(state),
                initial_accounting=plain(initial),endpoints=[])
            result=independent_audit(state,initial,None,scales);row['initial_independent']=result
            save(label+'_initialized');assert max(result['parity'].values())<=AUDIT_PARITY
            assert encoded(state.histories)==encoded(state.origins)
            assert encoded(state.origins)==encoded(owner._owner()._reference_origins)
            previous=initial;previous_independent=result;path=np.zeros(2)
            for index in range(count):
                candidate=owner.trial(state,dt=h);audit=owner.candidate_stage_audit(candidate)
                assert audit.predecessor_epoch==state.epoch and audit.endpoint_epoch==state.epoch+1 and audit.dt==h
                assert audit.stage_residual_tolerance==1e-11 and audit.stage_coordinates=='increment'
                assert audit.context_sha256==state.context_sha256 and audit.predecessor_sha256==candidate.predecessor_sha256
                assert not audit.unknown42.flags.writeable and not audit.residual_scales42.flags.writeable
                row['pending']=plain(audit);save(label+'_trial_'+str(index+1))
                path+=[np.linalg.norm(audit.increment42[18+3*c:21+3*c])for c in (0,1)];assert np.max(path)<1e-3
                state=owner.commit(candidate);account=owner.endpoint_accounting(state)
                q=coordinates(state,ref['reference']);v=state.physical_velocity24[p]
                qcont=amplitude*ref['mode15']*np.cos(ref['omega']*state.time);vcont=-amplitude*ref['omega']*ref['mode15']*np.sin(ref['omega']*state.time)
                qdisc=amplitude*ref['mode15']*np.cos((index+1)*theta);vdisc=-amplitude*ref['omega']*ref['mode15']*np.sin((index+1)*theta)
                def norm(value,scale):
                    square=float(value@ref['M15']@value);assert square>=0.;return float(np.sqrt(square)/scale)
                errors=dict(configuration_continuum=norm(q-qcont,scales['configuration']),velocity_continuum=norm(v-vcont,scales['velocity']),
                    configuration_discrete_midpoint=norm(q-qdisc,scales['configuration']),velocity_discrete_midpoint=norm(v-vdisc,scales['velocity']))
                step=dict(linear=float(np.linalg.norm(delta_account(account,previous,'linear'))/scales['momentum']),
                    angular=float(np.linalg.norm(delta_account(account,previous,'angular'))/scales['angular_momentum']),energy=abs(account.total_energy-previous.total_energy)/E)
                cumulative=dict(linear=float(np.linalg.norm(delta_account(account,initial,'linear'))/scales['momentum']),
                    angular=float(np.linalg.norm(delta_account(account,initial,'angular'))/scales['angular_momentum']),energy=abs(account.total_energy-initial.total_energy)/E)
                independent=independent_audit(state,account,audit,scales,previous)
                kin=independent['endpoint']['kinetic'];el=independent['endpoint']['elastic']
                pk=previous_independent['endpoint']['kinetic'];pe=previous_independent['endpoint']['elastic']
                ik=result['endpoint']['kinetic'];ie=result['endpoint']['elastic']
                independent['physical_step_defects']=dict(energy=(kin['T']+el['Psi']-pk['T']-pe['Psi'])/E,linear=(kin['P']-pk['P'])/scales['momentum'],angular=(kin['H']-pk['H'])/scales['angular_momentum'])
                independent['physical_cumulative_defects']=dict(energy=(kin['T']+el['Psi']-ik['T']-ie['Psi'])/E,linear=(kin['P']-ik['P'])/scales['momentum'],angular=(kin['H']-ik['H'])/scales['angular_momentum'])
                endpoint=dict(state=plain(state),accounting=plain(account),errors=errors,normalized_step_defects=step,
                    normalized_cumulative_defects=cumulative,independent=independent)
                row['endpoints'].append(endpoint);row.pop('pending');evidence['scientific_endpoints']+=1;save(label+'_endpoint_'+str(index+1))
                # Approved case energy/velocity and unchanged other predicates precede any later trial.
                facts=dict(label=label,amplitude=amplitude,steps=count,epoch=state.epoch,
                    dt_hex=float(h).hex(),time_hex=float(state.time).hex(),energy0_hex=float(E).hex(),
                    initial_state_sha256=initial.state_sha256,predecessor_state_sha256=previous.state_sha256,
                    endpoint_state_sha256=account.state_sha256,context_sha256=audit.context_sha256,
                    definition_sha256=sha256(definition.raw).hexdigest(),profile_sha256=sha256(PROFILE_BYTES).hexdigest())
                endpoint['energy_attempt']=dict(normalized_step_energy_hex=float(step['energy']).hex(),
                    normalized_cumulative_energy_hex=float(cumulative['energy']).hex(),binding=facts)
                endpoint['energy_verdict']=case_acceptance.check_energy(float(step['energy']),float(cumulative['energy']),facts)
                if endpoint['energy_verdict']['acceptance_used']:
                    evidence.setdefault('approved_case_energy_uses',[]).append(endpoint['energy_verdict'])
                    save('scoped_human_case_energy_acceptance_used')
                assert max(step[k]for k in ('linear','angular'))<=1e-6 and max(cumulative[k]for k in ('linear','angular'))<=1e-6
                assert account.elastic_consistency_witness/E<=1e-10
                assert candidate.stage_residual_norm<=1e-11 and candidate.endpoint_algebraic_norm<=1e-11
                assert errors['configuration_discrete_midpoint']<=.002
                endpoint['velocity_verdict']=case_acceptance.check_velocity(float(errors['velocity_discrete_midpoint']),facts)
                if endpoint['velocity_verdict']['acceptance_used']:
                    evidence.setdefault('approved_case_velocity_uses',[]).append(endpoint['velocity_verdict'])
                    save('scoped_human_case_velocity_acceptance_used')
                assert np.count_nonzero(audit.stage.inertia.mixed_inertia[z])==0 and np.count_nonzero(audit.stage.inertia.mixed_mass[z])==0
                assert encoded(state.histories)==encoded(initial_state_histories(row))
                assert max(independent['parity'].values())<=AUDIT_PARITY
                if label=='coarse':
                    checkpoint=owner.create_checkpoint()
                    checkpoint_path=output/('coarse-step-'+str(index+1)+'.checkpoint.json')
                    checkpoint_path.write_bytes(checkpoint)
                    row['endpoints'][-1]['checkpoint']=dict(path=str(checkpoint_path),sha256=sha256(checkpoint).hexdigest())
                    coarse_states.append((encoded(state),encoded(account),checkpoint))
                    if index==1:checkpoints['split']=coarse_states[-1][2]
                previous=account;previous_independent=independent
            assert state.epoch==count and abs(state.time-ref['duration'])<=8*np.finfo(float).eps*ref['duration']
        coarse=evidence['resolutions']['coarse']['endpoints'];fine=evidence['resolutions']['fine']['endpoints'];half=evidence['resolutions']['half_fine']['endpoints']
        continuum=('configuration_continuum','velocity_continuum');discrete=('configuration_discrete_midpoint','velocity_discrete_midpoint')
        maximum=lambda rows,keys:max(r['errors'][k]for r in rows for k in keys)
        comparisons=dict(continuum=compare(maximum(coarse,continuum),maximum(fine[1::2],continuum),1e-6,.4),
            amplitude=compare(maximum(fine,discrete),maximum(half,discrete),1e-6,.75))
        for key in ('linear','angular','energy'):
            comparisons[key]=compare(max(r['normalized_cumulative_defects'][key]for r in coarse),
                max(r['normalized_cumulative_defects'][key]for r in fine[1::2]),1e-10,.4)
        evidence['comparisons']=comparisons;save('original_comparisons');assert all(v['passes']for v in comparisons.values())
        assert evidence['scientific_endpoints']==20
        fresh=new_owner(definition);state=fresh.restore_checkpoint(checkpoints['split'],expected_sha256=sha256(checkpoints['split']).hexdigest())
        assert encoded(state)==coarse_states[1][0] and fresh.create_checkpoint()==checkpoints['split']
        for target in (2,3):
            trial=fresh.trial(state,dt=ref['duration']/4.);state=fresh.commit(trial);account=fresh.endpoint_accounting(state)
            evidence['replay_increments']+=1;save('fresh_split_replay_'+str(target+1))
            assert encoded(state)==coarse_states[target][0] and encoded(account)==coarse_states[target][1]
            replay_checkpoint=fresh.create_checkpoint()
            assert replay_checkpoint==coarse_states[target][2]
            evidence.setdefault('restart_correspondence',[]).append(dict(endpoint=target+1,state_sha256=sha256(encoded(state)).hexdigest(),account_sha256=sha256(encoded(account)).hexdigest(),checkpoint_sha256=sha256(replay_checkpoint).hexdigest()))
        evidence['case_envelope_acceptance']=case_acceptance.finish()
        evidence['status']=('PASS_CURRENT_PUBLIC_VIBRATION_DEVELOPMENT_ONLY'if evidence['case_envelope_acceptance']['strict_original_gates_pass']else'PASS_CURRENT_PUBLIC_VIBRATION_WITH_SCOPED_CASE_ENVELOPE_ACCEPTANCE_ONLY');save('complete');return evidence
    except BaseException as error:
        evidence.update(status='NONPASS',failure_type=type(error).__name__,failure=str(error))
        # Keep the primary exception even if publication itself fails.
        try:save('failure')
        except BaseException:pass
        raise


def initial_state_histories(row):
    return row['initial']['histories']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.output)


if __name__=='__main__':main()
