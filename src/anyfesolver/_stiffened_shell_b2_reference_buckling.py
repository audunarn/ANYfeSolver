"""Experimental conservative reference buckling of the physical B2 panel.

Spectral kinematics are deliberately separate from the affine static owner.
Section tractions are supplied physical data, never reconstructed from stiffness.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, ArpackError

POLICY = "B2_PHYSICAL_REFERENCE_GREEN_POINT_BUCKLING_V1_DEV"
PORT_POLICY = "B2_GEODESIC_MIDFRAME_PROPER_POLAR_SECOND_VARIATION_V1_DEV"
Q4_KERNEL_POLICY = "Q4_QUALIFIED_REFERENCE_RECOVERED_N_M_HOMOGENEOUS_SECOND_MOMENT_PHYSICAL_TO_GP_FRAME_V2_DEV"
S3_KERNEL_POLICY = "S3_V2D_FROZEN_THIN_UNIFORM_MEMBRANE_OMITS_PHYSICAL_SECOND_MOMENT"

class _BucklingNumericalFailure(RuntimeError):
    """A deliberately classified failed numerical spectral operation."""

def _sealed(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"finite {label} with shape {shape} required")
    return np.frombuffer(a.tobytes(), dtype=float).reshape(shape)

def _cross_matrix(v):
    return np.array(((0.,-v[2],v[1]),(v[2],0.,-v[0]),(-v[1],v[0],0.)))

@dataclass(frozen=True)
class B2SectionTractionQuadrature:
    """Physical traction source about the member reference axis, in its frame.

    ``traction_map[k] @ [N,Qy,Qz,Mx,My,Mz]`` gives S11,S12,S13.
    Transverse stresses are zero under this declared beam idealization.
    """
    member_id: str
    source_id: str
    coordinates_yz_m: tuple
    area_weights_m2: tuple
    traction_map: tuple
    datum: str = "member_reference_axis_local_yz"

    def __post_init__(self):
        if type(self.member_id) is not str or type(self.source_id) is not str or not self.member_id or not self.source_id or self.datum != "member_reference_axis_local_yz":
            raise ValueError("explicit source and member reference-axis datum required")
        n = len(self.area_weights_m2)
        yz = _sealed(self.coordinates_yz_m,(n,2),"section coordinates")
        w = _sealed(self.area_weights_m2,(n,),"section area weights")
        t = _sealed(self.traction_map,(n,3,6),"section traction reconstruction")
        if n < 3 or np.any(w <= 0.):
            raise ValueError("positive physical section area quadrature required")
        length=float(np.max(abs(yz)))
        if not np.isfinite(length) or length <= 0.: raise ValueError("nonzero physical section length required")
        result = np.zeros((6,6)); scale = np.zeros((6,6))
        for xy, weight, mapping in zip(yz,w,t):
            rho = np.r_[0.,xy/length]
            action = np.vstack((np.eye(3),_cross_matrix(rho)))
            # D R D^-1, with D=diag(I, I/length): dimensionless force /
            # moment virtual-work certificate in the declared physical datum.
            normalized=mapping.copy();normalized[:,3:]*=length
            with np.errstate(over="ignore",invalid="ignore"):
                result += weight*action@normalized
                scale += weight*abs(action)@abs(normalized)
        if not np.isfinite(result).all() or not np.isfinite(scale).all():
            raise ValueError("section six-resultant arithmetic not representable")
        # Elementwise backward work certificate; cancellation is measured, not hidden.
        error = abs(result-np.eye(6))
        if np.any(error > 128*np.finfo(float).eps*(scale+np.eye(6))):
            raise ValueError("section source does not reproduce six physical resultants")
        if np.max(np.sum(error,axis=1)) > 1e-10:
            raise ValueError("section source forward six-resultant identity failed")
        object.__setattr__(self,"coordinates_yz_m",yz)
        object.__setattr__(self,"area_weights_m2",w)
        object.__setattr__(self,"traction_map",t)

    @property
    def identity(self):
        h = hashlib.sha256((self.member_id+"\0"+self.source_id+"\0"+self.datum).encode())
        for a in (self.coordinates_yz_m,self.area_weights_m2,self.traction_map): h.update(a.tobytes())
        return h.hexdigest()

@dataclass(frozen=True)
class B2ReferenceBucklingRequest:
    preload: tuple
    section_tractions: tuple
    count: int = 1
    shift: float = 0.
    max_iterations: int = 1000

    def __post_init__(self):
        p = np.asarray(self.preload,dtype=float)
        if p.ndim != 1 or not np.isfinite(p).all(): raise ValueError("finite dead nodal force vector required")
        object.__setattr__(self,"preload",tuple(map(float,p)))
        sources = tuple(self.section_tractions)
        if any(type(s) is not B2SectionTractionQuadrature for s in sources): raise ValueError("typed physical traction sources required")
        if len({s.member_id for s in sources}) != len(sources): raise ValueError("one source per member required")
        object.__setattr__(self,"section_tractions",sources)
        if type(self.count) is not int or self.count < 1: raise ValueError("positive buckling count required")
        if not np.isfinite(self.shift) or self.shift < 0.: raise ValueError("finite nonnegative load-factor shift required")
        if type(self.max_iterations) is not int or self.max_iterations < 1: raise ValueError("positive iteration bound required")

@dataclass(frozen=True)
class B2ReferenceBucklingCapabilities:
    eligible: bool
    reasons: tuple = ()
    policy: str = POLICY
    production_qualified: bool = False
    scope: str = "zero-prescription Cartesian dead-force endpoint POINT Q4 / uniform-membrane S3"
    remaining: tuple = ("general S3 bending/director second moment/shear", "Q4 shear prestress", "interior loaded ports", "weak lines", "footprints", "prescribed prestress", "dead-couple potential", "full F12/B14 qualification")
    shell_kernel_policies: tuple = ()

@dataclass(frozen=True)
class B2ReferenceBucklingResult:
    status: str
    load_factors: tuple = ()
    modes: tuple = ()
    multiplier_modes: tuple = ()
    equation_residuals: tuple = ()
    constraint_residuals: tuple = ()
    preload_result: object = None
    identity: str = ""
    failure_reason: str | None = None
    policy: str = POLICY
    production_qualified: bool = False

def beam_reference_derivatives(operator, station):
    """Position/rotation derivatives of the objective geodesic-midframe trace.

    Return first derivatives and coordinate Hessians, plus reference-length
    derivatives. Coordinates are the existing twelve additive physical slots.
    """
    length = float(operator.length)
    poly = operator._trace_polynomials
    maps = np.array([[sum(float(c)*station**d for d,c in enumerate(p)) for p in row] for row in poly])
    slope = np.array([[sum(d*float(c)*station**(d-1)/length for d,c in enumerate(p) if d) for p in row] for row in poly])
    eye = np.eye(12)
    omega = .5*(eye[3:6]+eye[9:12]).T
    mean = .5*(eye[:3]+eye[6:9]).T
    first = np.zeros((12,12)); second = np.zeros((12,12,12))
    for node, pos in ((0,np.array((-length/2,0.,0.))),(1,np.array((length/2,0.,0.)))):
        base = 6*node
        delta = eye[base:base+3].T-mean
        theta = eye[base+3:base+6].T
        first[base:base+3] = (delta-np.cross(omega,pos)).T
        first[base+3:base+6] = (theta-omega).T
        for a in range(12):
            for b in range(12):
                second[base:base+3,a,b] = (-np.cross(omega[a],delta[b])-np.cross(omega[b],delta[a])
                    +.5*(np.cross(omega[a],np.cross(omega[b],pos))+np.cross(omega[b],np.cross(omega[a],pos))))
                second[base+3:base+6,a,b] = -.5*(np.cross(omega[a],theta[b])+np.cross(omega[b],theta[a]))
    d = maps@first; ds = slope@first
    h = np.einsum('ij,jab->iab',maps,second)
    hs = np.einsum('ij,jab->iab',slope,second)
    pos = np.array(((station-.5)*length,0.,0.)); axis=np.array((1.,0.,0.))
    for a in range(12):
        for b in range(12):
            h[:3,a,b] += .5*(np.cross(omega[a],np.cross(omega[b],pos))+np.cross(omega[b],np.cross(omega[a],pos)))+np.cross(omega[a],d[:3,b])+np.cross(omega[b],d[:3,a])
            hs[:3,a,b] += .5*(np.cross(omega[a],np.cross(omega[b],axis))+np.cross(omega[b],np.cross(omega[a],axis)))+np.cross(omega[a],ds[:3,b])+np.cross(omega[b],ds[:3,a])
            h[3:,a,b] += .5*(np.cross(omega[a],d[3:,b])+np.cross(omega[b],d[3:,a]))
            hs[3:,a,b] += .5*(np.cross(omega[a],ds[3:,b])+np.cross(omega[b],ds[3:,a]))
    actual = d.copy(); actual[:3] += mean.T+np.cross(omega,pos).T;actual[3:] += omega.T
    actual_s = ds.copy();actual_s[:3] += np.cross(omega,axis).T
    if not np.allclose(actual,maps,rtol=0.,atol=128*np.finfo(float).eps*max(1.,abs(maps).max())):
        raise ValueError("objective beam continuation does not reproduce reference trace")
    return maps,slope,h,hs

def member_green_hessian(member, displacement, source):
    """Full S11/S12/S13 Green sectional initial-stress work, tension sign."""
    operator=member.operator; length=float(operator.length)
    nodes, weights=np.polynomial.legendre.leggauss(4)
    stations=(nodes+1.)/2; result=operator.recover_sections(displacement,stations=stations).resultants
    made=np.zeros((12,12)); virtual=np.zeros(12)
    for s,w,resultants in zip(stations,weights,result):
        j,js,h,hs=beam_reference_derivatives(operator,float(s))
        for yz,area,traction_map in zip(source.coordinates_yz_m,source.area_weights_m2,source.traction_map):
            rho=np.r_[0.,yz]; stress=traction_map@resultants
            xs=js[:3].T+np.cross(js[3:].T,rho)
            dy=np.cross(j[3:].T,(0.,1.,0.)); dz=np.cross(j[3:].T,(0.,0.,1.))
            e11=xs[:,0]; g12=xs[:,1]+dy[:,0];g13=xs[:,2]+dz[:,0]
            virtual += w*length/2*area*(stress[0]*e11+stress[1]*g12+stress[2]*g13)
            for a in range(12):
                for b in range(12):
                    xsh=hs[:3,a,b]+np.cross(hs[3:,a,b],rho)+.5*(
                        np.cross(js[3:,a],np.cross(j[3:,b],rho))+np.cross(j[3:,a],np.cross(js[3:,b],rho))+
                        np.cross(js[3:,b],np.cross(j[3:,a],rho))+np.cross(j[3:,b],np.cross(js[3:,a],rho)))
                    dyh=np.cross(h[3:,a,b],(0.,1.,0.))+.5*(np.cross(j[3:,a],dy[b])+np.cross(j[3:,b],dy[a]))
                    dzh=np.cross(h[3:,a,b],(0.,0.,1.))+.5*(np.cross(j[3:,a],dz[b])+np.cross(j[3:,b],dz[a]))
                    v=(stress[0]*(xs[a]@xs[b]+xsh[0])+stress[1]*(xs[a]@dy[b]+xs[b]@dy[a]+xsh[1]+dyh[0])+
                       stress[2]*(xs[a]@dz[b]+xs[b]@dz[a]+xsh[2]+dzh[0]))
                    made[a,b] += w*length/2*area*v
    force=np.asarray(operator.evaluate(displacement).force)
    force=force.copy();virtual=virtual.copy()
    force.reshape(2,6)[:,3:]/=length;virtual.reshape(2,6)[:,3:]/=length
    if not np.isfinite(force).all() or not np.isfinite(virtual).all():
        raise ValueError("normalized sectional virtual work not representable")
    magnitude=max(float(np.max(abs(force))),float(np.max(abs(virtual))))
    # One common magnitude keeps all norms finite and introduces no unit floor.
    f=force/magnitude if magnitude else force;v=virtual/magnitude if magnitude else virtual
    denominator=np.linalg.norm(f)+np.linalg.norm(v)
    if np.linalg.norm(f-v) > 1e-10*denominator:
        raise ValueError("sectional first virtual work disagrees with physical force-flexibility trace")
    return made

def point_green_hessian(law, vertices, member, station, offset, multiplier):
    from ._ge_beam3_pose_joint import RigidPoseJoint
    width=law.shell_port.shape[1]; r=vertices-vertices.mean(axis=0)
    gram=sum((_cross_matrix(x).T@_cross_matrix(x) for x in r),np.zeros((3,3)))
    shell_h=np.zeros((6,width,width)); eye=np.eye(width)
    omega=law.shell_port[3:].T
    translations=np.stack([eye[6*i:6*i+3].T for i in range(len(r))])
    translations-=translations.mean(axis=0)
    for a in range(width):
        for b in range(width):
            rhs=np.zeros(3)
            for i,x in enumerate(r):
                rhs += np.cross(np.cross(omega[a],x),translations[i,b])+np.cross(np.cross(omega[b],x),translations[i,a])
                rhs += np.cross(.5*(np.cross(omega[a],np.cross(omega[b],x))+np.cross(omega[b],np.cross(omega[a],x))),x)
            shell_h[3:,a,b]=np.linalg.solve(gram,rhs)
    j,_,h,_=beam_reference_derivatives(member.operator,float(station))
    d=np.asarray(offset,dtype=float); local_h=h.copy()
    for a in range(12):
        for b in range(12):
            local_h[:3,a,b] += np.cross(h[3:,a,b],d)+.5*(np.cross(j[3:,a],np.cross(j[3:,b],d))+np.cross(j[3:,b],np.cross(j[3:,a],d)))
    frame=member.definition.local_frame_rows().T; transform=member.local_dof_map()
    beam_h=np.einsum('ij,jab,ak,bl->ikl',np.kron(np.eye(2),frame),local_h,transform,transform)
    port=np.zeros((12,width+12));port[:6,:width]=law.shell_port;port[6:,width:]=law.beam_port
    curvature=np.zeros((12,width+12,width+12));curvature[:6,:width,:width]=shell_h;curvature[6:,width:,width:]=beam_h
    joint=RigidPoseJoint(np.stack((law.shell_reference_m,law.beam_reference_m)),law.reference_frames)
    response=joint.evaluate(np.stack((law.shell_reference_m,law.beam_reference_m)),law.reference_frames,np.asarray(multiplier))
    return port.T@response.chart_hessian[:12,:12]@port+np.einsum('i,iab->ab',response.residual[:12],curvature)

@dataclass(frozen=True)
class B2PhysicalFootprintBucklingRequest(B2ReferenceBucklingRequest):
    """Opt-in five-row Q4 physical rigid-footprint spectral operation."""

@dataclass(frozen=True)
class B2PhysicalFootprintBucklingResult(B2ReferenceBucklingResult):
    policy: str = "B2_PHYSICAL_RIGID_FOOTPRINT_GREEN_BUCKLING_V1_DEV"

def physical_footprint_capabilities(owner):
    from ._stiffened_shell_b2_physical_rigid_footprint import BUCKLING_POLICY
    reasons=[]
    if not getattr(owner,"_physical_footprint_law",False): reasons.append("physical footprint family required")
    if owner._closed: reasons.append("owner closed")
    if owner.point_ids: reasons.append("mixed POINT / physical footprint spectral operation unavailable")
    if np.any(owner._u0 != 0.): reasons.append("prescribed prestress split unavailable")
    if any(len(shell.node_ids)!=4 for shell,*_ in owner._buckling_shells): reasons.append("S3/mixed physical footprint buckling unsupported; Q4 director chart required")
    return B2ReferenceBucklingCapabilities(not reasons,tuple(reasons),policy=BUCKLING_POLICY,
        scope="zero-prescription Cartesian dead-force Q4 endpoint complete physical rigid footprints",shell_kernel_policies=(Q4_KERNEL_POLICY+":NATIVE_PHYSICAL_EXP_DIRECTOR_FIRST_WORK_PULLBACK",))

def capabilities(owner):
    reasons=[]
    if owner._closed: reasons.append("owner closed")
    if owner._modal_only: reasons.append("modal-only boundary")
    if owner.footprints: reasons.append("footprint second variation unavailable")
    if owner._point_trace_policy is not None: reasons.append("loaded point trace unavailable")
    if np.any(owner._u0 != 0.): reasons.append("prescribed prestress split unavailable")
    if any(point.station not in (0.,1.) for point,*_ in owner._buckling_points): reasons.append("interior loaded port unavailable")
    kernels=tuple(sorted({S3_KERNEL_POLICY if len(shell.node_ids)==3 else Q4_KERNEL_POLICY for shell,*_ in owner._buckling_shells}))
    return B2ReferenceBucklingCapabilities(not reasons,tuple(reasons),shell_kernel_policies=kernels)

def _shell_resultants(owner, element, material, displacement, count):
    mesh=owner._buckling_model.mesh
    if count == 3:
        recovered=element.compute_variational_resultants(mesh,displacement,material)
    else:
        recovered=element.compute_stresses(mesh,displacement,material)
    return np.column_stack((recovered["membrane_resultants"],recovered["bending_resultants"],recovered["transverse_shear_resultants"]))

def compile_geometric(owner, request, preload, safe_point=lambda stage: None):
    """Assemble tensile-sign G; all constraint coordinates remain algebraic."""
    from ._stiffened_shell_b2_q4_panel_experimental import _scatter
    displacement=np.asarray(preload.displacements); multipliers=dict(preload.point_multipliers)
    sources={source.member_id:source for source in request.section_tractions}
    rows=[];cols=[];values=[];provenance=[]
    for member_id,(member,slots) in owner._members:
        safe_point("buckling member initial stress")
        transform=member.local_dof_map()
        local=member_green_hessian(member,transform@displacement[slots],sources[member_id])
        _scatter(rows,cols,values,slots,transform.T@local@transform)
        provenance.append((member_id,sources[member_id].identity))
    for shell,element,material_id,slots in owner._buckling_shells:
        safe_point("buckling shell preload recovery")
        material=owner._buckling_model.get_material(material_id)
        count=len(shell.node_ids); local_u=displacement[slots]
        recovered=_shell_resultants(owner,element,material,local_u,count)
        if not np.isfinite(recovered).all(): raise ValueError("nonfinite shell preload recovery")
        # Initially require exact zero forbidden resultants; no dimensional
        # floors or specimen-adjusted arithmetic tolerance conceal prestress.
        if np.any(recovered[:,6:] != 0.): raise ValueError("shell transverse-shear prestress unsupported")
        if count == 3:
            if np.any(recovered[:,3:6] != 0.): raise ValueError("S3 bending prestress unsupported")
            if not np.array_equal(recovered[:,:3],np.broadcast_to(recovered[0,:3],recovered[:,:3].shape)):
                raise ValueError("S3 nonuniform membrane preload unsupported")
            state={"membrane_compression":-recovered[0,:3]}
            kernel_policy=S3_KERNEL_POLICY
        else:
            from ._stiffened_shell_b2_physical_rigid_footprint import q4_kernel_frame_resultants
            kernel_resultants=q4_kernel_frame_resultants(element,owner._buckling_model.mesh,material,local_u,recovered)
            state={"membrane_compression_at_gauss":-kernel_resultants[:,:3],
                   "bending_compression_at_gauss":-kernel_resultants[:,3:6],
                   "stress_second_moment_at_gauss":-kernel_resultants[:,:3]*shell.thickness_m**2/12.}
            kernel_policy=Q4_KERNEL_POLICY
        # Existing shell KG uses compression positive; convert exactly once.
        local=-element.compute_geometric_stiffness_matrix(owner._buckling_model.mesh,material,state)
        if getattr(owner,"_physical_footprint_law",False):
            from ._stiffened_shell_b2_physical_rigid_footprint import q4_exp_director_chart_correction
            local=local+q4_exp_director_chart_correction(element,owner._buckling_model.mesh,material,local_u)
            kernel_policy += ":NATIVE_PHYSICAL_EXP_DIRECTOR_FIRST_WORK_PULLBACK"
        _scatter(rows,cols,values,slots,local)
        provenance.append((shell.shell_id,kernel_policy,hashlib.sha256(recovered.tobytes()).hexdigest()))
    for point,law,vertices,member,slots in owner._buckling_points:
        safe_point("buckling point multiplier curvature")
        local=point_green_hessian(law,vertices,member,point.station,point.offset_local_m,multipliers[point.point_id])
        _scatter(rows,cols,values,slots,local)
    if getattr(owner,"_physical_footprint_law",False):
        all_lambda=np.asarray(preload.constraint_multipliers)
        for item in owner._footprint_incidence:
            safe_point("buckling physical footprint multiplier curvature")
            slots=tuple(item.shell_slots)+tuple(item.beam_slots)
            local=item.multiplier_hessian(all_lambda[item.row_start:item.row_start+5])
            _scatter(rows,cols,values,slots,local)
            provenance.append((item.footprint_id,item.shell_node,"physical_director_five_row",hashlib.sha256(item.constraint_hessians.tobytes()).hexdigest()))
    g=sparse.coo_matrix((values,(rows,cols)),shape=owner._K.shape).tocsr()
    if not np.isfinite(g.data).all(): raise ValueError("initial-stress assembly not representable")
    digest=hashlib.sha256((POLICY+PORT_POLICY+repr(provenance)).encode())
    digest.update(owner._body_model_identity.encode())
    for matrix in (owner._K,owner._C,owner._T,g):
        for a in (matrix.indptr,matrix.indices,matrix.data): digest.update(a.tobytes())
    digest.update(np.asarray(request.preload).tobytes())
    return g,digest.hexdigest()

def solve(owner, request, *, cancellation=None):
    from .control import CancellationToken,SolveCancelled
    from .linalg import factorize_cached,MatrixClass
    from ._stiffened_shell_b2_q4_panel_experimental import _scaled_point_kkt,_modal_scaled_residual
    physical=type(request) is B2PhysicalFootprintBucklingRequest
    if type(request) not in (B2ReferenceBucklingRequest,B2PhysicalFootprintBucklingRequest): raise ValueError("typed physical B2 reference-buckling request required")
    result_type=B2PhysicalFootprintBucklingResult if physical else B2ReferenceBucklingResult
    admission=physical_footprint_capabilities(owner) if physical else capabilities(owner)
    if not admission.eligible: raise ValueError("; ".join(admission.reasons))
    load=np.asarray(request.preload)
    if load.shape != owner._u0.shape or np.any(load.reshape(-1,6)[:,3:] != 0.): raise ValueError("Cartesian dead nodal forces only; rotational couples unsupported")
    if {s.member_id for s in request.section_tractions} != {m for m,_ in owner._members}: raise ValueError("explicit physical traction source for every member required")
    if cancellation is not None and not isinstance(cancellation,CancellationToken): raise ValueError("CancellationToken required")
    n=owner._T.shape[1];q=owner._C.shape[0]
    if request.count >= min(n-q,n+q-1): raise ValueError("buckling count exceeds supported sparse spectrum")
    def safe(stage):
        if cancellation is not None: cancellation.checkpoint(stage)
    preload=None;identity=""
    try:
        safe("buckling before reference preload")
        preload=owner.solve_linear(load)
        safe("buckling after reference preload")
        if preload.status != "completed":
            return result_type("failed",preload_result=preload,
                failure_reason="reference preload status "+str(preload.status)+": "+str(preload.failure_reason))
        g,identity=compile_geometric(owner,request,preload,safe)
        k=(owner._T.T@owner._K@owner._T).tocsc(); c=(owner._C@owner._T).tocsc()
        destabilizing=-(owner._T.T@g@owner._T).tocsc()
        a=sparse.bmat(((k,c.T),(c,None)),format="csc")
        b=sparse.block_diag((destabilizing,sparse.csc_matrix((q,q))),format="csc")
        # Positive coordinate normalization belongs to reference K/C only.
        # A valid shifted indefinite pencil must never face the static guard.
        _,scale=_scaled_point_kkt(k,c)
        b.eliminate_zeros();b.sort_indices()
        scaling=sparse.diags(scale,format="csc");bs=(scaling@b@scaling).tocsc();bs.eliminate_zeros();bs.sort_indices()
        if (not np.isfinite(bs.data).all() or b.nnz != bs.nnz or
            not np.array_equal(b.indptr,bs.indptr) or not np.array_equal(b.indices,bs.indices) or
            np.any((b.data != 0.) & (bs.data == 0.))): raise _BucklingNumericalFailure("buckling scaling loses nonzero entries")
        raw_shifted=(a-request.shift*b).tocsc();raw_shifted.eliminate_zeros();raw_shifted.sort_indices()
        shifted=(scaling@raw_shifted@scaling).tocsc();shifted.eliminate_zeros();shifted.sort_indices()
        if (not np.isfinite(shifted.data).all() or shifted.nnz!=raw_shifted.nnz or
            not np.array_equal(shifted.indptr,raw_shifted.indptr) or not np.array_equal(shifted.indices,raw_shifted.indices) or
            np.any((raw_shifted.data!=0.)&(shifted.data==0.))):
            raise _BucklingNumericalFailure("shifted buckling scaling loses nonzero entries")
        safe("buckling before factorization")
        handle=factorize_cached(shifted,MatrixClass.GENERAL)
        if handle.status != "ok": raise _BucklingNumericalFailure("buckling factorization failed: "+str(handle.failure_reason))
        def action(v):
            safe("buckling iteration")
            try:
                answer=np.asarray(handle.solve(bs@v))
            except RuntimeError as exc:
                if (str(exc) == "Sparse solve produced NaN/Inf values" or
                    str(exc).startswith("Cannot solve with failed factorization:")):
                    raise _BucklingNumericalFailure("buckling sparse solve failed: "+str(exc)) from exc
                raise
            if not np.isfinite(answer).all(): raise _BucklingNumericalFailure("nonfinite buckling inverse action")
            return answer
        eigvalues,vectors=eigs(LinearOperator(a.shape,matvec=action,dtype=float),k=request.count,which="LM",v0=np.sin(np.arange(a.shape[0])+.23),tol=1e-11,maxiter=request.max_iterations)
        modes=[]
        for mu,y in zip(eigvalues,vectors.T):
            if not np.isfinite(mu) or abs(mu.imag)>1e-9*abs(mu.real) or mu.real == 0. or np.linalg.norm(y.imag)>1e-9*np.linalg.norm(y.real): raise _BucklingNumericalFailure("unresolved buckling pair")
            lam=request.shift+1./mu.real
            if not np.isfinite(lam) or lam <= 0.: raise _BucklingNumericalFailure("requested shift selected nonpositive buckling factor")
            x=scale*y.real;physical=np.asarray(owner._T@x[:n]);norm=np.linalg.norm(physical)
            if not np.isfinite(norm) or norm == 0.: raise _BucklingNumericalFailure("invalid physical buckling mode")
            x/=norm;physical/=norm
            residual=a@x-lam*(b@x);bound=abs(a)@abs(x)+abs(lam)*(abs(b)@abs(x))
            equation=_modal_scaled_residual(residual,bound,scale)
            constraint=_modal_scaled_residual(owner._C@physical,abs(owner._C)@abs(physical),scale[n:])
            balance=owner._K@physical+owner._C.T@x[n:]+lam*(g@physical)
            physical_bound=abs(owner._K)@abs(physical)+abs(owner._C.T)@abs(x[n:])+abs(lam)*(abs(g)@abs(physical))
            equation=max(equation,_modal_scaled_residual(owner._project(balance),abs(owner._T.T)@physical_bound,scale[:n]))
            if equation>1e-8 or constraint>1e-9: raise _BucklingNumericalFailure("buckling physical equation or constraint residual failed")
            modes.append((float(lam),tuple(physical),tuple(x[n:]),equation,constraint))
        modes.sort(key=lambda item:item[0]);safe("buckling before publication")
        return result_type("ok",*(tuple(m[i] for m in modes) for i in range(5)),preload,identity)
    except SolveCancelled as exc:
        return result_type("cancelled",preload_result=preload,identity=identity,failure_reason=str(exc))
    except (ArpackError,_BucklingNumericalFailure) as exc:
        return result_type("failed",preload_result=preload,identity=identity,failure_reason=str(exc))
