"""Matched geometry / common-work development tool, never a response comparator.

Public native reference-linear definition. The continuum's five collocated
stations and native four weak supports are deliberately NOT identified.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator
from scripts import r8_bl5_released_director_reference as external

IDENTITY = 'R8_BL5_MATCHED_NATIVE_REFERENCE_LINEAR_DEV_V1'
XS = (0., .25, .5, .75, 1.)


def rectangular_section(width=.04, height=.06, young=210e9, nu=.3, density=7850., terms=1000):
    """Exact area/inertias; parabolic complementary-energy shear; SV series.

    For long side a >= short side b, solve -Laplacian(Psi)=2 with Psi=0
    on the rectangle using the odd Fourier sine series. J=2 integral(Psi):
    ab^3/3 [1-192 b/(pi^5 a) sum_odd tanh(n pi a/(2b))/n^5].
    Dropped tanh <= 1 and sum_{n>N} n^-5 <= integral_N^inf x^-5 dx
    give the deliberately conservative absolute J error 16 b^4/(pi^5 N^4).
    Integrating tau=3V/(2A)(1-4z^2/h^2) gives integral tau^2 dA
    =6 V^2/(5A), hence k=5/6 in BOTH rectangular directions.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (width,height,young,density)) or not -1 < nu < .5:
        raise ValueError('positive rectangle/material and admissible Poisson ratio required')
    if type(terms) is not int or terms < 2:
        raise ValueError('at least two odd Fourier terms required')
    a,b=max(width,height),min(width,height); last=2*terms-1
    series=math.fsum(math.tanh(n*math.pi*a/(2*b))/n**5 for n in range(1,last+1,2))
    area=width*height; iy=width*height**3/12; iz=height*width**3/12
    j=a*b**3/3*(1-192*b/(math.pi**5*a)*series)
    g=young/(2*(1+nu)); k=5/6
    inertia=density*np.diag((area,area,area,iy+iz,iy,iz))
    stiffness=np.array((young*area,k*g*area,k*g*area,g*j,young*iy,young*iz))
    return dict(area=area,Iy=iy,Iz=iz,J=j,J_tail_bound=16*b**4/(math.pi**5*last**4),
                shear_factor=k,G=g,stiffness=stiffness,mass=inertia,density=density)


def definition(density=7850.):
    """Eight Q4 / two physical biaxial fibre B3 members; exact adopted geometry."""
    data=rectangular_section(density=density); e=210e9
    fibres=tuple(b3_ge.Fibre(str(i),sy*.04/math.sqrt(12),sz*.06/math.sqrt(12),
        data['area']/4,e,b3_ge.FlowCurve((0.,),(1e20,),1e9))
        for i,(sy,sz) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))))
    background=data['stiffness'].copy(); background[[0,4,5]]=0.
    section=b3_ge.PhysicalFibreSection(fibres,np.diag(np.sqrt(background)))
    positions=np.array([(x,0.,.04) for x in XS])
    beams=tuple(b3_ge.define_beam('b3-ge',i+1,(101+2*i,102+2*i,103+2*i),
        positions[2*i:2*i+3],np.tile(np.eye(3),(3,1,1)),section,data['mass']) for i in range(2))
    coordinates=tuple((x,y,0.) for y in (-.1,0.,.1) for x in XS)
    cells=tuple((1+5*j+i,2+5*j+i,7+5*j+i,6+5*j+i) for j in range(2) for i in range(4))
    panel=ss.ShellPanel(tuple(range(1,16)),coordinates,cells,.02,e,.3,density)
    line=ss.ExperimentalBL5Line((6,7,8,9,10),((0,0),(0,1),(1,0),(1,1)),
        multiplier_span=1,expected_beam_arm=(0.,0.,-.03),basis='p0')
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,panel,beams,(),
        tuple(ss.FixedCoordinate('shell',n,j) for n in (1,6,11) for j in range(6)),(line,),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        bl5_material_policy=ss.BL5_ELASTIC_NATIVE_FIBRE_V1)


def reference_state(owner):
    positions=np.array([owner._beam_nodes[n][0] for n in owner._beam_ids])
    return Mechanical(positions,np.zeros_like(positions),np.tile(np.eye(3),(5,1,1)),
                      np.tile(np.eye(3),(2,2,1,1)),np.zeros((2,18)))


def input_record(model):
    """Explicit public-definition inputs, virgin native bytes and physical measures."""
    data=rectangular_section(density=model.shell.density_kg_m3)
    return dict(identity=IDENTITY,profile=model.profile,shell=asdict(model.shell),
        beams=[dict(sha256=b.sha256,definition=json.loads(b.raw)) for b in model.beams],
        supports=[asdict(f) for f in model.fixed],lines=[line.descriptor() for line in model.lines],
        actions=[dict(domain='beam',node=105,wrench=(0.,0.,1.,0.,0.,0.))],
        q4_pullback_policy=model.q4_pullback_policy,bl5_material_policy=model.bl5_material_policy,
        section={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in data.items()},
        continuum_model_sha256='a22170179b6e257a47288ff825a3b69ad503e5e3c4a040fcd5f3b53933c86463',
        response_comparison_permitted=False)


def common_vector(owner, shell_translation, shell_spin, beam_translation, beam_spin, *, supported=False):
    """Reference virtual field: independent body translations / section spins.

    Each translation callable acts on its midsurface/centroid reference point.
    A single beam spin is shared by all nodal AND retained section coordinates.
    Thus the continuum five-station image exists. Arbitrary broken rotations
    are excluded; no averaging repairs their two-sided station values.
    This function alone constructs a NATIVE field, not an external lift proof.
    supported=True zeros the actual fixed shell coordinates (not a rigid body).
    Exact common global supported fields use global_common_field below.
    """
    q=np.zeros(owner._K.shape[0]); shell=q[:owner._shell_count].reshape(-1,6)
    for i,p in enumerate(owner._definition.shell.coordinates_m):
        shell[i,:3]=shell_translation(np.array(p)); shell[i,3:]=shell_spin
    for i,n in enumerate(owner._beam_ids):
        start=owner._shell_count+6*i
        q[start:start+3]=beam_translation(np.array(owner._beam_nodes[n][0])); q[start+3:start+6]=beam_spin
    retained=owner._shell_count+6*len(owner._beam_ids)
    for member in range(2):
        for cell in range(2): q[retained+24*member+3*cell:retained+24*member+3*cell+3]=beam_spin
    if supported: q[owner._fixed]=0.
    return q


def global_common_field(owner, shell_affine, shell_spin, beam_affine, beam_spin, *, supported=False):
    """Globally affine material-director / rigid bar-section common field.

    Affine arrays contain [constant, x-slope] per spatial component. Exact
    support admissibility requires zero shell constant AND shell spin; clipping
    arbitrary rotations at the native root would create a non-common kink.
    """
    sa,ba=np.asarray(shell_affine,dtype=float),np.asarray(beam_affine,dtype=float)
    ts,tb=np.asarray(shell_spin,dtype=float),np.asarray(beam_spin,dtype=float)
    if sa.shape != (3,2) or ba.shape != (3,2) or ts.shape != (3,) or tb.shape != (3,) or not all(np.isfinite(v).all() for v in (sa,ba,ts,tb)):
        raise ValueError('finite affine translations and section spins required')
    if supported and (np.any(sa[:,0]) or np.any(ts)):
        raise ValueError('common supported lift requires zero shell constant and spin')
    q=common_vector(owner,lambda p:sa[:,0]+p[0]*sa[:,1],ts,
                    lambda p:ba[:,0]+p[0]*ba[:,1],tb)
    _,model=external.build_reference(); field={}
    for n,p in model['physical_nodes'].items():
        if n in model['shell_nodes']:
            field[n]=sa[:,0]+p[0]*sa[:,1]+np.cross(ts,(0.,0.,p[2]))
        else:
            field[n]=ba[:,0]+p[0]*ba[:,1]+np.cross(tb,(0.,p[1],p[2]-.04))
    return q,model,field


def continuum_value(model,field,point,body):
    """Actual C3D20 interpolation at a material face point; fixed side rule."""
    candidates=[]
    for eid in model[body+'_elements']:
        ids=model['elements'][eid]; xyz=np.array([model['physical_nodes'][n] for n in ids])
        low,high=xyz.min(axis=0),xyz.max(axis=0)
        if np.all(np.array(point)>=low-1e-14) and np.all(np.array(point)<=high+1e-14):
            natural=2*(np.asarray(point)-low)/(high-low)-1
            shape,_=external._shape(natural)
            candidates.append(shape@np.array([field[n] for n in ids]))
    if not candidates: raise ValueError('point outside declared continuum material body')
    if max(np.max(abs(v-candidates[0])) for v in candidates)>1e-12:
        raise ValueError('global common material field has inconsistent shared faces')
    return candidates[0]


def global_correspondence(owner,q,model,field):
    """Physical five-channel gap on actual interpolated continuum face pairs."""
    state=reference_state(owner); rows=[]
    for op,slots,weight,edge,alpha in owner._line_samples:
        x=XS[edge]+.25*alpha
        faces={name:continuum_value(model,field,(x,0.,z),body) for name,z,body in
            (('shell_bottom',-.01,'shell'),('shell_top',.01,'shell'),
             ('beam_bottom',.01,'beam'),('beam_top',.07,'beam'))}
        station={name:i+1 for i,name in enumerate(faces)}
        eg=external.gap(station,{station[n]:v for n,v in faces.items()})
        j=op.evaluate(np.zeros((2,6)),state,np.zeros(5)).gap_jacobian
        rows.append((j@q[slots],eg,weight,edge))
    return rows


def concentrated_from_density(density):
    """Common affine-gap dual only: lambda_external=-L/2 adjacent densities.

    Native input is channel-major (5,4), units N/m and N. External output
    station-major (5,5), units N and Nm. This is a work pullback on the
    declared continuous common subspace, not an invertible row-space map.
    """
    density=np.asarray(density,dtype=float)
    if density.shape!=(5,4) or not np.isfinite(density).all():
        raise ValueError('finite five-channel/four-support densities required')
    result=np.zeros((5,5))
    for edge in range(4):
        result[edge]-=.125*density[:,edge]; result[edge+1]-=.125*density[:,edge]
    return result


def common_origin_wrench(nodes,action,origin=(0.,0.,0.)):
    """Sum material force/couple first, then transport the common origin.

    Algebraically sum((point-origin) cross force), with the common gauge term
    factored before subtraction. This avoids cancellation of large equal
    origin terms from opposing material face forces; no mechanical map changes.
    """
    force=np.array([math.fsum(float(v[j]) for v in action.values()) for j in range(3)])
    terms=[np.cross(nodes[n],v) for n,v in action.items()]
    moment=np.array([math.fsum(float(v[j]) for v in terms) for j in range(3)])
    return np.r_[force,moment-np.cross(np.asarray(origin),force)]


def external_station_field(shell_mid, shell_spin, beam_mid, beam_spin):
    """Material face-pair lift; input midsurface/centroid translations at ONE x."""
    sm,sb=np.asarray(shell_mid),np.asarray(beam_mid)
    ts,tb=np.asarray(shell_spin),np.asarray(beam_spin)
    return {'shell_bottom':sm+np.cross(ts,(0.,0.,-.01)),
            'shell_top':sm+np.cross(ts,(0.,0.,.01)),
            'beam_bottom':sb+np.cross(tb,(0.,0.,-.03)),
            'beam_top':sb+np.cross(tb,(0.,0.,.03))}


def external_gap_and_work(shell_mid,shell_spin,beam_mid,beam_spin,multiplier):
    """Original external physical rows and transpose, with material-face arms."""
    fields=external_station_field(shell_mid,shell_spin,beam_mid,beam_spin)
    station={name:i+1 for i,name in enumerate(fields)}
    u={station[name]:v for name,v in fields.items()}
    gap=external.gap(station,u); action=external.transpose_action(station,multiplier)
    work=math.fsum(float(action[n]@u[n]) for n in action)
    return gap,work,action


def correspondence(owner,q):
    """Local synthetic face-pair algebra at eight actual native Gauss samples.

    Alone this is not a global continuum representability proof; use the actual
    global_common_field / global_correspondence route for that distinct claim.
    """
    state=reference_state(owner); out=[]
    for op,slots,weight,edge,alpha in owner._line_samples:
        zero=op.evaluate(np.zeros((2,6)),state,np.zeros(5))
        native=zero.gap_jacobian@q[slots]
        shell=q[slots[:12]].reshape(2,6); s=op.station
        us=(1-s)*shell[0,:3]+s*shell[1,:3]; ts=(1-s)*shell[0,3:]+s*shell[1,3:]
        local=q[slots[12:]]; ub=(1-op.port.weight)*local[:3]+op.port.weight*local[3:6]; tb=local[6:]
        lam=np.array((2.,-3.,5.,7.,-11.)) # declared arbitrary work probe, not fitted
        eg,work,_=external_gap_and_work(us,ts,ub,tb,-lam)
        out.append(dict(edge=edge,alpha=alpha,weight=weight,
            gap_error=float(np.max(abs(native+eg))),
            work_error=abs(float(lam@native)-work),reference_arm=op.reference_arm.tolist()))
    return out


def station_sides(owner, x):
    """Exact endpoint coverage: never identifies two independent retained spins."""
    sides=[]; state=reference_state(owner)
    for edge,(member,cell) in enumerate(owner._definition.lines[0].beam_halfcells):
        for alpha in (0.,1.):
            if XS[edge]+alpha*.25 != x: continue
            ref=owner._beam_parts[member][0].operator.reference
            nodes=tuple(owner._beam_ids.index(n) for n in owner._beam_parts[member][0].node_ids)
            port=NativeHalfCellPort.from_reference(ref,nodes,member,cell,alpha,30,78)
            shell_ref=np.array([(XS[edge],0.,0.),(XS[edge+1],0.,0.)])
            op=BL5LocalOperator.at_reference(shell_ref,np.zeros((2,6)),.02,port,state,station=alpha)
            sides.append(op)
    return tuple(sides)


def calibration(owner):
    """ONE virgin public solve; no continuum displacement response scoring."""
    result=owner.solve_linear((ss.NodalAction('beam',105,(0.,0.,1.,0.,0.,0.)),))
    if result.status != 'completed': raise RuntimeError('native reference-linear solve did not complete')
    support=result.reactions[:owner._shell_count].reshape(-1,6)
    positions=np.array(owner._definition.shell.coordinates_m)
    wrench=np.r_[support[:,:3].sum(axis=0),
        (support[:,3:]+np.cross(positions,support[:,:3])).sum(axis=0)]
    applied=np.array((0.,0.,1.,0.,-1.,0.)); q=np.r_[result.shell_displacement,result.beam_coordinates]
    load=np.zeros_like(q); load[owner._shell_count+6*owner._beam_ids.index(105)+2]=1.
    applied_work=float(load@q); internal_work=float(q@(owner._K@q))
    interface_work=float(result.multipliers@(owner._C@q))
    denominator=max(abs(applied_work),1e-12) # declared J floor, never fitted
    return dict(identity=IDENTITY,status=result.status,production_qualified=False,comparison_permitted=False,
        free_equilibrium_ratio=result.free_equilibrium_ratio,constraint_max=result.constraint_max,
        support_wrench=wrench.tolist(),external_wrench_residual=(wrench+applied).tolist(),
        applied_virtual_work=applied_work,internal_pairing=internal_work,
        interface_pairing=interface_work,work_denominator_J=denominator,
        internal_work_relative_error=abs(internal_work-applied_work)/denominator,
        interface_work_relative_error=abs(interface_work)/denominator,
        shell_nodes=15,q4_cells=8,beam_nodal_dofs=30,retained_dofs=48,primal_dofs=len(q),
        native_rows=owner._C.shape[0],native_samples=len(owner._line_samples),
        native_measure='four half-cell supports: arclength-integrated five-channel densities',
        external_measure='five collocated station force/director moments, 25 rows',
        missing='body discretization and strong/weak multiplier-space convergence/correspondence; no response score')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv); args.output.mkdir(exist_ok=False)
    model=definition()
    (args.output/'input.json').write_text(json.dumps(input_record(model),indent=2)+'\n')
    owner=ss.create_analysis(definition=model)
    try:
        result=calibration(owner)
        (args.output/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    finally: owner.close()


if __name__ == '__main__': main()
