"""Registered parity grid/shared-native-chain source proof, DEVELOPMENT ONLY.

Sparse global dictionaries, local exact source work blocks; no numeric imports
at module load, global inverse/nullspace, candidate margin, or new physics.
"""
from dataclasses import replace
from fractions import Fraction as F
import math
from scripts import r8_f03_physical_interval_dev as physical
from scripts.r8_f03_finite_local_dev import (ExactBudget,Unresolved,zero,identity,solve,multiply,
    transpose,isotropic_shell,rectangle_work,local_reconstruction,native_halfcell,close_yaw,geometry_constant)
from scripts.r8_f03_finite_sparse_dev import normal

FAMILY={16:(16,4,16),32:(32,8,32),64:(64,16,64)}
CAPS={'global_dimension':8000,'global_entries':1_000_000,'exact_operations':20_000_000,'exact_bits':4096,
    'factor_fill':1_000_000,'interval_operations':30_000_000,'interval_bits':4096,
    'shell_cells':1024,'native_members':32}
ELL=F(2,5); TOP=F(7,400); ARM=F(-1,20); WIDTH=F(1,5)


def add(entries,i,j,value,budget=None):
    if value:
        made=entries.get((i,j),F())+value
        entries[i,j]=budget.check(made) if budget else normal(made)
        if len(entries)>CAPS['global_entries']:raise Unresolved('growing sparse entry cap')


def layout(level):
    if type(level)is not int or level not in FAMILY:raise Unresolved('registered parity levels only')
    n,ny,hc=FAMILY[level]; sc=6*(n+1)*(ny+1); nodal=6*(n+1); members=hc//2
    nfull=sc+nodal+6*members; np=18*members; m=5*members
    fixed=set(6*(row*(n+1))+axis for row in range(ny+1) for axis in range(6))|set(range(sc,sc+5))
    xf=tuple(i for i in range(nfull) if i not in fixed); free=xf+tuple(range(nfull,nfull+np+m))
    if len(free)>CAPS['global_dimension']:raise Unresolved('growing dimension cap')
    return {'level':level,'n':n,'ny':ny,'members':members,'shell_count':sc,'nodal':nodal,
        'nx_full':nfull,'nx':len(xf),'np':np,'m':m,'dim':len(free),'fixed':tuple(sorted(fixed)),
        'physical_free':xf,'full_free':free,'index':{old:new for new,old in enumerate(free)}}


def shell_norm(a,b,budget):
    """Local exact contribution of frozen five-field H1 and drilling Gz."""
    g=zero(24,24)
    for r,wr in ((F(-1),F(1,3)),(F(),F(4,3)),(F(1),F(1,3))):
        for s,ws in ((F(-1),F(1,3)),(F(),F(4,3)),(F(1),F(1,3))):
            signs=((-1,-1),(1,-1),(1,1),(-1,1))
            shape=[(1+ri*r)*(1+si*s)/4 for ri,si in signs]
            dx=[F(ri,4)*(1+si*s)/a for ri,si in signs];dy=[F(si,4)*(1+ri*r)/b for ri,si in signs]
            measure=a*b*wr*ws/F(1,5);disc={}
            for i in range(4):
                disc[6*i]=ELL*dy[i]/2;disc[6*i+1]=-ELL*dx[i]/2;disc[6*i+5]=shape[i]
                for j in range(4):
                    scalar=measure*(shape[i]*shape[j]+ELL**2*(dx[i]*dx[j]+dy[i]*dy[j]))
                    for axis in range(5):g[6*i+axis][6*j+axis]=budget.check(g[6*i+axis][6*j+axis]+scalar)
            for i,x in disc.items():
                for j,y in disc.items():g[i][j]=budget.check(g[i][j]+measure*x*y)
    return g


def geometry_maps(info,work,budget):
    """Common global y norm; each row-band proof uses its own subset once.

    Shared coordinates receive max(valid per-band bounds). Multiple coordinate
    inequalities may reuse a work row; no repeated energy is added to ||y||².
    """
    n,ny,sc=info['n'],info['ny'],info['shell_count'];h=F(1,n)
    rec=local_reconstruction(work['rows_z'],tuple(range(6,18)),tuple(range(6))+tuple(range(18,24)),budget)
    bounds=[F()]*info['nx_full']
    for band in range(ny):
        left=[F()]*12
        for col in range(n):
            right=[budget.check(sum(map(abs,inv),F())+sum((abs(v)*b for v,b in zip(tr,left)),F()))
                for inv,tr in zip(rec['inverse'],rec['transport'])]
            for side in range(2):
                node=(band+side)*(n+1)+col+1
                for axis in range(6):bounds[6*node+axis]=max(bounds[6*node+axis],right[6*side+axis])
            left=right
    phibase=sc+info['nodal']
    for cell in range(n):
        left,right,phi=sc+6*cell,sc+6*(cell+1),phibase+3*cell
        for axis in range(3):
            bounds[right+3+axis]=budget.check(bounds[left+3+axis]+1)
            bounds[phi+axis]=budget.check(bounds[left+3+axis]+F(3,2))
        for axis in range(3):
            extra=h/ELL*bounds[phi+(2 if axis==1 else 1)] if axis in (1,2) else 0
            bounds[right+axis]=budget.check(bounds[left+axis]+1+extra)
    actions=[]
    for member in range(info['members']):
        rows=[{} for _ in range(5)]
        def put(c,i,v):rows[c][i]=budget.check(rows[c].get(i,F())+v)
        for cell in (2*member,2*member+1):
            for node in (cell,cell+1):
                for axis in range(3):put(axis,6*node+axis,h/2);put(axis,sc+6*node+axis,-h/2)
                put(0,6*node+4,h*TOP/(2*ELL));put(1,6*node+3,-h*TOP/(2*ELL))
                put(3,6*node+4,h/(2*ELL));put(4,6*node+3,-h/(2*ELL))
            phi=phibase+3*cell
            put(0,phi+1,-ARM*h/ELL);put(1,phi,ARM*h/ELL)
            put(3,phi+1,-h/ELL);put(4,phi,h/ELL)
        actions.extend(rows)
    c=[F()]*info['nx_full'];rigid=[F()]*len(c)
    for member in range(info['members']):
        for i,v in actions[5*member+1].items():c[i]=budget.check(c[i]+v)
    for node in range(n+1):rigid[sc+6*node+1]=node*h;rigid[sc+6*node+5]=ELL
    for cell in range(n):rigid[phibase+3*cell+2]=ELL
    closed,cr=close_yaw(bounds,rigid,c,budget)
    if cr!=-F(1,2):raise Unresolved('actual global P0 ty yaw sum identity')
    pivots=[];minors=[]
    for member in range(info['members']):
        pivot=tuple(sc+6*(2*member+1)+axis for axis in range(3))+(phibase+6*member,phibase+6*member+1)
        minor=[[actions[5*member+i].get(j,F()) for j in pivot] for i in range(5)]
        inv=solve(minor,identity(5),budget,order_cap=5)
        if multiply(minor,inv,budget)!=identity(5):raise Unresolved('growing P0 local5 minor identity')
        for row,action in enumerate(actions):
            if row//5!=member and any(action.get(j,F()) for j in pivot):raise Unresolved('P0 minor cross-support contamination')
        pivots.extend(pivot);minors.append({'columns':pivot,'minor':minor,'inverse':inv})
    if len(set(pivots))!=info['m'] or any(j in info['fixed'] for j in pivots):raise Unresolved('supported distinct P0 pivot inventory')
    return {'bounds':closed,'unclosed_bounds':bounds,'rigid':rigid,'actions':actions,'yaw_row':c,'yaw_action':cr,
        'local_frontier':rec,'P0_minors':minors,'full_C_rank':info['m'],
        'multiplicity':'individual coordinate bound in one common ||y_all||2; repeated rows counted in absolute coefficients only'}


def exact_reference(level):
    info=layout(level);n,ny=info['n'],info['ny'];a,b=F(1,2*n),F(1,10*ny);h=F(1,n)
    budget=ExactBudget(operation_cap=CAPS['exact_operations'],bit_cap=CAPS['exact_bits'])
    constitutive,drill=isotropic_shell(F(210_000_000_000),F(3,10),F(7,200))
    work=rectangle_work(a,b,ELL,constitutive,drill,budget);maps=geometry_maps(info,work,budget)
    cn,cm=physical.original_native_section();native=native_halfcell(h,ELL,physical.diagonal(cn),physical.diagonal(cm),budget)
    bi,di,bproof=physical.native_exact_b_d(replace(physical.Inputs(),a=a),budget)
    kl=multiply(multiply(transpose(work['rows_z']),work['H_N_per_m'],budget),work['rows_z'],budget)
    gl=shell_norm(a,b,budget);full={};gfull={};sc=info['shell_count'];phibase=sc+info['nodal'];npbase=info['nx_full']
    for row in range(ny):
        for col in range(n):
            nodes=(row*(n+1)+col,row*(n+1)+col+1,(row+1)*(n+1)+col+1,(row+1)*(n+1)+col)
            slots=[6*node+axis for node in nodes for axis in range(6)]
            for i in range(24):
                for j in range(24):add(full,slots[i],slots[j],kl[i][j],budget);add(gfull,slots[i],slots[j],gl[i][j],budget)
    for cell in range(n):
        for i in range(2):
            for j in range(2):
                mass=h*F(2 if i==j else 1,6);grad=F(1 if i==j else -1)/h;scalar=mass+ELL**2*grad
                for axis in range(6):add(gfull,sc+6*(cell+i)+axis,sc+6*(cell+j)+axis,scalar,budget)
        phi=phibase+3*cell
        for axis in range(3):
            extra=(ARM**2 if axis in (0,1) else F())+(WIDTH**2/12 if axis in (0,2) else F())
            add(gfull,phi+axis,phi+axis,h*(1+extra/ELL**2),budget)
        for node in (cell,cell+1):
            for axis,u,value in ((0,1,-ARM),(1,0,ARM)):
                add(gfull,phi+axis,sc+6*node+u,h*value/(2*ELL),budget);add(gfull,sc+6*node+u,phi+axis,h*value/(2*ELL),budget)
    for member in range(info['members']):
        xslots=[sc+6*(2*member+node)+axis for node in range(3) for axis in range(6)]+list(range(phibase+6*member,phibase+6*member+6))
        for i in range(18):
            p=npbase+18*member+i
            for j in range(24):add(full,p,xslots[j],bi[i][j],budget);add(full,xslots[j],p,bi[i][j],budget)
            for j in range(18):add(full,p,npbase+18*member+j,-di[i][j],budget)
    for row,action in enumerate(maps['actions']):
        scale=ELL if row%5>=3 else F(1);lm=npbase+info['np']+row
        for j,v in action.items():add(full,lm,j,scale*v,budget);add(full,j,lm,scale*v,budget)
    idx=info['index'];ideal={(idx[i],idx[j]):v for (i,j),v in full.items() if i in idx and j in idx and v}
    g={(idx[i],idx[j]):v for (i,j),v in gfull.items() if i in idx and j in idx and v}
    if any(ideal.get((j,i),F())!=v for (i,j),v in ideal.items()):raise Unresolved('growing exact ideal symmetry')
    cgeom=geometry_constant([(i,j,v) for (i,j),v in gfull.items()],maps['bounds'],budget)
    k0=min(work['bound']['k0_N_per_m'],native['bound']['k0_N_per_m']);kappa=budget.check(k0/cgeom)
    proof={'ideal_symmetric':True,'known_ideal_inertia':(info['nx'],info['np']+info['m'],0),
        'supported_kernel':'connected shell rooted left edge; native shared nodal rotations/translations leave only global yaw; actual sumP0_ty closes it',
        'native_local':bproof,'full_C_rank':maps['full_C_rank'],'P0_rank_method':'disjoint supported exact5x5 minors; no offsupport columns',
        'inertia_basis':'independent exact H/reconstruction/G, C rank and D positivity; signed Schur identity',
        'physical_acceptance':None,'qualification':False}
    return {**info,'ideal':ideal,'G':g,'D_local':di,'B_local':bi,'shell_work':work,'native_work':native,'maps':maps,
        'k0':k0,'Cgeom':cgeom,'kappa':kappa,'proof':proof,'exact_operations':budget.operations,'a':a,'b':b}


def source_stored(reference):
    if reference['level']!=16:raise Unresolved('actual source assessment only level16 until reviewed readiness')
    import numpy as np
    from scipy import sparse
    from anyfesolver import _stiffened_shell_ge_experimental as compiler
    from anyfesolver._stiffened_shell_l1_assembly_experimental import compile_lines
    from scripts.b3_e4_bl5_coherent_aggregation_dev import model
    n,ny=reference['n'],reference['ny'];definition,policy=model(n,ny,n,2)
    sm,sids,xyz,si,sparts=compiler._shell_reference(definition.shell)
    bids,bindex,bnodes,bparts,bcount,_=compiler._beam_reference(definition.beams)
    sc=reference['shell_count'];_,ss,bs=compiler._reference_scales(xyz,bnodes,bparts,sc,bcount)
    sr,sk,se=compiler._assemble_shell(sm,sparts,sc,ss,definition.q4_pullback_policy)
    br,bk,be=compiler._assemble_beam(bparts,bnodes,bindex,bcount,bs)
    c,samples=compile_lines(definition.lines,shell_ids=sids,shell_xyz=xyz,shell_cells=definition.shell.q4_elements,
        thickness=definition.shell.thickness_m,beam_ids=bids,beam_nodes=bnodes,beam_parts=bparts,shell_count=sc,beam_count=bcount)
    raw=sparse.block_diag((sk,bk),format='coo');mapping={i:i for i in range(sc+reference['nodal'])}
    base=sc+reference['nodal'];phibase=base
    for member in range(reference['members']):
        for i in range(6):mapping[base+24*member+i]=phibase+6*member+i
        for i in range(18):mapping[base+24*member+6+i]=reference['nx_full']+18*member+i
    if raw.shape!=(sc+bcount,)*2 or len(mapping)!=raw.shape[0] or c.shape!=(reference['m'],raw.shape[0]):raise Unresolved('actual growing compiler role mapping')
    zscale={old:(1./float(ELL) if (new<sc+reference['nodal'] and new%6>=3) or (sc+reference['nodal']<=new<reference['nx_full']) else 1.) for old,new in mapping.items()}
    full={}
    for i,j,value in zip(raw.row,raw.col,raw.data):add(full,mapping[int(i)],mapping[int(j)],normal(normal(float(value)*zscale[int(i)])*zscale[int(j)]))
    co=c.tocoo()
    row_permutation={rawrow:5*(rawrow%reference['members'])+rawrow//reference['members'] for rawrow in range(reference['m'])}
    for row,j,value in zip(co.row,co.col,co.data):
        channel=int(row)//reference['members'];logical=row_permutation[int(row)]
        v=normal(normal(float(value)*(float(ELL) if channel>=3 else 1.))*zscale[int(j)])
        lm=reference['nx_full']+reference['np']+logical;add(full,lm,mapping[int(j)],v);add(full,mapping[int(j)],lm,v)
    idx=reference['index'];entries={(idx[i],idx[j]):v for (i,j),v in full.items() if i in idx and j in idx and v}
    expected_fixed=tuple(sorted(reference['fixed']))
    observed_fixed=tuple(sorted((si[f.node]*6+f.component) if f.domain=='shell' else sc+bindex[f.node]*6+f.component for f in definition.fixed))
    if expected_fixed!=observed_fixed:raise Unresolved('actual supported growing inventory mismatch')
    return {'entries':entries,'adapter':'UNCHANGED_ACTUAL_GROWING_LOCAL_COMPILERS','policy':policy,'source_mapping':mapping,'P0_channel_to_member_row_permutation':row_permutation,
        'source_native_local_shape':(42,42),'native_physical':reference['nodal']+6*reference['members'],'resultants':reference['np'],
        'shell_cells':len(sparts),'members':len(bparts),'samples':len(samples),'owner_factorizations':0,'owner_solves':0,
        'reference_residual_max':float(max(np.max(abs(sr),initial=0.),np.max(abs(br),initial=0.))),
        'local_scaled_reference_errors':(float(se),float(be))}


def congruence(ref):
    x=normal(1./math.sqrt(float(ref['k0'])));j=[x]*ref['nx']
    j += [normal(1./math.sqrt(float(ref['D_local'][i][i]))) for member in range(ref['members']) for i in range(18)]
    for row,action in enumerate(ref['maps']['actions']):
        scalar=ELL if row%5>=3 else F(1);size=max(abs(v*scalar) for v in action.values())
        j.append(normal(1./(float(size)*x)))
    if len(j)!=ref['dim'] or any(v<=0 for v in j):raise Unresolved('invertible growing computational J')
    return j


def sparse_float(entries,dimension):
    from scipy import sparse
    if dimension>CAPS['global_dimension'] or len(entries)>CAPS['global_entries']:raise Unresolved('growing sparse float construction cap')
    items=sorted((key,v) for key,v in entries.items() if v)
    return sparse.coo_matrix(([normal(float(v)) for _,v in items],([k[0] for k,_ in items],[k[1] for k,_ in items])),
        shape=(dimension,dimension),dtype='float64').tocsc()


def endpoint_intervals(reference,stored,j,t,caps):
    """Own registered-growing caps; exact same declared scalar interpretation."""
    from scripts.r8_f03_finite_sparse_dev import directed
    center,radius,exact={},{},{}
    shift=t*reference['kappa']
    for i,k in set(reference['ideal'])|set(stored['entries'])|set(reference['G']):
        key=(i,k);gv=reference['G'].get(key,F())
        exact[key]=(reference['ideal'].get(key,F())-shift*gv)*F(j[i])*F(j[k])
        value=normal(stored['entries'].get(key,0.)-float(shift)*float(gv));value=normal(normal(value*j[i])*j[k])
        if value:center[key]=value
        upper=directed(abs(exact[key]-F(value)),True,caps)
        if upper:radius[key]=upper
    return {'center_entries':center,'radius_entries':radius,'exact':exact}
