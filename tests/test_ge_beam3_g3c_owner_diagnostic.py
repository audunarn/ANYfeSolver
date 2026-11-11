"""Nonclassifying component diagnosis of the preserved d90aa3e failure."""
import json
import numpy as np
from anyfesolver._ge_beam3_g3c_owner import MixedGraphOwner
from anyfesolver._ge_beam3_generalized_static_boundary import chart_pullback

def test_component_directional_diagnostic():
    owner=MixedGraphOwner(); u=np.linspace(-.002,.003,48)
    mu=np.tile([.3,-.2,.4,-.1,.25,-.15],4)
    cmd=dict(kind='LOAD_STAGE',load_factor=0.,force_scale=.01,root_stage=0)
    z=np.r_[u,mu]; direction=np.sin(np.arange(72)+1.); direction/=np.linalg.norm(direction)
    initial=owner.snapshot_bytes()
    def components(z):
        trial=owner.trial(z[:48],z[48:],cmd); candidate=json.loads(trial['candidate'])
        rows={'full':(trial['residual'],trial['tangent'])}
        schur=[]
        for i,row in enumerate(candidate['native_rows']):
            p=row['payload']; response=p['response']; mapping=np.arange(18*i,18*(i+1))
            f,K,_=chart_pullback(np.array(response['residual']),np.array(response['tangent']),
                               z[mapping].reshape(3,6)[:,3:])
            force=np.zeros(72); tangent=np.zeros((72,72)); force[mapping]=f; tangent[np.ix_(mapping,mapping)]=K
            rows['native:'+str(row['element_id'])]=(force,tangent)
            r=np.array(response['full']['residual']); A=np.array(response['full']['jacobian'])
            solved=np.linalg.solve(A[18:,18:],np.c_[r[18:],A[18:,:18]])
            schur.append(dict(element_id=row['element_id'],internal_error=response['internal_error'],
                force_error=float(np.linalg.norm(np.array(response['residual'])-(r[:18]-A[:18,18:]@solved[:,0]))),
                tangent_error=float(np.linalg.norm(np.array(response['tangent'])-(A[:18,:18]-A[:18,18:]@solved[:,1:])))))
        local=trial['diagnostics'][11]; force=np.zeros(72); tangent=np.zeros((72,72))
        force[36:48]=local.chart_force; tangent[36:48,36:48]=local.chart_hessian
        rows['B2']=(force,tangent)
        rows['supports_and_joints']=owner._constraints(z[:48],np.tile(np.eye(3),(8,1,1)),z[48:],cmd)
        error_force=np.linalg.norm(rows['full'][0]-sum(v[0] for k,v in rows.items() if k!='full'))
        error_matrix=np.linalg.norm(rows['full'][1]-sum(v[1] for k,v in rows.items() if k!='full'))
        assert error_force<=1e-11 and error_matrix<=1e-11
        return rows,schur
    centre,schur=components(z)
    print('G3C_DIAGNOSTIC '+json.dumps(dict(kind='NONCLASSIFYING_COMPONENT_DECOMPOSITION',schur=schur),sort_keys=True),flush=True)
    for h in (1e-4,1e-5,1e-6):
        plus,_=components(z+h*direction); minus,_=components(z-h*direction)
        for name,(f,K) in centre.items():
            fd=(plus[name][0]-minus[name][0])/(2*h); analytic=K@direction; difference=fd-analytic
            error=float(np.linalg.norm(difference)/max(1.,np.linalg.norm(fd),np.linalg.norm(analytic)))
            print('G3C_DIAGNOSTIC '+json.dumps(dict(kind='NONCLASSIFYING_DIRECTIONAL_ERROR',component=name,h=h,
                error=error,threshold=1e-7,exceeds=error>1e-7,max_row=int(np.argmax(abs(difference))),
                max_difference=float(np.max(abs(difference)))),sort_keys=True),flush=True)
    assert owner.snapshot_bytes()==initial
