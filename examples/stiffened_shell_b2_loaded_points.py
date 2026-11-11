"""Explicit experimental Q4/B2 loaded interior point demonstration.

SI inputs; physical beam and plate mass are separate. Uses the existing
two-member definition, then explicitly selects fixed-end load-bubble coupling.
Finite-vertex point ports are not qualified welds or bonded lines. Mixed/S3,
footprints, finite gravity and spectra are outside this policy's current scope.
"""
import argparse
import json
from pathlib import Path
from anyfesolver import stiffened_shell as api
from stiffened_shell_b2_body_acceleration import definition


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cases",type=int,choices=(1,10,100),default=10)
    parser.add_argument("--consumer-path",type=Path)
    args=parser.parse_args()
    base=definition("q4")
    model=api.B2Q4PanelDefinition(base.profile,base.shell_nodes_m,base.shells,
        base.members,base.points,base.transformation,base.prescribed_offset,
        point_trace_policy=api.B2_FIXED_END_LOADED_POINT_TRACE_V1_DEV)
    accelerations=tuple((f"weight-{i:03d}",(0.,0.,-9.81*(1.+i/args.cases)))
                        for i in range(args.cases))
    selection=dict(selected_dofs=(49,50),selected_points=("right-port",),
                   recover_members={"beam-1":(.25,.5,.75)})
    analysis=api.create_analysis(definition=model)
    try:
        loads=tuple((name,analysis.body_acceleration_load(a)) for name,a in accelerations)
        result=analysis.solve_linear_many(loads,**selection)
        if result.status!="completed":
            print(json.dumps(dict(status=result.status,failure=result.failure_reason)))
            return 1
        assert result.factorization_count==1 and len(result.cases)==args.cases
        reopened=None
        if args.consumer_path is not None:
            from anyfem.solve.stiffened_shell_b2 import ExperimentalB2LoadedPointPanel,save_case,load_case
            consumer=ExperimentalB2LoadedPointPanel.from_definition(model,accelerations,body=True,**selection)
            reopened=load_case(save_case(consumer,args.consumer_path)).solve_linear_many()
            assert reopened==result
        print(json.dumps(dict(status=result.status,qualification="EXPERIMENTAL_NOT_QUALIFIED",
            policy=model.point_trace_policy,cases=len(result.cases),
            factorization_count=result.factorization_count,
            selected_last_displacement=result.cases[-1].selected_displacements,
            max_point_gap=max(abs(x) for c in result.cases for _,row in c.point_gaps for x in row),
            loaded_energy_last_J=result.cases[-1].loaded_strain_energy,
            consumer_roundtrip=reopened is not None)))
        return 0
    finally:
        analysis.close()


if __name__=="__main__":
    raise SystemExit(main())

