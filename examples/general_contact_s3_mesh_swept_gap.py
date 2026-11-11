"""S3 material friction with whole-chord absolute-gap geometry, not history-event qualification."""
from dataclasses import asdict
import argparse,json
from pathlib import Path
from examples.general_contact_s3_mesh_weak_pressure_friction import build_case as _endpoint_case
from examples.general_contact_s3_mesh_moving_friction import solve_case
from anyfesolver.general_contact import S3MeshWholeChordGapFrictionSettings

def build_case():
    case=list(_endpoint_case());case[3]=S3MeshWholeChordGapFrictionSettings(**asdict(case[3]))
    return tuple(case)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-step',type=int,default=6,choices=range(1,7))
    parser.add_argument('--restart',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();incoming=json.loads(args.restart.read_text()) if args.restart else None
    result=solve_case(build_case(),args.through_step,incoming)
    args.output.write_text(json.dumps(dict(status=result.status,info=result.info,
        checkpoint=result.to_restart_checkpoint() if result.status=='completed' else None),indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item()))
    if result.status!='completed':raise RuntimeError(result.info)

if __name__=='__main__':main()
