import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anyfesolver.sestra_deep_diagnostics import prepare_stage1,prepare_stage2

def main():
 p=argparse.ArgumentParser(description='Prepare fixed discriminating SESTRA development models; no external jobs.');p.add_argument('action',choices=['prepare','prepare-stage2','compare','probe']);p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args()
 if a.action=='prepare':result=prepare_stage1(a.input,a.output)
 elif a.action=='prepare-stage2':result=prepare_stage2(a.input,a.output)
 else:
  from anyfesolver.sestra_deep_diagnostics import run_diagnostics
  result=run_diagnostics(a.input,a.output,mode=a.action)
 print(json.dumps({'status':result.get('status','not_executed'),'output':a.output}));return 0 if result.get('status','not_executed') in {'passed','not_executed'} else 1

if __name__=='__main__':raise SystemExit(main())
