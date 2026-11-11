import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anyfesolver.sestra_structure_benchmarks import prepare_suite,run_suite
def main():
    p=argparse.ArgumentParser(description='Prepare/compare bounded combined-structure benchmarks; no external jobs.');p.add_argument('action',choices=['prepare','native','compare']);p.add_argument('input',nargs='?');p.add_argument('--output',required=True);a=p.parse_args()
    result=prepare_suite(a.output) if a.action=='prepare' else run_suite(a.input,a.output,a.action)
    print(json.dumps({'status':result.get('status','not_executed'),'output':a.output}));return 1 if result.get('status')=='failed' else 0
if __name__=='__main__':raise SystemExit(main())
