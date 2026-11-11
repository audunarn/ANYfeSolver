"""Prepare and sequentially compare the fixed three-case development regime."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anyfesolver.sestra_three_case import prepare_three_cases

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','compare']);p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args()
 if a.action=='prepare':result=prepare_three_cases(a.input,a.output)
 else:
  from anyfesolver.sestra_three_case import compare_suite
  result=compare_suite(a.input,a.output)
 print(json.dumps({'status':result.get('status','not_executed'),'order':result.get('order'),'output':a.output}))
 return 0 if result.get('status','not_executed') in {'passed','not_executed'} else 1

if __name__=='__main__':raise SystemExit(main())
