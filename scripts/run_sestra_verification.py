"""Run the bounded SESTRA development comparison (external execution is opt-in)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from anyfesolver.sestra_verification import run_verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--mode",choices=("prepare","compare","execute"),default="prepare")
    parser.add_argument("--output-root",default="reports/sestra_development")
    args = parser.parse_args()
    report = run_verification(args.manifest,args.output_root,mode=args.mode)
    print(f"{report['status']}: {report.get('reason',report.get('case_id',''))}\n{report['report_path']}")
    return 0 if report["status"] in {"passed","not_executed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
