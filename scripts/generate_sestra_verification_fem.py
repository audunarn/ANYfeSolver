"""Generate the complete formatted development shell FEM without GeniE."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from anyfesolver.sestra_verification import generate_hanging_shell


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args()
    report=generate_hanging_shell(args.output,overwrite=args.overwrite)
    print(f"{report.records_written} records, {report.bytes_written} bytes: {report.path}")


if __name__ == "__main__":
    main()
