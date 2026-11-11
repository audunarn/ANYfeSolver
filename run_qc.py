#!/usr/bin/env python3
"""
Comprehensive Quality Control Script for FE Solver

This script runs all quality control tests and test cases,
generates reports, and provides a complete verification of the solver.

Usage:
    python run_qc.py [--quick] [--no-save] [--verbose]

Options:
    --quick       Run only quick tests (no convergence, no performance)
    --no-save     Don't save results to files
    --verbose     Show detailed output
"""

import sys
import os
import argparse
import time
import json

# Support direct execution from a source checkout without requiring an
# editable install first.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from anyfesolver.quality_control import run_quality_control, run_quick_qc, run_full_qc, QCConfig
from anyfesolver.test_cases import run_all_demo_test_cases, TestCaseRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Quality Control Tests for FE Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qc.py                    # Run full QC
  python run_qc.py --quick            # Run quick QC only
  python run_qc.py --no-save          # Don't save results
  python run_qc.py --verbose          # Show detailed output
        """
    )

    parser.add_argument('--quick', action='store_true',
                        help='Run quick QC (analytical tests only)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--test-cases', action='store_true',
                        help='Run demonstration test cases')
    parser.add_argument('--only-test-cases', action='store_true',
                        help='Run only test cases, no QC')

    return parser.parse_args()


def run_full_verification(args):
    """Run complete verification suite."""
    print("=" * 80)
    print("FE SOLVER - COMPREHENSIVE QUALITY CONTROL & VERIFICATION")
    print("=" * 80)
    print()

    overall_start = time.time()

    # Run Quality Control Tests
    if not args.only_test_cases:
        print("PART 1: QUALITY CONTROL TESTS")
        print("-" * 80)

        if args.quick:
            qc_config = QCConfig(
                run_convergence=False,
                run_patch=False,
                run_performance=False,
                save_results=not args.no_save,
                verbose=args.verbose
            )
            qc_report = run_quality_control(qc_config)
        else:
            qc_config = QCConfig(
                save_results=not args.no_save,
                output_dir="qc_results",
                verbose=args.verbose
            )
            qc_report = run_quality_control(qc_config)

        qc_report.print_summary()

        if args.verbose:
            print("\nDetailed QC Results:")
            for result in qc_report.results:
                status = "PASS" if result.passed else "FAIL"
                print(f"  {status} {result.test_category}: {result.test_name}")
                if not result.passed:
                    print(f"    Error: {result.error:.2%}")

        print()

    # Run Test Cases
    if args.test_cases or args.only_test_cases:
        print("PART 2: DEMONSTRATION TEST CASES")
        print("-" * 80)

        test_case_results = run_all_demo_test_cases()
        TestCaseRunner.print_test_case_summary(test_case_results)

        if not args.no_save:
            TestCaseRunner.save_test_case_results(
                test_case_results,
                "test_case_results.json"
            )

        print()

    # Generate Summary Report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    total_time = time.time() - overall_start

    if not args.only_test_cases:
        qc_summary = qc_report.summary
        print(f"\nQuality Control Tests:")
        print(f"  Total: {qc_summary['total_tests']}")
        print(f"  Passed: {qc_summary['passed']} ({qc_summary['pass_rate']:.1f}%)")
        print(f"  Failed: {qc_summary['failed']}")

        if qc_summary['failed'] > 0:
            print("\n  Failed Tests:")
            for result in qc_report.results:
                if not result.passed:
                    print(f"    [FAIL] {result.test_name}: {result.error:.2%} error")

    if args.test_cases or args.only_test_cases:
        completed = sum(1 for r in test_case_results.values() if r.get('status') != 'FAILED')
        total = len(test_case_results)
        print(f"\nTest Cases:")
        print(f"  Completed: {completed}/{total}")

        if completed < total:
            print("\n  Failed Test Cases:")
            for name, result in test_case_results.items():
                if result.get('status') == 'FAILED':
                    print(f"    [FAIL] {name}: {result.get('error', 'Unknown error')}")

    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print("=" * 80)

    # Determine overall status
    overall_passed = True
    if not args.only_test_cases:
        overall_passed = overall_passed and (qc_summary['failed'] == 0)
    if args.test_cases or args.only_test_cases:
        overall_passed = overall_passed and (completed == total)

    if overall_passed:
        print("\nALL TESTS PASSED. The FE solver is verified and ready for use.")
        return 0
    else:
        print("\nSome tests failed. Please review the results above.")
        return 1


def main():
    """Main entry point."""
    args = parse_args()

    try:
        return run_full_verification(args)
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
