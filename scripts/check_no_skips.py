"""Fail when a junit report has no tests, or any skipped, failed or errored test."""
import sys
from xml.etree import ElementTree


def main(path):
    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    # pytest counts the test bodies it ran in ``tests``; --setup-only still writes testcase elements.
    executed = sum(int(suite.get("tests", "0")) for suite in root.iter("testsuite"))
    bad = [f"{case.get('classname')}::{case.get('name')}: {child.tag} {child.get('message') or ''}"
           for case in cases for child in case if child.tag in ("skipped", "failure", "error")]
    if not cases or executed != len(cases):
        bad.append(f"not every test ran: {executed} executed of {len(cases)} reported")
    for line in bad:
        print(line)
    print(f"{len(cases)} tests, {len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
