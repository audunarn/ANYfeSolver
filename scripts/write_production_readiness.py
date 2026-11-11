"""Write production-readiness capability and scope artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.production_readiness import DEFAULT_PRODUCTION_READINESS_DIR, write_production_readiness_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRODUCTION_READINESS_DIR)
    args = parser.parse_args()

    result = write_production_readiness_artifacts(args.output_dir)
    print(f"Wrote {result['capability_matrix']}")
    print(f"Wrote {result['verification_scope_json']}")
    print(f"Wrote {result['verification_scope_markdown']}")
    print(f"Production release status: {result['production_release_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
