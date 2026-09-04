from __future__ import annotations

import argparse
import json

from .experiment import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Lagrangian-relaxation unit-commitment benchmark")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--theta", type=float, default=1.6)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(iterations=args.iterations, theta=args.theta), indent=2))


if __name__ == "__main__":
    main()
