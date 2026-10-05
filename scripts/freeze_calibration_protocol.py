#!/usr/bin/env python3
"""Emit a frozen protocol from the chosen draft and user0/user1 tuning results."""
import argparse
import json
import sys
from pathlib import Path

from calibration_protocol import freeze_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tuning-result", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        result = freeze_protocol(json.loads(args.candidate.read_text()),
                                 [json.loads(path.read_text()) for path in args.tuning_result])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f"Protocol freeze failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
