"""Persist verification evidence without treating fixtures as clinical results."""

import json
import time
import unittest
from pathlib import Path


def main():
    directory = Path("audit")
    directory.mkdir(exist_ok=True)
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover("tests")
    with (directory / "backend_tests.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "Artificial software fixtures, not genuine patient end-to-end validation.",
    }
    (directory / "test_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
