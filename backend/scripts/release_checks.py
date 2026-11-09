"""Record actual local check outcomes, including unresolved failures."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root / "audit" / "release"
    directory.mkdir(parents=True, exist_ok=True)
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    python = sys.executable
    checks = [
        ("python_lint", [python, "-m", "ruff", "check", "backend", "tests"], root),
        ("python_format", [python, "-m", "ruff", "format", "--check", "backend", "tests"], root),
        ("python_compile", [python, "-m", "compileall", "-q", "backend", "tests"], root),
        ("python_dependencies", [python, "-m", "pip", "check"], root),
        ("backend_tests", [python, "-m", "backend.scripts.run_tests"], root),
        (
            "static_security",
            [python, "-m", "bandit", "-r", "backend", "-ll", "-f", "json", "-o", "audit/bandit.json"],
            root,
        ),
        (
            "python_advisories",
            [python, "-m", "pip_audit", "--format", "json", "--output", "audit/dependency_audit_final.json"],
            root,
        ),
        (
            "encoder_cuda",
            [
                python,
                "-m",
                "backend.scripts.verify_encoders",
                "--device",
                "cuda",
                "--output",
                "audit/encoder_gpu_verification.json",
            ],
            root,
        ),
        ("frontend_lint", [npm, "run", "lint"], root / "frontend"),
        ("frontend_types", [npm, "run", "typecheck"], root / "frontend"),
        ("frontend_build", [npm, "run", "build"], root / "frontend"),
        ("frontend_production_advisories", [npm, "audit", "--omit=dev", "--json"], root / "frontend"),
        ("frontend_all_advisories", [npm, "audit", "--json"], root / "frontend"),
    ]
    results = []
    for name, command, cwd in checks:
        print(f"RUN {name}", flush=True)
        start = time.perf_counter()
        env = dict(os.environ)
        if name == "encoder_cuda":
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        output = completed.stdout + completed.stderr
        (directory / f"{name}.txt").write_text(output, encoding="utf-8")
        if name.startswith("frontend_") and name.endswith("advisories"):
            (root / "audit" / f"{name}.json").write_text(completed.stdout, encoding="utf-8")
        result = {
            "check": name,
            "exit_code": completed.returncode,
            "seconds": time.perf_counter() - start,
            "log": f"audit/release/{name}.txt",
        }
        results.append(result)
        (root / "audit" / "release_checks.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    raise SystemExit(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    main()
