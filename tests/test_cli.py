import json
import subprocess
import sys


def test_cli_outputs_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "lagrangian_uc", "--iterations", "8"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    assert data["exact"]["objective"] > 0
    assert data["lagrangian"]["best_lower_bound"] <= data["exact"]["objective"] + 1e-7
