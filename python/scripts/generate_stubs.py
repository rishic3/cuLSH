import subprocess
import sys
from pathlib import Path


def main():
    """Generate .pyi stubs for _culsh_core"""
    output_dir = Path(__file__).parent.parent

    result = subprocess.run(
        [
            sys.executable, "-m", "pybind11_stubgen",
            "culsh._culsh_core",
            "-o", str(output_dir),
            "--ignore-all-errors",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"stubgen failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    stub_file = output_dir / "culsh" / "_culsh_core.pyi"
    if stub_file.exists():
        print(f"Generated {stub_file}")
    else:
        print("Warning: stub file not created", file=sys.stderr)


if __name__ == "__main__":
    main()
