"""Create a Python 3.11 environment for RNGen training and evaluation."""

import argparse
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tomllib
import venv


ROOT = Path(__file__).resolve().parent


def run(command):
    print(f"\n{shlex.join(map(str, command))}", flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument(
        "--torch-build", choices=("auto", "cpu", "cu118", "cu124", "cu126"),
        default="auto",
        help="PyTorch build; auto uses the platform default (CUDA 12.4 on Linux).",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Run this installer with Python 3.11.")
    if platform.system() == "Darwin" and args.torch_build.startswith("cu"):
        parser.error("CUDA builds require Linux or Windows; use auto on macOS.")

    env_dir = args.venv.expanduser().resolve()
    if env_dir == Path(sys.prefix).resolve():
        parser.error("Choose a separate --venv directory, not the running environment.")
    if env_dir.exists() and not (env_dir / "pyvenv.cfg").is_file():
        parser.error(f"Existing directory is not a virtual environment: {env_dir}")

    musicbert = ROOT / "musicbert_hf"
    if not (musicbert / "pyproject.toml").is_file():
        run(["git", "submodule", "sync", "--", "musicbert_hf"])
        run(["git", "submodule", "update", "--init", "--", "musicbert_hf"])

    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([python, "-c", "import sys; assert sys.version_info[:2] == (3, 11)"])
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    torch_requirement = next(item for item in project["dependencies"] if item.startswith("torch=="))
    torch_command = [python, "-m", "pip", "install", torch_requirement]
    if args.torch_build != "auto" and platform.system() != "Darwin":
        torch_command[-1] += f"+{args.torch_build}"
        torch_command += ["--index-url", f"https://download.pytorch.org/whl/{args.torch_build}"]
    run(torch_command)
    run([python, "-m", "pip", "install", "-r", "rngen_requirements.txt"])
    run([python, "-m", "pip", "check"])
    run([
        python, "-c",
        "import json, torch; import rngen.sft_ft, rngen.sft_eval; "
        "from rngen.event_vocabulary import load_event_vocab; "
        "from rngen.paths import TEMPLATES_PATH; "
        "assert load_event_vocab(); "
        "assert {'KEY_SFT', 'RNA_SFT', 'RNA_CURRICULUM_STAGE2'} <= "
        "json.loads(TEMPLATES_PATH.read_text()).keys(); "
        "print('RNGen imports, vocabulary, and templates: OK'); "
        "print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())",
    ])
    activation = env_dir / ("Scripts/Activate.ps1" if os.name == "nt" else "bin/activate")
    print(f"\nInstallation complete. Activate with: {'&' if os.name == 'nt' else 'source'} {shlex.quote(str(activation))}")


if __name__ == "__main__":
    main()
