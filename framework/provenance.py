"""
Where a result comes from: the code, the libraries and the hardware
"""

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("torch", "transformers", "accelerate", "sentence-transformers", "numpy", "pydantic")


def git_state(root: Path = ROOT) -> dict:
    """The commit of the code and whether it has uncommitted changes (None for both outside a git checkout)"""
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def library_versions() -> dict[str, str | None]:
    """Version of each library that matters for the results, None if it is not installed"""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware() -> dict:
    """The GPUs visible to this process (name, memory). Read only if torch is already loaded, which a run with a
    local model has done by the time it starts; a mock run does not pay for importing it"""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_loaded": False, "cuda": None, "gpus": []}
    if not torch.cuda.is_available():
        return {"torch_loaded": True, "cuda": None, "gpus": []}
    gpus = [{"name": torch.cuda.get_device_name(i), "memory_mb": torch.cuda.get_device_properties(i).total_memory >> 20}
            for i in range(torch.cuda.device_count())]
    return {"torch_loaded": True, "cuda": torch.version.cuda, "gpus": gpus}


def environment() -> dict:
    """Everything about the machine and code that a run or a sweep records about itself"""
    return {"git": git_state(), "python": sys.version.split()[0], "platform": platform.platform(),
            "libraries": library_versions(), "hardware": hardware()}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
