"""Central path configuration for the pipeline."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))


def construct_dataset(*parts: str) -> Path:
    return DATA_ROOT / "construct" / "dataset" / Path(*parts)


def random_dataset(*parts: str) -> Path:
    return DATA_ROOT / "random" / "dataset" / Path(*parts)
