import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

if str(TIAGO_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC_DIR))
