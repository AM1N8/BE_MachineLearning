"""Run the package without installing it."""

import os
import sys
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from copilote_b.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
