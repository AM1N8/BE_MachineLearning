"""Build the source archive and the notebook runner without embedding raw or private data."""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Any

from copilote.io import write_json
from copilote.settings import load_config, provenance

LOGGER = logging.getLogger(__name__)
EXCLUDED_SUFFIXES = (
    ".gz",
    ".joblib",
    ".ipynb_checkpoints",
    ".aux",
    ".log",
    ".fls",
    ".out",
    ".fdb_latexmk",
    ".synctex.gz",
)
EXCLUDED_PARTS = {
    "data",
    "outputs",
    ".venv",
    "artifacts",
    "__pycache__",
    ".git",
    "submissions",
    "experiments",
}


def run_notebook(config: dict[str, Any], notebook: str = "notebooks/copilote.ipynb") -> bool:
    """Execute the report notebook top-to-bottom, leaving executed outputs on disk."""
    import nbformat
    from nbclient import NotebookClient

    path = Path(notebook)
    if not path.exists():
        LOGGER.warning("Notebook %s absent; skipping execution", path)
        return False
    document = nbformat.read(path, as_version=4)
    NotebookClient(
        document,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    nbformat.write(document, path)
    return True


def archive(destination: str = "copilote-submission.zip") -> str:
    """Zip code, docs and measured artifacts while refusing raw data by construction."""
    target = Path(destination)
    roots = [
        "src",
        "tests",
        "configs",
        "docs",
        "reports/latex",
        "reports/results",
        "reports/figures",
        "notebooks",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "requirements.txt",
        "README.md",
        "PROGRESS.md",
        "AGENTS.md",
    ]
    written = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for root in roots:
            for path in sorted(Path().glob(root)):
                files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
                for file in files:
                    if EXCLUDED_PARTS & set(file.parts) or file.name.endswith(EXCLUDED_SUFFIXES):
                        continue
                    bundle.write(file, file.as_posix())
                    written += 1
    LOGGER.info("Archived %d files into %s", written, target)
    return str(target)


def main() -> None:
    """Entry point for `make archive`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["archive", "notebook"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--notebook", default="notebooks/copilote.ipynb")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.command == "notebook":
        ran = run_notebook(config, args.notebook)
        write_json(
            "reports/results/notebook.json",
            {"provenance": provenance(config), "executed": ran, "notebook": args.notebook},
        )
    else:
        write_json(
            "reports/results/archive.json", {"provenance": provenance(config), "archive": archive()}
        )


if __name__ == "__main__":
    main()
