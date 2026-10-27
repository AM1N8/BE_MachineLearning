"""Execute clean notebooks and save generated copies under outputs/."""

import argparse
from pathlib import Path
from typing import Any, Callable, cast

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/notebooks"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    read_notebook = cast(Callable[..., Any], nbformat.read)
    write_notebook = cast(Callable[..., Any], nbformat.write)
    for path in sorted((project / "notebooks").glob("*.ipynb")):
        notebook = read_notebook(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(project / "notebooks")}},
        )
        client.execute()
        write_notebook(notebook, args.output / path.name)
        print(f"Executed {path.name}", flush=True)


if __name__ == "__main__":
    main()
