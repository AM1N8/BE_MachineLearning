import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_documentation_links_resolve_inside_the_standalone_project() -> None:
    documents = [
        (path, path.read_text())
        for path in [
            PROJECT / "README.md",
            *PROJECT.glob("docs/*.md"),
            PROJECT / "data/README.md",
        ]
    ]
    for path in PROJECT.glob("notebooks/*.ipynb"):
        notebook = json.loads(path.read_text())
        documents.append(
            (
                path,
                "\n".join(
                    "".join(c["source"])
                    for c in notebook["cells"]
                    if c["cell_type"] == "markdown"
                ),
            )
        )
    checked = 0
    for path, text in documents:
        for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in destination or destination.startswith("#"):
                continue
            target = (path.parent / destination.split("#")[0]).resolve()
            assert target.is_file(), (path, destination)
            assert target.is_relative_to(PROJECT)
            checked += 1
    assert checked > 0


def test_notebook_sources_are_clean_valid_python() -> None:
    for path in PROJECT.glob("notebooks/*.ipynb"):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                compile("".join(cell["source"]), f"{path.name}:{index}", "exec")
