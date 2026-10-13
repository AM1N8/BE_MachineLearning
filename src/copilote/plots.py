"""Shared vector/raster export and accessible plotting defaults."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.prop_cycle": plt.cycler(
            color=[
                "#0072B2",
                "#E69F00",
                "#009E73",
                "#CC79A7",
                "#56B4E9",
                "#D55E00",
                "#F0E442",
                "#000000",
            ]
        ),
    }
)


def save_figure(name: str) -> None:
    """Save the current figure as PDF for LaTeX and PNG for notebooks."""
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for extension in ("pdf", "png"):
        plt.savefig(f"reports/figures/{name}.{extension}", bbox_inches="tight", dpi=160)
    plt.close()
