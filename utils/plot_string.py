"""
plot_string.py  —  Decode a Standard GE genotype and plot the phenotype tree.

Usage:
    python3 utils/plot_string.py "12,138,227,91,217,16,24,138,188,53,104"
    python3 utils/plot_string.py "12,138,227,91" --output out/my_tree.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

# Allow importing primitives from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import primitives as Primitives  # noqa: E402


def plot_individual(genotype: list, output_filename: str = "tree_plot.png") -> None:
    """
    Decode a flat GE genotype into its phenotype expression tree and save a plot.

    The figure contains:
      - The phenotype expression tree (top).
      - A text annotation with the codon count and flat genotype array (bottom).

    Args:
        genotype:        List of integer codons (0-255).
        output_filename: Path to save the output PNG image.
    """
    individual = Primitives.Individual(genotype)
    individual.decode()

    if individual.phenotype is None:
        print("Error: genotype could not be decoded into a valid phenotype.")
        print("  This may be caused by exceeding the wrap-around or node limits.")
        sys.exit(1)

    tree = individual.phenotype
    n = len(genotype)

    depth = tree.depth
    width = min(100, max(10, (2 ** depth) * 0.8))
    height = min(50, max(6, (depth + 1) * 1.5))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(f"Phenotype: {tree}", fontsize=11, pad=10, wrap=True)

    def draw_node(node, x, y, dx, dy):
        if hasattr(node, "left") and hasattr(node, "right"):
            ax.plot([x, x - dx], [y, y - dy], "k-", lw=1.5, zorder=1)
            ax.plot([x, x + dx], [y, y - dy], "k-", lw=1.5, zorder=1)
            draw_node(node.left,  x - dx, y - dy, dx / 2, dy)
            draw_node(node.right, x + dx, y - dy, dx / 2, dy)

        label = str(node.value) if hasattr(node, "value") else str(node)
        ax.text(
            x, y, label,
            ha="center", va="center",
            bbox=dict(facecolor="lightblue", edgecolor="black", boxstyle="round,pad=1"),
            fontsize=12,
            zorder=2,
        )

    draw_node(tree, 0, 0, width / 2, 1)

    # ── Genotype annotation ─────────────────────────────────────────────
    gt_label = f"Genotype ({n} codons): {genotype}"
    fig.text(
        0.5, 0.01, gt_label,
        ha="center", va="bottom",
        fontsize=9,
        wrap=True,
        bbox=dict(facecolor="lightyellow", edgecolor="grey", boxstyle="round,pad=0.5"),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
    print(f"Phenotype : {tree}")
    print(f"Genotype  : {genotype}")
    print(f"Saved tree visualization to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode a Standard GE genotype and plot the phenotype expression tree."
    )
    parser.add_argument(
        "genotype",
        type=str,
        help="Flat GE genotype as a comma-separated list of integers, e.g. '12,138,227,91'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tree_plot.png",
        help="Output image filename (default: tree_plot.png)",
    )

    args = parser.parse_args()

    try:
        genotype = [int(c.strip()) for c in args.genotype.split(",")]
    except ValueError:
        print("Error: genotype must be a comma-separated list of integers.")
        sys.exit(1)

    plot_individual(genotype, args.output)
