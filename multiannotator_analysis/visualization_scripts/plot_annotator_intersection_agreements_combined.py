import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from omegaconf import OmegaConf

# sys.path.append("../../")

sys.path.append("..")
from constants import IEEE_DATA_DESC, IEEE_DATA_DESC_COLOR_COMP

from utils import (
    filter_annotators_metrics,
    get_arbitrary_annotator_intersections,
    read_pairwise_IAA_metrics,
)

VIS_OUTPUT_DIR = Path(
    "../../output/visualizations/annotator_intersection_agreements/"
)
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATOR_INTERSECTIONS = [
    {"A00", "A03"},  # 1
    {"A04", "A06"},  # 2
    {"A01", "A02"},  # 3
    {"A01", "A05"},  # 4
    {"A00", "A03", "A08"},  # 5
    {"A02", "A05"},  # 6
    {"A00", "A09"},  # 7
    {"A02", "A03", "A04"},  # 8
    # {"A00", "A08"}, #9
]


def plot_boxswarmplots(
    ax: plt.Axes,
    annotator_intersection: set,
    df: pd.DataFrame,
    add_legend: bool = False,
) -> None:
    """
    Plot the box and swarm plots for the annotator intersection agreements.

    Parameters
    ----------
    ax : plt.Axes
        The axes object to plot the box and swarm plots on.
    annotator_intersection : set
        The annotator intersection to plot the box and swarm plots for.
    df : pd.DataFrame
        The DataFrame containing the metrics.
    add_legend : bool, optional
        Whether to add a legend to that subplot.

    Returns
    -------
    None
        The plot is created in the given axes object.
    """
    num_unique_images = df["img_id"].nunique()
    annotator_intersection_sorted = sorted(annotator_intersection)

    df_renamed = df.rename(
        columns={
            "dice_score": "Dice",
            "hd95_score_normalized": "HD95",
        }
    )
    df_long = pd.melt(
        df_renamed,
        id_vars=["img_id"],
        value_vars=["Dice", "HD95"],
        var_name="Metric",
        value_name="Value",
    )

    sns.set(style="ticks")

    sns.boxplot(
        data=df_long,
        x="Metric",
        y="Value",
        palette=[IEEE_DATA_DESC, IEEE_DATA_DESC_COLOR_COMP],
        notch=True,
        showcaps=False,
        flierprops={"marker": None},
        boxprops={"facecolor": (1, 1, 1, 1)},
        ax=ax,
        hue="Metric",
        dodge=True if add_legend else False,
    )

    sns.stripplot(
        size=2.0,
        alpha=0.9,
        # sns.swarmplot(
        # size=1.2,
        data=df_long,
        x="Metric",
        y="Value",
        palette=[IEEE_DATA_DESC, IEEE_DATA_DESC_COLOR_COMP],
        ax=ax,
        hue="Metric",
        dodge=True if add_legend else False,
    )

    # sns.despine(ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{{{', '.join(annotator_intersection_sorted)}}}: {num_unique_images} "
        f"images"
    )

    # Add legend if specified.
    if add_legend:
        legend_elements = [
            Patch(facecolor=IEEE_DATA_DESC, label="Dice"),
            Patch(facecolor=IEEE_DATA_DESC_COLOR_COMP, label="HD95"),
        ]
        ax.legend(
            handles=legend_elements,
            fontsize=12,
            loc="upper right",
            frameon=False,
        )


def main() -> None:
    """
    Main function.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Read the metrics DataFrame.
    metrics_df = read_pairwise_IAA_metrics(config)

    # Filter the metrics DataFrame to only include the metrics for the
    # individual annotators (i.e., exclude consensus labels).
    metrics_df = filter_annotators_metrics(metrics_df)

    # Create a figure with 2x4 subplots.
    fig, axs = plt.subplots(4, 2, figsize=(5.5, 10), sharex=True, sharey=True)

    # Plot the box and swarm plots for each annotator intersection agreement.
    for idx, annotator_intersection in enumerate(ANNOTATOR_INTERSECTIONS):
        filtered_metrics_df = get_arbitrary_annotator_intersections(
            metrics_df, annotator_intersection
        )
        plot_boxswarmplots(
            axs[idx // 2, idx % 2],
            annotator_intersection,
            filtered_metrics_df,
            add_legend=(
                idx == 0
            ),  # This adds the legend only for the first subplot.
        )

    plt.tight_layout()

    plt.savefig(VIS_OUTPUT_DIR / "all_annotator_intersections.png", dpi=600)
    plt.savefig(VIS_OUTPUT_DIR / "all_annotator_intersections.pdf", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
