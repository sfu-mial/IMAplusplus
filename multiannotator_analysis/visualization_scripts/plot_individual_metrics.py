import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

sys.path.append("..")
from constants import IEEE_DATA_DESC, METRIC_NAME_MAP

from utils import (
    create_custom_colormap,
    filter_annotators_metadata,
    filter_annotators_metrics,
    read_multiannotator_subset_seg_masks_metadata,
    read_pairwise_IAA_metrics,
    read_seg_masks_metadata,
)

VIS_OUTPUT_DIR = Path("../../output/visualizations/metrics/")
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Main function.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Read the pairwise IAA metrics file
    pairwise_IAA_metrics_df = read_pairwise_IAA_metrics(config)

    # Filter the pairwise IAA metrics DataFrame to only include the metrics
    # for the individual annotators (i.e., exclude consensus labels).
    pairwise_IAA_metrics_df = filter_annotators_metrics(
        pairwise_IAA_metrics_df
    )

    # Now, we will create a 1x5 grid of subplots, each showing the
    # distribution of the five metrics across all annotators.
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    sns.set(style="ticks")  # Set to 'ticks' style for a cleaner look

    for i, metric in enumerate(
        [
            "dice_score",
            "jaccard_score",
            "hd_score_normalized",
            "hd95_score_normalized",
            "assd_score_normalized",
        ]
    ):
        sns.histplot(
            pairwise_IAA_metrics_df[metric],
            ax=axs[i],
            color=IEEE_DATA_DESC,
            kde=True,
            alpha=1.0,
            # linewidth=2,
            # edgecolor="black",
            fill=False,
            # bins=30,
        )
        # axs[i].set_title(f"{METRIC_NAME_MAP[metric]} Distribution")
        axs[i].set_xlabel(METRIC_NAME_MAP[metric])
        axs[i].set_ylabel("Count")

    # For 2nd subplot onwards, remove the y-axis label.
    for i in range(1, 5):
        axs[i].set_ylabel("")

    # # Set the text size of the x-axis and y-axis labels.
    for ax in axs:
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    # Set the text size of x-axis and y-axis ticks.
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelsize=12)

    plt.tight_layout()
    plt.savefig(
        VIS_OUTPUT_DIR / "individual_metrics.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        VIS_OUTPUT_DIR / "individual_metrics.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    main()
