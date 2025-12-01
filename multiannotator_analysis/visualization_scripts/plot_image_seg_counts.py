import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

# sys.path.append("../../")

sys.path.append("..")
from utils import (
    create_custom_colormap,
    filter_annotators_metadata,
    filter_annotators_metrics,
    read_multiannotator_subset_seg_masks_metadata,
    read_seg_masks_metadata,
)

VIS_OUTPUT_DIR = Path("../../output/visualizations/dataset_statistics/")
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Main function.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Read the segmentation masks metadata file into a DataFrame.
    SEG_MASKS_METADATA_DF = read_seg_masks_metadata(config)

    # Group by 'image' and count the number of annotations per image
    image_counts = SEG_MASKS_METADATA_DF.groupby("img_filename").size()

    # Import custom colormap.
    custom_colormap = create_custom_colormap(
        num_colors=len(image_counts.unique())
    )

    # Plotting
    plt.figure(figsize=(6, 5))

    # Create a seaborn countplot for the distribution of segmentations per image
    sns.set(style="ticks")  # Set to 'ticks' style for a cleaner look
    ax = sns.countplot(
        x=image_counts,
        palette=custom_colormap,
        order=sorted(image_counts.unique()),
    )

    # Set the y-axis to a logarithmic scale
    ax.set_yscale("log")

    # Customize plot for publication-quality visualization
    # ax.set_title('Distribution of Number of Segmentations per Image', fontsize=20, fontweight='bold')
    ax.set_xlabel(
        "Number of Segmentations per Image", fontsize=21, labelpad=15
    )
    ax.set_ylabel("Number of Images", fontsize=21, labelpad=15)

    # Use sans-serif for a more professional look
    # plt.xticks(fontsize=12, family='Helvetica')
    # plt.yticks(fontsize=12, family='Helvetica')
    plt.xticks(fontsize=18, family="Arial")
    plt.yticks(fontsize=18, family="Arial")

    # Adding count labels on top of the bars
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=18,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Remove unnecessary gridlines and spines for a cleaner look
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(
        VIS_OUTPUT_DIR / "distribution.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        VIS_OUTPUT_DIR / "distribution.png", dpi=300, bbox_inches="tight"
    )

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
