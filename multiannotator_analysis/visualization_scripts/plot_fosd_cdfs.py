import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

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
    # Load the FOSD CDFs values from the CSV file.
    fosd_cdfs_df = pd.read_csv(
        Path("FOSD_CDF_values.csv"),
        header="infer",
        sep=",",
        low_memory=False,
    )

    # Plotting
    plt.figure(figsize=(6, 5))

    # Create a seaborn countplot for the distribution of segmentations per image
    sns.set(style="ticks")  # Set to 'ticks' style for a cleaner look
    ax = sns.lineplot(
        x="grid_value",
        y="Malignant",
        data=fosd_cdfs_df,
        label="Malignant",
        linewidth=3,
        color="#6e001e",
    )
    ax = sns.lineplot(
        x="grid_value",
        y="Benign",
        data=fosd_cdfs_df,
        label="Benign",
        linewidth=3,
        color="#006e50",
    )

    # Customize plot for publication-quality visualization
    # ax.set_title('Distribution of Number of Segmentations per Image', fontsize=20, fontweight='bold')
    ax.set_xlabel("IAA Score (Dice)", fontsize=21, labelpad=15)
    # ax.set_ylabel("Fraction of Images", fontsize=21, labelpad=15)
    ax.set_ylabel("", fontsize=21, labelpad=15)

    # Use sans-serif for a more professional look
    # plt.xticks(fontsize=12, family='Helvetica')
    # plt.yticks(fontsize=12, family='Helvetica')
    plt.xticks(fontsize=18, family="Arial")
    plt.yticks(fontsize=18, family="Arial")

    # Remove unnecessary gridlines and spines for a cleaner look
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(VIS_OUTPUT_DIR / "fosd_cdfs.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(VIS_OUTPUT_DIR / "fosd_cdfs.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
