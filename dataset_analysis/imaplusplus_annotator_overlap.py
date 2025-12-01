from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents

# Define the constants for the plot.
IMA_PLUS_PLUS_COLOR = "#90224190"

VIS_OUTPUT_DIR = Path("../output/visualizations/upset_plots/")
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_imaplusplus_annotator_overlap(
    annotator_image_counts: pd.Series,
) -> None:
    """
    Plot the overlap of the annotators in the IMA++ dataset.
    """
    # Set the style of the plot.
    # print(plt.style.available) # To print all the available styles.
    plt.style.use("seaborn-v0_8")

    # Set the figure size.
    upset_fig = plt.figure(figsize=(20, 10))

    # Create the upset plot object.
    upset = UpSet(
        from_contents(annotator_image_counts),
        subset_size="count",
        min_subset_size=1,
        sort_by="cardinality",
        show_counts=True,
    )

    # Plot the upset plot.
    upset.plot(fig=upset_fig)

    # Save the plot.
    upset_fig.savefig(
        VIS_OUTPUT_DIR / "imaplusplus_annotator_overlap.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    upset_fig.savefig(
        VIS_OUTPUT_DIR / "imaplusplus_annotator_overlap.svg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    upset_fig.savefig(
        VIS_OUTPUT_DIR / "imaplusplus_annotator_overlap.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def main() -> None:
    """
    Main function.
    """
    # First, read the the IMA++ segmentations' metadata.
    imapp_seg_metadata = pd.read_csv(
        "../output/metadata/IMAplusplus_seg_metadata.csv",
        header="infer",
        sep=",",
    )

    # Ensure that there are no duplicate image-annotator pairs.
    df_unique_image_annotator_pairs = imapp_seg_metadata[
        ["img_filename", "annotator"]
    ].drop_duplicates()

    # Then, group by annotator to get a list of images for each annotator.
    annotator_image_counts = df_unique_image_annotator_pairs.groupby(
        "annotator"
    )["img_filename"].apply(list)

    plot_imaplusplus_annotator_overlap(annotator_image_counts)


if __name__ == "__main__":
    main()
