from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents

# Define the constants for the plot.
# 902241 is the hex code for IEEE Data Descriptions.
IMA_PLUS_PLUS_COLOR = "#90224190"


def read_isic_ids(challenge_dataset_dir: Path) -> list[str]:
    """
    Given a directory containing multiple .txt files, each containing a list of
    ISIC IDs, return a list of all ISIC IDs.
    """
    isic_ids = []
    for file in challenge_dataset_dir.glob("*.txt"):
        with open(file, "r") as f:
            isic_ids.extend(f.read().splitlines())
    return isic_ids


def plot_isic_overlap(contents: dict[str, list[str]]) -> None:
    """
    Plot the overlap of the ISIC IDs.
    """
    # Set the style of the plot.
    # print(plt.style.available) # To print all the available styles.
    plt.style.use("seaborn-v0_8")

    # Set the figure size.
    upset_fig = plt.figure(figsize=(15, 8))

    # Create the upset plot object.
    upset = UpSet(
        from_contents(contents),
        subset_size="count",
        min_subset_size=1,
        sort_by="cardinality",
        show_counts=True,
    )

    # Highlight the IMA++ set.
    # https://shorturl.at/lVJyk
    upset.style_categories(
        "IMA++",
        shading_facecolor=IMA_PLUS_PLUS_COLOR,
    )

    # Plot the upset plot.
    upset.plot(fig=upset_fig)

    # # Add a title to the plot.
    # upset_fig.suptitle(
    #     "Overlap of ISIC IDs between ISIC 2016 - 2018 Segmentation Challenges and IMA++"
    # )

    # # Tilt the count lables.
    # This was done by modifying the plotting.py file in the upsetplot package.
    # miniconda3/envs/monai/lib/python3.10/site-packages/upsetplot/plotting.py
    # The modification is done in the `_label_sizes()` function under the
    # `where == "top"` condition.

    # Save the plot.
    upset_fig.savefig(
        "isic_overlap.png", dpi=300, bbox_inches="tight", pad_inches=0.1
    )
    # upset_fig.close()


def main() -> None:
    """
    Main function.
    """
    # First, read the IMA++ ISIC IDs.
    imapp_isic_ids = (
        pd.read_csv(
            "../output/metadata/IMAplusplus_seg_metadata.csv",
            header="infer",
            sep=",",
        )["ISIC_id"]
        .unique()
        .tolist()
    )

    # Read the ISIC IDs for each challenge.
    isic_ids_2016 = read_isic_ids(Path("data_files/ISIC2016"))
    isic_ids_2017 = read_isic_ids(Path("data_files/ISIC2017"))
    isic_ids_2018 = read_isic_ids(Path("data_files/ISIC2018"))

    contents = {
        "ISIC 2016": isic_ids_2016,
        "ISIC 2017": isic_ids_2017,
        "ISIC 2018": isic_ids_2018,
        "IMA++": imapp_isic_ids,
    }
    plot_isic_overlap(contents)


if __name__ == "__main__":
    main()
