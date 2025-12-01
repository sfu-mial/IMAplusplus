from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents

# Define the constants for the plot.
# 902241 is the hex code for IEEE Data Descriptions.
IMA_PLUS_PLUS_COLOR = "#90224190"

VIS_OUTPUT_DIR = Path("../output/visualizations/upset_plots/")
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_other_dataset_ids(challenge_dataset_dir: Path) -> list[str]:
    """
    Given a directory containing multiple .txt files, each containing a list of
    image IDs (ISIC IDs wherever applicable), return a list of all image IDs.
    """
    image_ids = []
    for file in challenge_dataset_dir.glob("*.txt"):
        with open(file, "r") as f:
            image_ids.extend(f.read().splitlines())
    return image_ids


def plot_other_datasets_overlap(contents: dict[str, list[str]]) -> None:
    """
    Plot the overlap of the image IDs.
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
        VIS_OUTPUT_DIR / "other_datasets_overlap.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    upset_fig.savefig(
        VIS_OUTPUT_DIR / "other_datasets_overlap.svg",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    upset_fig.savefig(
        VIS_OUTPUT_DIR / "other_datasets_overlap.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


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

    # Read the ISIC IDs for the ISIC 2016 - 2018 Segmentation Challenges.
    isic_ids_2016 = read_other_dataset_ids(Path("data_files/ISIC2016"))
    isic_ids_2017 = read_other_dataset_ids(Path("data_files/ISIC2017"))
    isic_ids_2018 = read_other_dataset_ids(Path("data_files/ISIC2018"))

    # Read the ISIC IDs for the HAM10000 and ISIC2019_seg datasets.
    ham10000_ids = read_other_dataset_ids(Path("data_files/HAM10000"))
    isic2019_seg_ids = read_other_dataset_ids(Path("data_files/ISIC2019_seg"))

    # Read the image IDs for non-ISIC datasets: PH2, DermoFit, SCD.
    ph2_ids = read_other_dataset_ids(Path("data_files/PH2"))
    dermafit_ids = read_other_dataset_ids(Path("data_files/DermoFit"))
    scd_ids = read_other_dataset_ids(Path("data_files/SCD"))

    contents = {
        "ISIC 2016": isic_ids_2016,
        "ISIC 2017": isic_ids_2017,
        "ISIC 2018": isic_ids_2018,
        "HAM10000": ham10000_ids,
        "ISIC2019-Seg": isic2019_seg_ids,
        "PH2": ph2_ids,
        "DermoFit": dermafit_ids,
        "SCD": scd_ids,
        "IMA++": imapp_isic_ids,
    }
    plot_other_datasets_overlap(contents)


if __name__ == "__main__":
    main()
