import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"create_multiannotator_subset_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def create_multiannotator_subset(config: OmegaConf) -> None:
    """
    Create the multiannotator subset of the dataset.
    """

    # Read the segmentation masks metadata file into a DataFrame.
    seg_masks_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(seg_masks_df)} rows of segmentation masks metadata."
    )

    # Next, we need to create a new subset of the dataset that only contains
    # the images that have more than 1 segmentation.
    # For this, first ensure that the column used for grouping exists.
    if "ISIC_id" not in seg_masks_df.columns:
        logger.error(
            "Required column `ISIC_id` not found in the segmentation "
            "masks metadata."
        )
        return
    # Then, group by the `ISIC_id` column and filter.
    seg_masks_df = seg_masks_df[
        seg_masks_df.groupby("ISIC_id")["seg_filename"].transform("nunique")
        > 1
    ]

    logger.info(
        f"Number of images that have been segmented by at least "
        f"2 annotators: {len(seg_masks_df.ISIC_id.unique())}"
    )
    logger.info(f"Number of segmentations in the subset: {len(seg_masks_df)}")

    # Next, we will export the subset of the dataset to a CSV file.
    seg_masks_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.multiannotator_subset_metadata_path),
        index=False,
    )
    logger.info(
        f"Exported {len(seg_masks_df)} rows of segmentation masks metadata to "
        f"{config.multiannotator_subset_metadata_path}."
    )


def main() -> None:
    """
    Main function.
    """
    # Read the config file.
    config = OmegaConf.load("config.yaml")
    create_multiannotator_subset(config)


if __name__ == "__main__":
    main()
