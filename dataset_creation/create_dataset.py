import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from constants import SKILL_LEVEL_MAPPING, SOURCE_TOOL_MAPPING
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append("..")
from utils.md5 import calculate_md5_file

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"create_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def create_seg_dataset_and_metadata(
    config: OmegaConf,
    output_seg_masks_metadata_path: Path,
    verbose: bool = False,
) -> None:
    """
    Create a segmentation dataset and metadata.
    """

    # Check if the image and segmentation metadata files exist before
    # attempting to read.
    raw_seg_masks_metadata_path = Path(config.raw_seg_masks_metadata_path)
    raw_img_metadata_path = Path(config.raw_img_metadata_path)

    if not raw_seg_masks_metadata_path.exists():
        logger.error(
            f"Segmentation masks metadata file not found at {raw_seg_masks_metadata_path}."
        )
        return

    if not raw_img_metadata_path.exists():
        logger.error(
            f"Images metadata file not found at {raw_img_metadata_path}."
        )
        return

    try:
        # If they do exist, read them into DataFrames.
        # We set `low_memory=False` since these are large files.
        raw_seg_masks_df = pd.read_csv(
            raw_seg_masks_metadata_path,
            header="infer",
            sep=",",
            low_memory=False,
        )
        raw_imgs_df = pd.read_csv(
            raw_img_metadata_path, header="infer", sep=",", low_memory=False
        )
        logger.info(
            f"Read {len(raw_seg_masks_df)} segmentation masks and {len(raw_imgs_df)} images."
        )
    except pd.errors.EmptyDataError as e:
        logger.error(f"Could not read CSV file(s): {e}")
        return
    except FileNotFoundError as e:
        # This case should be handled by the `exists()` check, but keeping it
        # for robustness.
        logger.error(f"File(s) not found during `read_csv`: {e}")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading CSV file(s): {e}")
        return

    # Print initial counts of the DataFrames.
    if verbose:
        logger.info(f"{len(raw_seg_masks_df)} total segmentations.")
        logger.info(f"{len(raw_imgs_df)} total images.")

    # Filter the segmentations to only retain those whose images have metadata,
    # since only those that have metadata have the images.

    # Ensure that the necessary columns are present in the DataFrames before
    # filtering.
    if (
        "ISIC_id" not in raw_seg_masks_df.columns
        or "isic_id" not in raw_imgs_df.columns
    ):
        logger.error(
            "Required columns (`ISIC_id` or `isic_id`) not found in the "
            "metadata files."
        )
        return

    # Now we filter the segmentations to only retain those whose images are
    # present.
    raw_seg_masks_df = raw_seg_masks_df[
        raw_seg_masks_df["ISIC_id"].isin(raw_imgs_df["isic_id"])
    ]

    # Exclude segmentations whose mskObjectIDs are in the exclude list.
    if hasattr(config, "exclude_mskObjectIDs"):
        raw_seg_masks_df = raw_seg_masks_df[
            ~raw_seg_masks_df["mskObjectID"].isin(config.exclude_mskObjectIDs)
        ]

    if verbose:
        logger.info(
            f"{len(raw_seg_masks_df)} segmentations remaining after filtering "
            f"for metadata."
        )

    # Anonymize the annotators.
    # Assign them names from `A00` to `Ann`, with annotator IDs assigned in
    # decreasing order of the number of images segmented.

    # First, we count how many segmentations each annotator (`creator`) has
    # produced.
    if "creator" not in raw_seg_masks_df.columns:
        logger.error(
            "Required column `creator` not found in the segmentation metadata."
        )
        return

    # Count the number of segmentations per annotator.
    annotator_counts = raw_seg_masks_df["creator"].value_counts()

    # Next, anonymize the `creator` column through a mapping of name -> A<xx>.
    annotator_name_mapping = {
        name: f"A{idx:02d}" for idx, name in enumerate(annotator_counts.index)
    }
    # Next, create a new column `annotator` in the segmentation metadata
    # DataFrame to store the anonymized names.
    raw_seg_masks_df["annotator"] = raw_seg_masks_df["creator"].map(
        annotator_name_mapping
    )

    # Next, check for the `mskObjectID` column. We will use
    # this to name our segmentation masks and also to find the corresponding
    # segmentation mask file in the `orig_segs_dir`.
    # Important Note: Truncating the `mskObjectID` column to the shortest
    # possible length is not possible because any truncation will result in
    # duplicate MD5 hashes, i.e., the output of
    # `utils.md5.calculate_shortest_md5_length()`` returns 24, which is the
    # same as the length of the `mskObjectID` column.
    if "mskObjectID" not in raw_seg_masks_df.columns:
        logger.error(
            "Required column `mskObjectID` not found in the segmentation metadata."
        )
        return

    # Next, calculate the MD5 hash of the segmentation masks and store it in a
    # new column `mask_md5`. For this, we will read the segmentation masks from
    # the `orig_segs_dir` and calculate the MD5 hash of the image data.

    seg_paths = [
        Path(config.orig_segs_dir) / x for x in raw_seg_masks_df["filename"]
    ]

    # We will use a ProcessPoolExecutor to parallelize the calculation of the
    # MD5 hashes.

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        raw_seg_masks_df["mask_md5"] = list(
            tqdm(
                executor.map(calculate_md5_file, seg_paths),
                total=len(seg_paths),
                desc="Calculating MD5 hashes",
            )
        )

    # Create column for tool used to segment the image..
    # We map the `source` column to a tool ID using the `SOURCE_TOOL_MAPPING`.
    if "source" not in raw_seg_masks_df.columns:
        logger.error("Required column `source` not found for tool mapping.")
        return
    raw_seg_masks_df["tool"] = raw_seg_masks_df["source"].map(
        SOURCE_TOOL_MAPPING
    )

    # Create column for skill level.
    # We map the `skill` column to a skill level ID using the
    # `SKILL_LEVEL_MAPPING`.
    if "skill" not in raw_seg_masks_df.columns:
        logger.error(
            "Required column `skill` not found for skill level mapping."
        )
        return
    raw_seg_masks_df["skill_level"] = raw_seg_masks_df["skill"].map(
        SKILL_LEVEL_MAPPING
    )

    # We did not end up truncating the `mask_md5` column to the shortest
    # possible length because they were already unique.
    # So we now store them as a separate column `mask_md5` in our CSV file.

    # # Next, truncate the `mask_md5` column to the shortest possible
    # # length to ensure that the truncated MD5 hash is unique.
    # logger.info("Truncating MD5 hashes to the shortest possible length...")
    # raw_seg_masks_df["mask_md5_trunc"] = raw_seg_masks_df["mask_md5"].str[
    #     : calculate_shortest_md5_length(raw_seg_masks_df, "mask_md5")
    # ]
    # logger.info("MD5 hashes truncated to the shortest possible length.")

    # Then, create a new column for storing the new filenames that contain
    # the following info:
    # - the `isic_id`
    # - the `annotator`
    # - the `tool`
    # - the `skill_level`
    # - the `mskObjectID`

    # We ideally would also want to store the `mask_md5` column in the new
    # filenames, but since it is 32 chars long, it would make the filenames
    # too long.

    # First, we ensure that the necessary columns are present in the DataFrame.
    if (
        "ISIC_id" not in raw_seg_masks_df.columns
        or "annotator" not in raw_seg_masks_df.columns
        or "mskObjectID" not in raw_seg_masks_df.columns
        or "tool" not in raw_seg_masks_df.columns
        or "skill_level" not in raw_seg_masks_df.columns
    ):
        logger.error(
            "Required columns not found in the segmentation metadata."
        )
        return

    # Then, we create the new filenames.
    raw_seg_masks_df["new_seg_filename"] = raw_seg_masks_df.apply(
        lambda row: f"{row['ISIC_id']}_"
        f"{row['annotator']}_"
        f"{row['tool']}_"
        f"{row['skill_level']}_"
        f"{row['mskObjectID']}"
        f"{Path(row.filename).suffix}",  # Retains the same file extension.
        axis=1,
    )

    # Show the anonymized annotator name mapping with the number of images
    # segmented by each annotator.
    if verbose:
        logger.info("\nAnonymized annotator name mapping:")
        logger.info(
            raw_seg_masks_df.groupby(["creator", "annotator"])
            .size()
            .reset_index(name="count")
            .sort_values("annotator")
        )

    # We also show how many images were segmented by how many annotators.
    # To achieve this, we use `value_counts()` twice:
    # - First, we group by `ISIC_id` to count the number of annotators per
    # image (`ISIC_id`).
    # - Then, we use it once again to count how many images were segmented by
    # 1 annotator, 2 annotators, 3 annotators, etc.
    if verbose:
        logger.info("\nDistribution of number of segmentations per image:")
        # Ensure that the necessary columns are present in the DataFrame.
        if "ISIC_id" not in raw_seg_masks_df.columns:
            logger.error(
                "Required column `ISIC_id` not found for distribution calculation."
            )
            pass
            # Note: This error will be logged, but the print might not happen
            # if verbose is False.
            # This is acceptable based on the requirement to print only if
            # verbose is True.
        else:
            logger.info(
                raw_seg_masks_df["ISIC_id"]
                .value_counts()
                .value_counts()
                .sort_index()
            )

    # Print statistics on the `tool` and `skill_level` columns.
    if verbose:
        logger.info("\nTool statistics:")
        logger.info(raw_seg_masks_df["tool"].value_counts().sort_index())
        logger.info("\nSkill level statistics:")
        logger.info(
            raw_seg_masks_df["skill_level"].value_counts().sort_index()
        )

    # Next, we prepare the segmentation dataset and metadata.

    # For that, we create columns for image and segmentation filenames.
    # Image name is just <ISIC_id>.JPG.
    # Segmentation name is the `new_seg_filename` column we just created.
    if (
        "new_seg_filename" not in raw_seg_masks_df.columns
        or "ISIC_id" not in raw_seg_masks_df.columns
    ):
        logger.error(
            "Required columns for image/segmentation filename creation not "
            "found."
        )
        return
    raw_seg_masks_df["img_filename"] = raw_seg_masks_df["ISIC_id"].map(
        lambda x: f"{x}.JPG"
    )
    raw_seg_masks_df["seg_filename"] = raw_seg_masks_df["new_seg_filename"]

    # Finally, we export the anonymized segmentations' metadata to a CSV file.
    COLUMNS_TO_EXPORT = [
        "ISIC_id",
        "img_filename",
        "seg_filename",
        "annotator",
        "tool",
        "skill_level",
        "mskObjectID",
        "mask_md5",
    ]

    # Ensure that the necessary columns exist in the DataFrame.
    if not all(col in raw_seg_masks_df.columns for col in COLUMNS_TO_EXPORT):
        missing_cols = [
            col
            for col in COLUMNS_TO_EXPORT
            if col not in raw_seg_masks_df.columns
        ]
        logger.error(f"Missing columns fror export: {missing_cols}")
        return

    # Then, we export the DataFrame to a CSV file.
    raw_seg_masks_df[COLUMNS_TO_EXPORT].to_csv(
        output_seg_masks_metadata_path, index=False
    )
    logger.info(
        f"Exported {len(raw_seg_masks_df)} anonymized segmentations' metadata "
        f"to {output_seg_masks_metadata_path}."
    )

    # As a last step, we return the unique `ISIC_id` values.
    # We will use this to create the image dataset and metadata.
    return raw_seg_masks_df["ISIC_id"].unique()


def create_img_dataset_and_metadata(
    config: OmegaConf,
    isic_ids: pd.Series,
    output_img_metadata_path: Path,
    verbose: bool = False,
) -> None:
    """
    Create an image dataset and metadata.
    """
    raw_img_metadata_path = Path(config.raw_img_metadata_path)
    # Check if the image metadata file exists before attempting to read.
    if not raw_img_metadata_path.exists():
        logger.error(
            f"Images metadata file not found at {raw_img_metadata_path}."
        )
        return

    # Read the image metadata file into a DataFrame.
    try:
        raw_imgs_df = pd.read_csv(
            raw_img_metadata_path,
            header="infer",
            sep=",",
            on_bad_lines="error",
            quotechar='"',
            engine="python",
        )
        logger.info(f"Read {len(raw_imgs_df)} images' metadata.")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Could not read images' metadata file: {e}")
        return
    except FileNotFoundError as e:
        # This case should be handled by the `exists()` check, but keeping it
        # for robustness.
        logger.error(f"Images' metadata file not found: {e}")
        return
    except Exception as e:
        logger.error(
            f"An error occurred while reading images' metadata file: {e}"
        )
        return

    # Filter the images to only retain those whose `isic_id` is in the
    # `isic_ids` list.
    raw_imgs_df = raw_imgs_df[raw_imgs_df["isic_id"].isin(isic_ids)]

    # As a last step, we export the image dataset and metadata to a CSV file.
    raw_imgs_df.to_csv(output_img_metadata_path, index=False)
    logger.info(
        f"Exported {len(raw_imgs_df)} images' metadata to "
        f"{config.output_img_metadata_path}."
    )


def main():
    """
    Main function.
    """
    try:
        config = OmegaConf.load("config.yaml")

        # Get the verbose flag from the config.
        # Default to False if not present.
        verbose_flag = config.get("verbose", False)

        # Create the output directory for the new dataset metadata.
        new_dataset_metadata_output_dir = Path(
            config.new_dataset_metadata_output_dir
        )
        new_dataset_metadata_output_dir.mkdir(parents=True, exist_ok=True)

        # Create the output path for the segmentations' and images' metadata.
        output_seg_masks_metadata_path = (
            new_dataset_metadata_output_dir
            / config.output_seg_masks_metadata_path
        )
        output_img_metadata_path = (
            new_dataset_metadata_output_dir / config.output_img_metadata_path
        )

        # Create the segmentation dataset and metadata.
        isic_ids_with_segs = create_seg_dataset_and_metadata(
            config, output_seg_masks_metadata_path, verbose=verbose_flag
        )

        # Create the image dataset and metadata.
        create_img_dataset_and_metadata(
            config,
            isic_ids_with_segs,
            output_img_metadata_path,
            verbose=verbose_flag,
        )
    except FileNotFoundError:
        logger.error("`config.yaml` not found.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
