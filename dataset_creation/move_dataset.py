import sys
from datetime import datetime
from pathlib import Path
from shutil import copy2

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"move_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def move_dataset(config: OmegaConf) -> None:
    """
    Move the dataset to the new location.
    """
    # Check if the source dataset directories (images and segs) and the
    # metadata files exist.
    # Check first for image directories.
    for img_ext, img_dirs in config.orig_imgs_dirs.items():
        for img_dir in img_dirs:
            if not Path(img_dir).exists():
                logger.error(f"Source image directory not found: {img_dir}")
                return

    # Next, check for segmentation directory.
    if not Path(config.orig_segs_dir).exists():
        logger.error(
            f"Source segmentation directory not found: {config.orig_segs_dir}"
        )
        return

    # Next, check for metadata files: both raw and new/processed.
    if not Path(config.raw_seg_masks_metadata_path).exists():
        logger.error(
            f"Source segmentation metadata file not found: {config.raw_seg_masks_metadata_path}"
        )
        return
    if not Path(
        Path(config.new_dataset_metadata_output_dir)
        / config.output_seg_masks_metadata_path
    ).exists():
        logger.error(
            f"Source segmentation metadata file not found: {config.new_dataset_metadata_output_dir / config.output_seg_masks_metadata_path}"
        )
        return

    # Next, check if the target data directory exists. If it does not, create
    # it.
    TARGET_DATA_DIR = Path(config.target_data_dir)
    TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create separate directories for images and masks if specified.
    if config.separate_images_and_masks:
        Path(TARGET_DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
        Path(TARGET_DATA_DIR / "masks").mkdir(parents=True, exist_ok=True)

    # Read the raw segmentation masks metadata file into a DataFrame.
    raw_seg_masks_df = pd.read_csv(
        config.raw_seg_masks_metadata_path,
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(raw_seg_masks_df)} rows of raw segmentation masks metadata."
    )

    # Read the new segmentation masks metadata file into a DataFrame.
    new_seg_masks_df = pd.read_csv(
        Path(
            Path(config.new_dataset_metadata_output_dir)
            / (config.output_seg_masks_metadata_path)
        ),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(new_seg_masks_df)} rows of new segmentation masks metadata."
    )

    # Now, let us copy.

    # For each row in the new metadata file, copy the image and the mask to
    # the target data directory.
    for _, row in tqdm(
        new_seg_masks_df.iterrows(),
        total=len(new_seg_masks_df),
        desc="Copying images and masks",
    ):
        # First, we need to find the segmtnation file for the current image.
        # Remember that the segmentations' original filenames are different
        # from what we want them to be (`seg_filename` column in the new
        # metadata file).
        # So, we load the raw segmentation metadata file, match the
        # `mskObjectID` column with the `mskObjectID` column in the raw
        # segmentation metadata file.
        # Then, we use the `filename` column in the raw segmentation metadata
        # to construct the path to the original segmentation file.

        # So, let's first find the corresponding row in the raw segmentation
        # metadata file.
        orig_seg_mask_filename = raw_seg_masks_df[
            raw_seg_masks_df["mskObjectID"] == row["mskObjectID"]
        ]["filename"].values[0]

        # Then, we construct the path to the original segmentation file.
        # We will use this path to first check if the segmentation file exists.
        # If it does, we perform the copy operation.
        src_seg_mask_path = Path(config.orig_segs_dir) / orig_seg_mask_filename
        if not src_seg_mask_path.exists():
            logger.error(
                f"Source segmentation mask file not found: {src_seg_mask_path}"
            )
            return

        # Now, we need to find the image file for the current row in the
        # metadata file.
        # Since the image filenames can be in any of the 4 source directories,
        # we need to check each of them. Remember that the file extension
        # differs between the source directories. See `config.yaml` for the
        # list of source directories and their corresponding file extensions.

        # We will use this flag to check if we found AND copied the image-mask
        # pair (for the current row in the new metadata file).
        found_and_copied = False

        # Check each of the source directories and their corresponding file
        # extensions.
        for img_ext, img_dirs in config.orig_imgs_dirs.items():
            for img_dir in img_dirs:
                # Construct the source image file path.
                # Depending on the source directory, choose the appropriate
                # file extension.
                src_img_path = (
                    Path(img_dir) / f"{Path(row.img_filename).stem}.{img_ext}"
                )

                if not src_img_path.exists():
                    # If the image file does not exist, this means that the
                    # image is not present in this source directory.
                    # So, we continue to the next source directory.
                    continue

                else:
                    # Construct the destination image and mask paths.
                    # All destination image paths will have the JPG extension
                    # for consistency.
                    if config.separate_images_and_masks:
                        # Image path.
                        dest_img_path = (
                            TARGET_DATA_DIR
                            / "images"
                            / f"{src_img_path.stem}.JPG"
                        )
                        # Mask path.
                        dest_seg_mask_path = (
                            TARGET_DATA_DIR
                            / "masks"
                            / Path(row.seg_filename).name
                        )
                    else:
                        # Image path.
                        dest_img_path = (
                            TARGET_DATA_DIR / f"{src_img_path.stem}.JPG"
                        )
                        # Mask path.
                        dest_seg_mask_path = (
                            TARGET_DATA_DIR / Path(row.seg_filename).name
                        )

                    # Copy the image and mask to the target data directory.
                    copy2(src_img_path, dest_img_path)
                    copy2(
                        src_seg_mask_path,
                        dest_seg_mask_path,
                    )
                    found_and_copied = True
                    break

        if not found_and_copied:
            logger.error(f"Image and mask not found for row {row.name}")
            return


def main():
    """
    Main function.
    """
    try:
        config = OmegaConf.load("config.yaml")
        move_dataset(config)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
