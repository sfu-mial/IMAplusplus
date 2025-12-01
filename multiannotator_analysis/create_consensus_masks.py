import os
import sys
from datetime import datetime
from pathlib import Path

import loguru
import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from omegaconf import OmegaConf
from skimage import io
from tqdm import tqdm

sys.path.append("..")
from utils.data import get_image_mask_pairs, load_mask
from utils.md5 import calculate_md5_file

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"create_consensus_masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def create_consensus_masks(config: OmegaConf) -> None:
    """
    Create the consensus masks (STAPLE and majority voting) for the dataset.

    Parameters
    ----------
    config : OmegaConf
        Configuration object.

    Returns
    -------
    None
        None.
    """
    # Specify the original segmentation masks directory.
    # This is also where we will save the consensus masks.
    orig_segs_dir = Path(config.new_dataset_masks_dir)
    if not orig_segs_dir.exists():
        logger.error(
            f"Original segmentation masks directory not found: {orig_segs_dir}"
        )
        return

    # First, read the metadata files for the dataset and for the
    # multiannotator subset into DataFrames.
    # We read both, but only use the multiannotator subset for consensus mask
    # calculation.
    # However, when the consensus masks are calculated and saved, we need to
    # add their corresponding metadata to *both* the metadata files.
    dataset_seg_masks_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(dataset_seg_masks_df)} rows of dataset segmentation masks "
        f"metadata."
    )
    multiannotator_subset_seg_masks_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.multiannotator_subset_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(multiannotator_subset_seg_masks_df)} rows of "
        f"multiannotator subset segmentation masks metadata."
    )

    # Group masks by image to create image-mask pairs for analysis.
    image_mask_pairs = get_image_mask_pairs(multiannotator_subset_seg_masks_df)
    logger.info(
        f"Created {len(image_mask_pairs)} image-mask pairs for "
        f"multiannotator subset."
    )

    # Create a list to store the consensus masks' metadata.
    consensus_masks_metadata = []

    # Next, for each image-mask pair, calculate the consensus mask.
    for img_id, mask_ids in tqdm(
        image_mask_pairs.items(),
        total=len(image_mask_pairs),
        desc="Calculating consensus masks",
    ):
        if len(mask_ids) < 2:
            logger.warning(
                f"Skipping image {img_id} because it has less than 2 masks."
            )
            continue

        # Load the masks for the current image.
        masks = []
        for mask_id in mask_ids:
            mask_path = orig_segs_dir / mask_id
            if not mask_path.exists():
                logger.error(f"Mask file not found: {mask_path}")
                continue
            mask = load_mask(mask_path.as_posix(), logger)
            if mask is not None:
                masks.append(mask.astype(np.uint8))

        # Skip if we did not load any masks.
        if not masks:
            logger.warning(
                f"No valid masks found for image {img_id}. Skipping."
            )
            continue

        # Calculate the STAPLE consensus mask.
        staple_mask = calculate_STAPLE_consensus_mask(masks, logger)
        if staple_mask is None:
            logger.error(
                f"Error calculating STAPLE consensus mask for image {img_id}."
            )
            continue

        # Calculate the majority voting consensus mask.
        majority_voting_mask = calculate_majority_voting_consensus_mask(
            masks, logger
        )
        if majority_voting_mask is None:
            logger.error(
                f"Error calculating majority voting consensus mask for image {img_id}."
            )
            continue

        # Save the consensus masks.
        staple_mask_path = (
            orig_segs_dir / f"{img_id.split('.')[0]}_ST_ST_ST_ST.png"
        )
        majority_voting_mask_path = (
            orig_segs_dir / f"{img_id.split('.')[0]}_MV_MV_MV_MV.png"
        )
        io.imsave(staple_mask_path, staple_mask * 255)
        io.imsave(majority_voting_mask_path, majority_voting_mask * 255)

        # Next, we need to calculate the MD5 hash of the consensus masks, and
        # store this in the metadata files.
        staple_mask_md5 = calculate_md5_file(staple_mask_path)
        majority_voting_mask_md5 = calculate_md5_file(
            majority_voting_mask_path
        )

        # Add the consensus masks metadata to the lists.
        consensus_masks_metadata.append(
            {
                "ISIC_id": img_id.split(".")[0],
                "img_filename": img_id,
                "seg_filename": f"{img_id.split('.')[0]}_ST_ST_ST_ST.png",
                "annotator": "ST",
                "tool": "ST",
                "skill_level": "ST",
                "mskObjectID": "ST",
                "mask_md5": staple_mask_md5,
            }
        )
        consensus_masks_metadata.append(
            {
                "ISIC_id": img_id.split(".")[0],
                "img_filename": img_id,
                "seg_filename": f"{img_id.split('.')[0]}_MV_MV_MV_MV.png",
                "annotator": "MV",
                "tool": "MV",
                "skill_level": "MV",
                "mskObjectID": "MV",
                "mask_md5": majority_voting_mask_md5,
            }
        )

    # Next, we need to add the consensus masks metadata to the dataset and the
    # multiannotator subset metadata files.
    dataset_seg_masks_df = pd.concat(
        [dataset_seg_masks_df, pd.DataFrame(consensus_masks_metadata)],
        ignore_index=True,
    )
    multiannotator_subset_seg_masks_df = pd.concat(
        [
            multiannotator_subset_seg_masks_df,
            pd.DataFrame(consensus_masks_metadata),
        ],
        ignore_index=True,
    )

    # Export the dataset and the multiannotator subset metadata files.
    dataset_seg_masks_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        index=False,
    )
    multiannotator_subset_seg_masks_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.multiannotator_subset_metadata_path),
        index=False,
    )

    logger.info(
        f"Exported {len(dataset_seg_masks_df)} rows of dataset segmentation masks "
        f"metadata to {config.output_seg_masks_metadata_path}."
    )
    logger.info(
        f"Exported {len(multiannotator_subset_seg_masks_df)} rows of "
        f"multiannotator subset segmentation masks metadata to "
        f"{config.multiannotator_subset_metadata_path}."
    )
    return None


def calculate_STAPLE_consensus_mask(
    masks: list[np.ndarray],
    logger: loguru.logger,
) -> np.ndarray:
    """
    Calculate the STAPLE consensus mask for the given masks.
    """
    # Convert the masks to SimpleITK format.
    sitk_masks = [sitk.GetImageFromArray(mask) for mask in masks]

    # Apply the STAPLE algorithm.
    try:
        stapler = sitk.STAPLEImageFilter()
        staple_output = stapler.Execute(sitk_masks)

        # Convert the STAPLE output back to a NumPy array and threshold it.
        staple_prob = sitk.GetArrayFromImage(staple_output)
        staple_binary = (staple_prob > 0.5).astype(np.uint8)

        return staple_binary
    except Exception as e:
        logger.error(f"Error calculating STAPLE consensus mask: {e}")
        return None


def calculate_majority_voting_consensus_mask(
    masks: list[np.ndarray],
    logger: loguru.logger,
) -> np.ndarray:
    """
    Calculate the majority voting consensus mask for the given masks.
    """

    # Convert the masks to SimpleITK format.
    sitk_masks = [sitk.GetImageFromArray(mask) for mask in masks]

    # Apply the SimpleITK LabelVotingImageFilter to the masks.
    try:
        label_voter = sitk.LabelVotingImageFilter()
        # For undecided pixels, we will assign a label of 1 (foreground).
        # This is because we want to include the undecided pixels in the consensus mask.
        label_voter.SetLabelForUndecidedPixels(1)
        consensus = label_voter.Execute(sitk_masks)

        # Convert the consensus back to a NumPy array and threshold it.
        consensus_prob = sitk.GetArrayFromImage(consensus)
        consensus_binary = (consensus_prob == 1).astype(np.uint8)

        return consensus_binary
    except Exception as e:
        logger.error(f"Error calculating majority voting consensus mask: {e}")
        return None


def main() -> None:
    """
    Main function to create the consensus masks for the dataset.
    """
    config = OmegaConf.load("config.yaml")
    create_consensus_masks(config)
    return None


if __name__ == "__main__":
    main()
