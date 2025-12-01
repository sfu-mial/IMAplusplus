import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from skimage import measure
from tqdm import tqdm

sys.path.append("..")
from utils.data import load_mask

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"mask_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def validate_masks_sequential_not_used(
    seg_metadata: pd.DataFrame, config: OmegaConf
) -> pd.DataFrame:
    """
    Validate masks for the fllowing issues:
    - missing/corrupted files (high severity)
    - empty masks (high severity)
    - masks covering the entire image (high severity)
    - disconnected regions (medium severity)
    - masks touching image borders (low severity)

    Parameters
    ----------
    seg_metadata : pandas.DataFrame
        DataFrame containing segmentation metadata with columns for image IDs
        ('img_filename'), mask IDs ('seg_filename'), and annotator information.
    config : omegaconf.OmegaConf
        Config object containing the paths to the dataset directories.

    Returns
    -------
    pandas.DataFrame
        DataFrame with details of all identified issues, including image ID, mask ID,
        issue description, and severity level. Empty DataFrame if no issues are found.
    """
    mask_dir = config.new_dataset_masks_dir
    issues = []

    # Process each mask with tqdm.
    for idx, row in tqdm(
        seg_metadata.iterrows(),
        total=len(seg_metadata),
        desc="Validating masks",
    ):
        mask_id = row["seg_filename"]
        mask_path = Path(mask_dir) / mask_id
        mask = load_mask(mask_path.as_posix(), logger)

        if mask is None:
            issues.append(
                {
                    "image": row["img_filename"],
                    "mask": mask_id,
                    "issue": "File not found or corrupted",
                    "severity": "high",
                }
            )
            continue

        # Check if mask is empty
        if np.sum(mask) == 0:
            issues.append(
                {
                    "image": row["img_filename"],
                    "mask": mask_id,
                    "issue": "Empty mask",
                    "severity": "high",
                }
            )
            continue

        # Check if mask covers the entire image
        if np.sum(mask) == mask.size:
            issues.append(
                {
                    "image": row["img_filename"],
                    "mask": mask_id,
                    "issue": "Mask covers entire image",
                    "severity": "high",
                }
            )
            continue

        # Check for disconnected regions
        labeled_mask, num_regions = measure.label(mask, return_num=True)

        if num_regions > 1:
            # Find sizes of all regions
            region_sizes = np.bincount(labeled_mask.flatten())[1:]

            # If the second largest region is significant (>5% of the largest)
            if (
                len(region_sizes) > 1
                and region_sizes[1] > 0.05 * region_sizes[0]
            ):
                issues.append(
                    {
                        "image": row["img_filename"],
                        "mask": mask_id,
                        "issue": f"Multiple disconnected regions ({num_regions})",
                        "severity": "medium",
                    }
                )

        # Check if mask touches image borders
        border_touch = (
            np.any(mask[0, :])  # Top border
            or np.any(mask[-1, :])  # Bottom border
            or np.any(mask[:, 0])  # Left border
            or np.any(mask[:, -1])  # Right border
        )

        if border_touch:
            issues.append(
                {
                    "image": row["img_filename"],
                    "mask": mask_id,
                    "issue": "Mask touches image border",
                    "severity": "low",
                }
            )

    # Convert issues to DataFrame
    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()

    # Log summary
    if not issues_df.empty:
        logger.warning(
            f"Found {len(issues_df)} issues in {issues_df['mask'].nunique()} masks"
        )

        # Count by severity
        severity_counts = issues_df["severity"].value_counts()
        for severity, count in severity_counts.items():
            logger.warning(f"  - {severity}: {count}")
    else:
        logger.info("No issues found in masks")

    return issues_df


def _validate_single_mask(
    seg_metadata_row: Dict[str, Any], mask_dir: os.PathLike
) -> List[Dict[str, Any]]:
    """
    Worker function to validate a *single* mask.
    Since this function is called in parallel using `ProcessPoolExecutor`,
    we need to ensure that the function is picklable.
    This means that it must be a "top-level" function, i.e. not defined inside
    another function.
    This also means that it must import its own dependencies.

    Parameters
    ----------
    seg_metadata_row : Dict[str, Any]
        Dict containing 1 row of segmentation metadata with keys for
        image IDs ('img_filename'), mask IDs ('seg_filename'), and annotator information.
    mask_dir : os.PathLike
        Path to the directory containing the masks.

    Returns
    -------
    List[Dict[str, Any]]
        List with details of all identified issues, including image ID, mask ID,
        issue description, and severity level. Empty List if no issues are found.
    """
    # Imports needed by each worker process.
    from pathlib import Path

    import numpy as np
    from skimage import measure

    issues = []
    mask_id = seg_metadata_row["seg_filename"]
    img_filename = seg_metadata_row["img_filename"]
    mask_path = Path(mask_dir) / mask_id

    mask = load_mask(mask_path.as_posix(), logger)
    # Check if the mask does not exist.
    if mask is None:
        issues.append(
            {
                "image": img_filename,
                "mask": mask_id,
                "issue": "File not found or corrupted",
                "severity": "high",
            }
        )
        # If yes, no need to continue checking for other issues.
        return issues

    # Check if the mask is empty.
    if np.sum(mask) == 0:
        issues.append(
            {
                "image": img_filename,
                "mask": mask_id,
                "issue": "Empty mask",
                "severity": "high",
            }
        )
        # If yes, no need to continue checking for other issues.
        return issues

    # Check if the mask covers the entire image.
    if np.sum(mask) == mask.size:
        issues.append(
            {
                "image": img_filename,
                "mask": mask_id,
                "issue": "Mask covers entire image",
                "severity": "high",
            }
        )
        # If yes, no need to continue checking for other issues.
        return issues

    # Check for (multiple) disconnected regions.
    labeled_mask, num_regions = measure.label(mask, return_num=True)
    # If there are multiple regions, check if the second largest region is
    # significant (>5% of the largest region).
    if num_regions > 1:
        # Find sizes of all regions.
        region_sizes = np.bincount(labeled_mask.flatten())[1:]

        # If there are multiple regions and the second largest region is
        # significant (>5% of the largest region), add an issue.
        if len(region_sizes) > 1 and region_sizes[1] > 0.05 * region_sizes[0]:
            issues.append(
                {
                    "image": img_filename,
                    "mask": mask_id,
                    "issue": f"Multiple disconnected regions ({num_regions})",
                    "severity": "medium",
                }
            )

    # Check if the mask touches the image borders.
    border_touch = (
        np.any(mask[0, :])  # Top border
        or np.any(mask[-1, :])  # Bottom border
        or np.any(mask[:, 0])  # Left border
        or np.any(mask[:, -1])  # Right border
    )
    if border_touch:
        issues.append(
            {
                "image": img_filename,
                "mask": mask_id,
                "issue": "Mask touches image border",
                "severity": "low",
            }
        )
    return issues


def validate_masks_parallel(
    seg_metadata: pd.DataFrame, config: OmegaConf
) -> pd.DataFrame:
    """
    Parallelized version of `validate_masks()` defined above. Uses
    `concurrent.futures.ProcessPoolExecutor` to run validations in parallel.

    Validate masks for the following issues:
    - missing/corrupted files (high severity)
    - empty masks (high severity)
    - masks covering the entire image (high severity)
    - disconnected regions (medium severity)
    - masks touching image borders (low severity)

    Parameters
    ----------
    seg_metadata : pandas.DataFrame
        DataFrame containing segmentation metadata with columns for image IDs
        ('img_filename'), mask IDs ('seg_filename'), and annotator information.
    config : omegaconf.OmegaConf
        Config object containing the paths to the dataset directories.

    Returns
    -------
    pandas.DataFrame
        DataFrame with details of all identified issues, including image ID, mask ID,
        issue description, and severity level. Empty DataFrame if no issues are found.
    """
    mask_dir = config.new_dataset_masks_dir
    issues = []

    # Use `ProcessPoolExecutor` to run validations in parallel.
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Prepare the arguments for `executor.map()`.
        # 1. An iterator for row dicts (plain Python dicts are picklable).
        row_tuples = seg_metadata[["img_filename", "seg_filename"]].to_dict(
            orient="records"
        )
        # 2. An iterable for the mask directory.
        mask_dir_iter = repeat(mask_dir)

        # Wrao the results of `executor.map()` in a tqdm progress bar.
        results_iter = executor.map(
            _validate_single_mask, row_tuples, mask_dir_iter
        )

        all_issues_list = list(
            tqdm(
                results_iter, total=len(seg_metadata), desc="Validating masks"
            )
        )

    # The result will be a list of lists, each containing the issues for a
    # single mask.
    # We need to flatten this list to get a single list of issues.
    issues = [issue for sublist in all_issues_list for issue in sublist]

    # Convert the list of issues to a DataFrame.
    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()

    # Log summary
    if not issues_df.empty:
        logger.warning(
            f"Found {len(issues_df)} issues in {issues_df['mask'].nunique()} masks"
        )

        # Count by severity
        severity_counts = issues_df["severity"].value_counts()
        for severity, count in severity_counts.items():
            logger.warning(f"  - {severity}: {count}")
    else:
        logger.info("No issues found in masks")

    return issues_df


def sanitize_mask_issues_df(
    issues_df: pd.DataFrame, config: OmegaConf
) -> pd.DataFrame:
    """
    Sanitize the mask issues DataFrame by removing masks and the corresponding metadata
    for the following issues:
    - File not found or corrupted
    - Empty mask

    Parameters
    ----------
    issues_df : pandas.DataFrame
        DataFrame containing the mask issues.
    config : omegaconf.OmegaConf
        Config object containing the paths to the dataset directories.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the sanitized mask issues.
    """
    # Read the segmentation masks and images metadata files into DataFrames.
    seg_metadata = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    img_metadata = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_img_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )

    # Remove masks and the corresponding metadata for the following issues:
    # - File not found or corrupted
    # - Empty mask
    issues_df_to_remove = issues_df[
        issues_df["issue"].isin(["File not found or corrupted", "Empty mask"])
    ]

    # For each mask in the issues to remove, remove the image and the mask
    # file and the corresponding metadata rows from the segmentation masks and
    # images metadata files.
    for idx, row in issues_df_to_remove.iterrows():
        img_id = row["image"]
        mask_id = row["mask"]
        img_path = Path(config.new_dataset_images_dir) / img_id
        mask_path = Path(config.new_dataset_masks_dir) / mask_id
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            continue
        if not mask_path.exists():
            logger.error(f"Mask file not found: {mask_path}")
            continue
        os.remove(img_path.as_posix())
        os.remove(mask_path.as_posix())
        # Find the corresponding row in the segmentation masks metadata file.
        seg_metadata_row = seg_metadata[
            seg_metadata["seg_filename"] == mask_id
        ]
        seg_metadata.drop(seg_metadata_row.index, inplace=True)
        # Find the corresponding row in the images metadata file.
        img_metadata_row = img_metadata[
            img_metadata["isic_id"] == img_id.split(".")[0]
        ]
        img_metadata.drop(img_metadata_row.index, inplace=True)

    # Export the sanitized segmentation masks and images metadata files.
    seg_metadata.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        index=False,
    )
    logger.info(
        f"Exported {len(seg_metadata)} rows of segmentation masks metadata to "
        f"{config.output_seg_masks_metadata_path}."
    )
    img_metadata.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_img_metadata_path),
        index=False,
    )
    logger.info(
        f"Exported {len(img_metadata)} rows of images metadata to "
        f"{config.output_img_metadata_path}."
    )
    # Export the sanitized mask issues DataFrame.
    issues_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_qa_results_path),
        index=False,
    )
    logger.info(
        f"Exported {len(issues_df)} rows of segmentation masks metadata issues to "
        f"{config.output_seg_masks_metadata_qa_results_path}."
    )

    return issues_df_to_remove


def main() -> None:
    """
    Main function.
    """
    # Read the config file.
    config = OmegaConf.load("config.yaml")

    # Read the segmentation masks metadata file into a DataFrame.
    seg_metadata = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    issues_df = validate_masks_parallel(seg_metadata, config)

    issues_df_to_remove = sanitize_mask_issues_df(issues_df, config)
    logger.info(
        f"Sanitized {len(issues_df_to_remove)} rows of segmentation masks metadata issues."
    )


if __name__ == "__main__":
    main()
