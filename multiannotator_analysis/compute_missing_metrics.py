"""
Compute the missing metrics for the multiannotator subset of the dataset.

For some ISIC_ids, because at least one of the masks is empty, the pairwise
IAA metrics were not computed for **any** mask pairs. In this script, we will
compute the metrics for valid mask pairs, and for pairs where one mask is
empty, we will set the metrics to 0 (for overlap metrics), NaN (for boundary
metrics), and 1 (for the normalized metrics).

We will edit the existing pairwise IAA metrics file and insert new rows in the
same file at the appropriate locations.
"""

import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append("..")

from utils.data import get_image_mask_pairs, load_mask
from utils.metrics import compute_boundary_metrics, compute_overlap_metrics

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"compute_missing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def compute_missing_metrics(config: OmegaConf) -> list[dict]:
    """
    Compute the missing metrics for the multiannotator subset of the dataset.

    Parameters
    ----------
    config : OmegaConf
        Configuration object.

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing the missing metrics for an
        image with a given ISIC_id.
    """
    # Read the multiannotator subset segmentation masks metadata file into a
    # DataFrame.
    seg_metadata_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.multiannotator_subset_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(seg_metadata_df)} rows of multiannotator subset segmentation "
        f"masks metadata."
    )

    # Read the missing metrics ISIC_ids list from the config.
    missing_metrics_ISIC_ids = config.missing_metrics_ISIC_ids
    logger.info(
        f"Read {len(missing_metrics_ISIC_ids)} missing metrics ISIC_ids."
    )

    if len(missing_metrics_ISIC_ids) == 0:
        logger.info("No missing metrics ISIC_ids found. Exiting.")
        return

    overall_results_list: list[dict] = []

    for ISIC_id in missing_metrics_ISIC_ids:
        logger.info(f"Processing ISIC_id {ISIC_id}.")

        # Get the mask pairs for the ISIC_id.
        image_mask_pairs = get_image_mask_pairs(
            seg_metadata_df[seg_metadata_df["ISIC_id"] == ISIC_id]
        )

        allzeros_mask_ids = missing_metrics_ISIC_ids[ISIC_id]
        logger.info(
            f"Found {len(allzeros_mask_ids)} empty masks for ISIC_id {ISIC_id}."
        )

        for img_id, mask_ids in tqdm(
            image_mask_pairs.items(),
            total=len(image_mask_pairs),
            desc=f"Processing ISIC_id {ISIC_id}",
        ):
            image_results_list: list[dict] = []

            masks = {}
            for mask_id in mask_ids:
                mask_path = Path(config.new_dataset_masks_dir) / mask_id
                if not mask_path.exists():
                    logger.error(f"Mask file not found: {mask_path}")
                    continue
                masks[mask_id] = load_mask(mask_path.as_posix(), logger)

            for mask1, mask2 in combinations(mask_ids, 2):
                mask1_data, mask2_data = masks.get(mask1), masks.get(mask2)
                # Check if one of the masks is empty.
                if mask1_data is None or mask2_data is None:
                    logger.warning(
                        f"One or both masks for image {img_id} not found: "
                        f"{mask1} and {mask2}. Skipping."
                    )

                # Get annotator IDs from metadata.
                annotator1_series = seg_metadata_df[
                    seg_metadata_df["seg_filename"] == mask1
                ]["annotator"]
                annotator2_series = seg_metadata_df[
                    seg_metadata_df["seg_filename"] == mask2
                ]["annotator"]
                if annotator1_series.empty or annotator2_series.empty:
                    logger.warning(
                        f"Annotator info missing for for mask pair: {mask1} "
                        f"and {mask2}. Skipping."
                    )
                    continue

                annotator1 = annotator1_series.values[0]
                annotator2 = annotator2_series.values[0]

                # Check if one of the masks is all zeros.
                if mask1 in allzeros_mask_ids or mask2 in allzeros_mask_ids:
                    image_results_list.append(
                        {
                            "img_id": img_id,
                            "mask1": mask1,
                            "mask2": mask2,
                            "annotator1": annotator1,
                            "annotator2": annotator2,
                            "dice_score": 0,
                            "jaccard_score": 0,
                            "hd_score": float("nan"),
                            "hd95_score": float("nan"),
                            "assd_score": float("nan"),
                            "hd_score_normalized": 1,
                            "hd95_score_normalized": 1,
                            "assd_score_normalized": 1,
                        }
                    )
                    continue

                # Compute the metrics.
                overlap_metrics = compute_overlap_metrics(
                    mask1_data, mask2_data
                )
                boundary_metrics = compute_boundary_metrics(
                    mask1_data, mask2_data
                )

                image_results_list.append(
                    {
                        "img_id": img_id,
                        "mask1": mask1,
                        "mask2": mask2,
                        "annotator1": annotator1,
                        "annotator2": annotator2,
                        **overlap_metrics._asdict(),
                        **boundary_metrics._asdict(),
                    }
                )

            overall_results_list.extend(image_results_list)

    return overall_results_list


def main() -> None:
    """
    Main function.
    """
    config = OmegaConf.load("config.yaml")

    # Compute the missing metrics.
    missing_metrics_list = compute_missing_metrics(config)
    logger.info(f"Computed {len(missing_metrics_list)} missing metrics.")

    # Read the existing pairwise IAA metrics file into a DataFrame.
    pairwise_IAA_metrics_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.all_pairwise_IAA_metrics_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(
        f"Read {len(pairwise_IAA_metrics_df)} rows of pairwise IAA metrics."
    )

    # Concatenate the existing pairwise IAA metrics DataFrame with the
    # missing metrics DataFrame.
    pairwise_IAA_metrics_df = pd.concat(
        [pairwise_IAA_metrics_df, pd.DataFrame(missing_metrics_list)],
        ignore_index=True,
    )
    logger.info(
        f"Concatenated {len(pairwise_IAA_metrics_df)} rows of pairwise IAA metrics "
        f"with {len(missing_metrics_list)} missing metrics."
    )

    # Sort the pairwise IAA metrics DataFrame by img_id and mask1.
    pairwise_IAA_metrics_df = pairwise_IAA_metrics_df.sort_values(
        by=["img_id", "mask1"]
    )
    logger.info(
        f"Sorted {len(pairwise_IAA_metrics_df)} rows of pairwise IAA metrics "
        f"by img_id and mask1."
    )

    # Overwrite the pairwise IAA metrics DataFrame to a file.
    pairwise_IAA_metrics_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.all_pairwise_IAA_metrics_path),
        index=False,
        sep=",",
    )

    logger.info(
        f"Saved {len(pairwise_IAA_metrics_df)} rows of pairwise IAA metrics to "
        f"{config.all_pairwise_IAA_metrics_path}."
    )

    return None


if __name__ == "__main__":
    main()
