import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append("..")
from utils.data import get_image_mask_pairs, load_mask
from utils.metrics import (
    compute_boundary_metrics,
    compute_overlap_metrics,
    summarize_metrics,
)

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"compute_IAA_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)

# -------------------------
# Parallel worker utilities
# -------------------------


def _process_single_image(
    img_id: str,
    mask_ids: list[str],
    segs_dir_str: str,
    annotator_map: dict[str, str],
) -> list[dict]:
    """
    Compute all pairwise metrics for one image.

    Returns a list of result dicts for this image.
    """
    segs_dir = Path(segs_dir_str)

    # Load all masks for the image
    masks: dict[str, object] = {}
    for mask_id in mask_ids:
        mask_path = segs_dir / mask_id
        if not mask_path.exists():
            # Skip missing masks silently in worker
            continue
        masks[mask_id] = load_mask(mask_path.as_posix(), logger)

    image_results_list: list[dict] = []

    for mask1, mask2 in combinations(mask_ids, 2):
        mask1_data = masks.get(mask1)
        mask2_data = masks.get(mask2)
        if mask1_data is None or mask2_data is None:
            continue

        annotator1 = annotator_map.get(mask1)
        annotator2 = annotator_map.get(mask2)
        if annotator1 is None or annotator2 is None:
            continue

        overlap_metrics = compute_overlap_metrics(mask1_data, mask2_data)
        boundary_metrics = compute_boundary_metrics(mask1_data, mask2_data)

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

    return image_results_list


# -------------------------
# Main API
# -------------------------


def compute_IAA_metrics(
    config: OmegaConf,
    calc_mode: Literal["memory", "file"] = "memory",
    tmp_metrics_dir: os.PathLike | None = None,
    output_subdir_tag: str | None = None,
    num_workers: int | None = None,
) -> pd.DataFrame:
    """
    Compute the inter-annotator agreement (IAA) metrics for all image-mask
    pairs in the (multiannotator) segmentation masks metadata DataFrame.

    Computes the pairwise IAA metrics between all masks for each image.
    Depends on the `calc_mode` parameter, results are either accumulated in
    memory or written to disk as per-image JSON files in the `tmp_metrics_dir`.

    The following IAA metrics are computed:

    - Dice coefficient
    - Jaccard coefficient
    - Hausdorff distance
    - 95th percentile Hausdorff distance
    - Average symmetric surface distance

    The last three metrics are normalized by the image diagonal length.

    Parameters
    ----------
    config : OmegaConf
        Configuration object.
    calc_mode : Literal["memory", "file"], optional
        Calculation mode. Either "memory" or "file". Default is "memory".
    tmp_metrics_dir : os.PathLike, optional
        Path to the directory to write the per-image JSON files to.
    output_subdir_tag : str, optional
        Subdirectory tag to append to the `tmp_metrics_dir`.
    num_workers : int, optional
        If > 1, enable process-based parallelism across images using that many
        worker processes. Defaults to os.cpu_count() when None.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the overlap-based metrics for all image-mask
        pairs if `calc_mode` is "memory", otherwise an empty DataFrame.
    """
    # Specify the segmentation masks directory.
    segs_dir = Path(config.new_dataset_masks_dir)
    if not segs_dir.exists():
        logger.error(f"Segmentation masks directory not found: {segs_dir}")
        return pd.DataFrame()

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
        f"Read {len(seg_metadata_df)} rows of multiannotator subset segmentation masks metadata."
    )

    # Build annotator map to avoid sending entire DataFrame to workers
    if (
        "seg_filename" not in seg_metadata_df.columns
        or "annotator" not in seg_metadata_df.columns
    ):
        logger.error(
            "Expected columns 'seg_filename' and 'annotator' not found in metadata."
        )
        return pd.DataFrame()
    annotator_map: dict[str, str] = dict(
        zip(
            seg_metadata_df["seg_filename"].astype(str),
            seg_metadata_df["annotator"].astype(str),
        )
    )

    output_path = None
    if calc_mode == "file":
        if tmp_metrics_dir is None:
            logger.error(
                "`tmp_metrics_dir` must be provided if `calc_mode` is 'file'."
            )
            return pd.DataFrame()
        output_path = (
            Path(tmp_metrics_dir) / f"overlap_metrics_{output_subdir_tag}.json"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing overlap metrics to {output_path}.")

    # Group the masks by image.
    image_mask_pairs = get_image_mask_pairs(seg_metadata_df)
    logger.info(
        f"Found metadata for {len(image_mask_pairs)} image-mask pairs."
    )

    overall_results_list: list[dict] = []

    # Decide on execution mode
    use_parallel = (num_workers or 0) != 1
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    items: Iterable[tuple[str, list[str]]] = image_mask_pairs.items()

    if use_parallel and num_workers > 1:
        logger.info(
            f"Processing images in parallel with {num_workers} workers..."
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_image,
                    img_id,
                    mask_ids,
                    segs_dir.as_posix(),
                    annotator_map,
                ): img_id
                for img_id, mask_ids in items
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing images",
            ):
                img_id = futures[fut]
                try:
                    image_results_list = fut.result()
                except Exception as e:
                    logger.error(f"Error processing image {img_id}: {e}")
                    image_results_list = []

                if calc_mode == "file" and image_results_list:
                    assert output_path is not None
                    file_path = output_path / f"{img_id}.json"
                    try:
                        with open(file_path, "w") as f:
                            json.dump(image_results_list, f, indent=4)
                    except Exception as e:
                        logger.error(
                            f"Error writing metrics for image {img_id} to {file_path}: {e}"
                        )
                elif calc_mode == "memory":
                    overall_results_list.extend(image_results_list)
                else:
                    logger.error(f"Invalid calculation mode: {calc_mode}")
                    raise ValueError(f"Invalid calculation mode: {calc_mode}")
    else:
        # Fallback to sequential processing
        for img_id, mask_ids in tqdm(
            items, total=len(image_mask_pairs), desc="Processing images"
        ):
            logger.info(
                f"Processing image {img_id} with {len(mask_ids)} masks."
            )
            image_results_list = _process_single_image(
                img_id, mask_ids, segs_dir.as_posix(), annotator_map
            )

            if calc_mode == "file" and image_results_list:
                assert output_path is not None
                file_path = output_path / f"{img_id}.json"
                try:
                    with open(file_path, "w") as f:
                        json.dump(image_results_list, f, indent=4)
                    logger.debug(
                        f"Wrote metrics for image {img_id} to {file_path}."
                    )
                except Exception as e:
                    logger.error(
                        f"Error writing metrics for image {img_id} to {file_path}: {e}"
                    )
            elif calc_mode == "memory":
                overall_results_list.extend(image_results_list)
            else:
                logger.error(f"Invalid calculation mode: {calc_mode}")
                raise ValueError(f"Invalid calculation mode: {calc_mode}")

    if calc_mode == "memory":
        return pd.DataFrame(overall_results_list)
    elif calc_mode == "file":
        return pd.DataFrame()
    else:
        logger.error(f"Invalid calculation mode: {calc_mode}")
        return pd.DataFrame()


def main() -> None:
    """
    Main function to compute the inter-annotator agreement (IAA) metrics for the dataset.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Compute the IAA metrics.
    metrics_df = compute_IAA_metrics(config, calc_mode="memory", num_workers=8)
    logger.info(f"Computed {len(metrics_df)} rows of IAA metrics.")

    # Summarize the metrics.
    summary_df = summarize_metrics(metrics_df)

    # Save the metrics.
    metrics_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / ("IMAplusplus_multiannotator_subset_IAA_metrics.csv"),
        index=False,
        sep=",",
    )
    logger.info(
        f"Saved {len(metrics_df)} rows of IAA metrics to "
        f"{config.new_dataset_metadata_output_dir}/"
        f"IMAplusplus_multiannotator_subset_IAA_metrics.csv."
    )

    # Save the summary.
    summary_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / ("IMAplusplus_multiannotator_subset_IAA_metrics_summary.csv"),
        index=False,
        sep=",",
    )
    logger.info(
        f"Saved {len(summary_df)} rows of IAA metrics summary to "
        f"{config.new_dataset_metadata_output_dir}/"
        f"IMAplusplus_multiannotator_subset_IAA_metrics_summary.csv."
    )

    return None


if __name__ == "__main__":
    main()
