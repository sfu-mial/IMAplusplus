import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

sys.path.append("..")

# Configure loguru to output ERROR level messages and above to the console.
logger.remove()
# Log ERROR level messages and above to the console.
logger.add(sys.stderr, level="ERROR")
# Log INFO level messages and above to the console.
logger.add(sys.stdout, level="INFO")
# Log DEBUG level messages and above to a timestamped log file.
logger.add(
    f"compute_image_level_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def compute_image_level_metrics(config: OmegaConf) -> None:
    """
    Compute the image-level metrics for the multiannotator subset of the
    dataset.

    Given the pairwise IAA metrics, we report the mean and standard deviation
    of the IAA metrics for each image, **excluding** metrics computed between
    consensus labels.

    Parameters
    ----------
    config : OmegaConf
        Configuration object.

    Returns
    -------
    None
    """

    # Read the pairwise IAA metrics file into a DataFrame.
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

    # List the metrics columns to be summarized.
    metrics_cols = [
        "dice_score",
        "jaccard_score",
        "hd_score",
        "hd95_score",
        "assd_score",
        "hd_score_normalized",
    ]

    # Filter the metrics to only include rows where both annotators are not
    # consensus labels.
    consensus_labels = ["MV", "ST"]
    pairwise_IAA_metrics_df = pairwise_IAA_metrics_df[
        ~(
            pairwise_IAA_metrics_df["annotator1"].isin(consensus_labels)
            | pairwise_IAA_metrics_df["annotator2"].isin(consensus_labels)
        )
    ]
    logger.info(
        f"Filtered {len(pairwise_IAA_metrics_df)} rows of pairwise IAA metrics "
        f"to only include rows where both annotators are not consensus labels."
    )

    # Group by img_id and calculate the mean and standard deviation of the metrics columns.
    image_level_agg_metrics = pairwise_IAA_metrics_df.groupby("img_id")[
        metrics_cols
    ].agg(["mean", "std"])

    # Flatten the multi-index/multi-level columns so that the column names are
    # as descriptive. Handle any number of levels in the MultiIndex.
    if isinstance(image_level_agg_metrics.columns, pd.MultiIndex):
        image_level_agg_metrics.columns = [
            "_".join(str(level) for level in col_tuple)
            for col_tuple in image_level_agg_metrics.columns.to_flat_index()
        ]
    else:
        # If columns are not MultiIndex, use them as-is.
        image_level_agg_metrics.columns = [
            f"{col}_mean" if i % 2 == 0 else f"{col}_std"
            for i, col in enumerate(image_level_agg_metrics.columns)
        ]

    # Reset the index so that `img_id` is a column.
    image_level_agg_metrics = image_level_agg_metrics.reset_index()

    # Save the image-level metrics to a file.
    image_level_agg_metrics.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.image_level_metrics_path),
        index=False,
        sep=",",
    )

    logger.info(
        f"Saved {len(image_level_agg_metrics)} rows of image-level metrics to "
        f"{config.image_level_metrics_path}."
    )

    return None


def main() -> None:
    """
    Main function.
    """
    config = OmegaConf.load("config.yaml")
    try:
        compute_image_level_metrics(config)
    except Exception as e:
        logger.error(f"Error computing image-level metrics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
