from typing import NamedTuple

import numpy as np
import pandas as pd
from medpy.metric.binary import assd, dc, hd, hd95, jc


class OverlapMetrics(NamedTuple):
    dice_score: float
    jaccard_score: float


class BoundaryMetrics(NamedTuple):
    hd_score: float
    hd95_score: float
    assd_score: float
    hd_score_normalized: float
    hd95_score_normalized: float
    assd_score_normalized: float


# class IAAMetrics(NamedTuple):
#     overlap_metrics: OverlapMetrics
#     boundary_metrics: BoundaryMetrics


def compute_overlap_metrics(
    mask1_data: np.ndarray,
    mask2_data: np.ndarray,
) -> OverlapMetrics:
    """
    Compute the overlap-based metrics (Dice, Jaccard) for a pair of masks.

    Parameters
    ----------
    mask1_data : np.ndarray
        First mask.
    mask2_data : np.ndarray
        Second mask.

    Returns
    -------
    OverlapMetrics
        An instance of the OverlapMetrics NamedTuple containing the Dice and
        Jaccard scores.
    """

    # Check that the masks have the same spatial dimensions.
    if mask1_data.shape != mask2_data.shape:
        raise ValueError(
            f"Masks must have the same shape. "
            f"Got {mask1_data.shape} vs {mask2_data.shape}."
        )

    # Compute the overlap-based metrics.
    dice_score = dc(mask1_data, mask2_data)
    jaccard_score = jc(mask1_data, mask2_data)

    return OverlapMetrics(dice_score=dice_score, jaccard_score=jaccard_score)


def compute_boundary_metrics(
    mask1_data: np.ndarray,
    mask2_data: np.ndarray,
) -> BoundaryMetrics:
    """
    Compute the boundary-based metrics (Hausdorff distance, 95th percentile Hausdorff distance, Average symmetric surface distance) for a pair of masks.

    Parameters
    ----------
    mask1_data : np.ndarray
        First mask.
    mask2_data : np.ndarray
        Second mask.

    Returns
    -------
    BoundaryMetrics
        An instance of the BoundaryMetrics NamedTuple containing the Hausdorff
        distance, 95th percentile Hausdorff distance, and Average symmetric
        surface distance.
    """

    # Check that the masks have the same spatial dimensions.
    if mask1_data.shape != mask2_data.shape:
        raise ValueError(
            f"Masks must have the same shape. "
            f"Got {mask1_data.shape} vs {mask2_data.shape}."
        )

    # Calculate the diagonal length of the masks to compute the normalized metrics.
    mask_diagonal_length = np.hypot(mask1_data.shape[0], mask1_data.shape[1])

    # Compute the boundary-based metrics.
    hd_score = hd(mask1_data, mask2_data)
    hd95_score = hd95(mask1_data, mask2_data)
    assd_score = assd(mask1_data, mask2_data)

    return BoundaryMetrics(
        hd_score=hd_score,
        hd95_score=hd95_score,
        assd_score=assd_score,
        hd_score_normalized=hd_score / mask_diagonal_length,
        hd95_score_normalized=hd95_score / mask_diagonal_length,
        assd_score_normalized=assd_score / mask_diagonal_length,
    )


def summarize_metrics(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the metrics across the entire dataset.
    Reports the following for each metric:
    - Mean
    - Standard deviation
    - Minimum
    - Maximum
    - Median
    - IQR

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing the metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the summarized metrics.
    """
    summary_df = pd.DataFrame()

    # Overlap metrics.
    overlap_cols = ["dice_score", "jaccard_score"]
    for metric in overlap_cols:
        if metric in metrics_df.columns and pd.api.types.is_numeric_dtype(
            metrics_df[metric]
        ):
            summary_df.loc["Mean", metric] = metrics_df[metric].mean()
            summary_df.loc["Standard deviation", metric] = metrics_df[
                metric
            ].std()
            summary_df.loc["Minimum", metric] = metrics_df[metric].min()
            summary_df.loc["Maximum", metric] = metrics_df[metric].max()
            summary_df.loc["Median", metric] = metrics_df[metric].median()
            summary_df.loc["IQR", metric] = metrics_df[metric].quantile(
                0.75
            ) - metrics_df[metric].quantile(0.25)

    # Boundary metrics.
    boundary_cols = [
        "hd_score",
        "hd95_score",
        "assd_score",
        "hd_score_normalized",
        "hd95_score_normalized",
        "assd_score_normalized",
    ]
    for metric in boundary_cols:
        if metric in metrics_df.columns and pd.api.types.is_numeric_dtype(
            metrics_df[metric]
        ):
            summary_df.loc["Mean", metric] = metrics_df[metric].mean()
            summary_df.loc["Standard deviation", metric] = metrics_df[
                metric
            ].std()
            summary_df.loc["Minimum", metric] = metrics_df[metric].min()
            summary_df.loc["Maximum", metric] = metrics_df[metric].max()
            summary_df.loc["Median", metric] = metrics_df[metric].median()
            summary_df.loc["IQR", metric] = metrics_df[metric].quantile(
                0.75
            ) - metrics_df[metric].quantile(0.25)

    return summary_df
