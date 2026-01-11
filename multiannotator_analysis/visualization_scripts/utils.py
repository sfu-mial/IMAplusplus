import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from constants import METRIC_NAME_MAP
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from omegaconf import OmegaConf


def read_img_metadata(config: OmegaConf) -> pd.DataFrame:
    """
    Read the image metadata file into a DataFrame.
    """
    return pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_img_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )


def read_seg_masks_metadata(config: OmegaConf) -> pd.DataFrame:
    """
    Read the segmentation masks metadata file into a DataFrame.
    """
    return pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_seg_masks_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )


def read_pairwise_IAA_metrics(config: OmegaConf) -> pd.DataFrame:
    """
    Read the pairwise IAA metrics file into a DataFrame.
    """
    return pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.all_pairwise_IAA_metrics_path),
        header="infer",
        sep=",",
        low_memory=False,
    )


def read_image_level_metrics(config: OmegaConf) -> pd.DataFrame:
    """
    Read the image-level metrics file into a DataFrame.
    """
    return pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.image_level_metrics_path),
        header="infer",
        sep=",",
        low_memory=False,
    )


def read_multiannotator_subset_seg_masks_metadata(
    config: OmegaConf,
) -> pd.DataFrame:
    """
    Read the multiannotator subset segmentation masks metadata file into a DataFrame.
    """
    return pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.multiannotator_subset_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )


def filter_annotators_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility function to drop all rows that have metrics corresponding to
    consensus labels: either majority voting (MV) or STAPLE (ST).
    """
    consensus_labels = ["MV", "ST"]
    return df[
        ~(
            df["annotator1"].isin(consensus_labels)
            | df["annotator2"].isin(consensus_labels)
        )
    ]


def filter_annotators_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility function to drop all rows that have metadata corresponding to
    consensus labels: either majority voting (MV) or STAPLE (ST).
    """
    consensus_labels = ["MV", "ST"]
    return df[~df["annotator"].isin(consensus_labels)]


def report_stats(scores1: list[float], scores2: list[float]) -> None:
    """
    Report the following statistics of two sets of scores:
    - Mean
    - Standard deviation
    - Cohen's d
    - Kolmogorov-Smirnov 2-sample test p-value
    - Mann-Whitney U test p-value
    """
    num_s1, num_s2 = len(scores1), len(scores2)
    np_s1, np_s2 = np.array(scores1), np.array(scores2)

    mean_s1, mean_s2 = np.mean(np_s1), np.mean(np_s2)
    var_s1, var_s2 = np.var(np_s1, ddof=1), np.var(np_s2, ddof=1)

    cohen_denominator = np.sqrt(
        ((num_s1 - 1) * (var_s1**2) + (num_s2 - 1) * (var_s2**2))
        / (num_s1 + num_s2 - 2)
    )

    cohen_d = (abs(mean_s1 - mean_s2)) / cohen_denominator

    # Kolmogorov-Smirnov 2-sample test.
    pval_KS = stats.ks_2samp(scores1, scores2).pvalue
    # Mann-Whitney U test.
    pval_MW = stats.mannwhitneyu(scores1, scores2).pvalue

    print(f"Score 1: {mean_s1:.4f} ± {np.sqrt(var_s1):.4f}")
    print(f"Score 2: {mean_s2:.4f} ± {np.sqrt(var_s2):.4f}")
    print(f"p-val KS: {pval_KS:.2e} \t p-val MW: {pval_MW:.2e}")
    print(f"Cohen's d: {cohen_d:4f}")


def create_custom_colormap(num_colors: int = 50) -> LinearSegmentedColormap:
    """
    Create a custom seaborn colormap palette using two colors.
    # https://shorturl.at/HWJkP
    """
    # Relative positions of colors in the colormap
    pos = [0.0, 1.0]
    colors = ["#ed0041", "#190007"]

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("my_cmap", list(zip(pos, colors)))

    # Use the colormap directly in seaborn
    palette = sns.color_palette(cmap(np.linspace(0, 1, num_colors)))

    # # Plot the palette
    # sns.palplot(palette)
    # plt.show()

    return palette


def generate_variability_heatmap(
    df: pd.DataFrame,
    factor_col: str,
    metric_col: str = "dice_score",
    custom_order: list[str] | None = None,
    annot_metric: bool = True,
    annot_pval: bool = True,
    show_metric_cbar: bool = True,
    show_pval_cbar: bool = True,
    metric_cmap: str = "viridis",
    pval_cmap: str = "viridis",
    annot_metric_fontsize: int = 9,
    annot_pval_fontsize: int = 9,
    save_fig: bool = True,
    output_dir: os.PathLike = ".",
    filename: str = "variability_heatmap",
    figure_size: str = "small",
) -> None:
    """
    Generates a combined heatmap for a given factor with extensive customization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics. The p-values will be calculated
        based on the metrics.
    factor_col : str
        The column prefix for the factor (e.g., "tool").
    metric_col : str, optional
        Column name of the metric to plot. Default is "dice_score".
    custom_order : list[str], optional
        Defines a specific order for the factor levels.
    annot_metric : bool, optional
        Whether to annotate the metric values on the heatmap.
    annot_pval : bool, optional
        Whether to annotate the p-values on the heatmap.
    show_metric_cbar : bool, optional
        Whether to show the metric colorbar.
    show_pval_cbar : bool, optional
        Whether to show the p-value colorbar.
    metric_cmap : str, optional
        The colormap for the metric values.
    pval_cmap : str, optional
        The colormap for the p-values.
    annot_metric_fontsize : int, optional
        The fontsize for the metric values' annotations.
    annot_pval_fontsize : int, optional
        The fontsize for the p-value annotations.
    save_fig : bool, optional
        Whether to save the figure.
    output_dir : os.PathLike, optional
        The directory to save the figure.
    filename : str, optional
        The filename for the saved heatmap.
    """
    col1, col2 = f"{factor_col}1", f"{factor_col}2"

    if custom_order:
        levels = custom_order
    else:
        levels = sorted(pd.concat([df[col1], df[col2]]).unique())

    n = len(levels)

    metric_matrix = pd.DataFrame(
        np.full((n, n), np.nan), index=levels, columns=levels
    )
    pval_matrix = pd.DataFrame(
        np.full((n, n), np.nan), index=levels, columns=levels
    )

    for i, level1 in enumerate(levels):
        for j, level2 in enumerate(levels):
            if not (
                level1 in metric_matrix.index
                and level2 in metric_matrix.columns
            ):
                continue

            if j <= i:
                if (
                    level1 in ["MV", "ST"]
                    and level2 in ["MV", "ST"]
                    and level1 == level2
                ):
                    metric_matrix.loc[level1, level2] = np.nan
                else:
                    pairwise_mask = (
                        (df[col1] == level1) & (df[col2] == level2)
                    ) | ((df[col1] == level2) & (df[col2] == level1))
                    if pairwise_mask.any():
                        metric_matrix.loc[level1, level2] = df.loc[
                            pairwise_mask, metric_col
                        ].mean()

            if j > i:
                if (
                    level1 in ["MV", "ST"]
                    and level2 in ["MV", "ST"]
                    and level1 == level2
                ):
                    pval_matrix.loc[level1, level2] = np.nan
                # elif level1 in ["MV", "ST"] or level2 in ["MV", "ST"]:
                #     # Skip p-value calculation if one level is a consensus method
                #     # (consensus methods don't have skill levels in the dataframe)
                #     pval_matrix.loc[level1, level2] = np.nan
                else:
                    dist1 = df.loc[
                        (df[col1] == level1) | (df[col2] == level1), metric_col
                    ]
                    dist2 = df.loc[
                        (df[col1] == level2) | (df[col2] == level2), metric_col
                    ]

                    # Filter out NaN values and ensure we have valid data
                    dist1_clean = dist1.dropna()
                    dist2_clean = dist2.dropna()

                    if len(dist1_clean) > 1 and len(dist2_clean) > 1:
                        try:
                            p_value = stats.mannwhitneyu(
                                dist1_clean, dist2_clean
                            ).pvalue
                            pval_matrix.loc[level1, level2] = p_value
                        except (ValueError, AttributeError):
                            # Handle cases where the test cannot be performed.
                            pval_matrix.loc[level1, level2] = np.nan

    # Visualize the heatmaps.
    if figure_size == "small":
        fig, ax = plt.subplots(figsize=(max(n * 0.9, 12), max(n * 0.9, 10)))
    elif figure_size == "large":
        fig, ax = plt.subplots(figsize=(max(n * 0.9, 12), max(n * 0.9, 16)))

    sns.set(style="ticks")  # Set to 'ticks' style for a cleaner look.

    # Mask for metric: hide upper triangle (show lower triangle).
    mask_metric = np.triu(np.ones_like(metric_matrix, dtype=bool), k=1)
    # Mask for p-values: hide lower triangle (show upper triangle).
    mask_pval = np.tril(np.ones_like(pval_matrix, dtype=bool), k=0)

    # Create custom annotation arrays that show "-" for NaN values.
    # Convert DataFrames to object dtype (string) DataFrames for annotations.
    if annot_metric:
        annot_metric_array = metric_matrix.copy().astype(object)
        # Replace NaN with "-" and format non-NaN values
        mask_nan = pd.isna(metric_matrix)
        annot_metric_array[mask_nan] = "-"
        annot_metric_array[~mask_nan] = [
            f"{val:.3f}" for val in metric_matrix.values[~mask_nan.values]
        ]
    else:
        annot_metric_array = False

    if annot_pval:
        annot_pval_array = pval_matrix.copy().astype(object)
        # Replace NaN with "-" and format non-NaN values
        mask_nan = pd.isna(pval_matrix)
        annot_pval_array[mask_nan] = "-"
        annot_pval_array[~mask_nan] = [
            f"{val:.2e}" for val in pval_matrix.values[~mask_nan.values]
        ]
    else:
        annot_pval_array = False

    # Calculate vmin/vmax for heatmaps (excluding NaN)
    metric_vmin = (
        metric_matrix.min().min() if metric_matrix.notna().any().any() else 0
    )
    metric_vmax = (
        metric_matrix.max().max() if metric_matrix.notna().any().any() else 1
    )

    # For NaN cells to be rendered, replace with sentinel value outside vmin/vmax range
    metric_matrix_for_plot = metric_matrix.copy()
    metric_matrix_for_plot = metric_matrix_for_plot.fillna(
        metric_vmin - (metric_vmax - metric_vmin) * 0.1
    )

    pval_matrix_for_plot = pval_matrix.copy()
    if pval_matrix.notna().any().any():
        pval_positive = pval_matrix[(pval_matrix > 0) & (pval_matrix.notna())]
        if len(pval_positive) > 0:
            pval_vmin = pval_positive.min().min()
            pval_matrix_for_plot = pval_matrix_for_plot.fillna(
                pval_vmin * 0.1 if pval_vmin > 0 else 1e-10
            )

    sns.heatmap(
        metric_matrix_for_plot,
        mask=mask_metric,
        cmap=metric_cmap,
        vmin=metric_vmin,
        vmax=metric_vmax,
        annot=annot_metric_array if annot_metric else False,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": annot_metric_fontsize},
    )
    sns.heatmap(
        pval_matrix_for_plot,
        mask=mask_pval,
        cmap=pval_cmap,
        annot=annot_pval_array if annot_pval else False,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": annot_pval_fontsize},
    )

    # Manually add "-" text for NaN cells that are not masked
    # Seaborn heatmap uses 0.5 offset for centering text in cells
    if annot_metric:
        for i, idx in enumerate(metric_matrix.index):
            for j, col in enumerate(metric_matrix.columns):
                if not mask_metric[i, j] and pd.isna(
                    metric_matrix.loc[idx, col]
                ):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "-",
                        ha="center",
                        va="center",
                        fontsize=annot_metric_fontsize,
                        color="black",
                        weight="normal",
                    )

    if annot_pval:
        for i, idx in enumerate(pval_matrix.index):
            for j, col in enumerate(pval_matrix.columns):
                if not mask_pval[i, j] and pd.isna(pval_matrix.loc[idx, col]):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "-",
                        ha="center",
                        va="center",
                        fontsize=annot_pval_fontsize,
                        color="black",
                        weight="normal",
                    )

    # Create custom colorbars for the metric and p-value heatmaps.
    if show_metric_cbar:
        norm_metric = (
            Normalize(
                vmin=metric_matrix.min().min(), vmax=metric_matrix.max().max()
            )
            if metric_col in ["dice_score", "jaccard_score"]
            else Normalize(
                vmin=metric_matrix.min().min(), vmax=metric_matrix.max().max()
            )
        )
        sm_metric = cm.ScalarMappable(norm=norm_metric, cmap=metric_cmap)
        cbar_metric = fig.colorbar(sm_metric, ax=ax, shrink=0.8, pad=0.02)
        cbar_metric.set_label(
            f"Mean {METRIC_NAME_MAP[metric_col]}",
            fontsize=annot_metric_fontsize,
            weight="bold",
        )
        cbar_metric.ax.tick_params(labelsize=annot_metric_fontsize)

    if show_pval_cbar:
        # Filter out NaN, 0, and negative values for p-value calculation.
        pval_positive = pval_matrix[(pval_matrix > 0) & (pval_matrix.notna())]
        if len(pval_positive) > 0:
            min_pval = pval_positive.min().min()
            max_pval = pval_matrix[pval_matrix.notna()].max().max()
            if pd.notna(min_pval) and pd.notna(max_pval) and min_pval > 0:
                norm_pval = LogNorm(vmin=min_pval, vmax=max_pval)
                sm_pval = cm.ScalarMappable(cmap=pval_cmap, norm=norm_pval)
                cbar_pval = fig.colorbar(sm_pval, ax=ax, shrink=0.8, pad=0.08)
                cbar_pval.set_label(
                    "p-value (MW U test, log scale)",
                    fontsize=annot_pval_fontsize,
                    weight="bold",
                )
                cbar_pval.ax.tick_params(labelsize=annot_pval_fontsize)

    # Final "touches".
    # ax.set_title(
    #     f"Inter- and Intra-{factor_col.title()} Agreement",
    #     fontsize=20,
    #     weight="bold",
    #     pad=20,
    # )
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save_fig:
        if output_dir is None or filename is None:
            raise ValueError(
                "output_dir and filename must be provided if save_fig is True"
            )
        output_path_pdf = Path(output_dir) / f"{filename}.pdf"
        output_path_png = Path(output_dir) / f"{filename}.png"
        plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
        plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure to {output_path_pdf} and {output_path_png}")
    return None


def get_arbitrary_annotator_intersections(
    df: pd.DataFrame, target_annotators: dict
):
    """
    Given the metrics DataFrame, get the rows corresponding to images annotated
    by the exact set of target annotators.

    Parameters
    ----------
    df : pd.DataFrame
        The metrics DataFrame.
    target_annotators : dict
        A dictionary of target annotators. The keys are the annotator names,
        and the values are the corresponding annotator IDs.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    # print(
    #     f"\nFiltering for images annotated by exactly in: {target_annotators}."
    # )

    # First, we `melt`` the `annotator1` and `annotator2` columns into one
    # long list associated with the `img_id` column.
    df_long = pd.melt(
        df,
        id_vars=["img_id"],
        value_vars=["annotator1", "annotator2"],
        value_name="annotator",
    )

    # Next, group the DataFrame by `img_id` and get the unique set of
    # annotators' combinations.
    annotators_per_image = (
        df_long.groupby("img_id")["annotator"].unique().apply(set)
    )

    # Next, find images that have the set of target annotators.
    # For this, we need to check if `target_annotators` is an exact match of
    # the image's full annotator set.

    def has_exact_target_annotators(image_annotator_set: set) -> bool:
        """
        Check if a set of annotators has the exact target annotators.
        """
        # return target_annotators.issubset(image_annotator_set)
        return target_annotators == image_annotator_set

    # Use this check on our groupby object.
    matching_images_mask = annotators_per_image.apply(
        has_exact_target_annotators
    )

    # Get the list of images (`img_id`s) that match the target annotators.
    target_image_ids = annotators_per_image[matching_images_mask].index

    print(
        f"Found {len(target_image_ids)} images annotated by exactly in: "
        f"{target_annotators}."
    )

    # Finally, use these image IDs to filter the original DataFrame.
    filtered_df = df[df["img_id"].isin(target_image_ids)].copy()

    if filtered_df.empty:
        print(f"No images found annotated by exactly in: {target_annotators}.")
        return None

    return filtered_df


def generate_variability_heatmap_lower_triangle_only_pairs(
    df: pd.DataFrame,
    factor_col: str,
    metric_cols: list[str] = ["dice_score", "hd95_score_normalized"],
    custom_order: list[str] | None = None,
    annot_metric: bool = True,
    show_metric_cbar: bool = True,
    metric_cmap: str = "viridis",
    annot_metric_fontsize: int = 9,
    save_fig: bool = True,
    output_dir: os.PathLike = ".",
    filename: str = "variability_heatmap_LT_only_pairs",
    figure_size: str = "small",
) -> None:
    """
    Generates a combined heatmap for a given factor with extensive customization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics. The p-values will be calculated
        based on the metrics.
    factor_col : str
        The column prefix for the factor (e.g., "tool").
    metric_col : str, optional
        Column name of the metric to plot. Default is "dice_score".
    custom_order : list[str], optional
        Defines a specific order for the factor levels.
    annot_metric : bool, optional
        Whether to annotate the metric values on the heatmap.
    show_metric_cbar : bool, optional
        Whether to show the metric colorbar.
    metric_cmap : str, optional
        The colormap for the metric values.
    annot_metric_fontsize : int, optional
        The fontsize for the metric values' annotations.
    save_fig : bool, optional
        Whether to save the figure.
    output_dir : os.PathLike, optional
        The directory to save the figure.
    filename : str, optional
        The filename for the saved heatmap.
    """

    # Check that the two metrics are specified.
    if not isinstance(metric_cols, list):
        metric_cols = [metric_cols]
    assert len(metric_cols) == 2, (
        "`metric_cols` must be a list of two metric columns."
        "Got {len(metric_cols)} metric columns."
    )
    assert all(col in df.columns for col in metric_cols), (
        f"All metric columns must be in the DataFrame. Got {metric_cols}."
    )

    col1, col2 = f"{factor_col}1", f"{factor_col}2"

    if custom_order:
        levels = custom_order
    else:
        levels = sorted(pd.concat([df[col1], df[col2]]).unique())

    n = len(levels)

    N_METRICS = len(metric_cols)

    # Visualize the heatmaps.
    if figure_size == "small":
        fig, axes = plt.subplots(
            1,
            N_METRICS,
            figsize=(max(n * 0.9, 12), max(n * 0.9, 4)),
        )
    elif figure_size == "large":
        fig, axes = plt.subplots(
            1,
            N_METRICS,
            figsize=(max(n * 0.9, 10) * N_METRICS, max(n * 0.9, 6)),
        )

    # Ensure axes is always a list (even for 1 metric)
    if N_METRICS == 1:
        axes = [axes]

    # Compute overall min and max values for the metrics.
    all_values = pd.concat([df[m].dropna() for m in metric_cols])
    vmin, vmax = all_values.min(), all_values.max()

    # Plot each metric in its own subplot.
    for ax, metric in zip(axes, metric_cols):
        # Build lower-triangle metric matrix
        metric_matrix = pd.DataFrame(
            np.full((n, n), np.nan), index=levels, columns=levels
        )
        for i, level1 in enumerate(levels):
            for j, level2 in enumerate(levels):
                if j <= i:
                    if (
                        level1 in ["MV", "ST"]
                        and level2 in ["MV", "ST"]
                        and level1 == level2
                    ):
                        metric_matrix.loc[level1, level2] = np.nan
                    else:
                        pairwise_mask = (
                            (df[col1] == level1) & (df[col2] == level2)
                        ) | ((df[col1] == level2) & (df[col2] == level1))
                        if pairwise_mask.any():
                            metric_matrix.loc[level1, level2] = df.loc[
                                pairwise_mask, metric
                            ].mean()

        # Reindex to enforce custom order
        metric_matrix = metric_matrix.reindex(index=levels, columns=levels)

        # Create colormap with extra color for NaN values.
        cmap = plt.get_cmap(metric_cmap).copy()
        cmap.set_bad("lightgray")

        # Masks
        # mask_upper_triangle = np.triu(
        #     np.ones_like(metric_matrix, dtype=bool), k=1
        # )  # upper triangle mask
        masked_data = np.ma.array(
            metric_matrix.values, mask=np.isnan(metric_matrix.values)
        )  # NaNs masked for lightgray

        # Annotations
        if annot_metric:
            annot_metric_array = metric_matrix.copy().astype(object)
            mask_nan = pd.isna(metric_matrix)
            annot_metric_array[mask_nan] = "-"
            annot_metric_array[~mask_nan] = [
                f"{val:.2f}" for val in metric_matrix.values[~mask_nan.values]
            ]
        else:
            annot_metric_array = False

        # metric_matrix = metric_matrix.reindex(index=levels, columns=levels)
        metric_matrix = metric_matrix.loc[levels, levels]

        # Heatmap
        sns.heatmap(
            masked_data,
            # mask=mask_upper_triangle,
            mask=np.triu(np.ones_like(metric_matrix, dtype=bool), k=1),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annot_metric_array if annot_metric else False,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=False,  # shared colorbar later
            ax=ax,
            annot_kws={"fontsize": annot_metric_fontsize},
        )

        # Make upper-triangle cells fully white
        for i in range(n):
            for j in range(i + 1, n):
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=True, color="white", lw=0)
                )

        # Explicitly set tick labels to ensure custom order is shown
        ax.set_xticks(np.arange(len(levels)) + 0.5)
        ax.set_xticklabels(
            levels, rotation=45, ha="right", fontsize=annot_metric_fontsize
        )
        ax.set_yticks(np.arange(len(levels)) + 0.5)
        ax.set_yticklabels(levels, rotation=0, fontsize=annot_metric_fontsize)

        if figure_size == "large":
            ax.set_title(METRIC_NAME_MAP[metric], fontsize=18)
        else:
            ax.set_title(METRIC_NAME_MAP[metric])
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.setp(ax.get_yticklabels(), rotation=0)

    # Shared colorbar
    if show_metric_cbar:
        sm = plt.cm.ScalarMappable(
            norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )
        fig.colorbar(
            sm,
            ax=axes,
            orientation="vertical",
            fraction=0.03,
            pad=0.04,
            # label="Metric value",
            label="",
        )

    # Save figure
    if save_fig:
        if output_dir is None or filename is None:
            raise ValueError(
                "output_dir and filename must be provided if save_fig is True"
            )
        output_path_pdf = Path(output_dir) / f"{filename}.pdf"
        output_path_png = Path(output_dir) / f"{filename}.png"
        output_path_svg = Path(output_dir) / f"{filename}.svg"
        plt.savefig(output_path_pdf, dpi=600, bbox_inches="tight")
        plt.savefig(output_path_png, dpi=600, bbox_inches="tight")
        plt.savefig(output_path_svg, dpi=600, bbox_inches="tight")
        plt.close()
        print(
            f"Saved figure to {output_path_pdf} and {output_path_png} and {output_path_svg}"
        )
    return None
