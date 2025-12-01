import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

# sys.path.append("../../")

sys.path.append("..")
from utils import (
    generate_variability_heatmap,
    generate_variability_heatmap_lower_triangle_only_pairs,
    read_pairwise_IAA_metrics,
)

VIS_OUTPUT_DIR = Path("../../output/visualizations/metrics/")
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Main function.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Read the pairwise IAA metrics file
    pairwise_IAA_metrics_df = read_pairwise_IAA_metrics(config)

    # Generate the variability heatmaps.
    # generate_variability_heatmap(
    #     df=pairwise_IAA_metrics_df,
    #     factor_col="tool",
    #     metric_col="dice_score",
    #     custom_order=["T1", "T2", "T3", "MV", "ST"],
    #     annot_metric=True,
    #     annot_pval=True,
    #     show_metric_cbar=True,
    #     show_pval_cbar=True,
    #     metric_cmap="magma_r",
    #     pval_cmap="Blues",
    #     annot_metric_fontsize=12,
    #     annot_pval_fontsize=12,
    #     save_fig=True,
    #     output_dir=VIS_OUTPUT_DIR,
    #     filename="inter_intra_tool_agreement",
    #     figure_size="small",
    # )

    # generate_variability_heatmap(
    #     df=pairwise_IAA_metrics_df,
    #     factor_col="skill_level",
    #     metric_col="dice_score",
    #     custom_order=["S1", "S2", "MV", "ST"],
    #     annot_metric=True,
    #     annot_pval=True,
    #     show_metric_cbar=True,
    #     show_pval_cbar=True,
    #     metric_cmap="magma_r",
    #     pval_cmap="Blues",
    #     annot_metric_fontsize=12,
    #     annot_pval_fontsize=12,
    #     save_fig=True,
    #     output_dir=VIS_OUTPUT_DIR,
    #     filename="inter_intra_skill_level_agreement",
    #     figure_size="small",
    # )

    # generate_variability_heatmap(
    #     df=pairwise_IAA_metrics_df,
    #     factor_col="annotator",
    #     metric_col="dice_score",
    #     custom_order=None,
    #     annot_metric=True,
    #     annot_pval=False,
    #     show_metric_cbar=True,
    #     show_pval_cbar=True,
    #     metric_cmap="magma_r",
    #     pval_cmap="Greens",
    #     annot_metric_fontsize=10,
    #     annot_pval_fontsize=10,
    #     save_fig=True,
    #     output_dir=VIS_OUTPUT_DIR,
    #     filename="inter_intra_annotator_agreement",
    #     figure_size="large",
    # )

    generate_variability_heatmap_lower_triangle_only_pairs(
        df=pairwise_IAA_metrics_df,
        factor_col="tool",
        metric_cols=["dice_score", "hd95_score_normalized"],
        custom_order=["T1", "T2", "T3", "MV", "ST"],
        annot_metric=True,
        show_metric_cbar=True,
        metric_cmap="magma_r",
        annot_metric_fontsize=12,
        save_fig=True,
        output_dir=VIS_OUTPUT_DIR,
        filename="inter_intra_tool_agreement_LT_only_pairs",
        figure_size="small",
    )

    generate_variability_heatmap_lower_triangle_only_pairs(
        df=pairwise_IAA_metrics_df,
        factor_col="skill_level",
        metric_cols=["dice_score", "hd95_score_normalized"],
        custom_order=["S1", "S2", "MV", "ST"],
        annot_metric=True,
        show_metric_cbar=True,
        metric_cmap="magma_r",
        annot_metric_fontsize=12,
        save_fig=True,
        output_dir=VIS_OUTPUT_DIR,
        filename="inter_intra_skill_agreement_LT_only_pairs",
        figure_size="small",
    )

    generate_variability_heatmap_lower_triangle_only_pairs(
        df=pairwise_IAA_metrics_df,
        factor_col="annotator",
        metric_cols=["dice_score", "hd95_score_normalized"],
        custom_order=None,
        annot_metric=True,
        show_metric_cbar=False,
        metric_cmap="magma_r",
        annot_metric_fontsize=14,
        save_fig=True,
        output_dir=VIS_OUTPUT_DIR,
        filename="inter_intra_annotator_agreement_LT_only_pairs",
        figure_size="large",
    )

    return None


if __name__ == "__main__":
    main()
