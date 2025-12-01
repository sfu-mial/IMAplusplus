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
    f"extend_IAA_metrics_columns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def mask_name_to_tool(mask_name: str) -> str:
    """
    Extract the tool from the mask name.
    """
    return mask_name.split("_")[3]


def mask_name_to_skill_level(mask_name: str) -> str:
    """
    Extract the skill level from the mask name.
    """
    return mask_name.split("_")[4]


def add_metrics_columns_for_tool_skill(config: OmegaConf) -> None:
    """
    Add columns to the metrics DataFrame for {tool, skill_level} pairs.
    """
    # Read the pairwise IAA metrics file.
    metrics_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.all_pairwise_IAA_metrics_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    logger.info(f"Read {len(metrics_df)} rows of pairwise IAA metrics.")

    # If a mask (either the `mask1` or `mask2` column) has the name:
    # ISIC_0000004_A04_T1_S2_545d08aabae47821f8802550.png,
    # this follows the format:
    # <ISIC_id>_<annotator>_<tool>_<skill_level>_<mskObjectID>.png,
    # then we can extract the <ISIC_id>, <annotator>, <tool>, and <skill_level>
    # from the mask name.
    # We already have `annotator1` and `annotator2` columns, so we need to add
    # {tool, skill_level} columns to the DataFrame.
    # `mask1` will lead to {tool1, skill_level1}, and
    # `mask2` will lead to {tool2, skill_level2}.

    # First, drop these columns if they exist.
    cols_to_drop = ["tool1", "skill_level1", "tool2", "skill_level2"]
    if all(col in metrics_df.columns for col in cols_to_drop):
        metrics_df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped {cols_to_drop} columns from metrics DataFrame.")

    metrics_df["tool1"] = metrics_df["mask1"].apply(mask_name_to_tool)
    metrics_df["skill_level1"] = metrics_df["mask1"].apply(
        mask_name_to_skill_level
    )
    metrics_df["tool2"] = metrics_df["mask2"].apply(mask_name_to_tool)
    metrics_df["skill_level2"] = metrics_df["mask2"].apply(
        mask_name_to_skill_level
    )

    # Now, write the extended metrics DataFrame to the same file.
    metrics_df.to_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.all_pairwise_IAA_metrics_path),
        index=False,
        sep=",",
    )
    logger.info(
        f"Wrote {len(metrics_df)} rows of extended metrics to "
        f"{config.all_pairwise_IAA_metrics_path}."
    )
    return None


def main() -> None:
    """
    Main function.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Add metrics columns for tool skill.
    add_metrics_columns_for_tool_skill(config)

    return None


if __name__ == "__main__":
    main()
