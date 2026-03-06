import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
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
    f"compute_image_metadata_completeness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level="DEBUG",
)


def count_rows_with_values(
    df: pd.DataFrame, cols: List[str], *, empty_as_nan: bool = True
) -> int:
    """
    Count rows in a DataFrame that have at least one non-missing value in the given columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to operate on.
    cols : List[str]
        Column names to consider.
    empty_as_nan : bool, default True
        If True, treat empty/blank strings as missing values.

    Returns
    -------
    int
        Number of rows with at least one non-missing value among the specified columns.
    """
    sub = df[cols].copy()
    if empty_as_nan:
        # Treat empty strings (and whitespace-only strings) as NaN.
        sub = sub.replace(r"^\s*$", np.nan, regex=True)

    mask = sub.notna().any(axis=1)
    return int(mask.sum())


def main() -> None:
    """
    Main function to compute the image metadata completeness for the dataset.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Load the image metadata file into a DataFrame.
    img_metadata_df = pd.read_csv(
        Path(config.new_dataset_metadata_output_dir)
        / (config.output_img_metadata_path),
        header="infer",
        sep=",",
        low_memory=False,
    )
    total_rows = len(img_metadata_df)
    logger.info(f"Read {total_rows} rows of image metadata.")

    # Count the number of rows with any level of recorded diagnosis.
    diagnosis_cols = [
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
        "diagnosis_5",
    ]
    num_rows_with_diagnosis = count_rows_with_values(
        img_metadata_df, diagnosis_cols
    )
    logger.info(
        f"Found {num_rows_with_diagnosis} rows ({num_rows_with_diagnosis / total_rows * 100:.2f}%) with any level of recorded diagnosis."
    )

    # Count the number of rows with recorded age.
    age_cols = ["age_approx"]
    num_rows_with_age = count_rows_with_values(img_metadata_df, age_cols)
    logger.info(
        f"Found {num_rows_with_age} rows"
        f"({num_rows_with_age / total_rows * 100:.2f}%) with recorded age."
    )

    # Count the number of rows with recorded sex.
    sex_cols = ["sex"]
    num_rows_with_sex = count_rows_with_values(img_metadata_df, sex_cols)
    logger.info(
        f"Found {num_rows_with_sex} rows"
        f"({num_rows_with_sex / total_rows * 100:.2f}%) with recorded sex."
    )

    # Count the number of rows with recorded anatomic_site.
    anatomic_site_cols = ["anatom_site_general"]
    num_rows_with_anatomic_site = count_rows_with_values(
        img_metadata_df, anatomic_site_cols
    )
    logger.info(
        f"Found {num_rows_with_anatomic_site} rows"
        f"({num_rows_with_anatomic_site / total_rows * 100:.2f}%) with recorded anatomic site."
    )

    # Count the number of rows with recorded malignancy status.
    malignancy_status_cols = ["benign_malignant"]
    num_rows_with_malignancy_status = count_rows_with_values(
        img_metadata_df, malignancy_status_cols
    )
    logger.info(
        f"Found {num_rows_with_malignancy_status} rows"
        f"({num_rows_with_malignancy_status / total_rows * 100:.2f}%) with recorded malignancy status."
    )

    # Count the number of rows with recorded diagnosis_confirm_type.
    diagnosis_confirm_type_cols = ["diagnosis_confirm_type"]
    num_rows_with_diagnosis_confirm_type = count_rows_with_values(
        img_metadata_df, diagnosis_confirm_type_cols
    )
    logger.info(
        f"Found {num_rows_with_diagnosis_confirm_type} rows"
        f"({num_rows_with_diagnosis_confirm_type / total_rows * 100:.2f}%) with recorded diagnosis confirm type."
    )

    return None


if __name__ == "__main__":
    main()
