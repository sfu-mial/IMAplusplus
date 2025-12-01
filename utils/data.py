import os

import loguru
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte


def load_image(
    image_path: os.PathLike, logger: loguru.logger
) -> np.ndarray | None:
    """
    Load a single dermoscopic image from file into a NumPy array.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    logger : loguru.logger
        Logger object to log errors.
    Returns
    -------
    numpy.ndarray
        Image array with shape (height, width, 3) for RGB images, or None if loading fails.
    """
    try:
        img = io.imread(image_path)
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def load_mask(
    mask_path: os.PathLike, logger: loguru.logger
) -> np.ndarray | None:
    """
    Load a single binary segmentation mask from file and ensure it is binary.


    Parameters
    ----------
    mask_path : str
        Path to the mask file.
    logger : loguru.logger
        Logger object to log errors.

    Returns
    -------
    numpy.ndarray
        Binary mask array with values 0 and 1, or None if loading fails.
    """
    try:
        mask = io.imread(mask_path)
        # Ensure the mask is binary
        if mask.ndim > 2:  # If RGB, convert to binary
            mask = mask.mean(axis=2) > 0
        else:
            mask = mask > 0
        return mask.astype(np.uint8)
    except Exception as e:
        logger.error(f"Error loading mask {mask_path}: {e}")
        return None


def load_image_with_mask(
    image_path: os.PathLike, mask_path: os.PathLike, logger: loguru.logger
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load an image and its corresponding mask from file into NumPy arrays.
    """
    image = load_image(image_path, logger)
    image = img_as_ubyte(image)
    segmentation_image = load_image(mask_path, logger)
    mask = np.where(segmentation_image > 250, 1, 0).astype(bool)
    mask = img_as_ubyte(mask)
    return image, mask


def resize_image_and_mask_square_not_used(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize the image and mask to a square of the given size.
    The image is resized using bicubic interpolation, and the mask is resized
    using nearest neighbor interpolation.
    The mask is then converted to a boolean array. This is important because
    nearest neighbor interpolation may cause the mask to be non-binary.

    Args:
        image (np.ndarray): The image to resize.
        mask (np.ndarray): The mask to resize.
        target_size (int): The size of the square to resize the image and mask to.

    Returns:
        tuple: A tuple containing the resized image and mask.
    """
    # Resize the image and the mask.
    img_resized = resize(image, (target_size, target_size), anti_aliasing=True)
    mask_resized = resize(
        mask, (target_size, target_size), order=0, preserve_range=True
    )

    # Convert the image to uint8. For skimage, `img_as_ubyte` is needed to
    # scale the image to the range [0, 255] and avoid overflow.
    img_resized = img_as_ubyte(img_resized)

    # Convert the mask to a boolean array.
    mask_resized = mask_resized > 0.5

    mask_resized = img_as_ubyte(mask_resized)

    # # Keep the largest region in the mask.
    # mask_resized = keep_largest_region(mask_resized)

    # # Converting to bool is not needed here since the mask is already a boolean
    # # array after the > 0.5 threshold.
    # mask_resized = mask_resized.astype(bool)

    return img_resized, mask_resized


def get_image_mask_pairs(seg_metadata: pd.DataFrame) -> dict[str, list[str]]:
    """
    Group masks by image to create image-mask pairs for analysis.

    This function organizes the flat segmentation metadata into a hierarchical structure
    where each image is associated with multiple mask IDs. This organization is critical
    for inter-annotator analysis since we need to know which masks belong to the same image
    to make valid comparisons. The resulting dictionary makes it easy to iterate through
    images and access all masks for each image, which is a common pattern in the analysis
    pipeline for computing pairwise metrics and consensus masks.

    Parameters
    ----------
    seg_metadata : pandas.DataFrame
        DataFrame containing image and mask information.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with image IDs as keys and lists of mask IDs as values.
    """
    image_mask_pairs: dict[str, list[str]] = {}

    for _, row in seg_metadata.iterrows():
        img_id = row["img_filename"]
        mask_id = row["seg_filename"]

        if img_id not in image_mask_pairs:
            image_mask_pairs[img_id] = []

        image_mask_pairs[img_id].append(mask_id)

    return image_mask_pairs
