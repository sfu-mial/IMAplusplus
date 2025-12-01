import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from skimage import segmentation
from tqdm import tqdm

# sys.path.append("../../")

sys.path.append("..")
from utils import read_image_level_metrics

VIS_OUTPUT_DIR = Path(
    "../../output/visualizations/overlaid_segs/zero_overlap_images/"
)
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPATIAL_SIZE_FOR_VIS = (384, 384)

MASK_COLORS = [
    ## (1, 1, 0),  # Yellow
    ## (1, 1, 1),  # White
    (0, 0, 0),  # Black
    (1, 0, 1),  # Magenta
    # (0, 0, 1),  # Blue
    ## (1, 0, 0),  # Red
    ## (0, 1, 0),  # Green
    # (0, 1, 1),  # Cyan
    # (1, 0.5, 0),  # Orange
]


def isic_id_from_filename(img_path: Path) -> str:
    """Extracts the ISIC ID from a skin lesion image filename.

    The filename format is expected to be "ISIC_0000112.JPG".

    Parameters
    ----------
    img_path : Path
        The path to the image file.

    Returns
    -------
    str
        The ISIC ID (e.g., "ISIC_0000112").
    """
    return img_path.stem


def overlay_segmentations(
    image: np.ndarray, seg_paths: list[Path], colors: list[tuple]
) -> np.ndarray:
    """
    Overlays multiple segmentation mask boundaries onto a base image.

    To ensure that contours along the image edges are visible, the image and
    masks are padded before marking boundaries and then cropped back.

    Parameters
    ----------
    image : np.ndarray
        The base image (in RGB float32 format).
    seg_paths : list[Path]
        A list of paths to the binary segmentation masks.
    colors : list[tuple]
        A list of RGB tuples for the contour colors.

    Returns
    -------
    np.ndarray
        The image with segmentation boundaries overlaid.
    """
    # Pad image to make edge contours visible. 'edge' mode avoids black
    # borders.
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result_image = padded_image

    for i, seg_path in enumerate(seg_paths):
        mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, SPATIAL_SIZE_FOR_VIS)
        binary_mask = (mask > 127).astype(int)  # Ensure binary {0, 1}.

        # Pad the mask to match the padded image dimensions.
        padded_mask = np.pad(
            binary_mask, 1, mode="constant", constant_values=0
        )

        color = colors[i % len(colors)]

        # Overlay the segmentation mask boundaries onto the result image.
        result_image = segmentation.mark_boundaries(
            result_image,
            padded_mask,
            color=color,
            outline_color=color,
            mode="thick",
        )

    # Crop the image back to its original size to remove the padding.
    cropped_result = result_image[1:-1, 1:-1, :]
    return cropped_result


def save_image_with_title_bar(
    image_bgr: np.ndarray, title: str, output_path: Path
) -> None:
    """
    Adds a white rectangle bar above the image with black text (title),
    then saves the result to output_path.
    """
    TITLE_BAR_HEIGHT = 80

    h, w = image_bgr.shape[:2]

    # --- Create a white bar ---
    bar = (
        np.ones((TITLE_BAR_HEIGHT, w, 3), dtype=np.uint8) * 255
    )  # white RGB(255,255,255)

    # --- Put black text centered on the bar ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = TITLE_BAR_HEIGHT / 50  # scale text relative to bar height
    thickness = int(max(1, TITLE_BAR_HEIGHT / 25))
    text_color = (0, 0, 0)  # black

    (text_w, text_h), baseline = cv2.getTextSize(
        title, font, font_scale, thickness
    )
    text_x = (w - text_w) // 2
    text_y = (TITLE_BAR_HEIGHT + text_h) // 2  # vertically centered

    cv2.putText(
        bar,
        title,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    # --- Combine bar + image vertically ---
    combined = np.vstack((bar, image_bgr))

    # --- Save the combined image ---
    cv2.imwrite(str(output_path), combined)


def process_image(
    img_path: Path, colors: list[tuple], config: OmegaConf
) -> None:
    """
    Processes a single image: finds segmentations, overlays them, and saves.

    Parameters
    ----------
    img_path : Path
        The path to the input image.
    colors : list[tuple]
        A list of RGB tuples for the contour colors.
    config : OmegaConf
        The configuration object.

    Returns
    -------
    None
    """
    try:
        isic_id = isic_id_from_filename(img_path)

        SEGS_DIR = Path(config.new_dataset_masks_dir)

        seg_paths = sorted(list(SEGS_DIR.glob(f"{isic_id}*.png")))
        num_segmentations = len(seg_paths)

        if num_segmentations < 4:
            # We check for at least 4 segmentations because MV and STAPLE
            # count as two segmentations.
            print(
                f"Warning: Found {num_segmentations} masks for {isic_id}. "
                f"Skipping."
            )
            return

        # Load the image and convert to float RGB for scikit-image.
        image_bgr = cv2.imread(str(img_path))
        image_bgr = cv2.resize(image_bgr, SPATIAL_SIZE_FOR_VIS)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0

        # Overlay all segmentations.
        result_image_float = overlay_segmentations(
            image_float, seg_paths, colors
        )

        # Construct the new filename and save the result.
        # Again, we subtract 2 because MV and STAPLE count as two
        # segmentations, but we only want to count the number of
        # actual annotator-provided segmentations.
        new_filename = f"{isic_id}_{num_segmentations - 2}segs.png"
        output_path = VIS_OUTPUT_DIR / new_filename

        # Convert back to uint8 BGR for saving with OpenCV.
        result_image_uint8 = (result_image_float * 255).astype(np.uint8)
        result_image_bgr = cv2.cvtColor(result_image_uint8, cv2.COLOR_RGB2BGR)

        # Write to file with OpenCV.
        # cv2.imwrite(str(output_path), result_image_bgr)

        # Write to file with OpenCV, but add a title bar containing the ISIC
        # ID.
        # save_image_with_title_bar(
        #     image_bgr=result_image_bgr,
        #     title=isic_id,
        #     output_path=output_path,
        # )

        # Write to file with matplotlib.
        # Convert BGR to RGB for matplotlib.
        result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(result_image_rgb)
        plt.title(isic_id, fontsize=36)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return


def main() -> None:
    """
    Main function.
    """
    # Load the configuration.
    config = OmegaConf.load("config.yaml")

    # Read the image-level IAA metrics file into a DataFrame.
    image_level_IAA_metrics_df = read_image_level_metrics(config)

    # Filter the image-level IAA metrics DataFrame to only include images with
    # zero overlap.
    image_level_IAA_metrics_df = image_level_IAA_metrics_df[
        image_level_IAA_metrics_df["dice_score_mean"] == 0
    ]

    print(
        f"Found {len(image_level_IAA_metrics_df)} images with zero "
        f"overlap across all annotators."
    )

    # Process each image.
    for _, row in tqdm(
        image_level_IAA_metrics_df.iterrows(),
        total=len(image_level_IAA_metrics_df),
        desc="Visualizing zero overlap images",
    ):
        img_path = Path(config.new_dataset_images_dir) / row["img_id"]
        process_image(img_path, MASK_COLORS, config)
    return None


if __name__ == "__main__":
    main()
