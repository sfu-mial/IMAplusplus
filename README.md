# ISIC MultiAnnot++ (IMAplusplus)

This repository contains the code for creating and analyzing **ISIC MultiAnnot++**, a large public multi-annotator skin lesion segmentation dataset for images from the ISIC Archive. The final dataset contains **17,684 segmentation masks** spanning **14,967 dermoscopic images**, where **2,394 dermoscopic images** have 2-5 segmentations per image, making it the largest publicly available skin lesion segmentation (SLS) dataset.

The segmentations contain metadata corresponding to the annotators' skill levels as well as the tool used to perform the segmentation, enabling several kinds of research including, but not limited to, annotator-specific preference modeling for segmentation and annotator metadata analysis.

## Dataset Overview

- **Total Images**: 14,967 dermoscopic images
- **Total Segmentations**: 17,684 segmentation masks
- **Multi-annotator Images**: 2,394 images with 2-5 segmentations per image
- **Annotator Metadata**: Skill levels (expert/novice) and segmentation tools
- **Source**: ISIC Archive

## Repository Structure

```
IMAplusplus/
├── dataset_creation/          # Dataset creation and preprocessing
│   ├── create_dataset.py       # Creates dataset metadata and anonymizes annotators
│   ├── move_dataset.py         # Copies images and masks to target directory
│   ├── constants.py           # Tool and skill level mappings
│   └── config.yaml             # Configuration for dataset creation
│
├── dataset_analysis/           # Dataset quality assurance and visualization
│   ├── mask_qa.py              # Validates masks for quality issues
│   ├── other_datasets_overlap.py  # Visualizes overlap with other datasets
│   ├── imaplusplus_annotator_overlap.py  # Visualizes annotator overlap
│   └── config.yaml             # Configuration for analysis
│
├── multiannotator_analysis/    # Multi-annotator analysis and metrics
│   ├── create_multiannotator_subset.py  # Creates subset with multiple annotations
│   ├── create_consensus_masks.py        # Generates STAPLE and majority voting masks
│   ├── compute_IAA_metrics.py           # Computes inter-annotator agreement metrics
│   ├── compute_image_level_metrics.py  # Aggregates metrics per image
│   ├── visualization_scripts/          # Scripts for generating visualizations
│   └── config.yaml                     # Configuration for analysis
│
├── utils/                      # Utility functions
│   ├── data.py                 # Data loading utilities
│   ├── metrics.py              # Metric computation functions
│   └── md5.py                  # MD5 hash utilities
│
├── output/                     # Generated outputs
│   ├── metadata/               # CSV metadata files
│   ├── seg_masks/              # Segmentation masks
│   └── visualizations/         # Generated plots and figures
│
└── overall_script.sh          # Main pipeline script
```

## Installation

### Dependencies

The codebase requires Python 3.x and the following packages:

```bash
pip install pandas numpy scikit-image SimpleITK medpy omegaconf loguru tqdm matplotlib upsetplot
```

Key dependencies:
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations
- **scikit-image**: Image processing and mask operations
- **SimpleITK**: STAPLE consensus mask computation
- **medpy**: Medical image metrics (Dice, Jaccard, Hausdorff distance)
- **omegaconf**: Configuration management
- **loguru**: Logging
- **tqdm**: Progress bars
- **matplotlib**: Plotting
- **upsetplot**: UpSet plots for set visualization

## Usage

### Quick Start

Run the complete pipeline using the provided script:

```bash
bash overall_script.sh
```

This script executes the following steps in order:

1. **Dataset Creation** (`dataset_creation/`)
   - Creates anonymized metadata from raw ISIC data
   - Maps annotators to anonymized IDs (A00, A01, ...)
   - Maps tools and skill levels to standardized codes
   - Calculates MD5 hashes for masks
   - Generates standardized filenames

2. **Dataset Analysis** (`dataset_analysis/`)
   - Performs quality assurance on masks
   - Validates for missing/corrupted files, empty masks, etc.
   - Visualizes overlap with other datasets (ISIC 2016-2019, HAM10000, PH2, etc.)
   - Visualizes annotator overlap patterns

3. **Multi-annotator Analysis** (`multiannotator_analysis/`)
   - Creates subset of images with multiple annotations
   - Generates consensus masks (STAPLE and majority voting)
   - Computes inter-annotator agreement (IAA) metrics

### Individual Scripts

You can also run individual scripts separately:

#### Dataset Creation

```bash
cd dataset_creation/
python create_dataset.py  # Creates metadata
python move_dataset.py    # Copies files to target directory
```

#### Dataset Analysis

```bash
cd dataset_analysis/
python mask_qa.py                        # Quality assurance
python other_datasets_overlap.py         # Dataset overlap visualization
python imaplusplus_annotator_overlap.py  # Annotator overlap visualization
```

#### Multi-annotator Analysis

```bash
cd multiannotator_analysis/
python create_multiannotator_subset.py  # Create multi-annotator subset
python create_consensus_masks.py        # Generate consensus masks
python compute_IAA_metrics.py          # Compute IAA metrics
python compute_image_level_metrics.py   # Aggregate metrics per image
```

## Configuration

Each module uses a `config.yaml` file for configuration. Key settings include:

- **Paths**: Source and target directories for images and masks
- **Metadata**: Paths to input and output metadata CSV files
- **Processing options**: Verbose logging, parallel processing settings

Example configuration structure:

```yaml
# dataset_creation/config.yaml
orig_imgs_dirs:
  jpg: ["/path/to/images/"]
orig_segs_dir: "/path/to/masks/"
raw_img_metadata_path: "./original_metadata_files/raw_ISIC_images_metadata.csv.gz"
raw_seg_masks_metadata_path: "./original_metadata_files/raw_ISIC_segmasks_metadata.csv"
target_data_dir: "/path/to/output/"
```

## Output Files

### Metadata Files

All metadata files are saved in `output/metadata/`:

- `IMAplusplus_seg_metadata.csv`: Complete segmentation metadata
- `IMAplusplus_img_metadata.csv`: Image metadata
- `IMAplusplus_multiannotator_subset_seg_metadata.csv`: Multi-annotator subset metadata
- `IMAplusplus_multiannotator_subset_IAA_metrics.csv`: Pairwise IAA metrics
- `IMAplusplus_multiannotator_subset_IAA_metrics_summary.csv`: Summary statistics
- `IMAplusplus_multiannotator_subset_image_level_metrics.csv`: Per-image aggregated metrics
- `IMAplusplus_seg_metadata_qa_results.csv`: Quality assurance results

### Segmentation Metadata Schema

Each segmentation mask has the following metadata:

- `ISIC_id`: ISIC image identifier
- `img_filename`: Image filename
- `seg_filename`: Segmentation mask filename
- `annotator`: Anonymized annotator ID (A00, A01, ..., ST, MV)
- `tool`: Segmentation tool (T1: manual pointlist, T2: unknown, T3: autofill)
- `skill_level`: Annotator skill level (S1: expert, S2: novice)
- `mskObjectID`: Original mask object ID
- `mask_md5`: MD5 hash of the mask file

### Consensus Masks

For images with multiple annotations, two consensus masks are generated:

- **STAPLE** (`*_ST_ST_ST_ST.png`): STAPLE algorithm consensus
- **Majority Voting** (`*_MV_MV_MV_MV.png`): Majority voting consensus

### Inter-annotator Agreement Metrics

The following metrics are computed for all pairwise mask comparisons:

**Overlap Metrics:**
- Dice coefficient
- Jaccard coefficient

**Boundary Metrics:**
- Hausdorff distance (HD)
- 95th percentile Hausdorff distance (HD95)
- Average symmetric surface distance (ASSD)
- Normalized versions (by image diagonal length)

## Key Features

### Annotator Anonymization

Annotators are anonymized to IDs (A00, A01, ...) based on the number of segmentations they produced, sorted in decreasing order.

### Tool and Skill Level Mapping

- **Tools**: 
  - T1: Manual pointlist
  - T2: Unknown/unspecified
  - T3: Autofill

- **Skill Levels**:
  - S1: Expert
  - S2: Novice

### Quality Assurance

The `mask_qa.py` script validates masks for:

- Missing or corrupted files (high severity)
- Empty masks (high severity)
- Masks covering entire image (high severity)
- Disconnected regions (medium severity)
- Masks touching image borders (low severity)

### Parallel Processing

Many scripts support parallel processing for improved performance:

- MD5 hash calculation
- Mask validation
- IAA metric computation

## Visualization

The repository includes scripts for generating various visualizations:

- **UpSet plots**: Annotator and dataset overlap patterns.
- **Distribution plots**: Dataset statistics.
- **Metric visualizations**: IAA metrics, agreement patterns.
- **Overlaid segmentations**: Visualizing segmentations that have zero overlap.

Visualizations are saved in `output/visualizations/` in multiple formats (PNG, PDF, SVG).

<!-- ## Citation

If you use this dataset or code, please cite:

```bibtex
@article{imaplusplus2024,
  title={ISIC MultiAnnot++: A Large Public Multi-Annotator Skin Lesion Segmentation Dataset},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
``` -->

