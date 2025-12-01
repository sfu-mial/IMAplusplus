#! /bin/bash

# Exit the script if any command fails.
# Source: https://sipb.mit.edu/doc/safe-shell/.
set -euf -o pipefail

# Create the dataset and copy files to the new location.
cd dataset_creation/
python create_dataset.py
python move_dataset.py

# Analyze the dataset.
cd ../dataset_analysis/
# Perform quality validation on the masks.
python mask_qa.py
# Visualize the overlap between IMA++ and other datasets.
python other_datasets_overlap.py
# Visualize the overlap among annotators.
python imaplusplus_annotator_overlap.py

# Create the multiannotator subset and calculate the consensus masks.
cd ../multiannotator_analysis/
# Create the multiannotator subset of IMA++.
python create_multiannotator_subset.py
# Calculate the consensus masks (STAPLE and majority voting) for the 
# multiannotator subset and save them to the same location as the masks.
# Also, add the consensus masks metadata to the dataset and the multiannotator
# subset metadata files.
python create_consensus_masks.py
# Compute the inter-annotator agreement (IAA) metrics for the multiannotator subset.
python compute_IAA_metrics.py