#!/bin/bash
#
# Script to create a smaller subset of the original dataset.
# The primary purpose of this dataset is to enable faster training.
# Note the users should register the Cityscapes dataset website at
# https://www.cityscapes-dataset.com/downloads/ to download the dataset.
#
# Usage:
#   sh create_data_subset.sh /path/to/Cityscapes/Root
# There should not be a slash at the end of the path string.
#
# The folder structure inside the given directory is assumed to be:
#  + <source_path>
#    + gtFine
#      + train
#      + val
#      + test
#    + leftImg8bit
#      + train
#      + val
#      + test

# Exit immediately if a command exits with a non-zero status.
set -e

# Checking proper number of arguments before proceeding
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: sh create_data_subset.sh /path/to/Cityscapes/Root"
    exit 1
fi


CITYSCAPES_ROOT=$1
OUTPUT_DIR_ROOT="$CITYSCAPES_ROOT/../Cityscapes_Small"

# Checking if the output directory exists.
# If it does exist, delete it.
# Create new output directory
if [ -d "$OUTPUT_DIR_ROOT" ]; then
    rm -rf $OUTPUT_DIR_ROOT
fi

mkdir $OUTPUT_DIR_ROOT

# Procedure to create the train test and val directories
# in the current directory.
make_dirs()
{
    mkdir train
    mkdir test
    mkdir val

}

# Create the directory structure in the output directory
cd $OUTPUT_DIR_ROOT
mkdir gtFine
mkdir leftImg8bit
cd gtFine
make_dirs
cd ../leftImg8bit
make_dirs


# Copy the required folders from the source directory to the destination directories
dirs='/train/bochum /train/jena /train/ulm /test/bonn /val/lindau'
for i in $dirs; do   # The quotes are necessary here
    source_image="$CITYSCAPES_ROOT/leftImg8bit$i"
    source_label="$CITYSCAPES_ROOT/gtFine$i"
    reduced=${i%/*}
    destination_image="$OUTPUT_DIR_ROOT/leftImg8bit$reduced"
    destination_label="$OUTPUT_DIR_ROOT/gtFine$reduced"
    cp -r $source_image $destination_image
    cp -r $source_label $destination_label
    echo "Copied $i"
done