# Script to create a smaller subset of the original dataset as well
# the toy dataset which is a scaled down version of the subset.
#
# Usage:
#   sh create_intermediate_and_toy_sets.sh /path/to/Cityscapes/Root 
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
SUBSET_DIR_ROOT="$CITYSCAPES_ROOT/../Cityscapes_Small"
TOY_DIR_ROOT="$CITYSCAPES_ROOT/../Cityscapes_Toy"

# If toy directory root does not exist, create it.
if [ ! -d "$TOY_DIR_ROOT" ]; then
    mkdir $TOY_DIR_ROOT
fi

# Creating the data subset
sh create_data_subset.sh $CITYSCAPES_ROOT

# Creating the toy dataset
python create_toy_dataset.py --source $SUBSET_DIR_ROOT --destination $TOY_DIR_ROOT --approach 2