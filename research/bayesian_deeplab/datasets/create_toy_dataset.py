"""Script to create a toy dataset from a given set of images. This script
has been tailored for the Cityscapes dataset where the images and labels are
present in .png format and have dimensions 1024 x 2048.
There are two approaches followed to create the toy datasets.

Approach 1:
Step 1: Crop the image to a size which is original_size/4.
Step 2: Resize cropped image to a size which is cropped_size/4.

Approach 2:
Step 1: Resize the image to a size which is original_size/16.

The directory structure required for the source directory is:
  + <source_path>
     + gtFine
       + train
       + val
       + test
     + leftImg8bit
       + train
       + val
       + test

The destination directory should be empty.
The user has the option to choose the toy dataset creation mechanism.
"""
import os
import glob
import argparse
import sys

#TODO: Will the below package be in the same directory ???? 
from image_manipulation_utils import crop_image_by_factor_from_file
from image_manipulation_utils import resize_image_by_factor_from_file
from image_manipulation_utils import resize_image_by_factor
from image_manipulation_utils import save_image

parser=argparse.ArgumentParser()

parser.add_argument('--source', '-s', required=True, help='Path of the source directory to get the images from')
parser.add_argument('--destination','-d', required=True, help='Path of the destination directory to store the processed images in')
parser.add_argument('--approach', '-a', default=1, type=int, help='Approach 1 (Crop (/4) and resize (/4)) or Approach 2 (Resize /16)')

args=parser.parse_args()

source_path = args.source
destination_path = args.destination
approach = args.approach

# Create folder structure in destination path
for dirpath, _, _ in os.walk(source_path):
    structure = destination_path + dirpath[len(source_path):]
    if not os.path.isdir(structure):
        os.mkdir(structure)

# Get the file list from the source path
annotation_path_format = os.path.join(source_path, "gtFine", "*", "*", "*_gtFine_labelTrainIds.png")
image_path_format = os.path.join(source_path, "leftImg8bit", "*", "*", "*_leftImg8bit.png")

annotation_file_paths = glob.glob(annotation_path_format)
image_file_paths = glob.glob(image_path_format)

all_paths = annotation_file_paths + image_file_paths


# Processing the images
# Approach 1
progress = 0
if approach == 1:
    print("Progress: {:>3} %".format( progress * 100 / len(all_paths) ), end=' ')
    for src_path in all_paths:
        dst_path = src_path.replace(source_path, destination_path)

        # Crop image
        cropped_image = crop_image_by_factor_from_file(4, src_path, dst_path=None)
        
        # Resize image
        resized_image = resize_image_by_factor(4, cropped_image)

        # Save image
        save_image(resized_image, dst_path)

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(all_paths) ), end=' ')
        sys.stdout.flush()

elif approach == 2:
    for src_path in all_paths:
        dst_path = src_path.replace(source_path, destination_path)

        # Resize image
        resized_image = resize_image_by_factor_from_file(16, src_path, dst_path=dst_path)

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(all_paths) ), end=' ')
        sys.stdout.flush()

else:
    print ("There are only approaches 1 and 2. I'm sorry.")