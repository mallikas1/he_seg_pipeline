import os
import zarr
from pathlib import Path
import numpy as np
from tifffile import TiffFile, imread, imwrite
import ome_types
import dask.array as da
import shutil
# use nnunet env
from batchgenerators.utilities.file_and_folder_operations import save_json, join
# MOVE THE SCRIPT TO NNUNET FOLDER OR RUN FROM TERMINAL WITH PATH SET TO NNUNET FOLDER
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import scipy.ndimage
import argparse
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument('data_directory', type=str,
                    help="Data directory with original images")
parser.add_argument('save_directory', type=str, help= 'directory where the images will be saved')

args = parser.parse_args()

src_dir = Path(args.data_directory)
src_dir = Path("/home/fmallika/Desktop/a")

dst_dir = Path("/home/fmallika/Desktop/a") 

downsampling_factor = 4

## downsampling images and saving them as tiff images rather than ome.tiff
def downsamp2tiff(file, downsampling_factor=4):
    img_store = imread(file, aszarr=True)
    img_arr = zarr.open(img_store, mode='r')
    # downsample 
    img_arr = img_arr[::downsampling_factor, ::downsampling_factor]
    return img_arr



def rename4nnunet(img_ls, seg_ls, train=True, json=False):
    df = pd.DataFrame()
    for i, (im, se) in enumerate(zip(img_ls, seg_ls)):
        #   print(i, im, se)
        if train:
            x = 'imagesTr'
            y = 'labelsTr'
        else:
            x = 'imagesTs'
            y = 'labelsTs'
        target_name = f'HE_image_{i:03d}'
        shutil.copy(im, join(dst_dir, x, target_name + '_0000.tiff'))
        df = df._append({'og_name': im, 'new_name': target_name, 
                        'seg_name': se}, ignore_index=True)
        # segmentation
        shutil.copy(se, join(dst_dir, y, target_name + '.tiff'))
        if json:
            # spacing file!
            pixel_spacing = (0.172, 0.172) ### need to automate (just checked it)
            save_json({'spacing': pixel_spacing}, join(dst_dir,x, target_name + '.json'))
    if train:
        df.to_csv(rf"{dst_dir}/train_names.csv")
    else:
        df.to_csv(rf"{dst_dir}/test_names.csv")



#### Sample RUN

img_ls = sorted(src_dir.rglob('imagesTr/*.tif*'))
label_ls = sorted(src_dir.rglob('labelsTr/*.tif*'))
for file_img, file_seg in tqdm(zip(img_ls, label_ls)):
    name_img = file.stem#.split('.')[:-1][0]
    # downsample
    img_arr = downsamp2tiff(file_img, downsampling_factor=downsampling_factor)
    mask_arr = downsamp2tiff(file_seg, downsampling_factor=downsampling_factor)
    # # save to tiff
    img_tgt = rf"{dst_dir}/imagesTr/{name_img}.tiff"
    imwrite(str(img_tgt), img_arr)
    mask_tgt = rf"{dst_dir}/labelsTr/{name_img}_SEG.tiff"
    imwrite(str(mask_tgt), mask_arr)



## RENAMING THE DATA TO FIT NNUNET FORMAT
# train_set
imagestr = sorted(dst_dir.rglob('imagesTr/*.tif*'))
labelstr = sorted(dst_dir.rglob('labelsTr/*.tif*'))

rename4nnunet(imagestr, label_ls, train=True, json=False)

#test_set
imagests = sorted(dst_dir.rglob('imagesTs/*.tif*'))
labelsts = sorted(dst_dir.rglob('labelsTs/*.tif*'))

rename4nnunet(imagests, labelsts, train=False, json=False)

# # ## generate json for entire dataset -> NNUNET FUNCTION 

# generate_dataset_json(output_folder=rf"{dst_dir}", channel_names={0:'R', 1:'G', 2:'B'}, 
#                       labels={'background': 0, 'fascicle': 1, 'perineuirum':2, 'epineurium':3},
#                       num_training_cases=26, file_ending='.tiff', overwrite_image_reader_writer='NaturalImage2DIO')

