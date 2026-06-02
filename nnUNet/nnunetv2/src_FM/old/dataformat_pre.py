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



## downsampling images and saving them as 3d for nnunet data formating

parser = argparse.ArgumentParser()
parser.add_argument('data_directory', type=str,
                    help="Data directory with original images")
parser.add_argument('save_directory', type=str, help= 'directory where the images will be saved')

args = parser.parse_args()

src_dir = Path(args.data_directory)
src_dir = Path("/home/fmallika/Documents/nnUNet/nnUNet_raw/Dataset995_HE/masks")
# src_dir = Path("/home/fmallika/Documents/nnUNet/nnUNet_raw/test_img/un_segmented_org_images")

# dst_dir = Path("/home/fmallika/Documents/nnUNet/nnUNet_raw/test_img")
dst_dir = Path("/home/fmallika/Documents/nnUNet/nnUNet_raw/Dataset995_HE/labelsTr") 

# original pixel spacing (0.172, 0.172) 
# need to automate (just checked it since the sample size was small)

original_spacing = (0.172, 0.172)
downsampling_factor = 4
pixel_spacing = tuple(downsampling_factor * s for s in original_spacing)  

# img_ls = sorted(src_dir.rglob('imagesTr1/*.tif*'))

# for file in tqdm(img_ls):
#     name = file.stem.split('.')[0] #[:-1][0]
#     print(name)
#     img_store = imread(file, aszarr=True)
#     img_arr = zarr.open(img_store, mode='r')
#     # downsample 
#     img_arr = img_arr[::downsampling_factor, ::downsampling_factor]
#     # # save to tiff
#     # img_tgt = dst_dir / 'imagesTr'/ f'{name}.tiff'
#     img_tgt = rf"{dst_dir}/{name}.tiff"
#     imwrite(str(img_tgt), img_arr)
    


img_ls = sorted(src_dir.rglob('*.tif*'))
for file in tqdm(img_ls):
    name = file.stem.split('.')[:-1][0]
    print(name)
    img_store = imread(file, aszarr=True)
    img_arr = zarr.open(img_store, mode='r')
    # downsample 
    img_arr = img_arr[::downsampling_factor, ::downsampling_factor]
    # # save to tiff
    img_tgt = dst_dir / f'{name}.tiff'
    imwrite(str(img_tgt), img_arr)
    


label_ls = sorted(src_dir.rglob('labelsTr/*.tif*'))
for file in tqdm(label_ls):
    name = file.stem.split('.')[0]
    print(name)
    mask_store = imread(file, aszarr=True)
    mask_arr = zarr.open(store=mask_store, mode="r")
    # downsample 
    mask_arr = mask_arr[::downsampling_factor, ::downsampling_factor]
    # save to tiff
    mask_tgt = dst_dir / 'labelsTr'/ f'{name}.tiff'
    imwrite(str(mask_tgt), mask_arr)


# manualing split the data (5 images for testing - SR004 (2) and SR0011 (3) images ) and 22 images for training

## RENAMING THE DATA TO FIT NNUNET FORMAT

original_spacing = 0.172
downsampling_factor = 4
pixel_spacing = (original_spacing*downsampling_factor, original_spacing*downsampling_factor)
# pixel_spacing  = (1, original_spacing*downsampling_factor, original_spacing*downsampling_factor)




# train_set
imagestr = sorted(dst_dir.rglob('imagesTr/*.tif*'))
labelstr = sorted(dst_dir.rglob('labelsTr/*.tif*'))
df = pd.DataFrame()
for i, (im, se) in enumerate(zip(imagestr, labelstr)):
    #   print(i, im, se)
    target_name = f'HE_image_{i:03d}'
    shutil.copy(im, join(dst_dir, 'imagesTr', target_name + '_0000.tiff'))
    df = df._append({'og_name': im, 'new_name': target_name, 
                     'seg_name': se}, ignore_index=True)
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'imagesTs', target_name + '.json'))
    # segmentation
    shutil.copy(se, join(dst_dir, 'labelsTr', target_name + '.tiff'))
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'labelsTs', target_name + '.json'))

df.to_csv(rf"{dst_dir}/train_names.csv")

# test_set
imagests = sorted(dst_dir.rglob('imagesTs/*.tif*'))
df = pd.DataFrame()
for i, im in enumerate(imagests):
    #   print(i, im, se)
    target_name = f'HE_image_{i:03d}'
    shutil.copy(im, join(dst_dir, 'imagesTs', target_name + '_0000.tiff'))
    df = df._append({'og_name': im, 'new_name': target_name}, ignore_index=True)
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'imagesTs', target_name + '.json'))


df.to_csv(rf"{dst_dir}/test_names.csv")

# ## generate json for entire dataset -> NNUNET FUNCTION 
# # generate_dataset_json(output_folder: str,
# #                           channel_names: dict,
# #                           labels: dict,
# #                           num_training_cases: int,
# #                           file_ending: str,
# #                           regions_class_order: Tuple[int, ...] = None,
# #                           dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
# #                           description: str = None,
# #                           overwrite_image_reader_writer: str = None, **kwargs):
# # overwrite_image_reader_writer='NaturalImage2DIO' for 2D images
# # overwrite_image_reader_writer='Tiff3DIO' for 3D images

generate_dataset_json(output_folder=rf"{dst_dir}", channel_names={0:'R', 1:'G', 2:'B'}, 
                      labels={'background': 0, 'fascicle': 1, 'perineuirum':2, 'epineurium':3},
                      num_training_cases=26, file_ending='.tiff', overwrite_image_reader_writer='NaturalImage2DIO')



generate_dataset_json(output_folder=rf"{dst_dir}", channel_names={0:'Grayscale'}, 
                      labels={'background': 0, 'fascicle': 1, 'perineuirum':2, 'epineurium':3},
                      num_training_cases=26, file_ending='.tiff', overwrite_image_reader_writer='NaturalImage2DIO')



