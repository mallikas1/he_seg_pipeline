'''
This script needs to run before running the nnunet testing script

 '''

#%%
import os
import zarr
from pathlib import Path
import numpy as np
from tifffile import TiffFile, imread, imwrite
import dask.array as da
# use nnunet env
from batchgenerators.utilities.file_and_folder_operations import save_json, join
from tqdm import tqdm
import pandas as pd
import argparse


def load_image(path):
    img_store = imread(path, aszarr=True)
    image = zarr.open(img_store, mode='r')
    image = image.astype(np.uint8)
    return np.array(image)


original_spacing = (0.172, 0.172)
downsampling_factor = 4
pixel_spacing = tuple(downsampling_factor * s for s in original_spacing)  

parser = argparse.ArgumentParser(
                    prog='Pre-Segmentation',
                    description='Downsampling and renaming to match nnunet data standards')
parser.add_argument('-main_dir', type=str, default=rf'/mnt/nas_histology/globus-donotchange/primary', help='this is the nas_histology path where all the files are')
parser.add_argument('-subject_id', type=str, help='eg: 01 or 26, ...')
parser.add_argument('-save_dir', type=str, help='where the final files would be saved for nnunet to access')
args = parser.parse_args()

print(args)


src_dir = args.main_dir + '/sub-SR0' + str(args.subject_id)
dst_dir = args.save_dir
os.makedirs(dst_dir, exist_ok=True)

src_dir = Path(src_dir)
dst_dir = Path(dst_dir)
img_ls = sorted(src_dir.rglob('*HE*.tif*'))
print(src_dir)
print(dst_dir)

for file in tqdm(img_ls):
    name = file.stem.split('.')[0]#[:-1][0]
    print(name)
    img_store = imread(file, aszarr=True)
    img_arr = zarr.open(img_store, mode='r')
    # downsample by a factor of 4
    img_arr = img_arr[::downsampling_factor, ::downsampling_factor]
    # # save to tiff
    img_tgt = dst_dir / f'{name}.tiff'
    imwrite(str(img_tgt), img_arr)
    

## RENAME 
# test_set
imagests = sorted(dst_dir.rglob('*.tif*'))
df = pd.DataFrame()
for i, im in enumerate(imagests):
    target_name = f'HE_image_{i:03d}'
    os.rename(im, join(dst_dir, target_name + '_0000.tiff'))
    df = df._append({'og_name': im, 'new_name': target_name}, ignore_index=True)


# save a csv with original and updates file names for post-seg renaming 
df.to_csv(rf"{dst_dir}/test_names_{str(args.subject_id)}.csv")


