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

## downsampling images and saving them as 3d for nnunet data formating

# src_dir = Path("/media/fmallika/10TB/Mallika/autoseg")
src_dir = Path("/home/fmallika/Desktop/aa")

dst_dir = Path("/home/fmallika/Desktop/aa")
# dst_dir = Path("/home/fmallika/Documents/nnUNet/nnUNet_raw/Dataset991_HE4/") 

# original pixel spacing (0.172, 0.172) 
# need to automate (just checked it since the sample size was small)

original_spacing = (0.172, 0.172)
downsampling_factor = 4
pixel_spacing = tuple(downsampling_factor * s for s in original_spacing)  

img_ls = sorted(src_dir.rglob('*.tif*')) #img/*.tif*'))
for file in tqdm(img_ls):
    name = file.stem.split('.')[:-1][0]
    # name = name.split('-')[:5]
    # name = '-'.join(name)
    print(name)
    img_store = imread(file, aszarr=True)
    img_arr = zarr.open(img_store, mode='r')
    # downsample 
    img_arr = img_arr[::downsampling_factor, ::downsampling_factor]
    # # save to tiff
    # img_tgt = dst_dir / 'un_segmented_org_images'/ f'{name}.tiff'
    img_tgt = dst_dir / f'{name}.tiff'
    imwrite(str(img_tgt), img_arr)
    # imwrite(str(img_tgt), img_stack)
    # print(name, img_stack.shape)
    

# img_ls = sorted(src_dir.rglob('imagesTs/SR010*.tif*')) #img/*.tif*'))
# for file in tqdm(img_ls):
#     name = file.stem.split('.')[:-1][0]
#     # name = name.split('-')[:5]
#     # name = '-'.join(name)
#     print(name)
#     img_store = imread(file, aszarr=True)
#     img_arr = zarr.open(img_store, mode='r')
#     # # save to tiff
#     img_tgt = dst_dir / 'imagesTs'/ f'{name}.tiff'
#     imwrite(str(img_tgt), img_arr)
#     # imwrite(str(img_tgt), img_stack)
#     # print(name, img_stack.shape)
    


label_ls = sorted(src_dir.rglob('label/*.tif*'))
for file in tqdm(label_ls):
    name = file.stem.split("-")[:5]
    name = "-".join(name)
    print(name)
    mask_store = imread(file, aszarr=True)
    mask_arr = zarr.open(store=mask_store, mode="r")
    # downsample 
    mask_arr = mask_arr[::downsampling_factor, ::downsampling_factor]

    # save to tiff
    mask_tgt = dst_dir / 'labelsTr'/ f'{name}.tiff'
    imwrite(str(mask_tgt), mask_arr)
    # imwrite(str(mask_tgt), mask_stack)
    # print(name, mask_stack.shape)



# # resize:

label_ls = sorted(src_dir.rglob('*.tif*'))
for file in tqdm(label_ls):
    name = file.stem.split("-")[:5]
    name = "-".join(name)
    print(name)
    mask_store = imread(file, aszarr=True)
    mask_arr = zarr.open(store=mask_store, mode="r")
    # downsample 
    mask_arr = scipy.ndimage.zoom(mask_arr, downsampling_factor, order=0)
    # save to tiff
    mask_tgt = rf"{dst_dir}/{name}.tiff"
    imwrite(str(mask_tgt), mask_arr)
    # imwrite(str(mask_tgt), mask_stack)
    # print(name, mask_stack.shape)



# manualing split the data (5 images for testing - SR004 (2) and SR0011 (3) images ) and 22 images for training

## RENAMING THE DATA TO FIT NNUNET FORMAT

original_spacing = 0.172
downsampling_factor = 4
pixel_spacing = (original_spacing*downsampling_factor, original_spacing*downsampling_factor)
# pixel_spacing  = (1, original_spacing*downsampling_factor, original_spacing*downsampling_factor)

# train_set
imagestr = sorted(dst_dir.rglob('imagesTr/*.tif*'))
labelstr = sorted(dst_dir.rglob('labelsTr/*.tif*'))

for i, (im, se) in enumerate(zip(imagestr, labelstr)):
    #   print(i, im, se)
    target_name = f'HE_image_{i:03d}'
    shutil.copy(im, join(dst_dir, 'imagesTr', target_name + '_0000.tiff'))
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'imagesTr', target_name + '.json'))
    # segmentation
    shutil.copy(se, join(dst_dir, 'labelsTr', target_name + '.tiff'))
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'labelsTr', target_name + '.json'))
    


# test_set
import pandas as pd

df = pd.DataFrame()
imagests = sorted(dst_dir.rglob('imagesTs/*.tif*'))
labelsts = sorted(dst_dir.rglob('labelsTs/*.tif*'))

for i, (im, se) in enumerate(zip(imagests, labelsts)):
    #   print(i, im, se)
    target_name = f'HE_image_{i:03d}'
    shutil.copy(im, join(dst_dir, 'imagesTs', target_name + '_0000.tiff'))
    df = df._append({'og_name': im, 'new_name': target_name}, ignore_index=True)
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'imagesTs', target_name + '.json'))
    # segmentation
    shutil.copy(se, join(dst_dir, 'labelsTs', target_name + '.tiff'))
    # spacing file!
    # save_json({'spacing': pixel_spacing}, join(dst_dir, 'labelsTs', target_name + '.json'))

df.to_csv(rf"{dst_dir}/imagests/names.csv")


## generate json for entire dataset -> NNUNET FUNCTION 
# generate_dataset_json(output_folder: str,
#                           channel_names: dict,
#                           labels: dict,
#                           num_training_cases: int,
#                           file_ending: str,
#                           regions_class_order: Tuple[int, ...] = None,
#                           dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
#                           description: str = None,
#                           overwrite_image_reader_writer: str = None, **kwargs):
# overwrite_image_reader_writer='NaturalImage2DIO' for 2D images
# overwrite_image_reader_writer='Tiff3DIO' for 3D images

generate_dataset_json(output_folder=rf"{dst_dir}", channel_names={0:'R', 1:'G', 2:'B'}, 
                      labels={'background': 0, 'fascicle': 1, 'perineuirum':2, 'epineurium':3},
                      num_training_cases=24, file_ending='.tiff', overwrite_image_reader_writer='NaturalImage2DIO')



### resize to og resolution and file name

import pandas as pd
import os
import zarr
import glob
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


src_dir = Path(rf"/home/fmallika/Documents/nnUNet/nnUNet_raw/test_img/seg")
dst_dir = Path(rf"/media/fmallika/10TB/Mallika/autoseg")

df = pd.read_csv(rf"{src_dir}/names.csv")
label_ls = sorted(src_dir.rglob('*.tif*'))
og_name = [i.split('imagesTs')[-1].split('.tiff')[0] for i in df['og_name']]
new_names = [i for i in df['new_name']]

for file in tqdm(label_ls):
    name = file.stem.split("-")[:5]
    name = "-".join(name)
    print(name)
    mask_store = imread(file, aszarr=True)
    mask_arr = zarr.open(store=mask_store, mode="r")
    # downsample 
    mask_arr = scipy.ndimage.zoom(mask_arr, downsampling_factor, order=0)
    # save to tiff
    # idx = new_names.index(name)
    # name = og_name[idx]        
    mask_tgt = rf"{dst_dir}/{name}_SEG.tiff"
    imwrite(str(mask_tgt), mask_arr)
    # imwrite(str(mask_tgt), mask_stack)
    # print(name, mask_stack.shape)





### create folders and subfolder:
# and move the files to the respective folders

src_dst = rf"/media/fmallika/10TB/Mallika/autoseg"

files= glob.glob(rf"{src_dst}/*ome.tif*")

for file in files:
    name = file.split('.ome')[0]
    if not os.path.isdir(name):
        os.mkdir(name)
    shutil.move(file, name)
    shutil.move(rf'{name}_SEG.tiff', name)


## for each folder:

src_dst = rf"/media/fmallika/10TB/Mallika/autoseg"

files= os.listdir(src_dst)

for file in files:
    name = glob.glob(rf"{src_dst}/{file}/*SEG*")
    print(file, name)
    main_seg = imread(name[0])
    fasc = (main_seg==1)*1
    peri = (main_seg==2)*1
    epi = (main_seg==3)*1
    imwrite(rf"{src_dst}/{file}/{file}_fascicles.tiff", fasc, compression ='zlib')
    imwrite(rf"{src_dst}/{file}/{file}_perineurium.tiff", peri, compression ='zlib')
    imwrite(rf"{src_dst}/{file}/{file}_epineurium.tiff", epi, compression ='zlib')
    




### get metadata:

import pyometiff
import pathlib 
from pyometiff import OMETIFFReader
import json 


tiff_path = rf""
img_fpath = pathlib.Path(tiff_path)
reader = OMETIFFReader(fpath=img_fpath)
img_array, metadata, xml_metadata = reader.read()


json_output_path = rf""
with open(json_output_path, 'w') as json_file:
    json.dump(xml_metadata, json_file, indent=4)

