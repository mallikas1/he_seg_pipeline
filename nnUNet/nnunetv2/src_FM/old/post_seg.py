## post-segmentation steps to do:

# 1. rename file to original file name + resize to original resolution
# 2. move files (original image and segmentation) into respective folder
# 3. split segmentation into its masks (fascicle, perineurium, epineurium) save as compressed files 
# 4. create a snapshot of image and segmentations
# 5. Save metadata.json for each folder








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
from PIL import Image
import cv2



src_dir = Path(rf"/home/fmallika/Desktop/For_Duke/imagesTs")
dst_dir = Path(rf"/home/fmallika/Desktop/For_Duke/imagesTs")

df = pd.read_csv(rf"{src_dir}/test_names.csv")
label_ls = sorted(dst_dir.rglob('*.tif*'))
og_name = [i.split('a/')[-1].split('.tiff')[0] for i in df['og_name']]
new_names = [i for i in df['new_name']]

def load_image(path):
    img_store = imread(path, aszarr=True)
    image = zarr.open(img_store, mode='r')
    image = image.astype(np.uint8)
    return np.array(image)


## resampled_based_on src_image:

for file in tqdm(label_ls):
    name = file.stem.split("-")[:5]
    name = "-".join(name)
    idx = new_names.index(name)
    name = og_name[idx]  
    name = name.split("SR")[-1]
    name = "SR"+name
    print(name)
    mask_arr = load_image(file)
    mask_img = Image.fromarray(mask_arr)
    img = load_image(rf"/home/fmallika/Desktop/For_Duke/og_ome_tiff/SR020/{name}.ome.tiff")
    rgb_image = Image.fromarray(img)
    target_size = rgb_image.size
    resampled_mask = mask_img.resize(target_size, resample=Image.NEAREST)
    mask_arr = np.asarray(resampled_mask)
    # save to tiff   
    mask_tgt = rf"{dst_dir}/{name}_SEG.tiff"
    imwrite(str(mask_tgt), mask_arr, compression ='zlib')
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

src_dst = rf"/home/fmallika/Desktop/cleaned_seg/SR013"

files= os.listdir(src_dst)

for file in files:
    print(file)
    main_seg = imread(rf"{src_dst}/{file}")
    fasc = (main_seg == 1) * 255
    peri = (main_seg == 2) * 255
    epi = (main_seg == 3) * 255
    name = file.split('_SEG')[0]
    fasc = fasc.astype(np.uint8)
    peri = peri.astype(np.uint8)
    epi = epi.astype(np.uint8)
    cv2.imwrite(rf"{src_dst}/{name}_fascicles.tiff", fasc)
    cv2.imwrite(rf"{src_dst}/{name}_perineurium.tiff", peri)
    cv2.imwrite(rf"{src_dst}/{name}_epineurium.tiff", epi)

    # imwrite(rf"{src_dst}/{file}/{file}_fascicles.tiff", fasc, compression ='zlib')
    # imwrite(rf"{src_dst}/{file}/{file}_perineurium.tiff", peri, compression ='zlib')
    # imwrite(rf"{src_dst}/{file}/{file}_epineurium.tiff", epi, compression ='zlib')
    

# ### get metadata:

import pyometiff
import pathlib 
from pyometiff import OMETIFFReader
import json 

dir_path = rf"/home/fmallika/Desktop/For_Duke/og_ome_tiff/SR013"
file = rf"SR013-EA2-03d04d-136um-HE-20240805-s1"
tiff_path = rf"{dir_path}/{file}/{file}.ome.tiff"
img_fpath = pathlib.Path(tiff_path)
reader = OMETIFFReader(fpath=img_fpath)
img_array, metadata, xml_metadata = reader.read()

def remove_key(data, key_to_remove):
    if isinstance(data, dict):
        # Recursively clean dictionaries
        return {k: remove_key(v, key_to_remove) for k, v in data.items() if k != key_to_remove}
    elif isinstance(data, list):
        # Recursively clean lists
        return [remove_key(item, key_to_remove) for item in data]
    else:
        # Return the value as is for non-dict/list types
        return data

# Remove "Directory" from metadata
cleaned_metadata = remove_key(metadata, "Directory")


json_output_path = rf"{dir_path}/{file}/{file}_metadata.json"
with open(json_output_path, 'w') as json_file:
    json.dump(cleaned_metadata, json_file, indent=4)

xml_output_path = rf"{dir_path}/{file}/{file}_metadata.xml"
with open(xml_output_path, "w", encoding="utf-8") as xml_file:
    xml_file.write(xml_metadata)

