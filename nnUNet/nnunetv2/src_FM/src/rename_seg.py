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


## resize nnunet

def load_image(path):
    img_store = imread(path, aszarr=True)
    image = zarr.open(img_store, mode='r')
    image = image.astype(np.uint8)
    return np.array(image)

size_limit = 15 * 1024 * 1024 * 1024 


base_dir = rf"/media/fm/16TB/REVA_analysis/needs_post/"
subjects_list = ['09', '10', '11', '12', '13', '14', '15', '16']
for sub_id in subjects_list:
    print('On subject: ', sub_id)
    src_dir = Path(rf"{base_dir}/test_sr0{sub_id}/")
    dst_dir = Path(rf"{base_dir}/seg_sr0{sub_id}/publish")
    # dst_dir = Path(rf"{base_dir}/seg_sr0{sub_id}/good")

    save_dir1 = Path(rf"/media/fm/16TB/training_data")
    os.makedirs(save_dir1, exist_ok=True)

    df = pd.read_csv(rf"{src_dir}/test_names_{sub_id}.csv")
    label_ls = sorted(dst_dir.rglob('*.tif*'))
    og_name = [i.split('a/')[-1].split('.tiff')[0] for i in df['og_name']]
    new_names = [i for i in df['new_name']]

    # print(label_ls)
    ## resampled_based_on src_image:
    for file in tqdm(label_ls):
        try:
            name = file.stem.split("-")[:5]
            print(name)
            name = "-".join(name)
            idx = new_names.index(name)
            name = og_name[idx]  
            name = name.split("SR")[-1]
            name = "SR"+name
            print(name)
            file_name = f'{src_dir}'+ '/'+new_names[idx]+'_0000.tiff'
            os.makedirs(f'{save_dir1}/labels', exist_ok=True)
            os.makedirs(f'{save_dir1}/images', exist_ok=True)
            shutil.copy(file_name, rf"{save_dir1}/images/{name}.tiff")
            shutil.copy(file, rf"{save_dir1}/labels/{name}_seg.tiff")

            # print(id, region)
            # img_path = rf"/mnt/nas_histology/globus-donotchange/primary/sub-{id}/sam-{id}-{region}/HE/{name}.ome.tiff"
            # file_size = os.path.getsize(img_path)
            #print(img_path, file_size)
            # if file_size < size_limit:
            #     #print('here')
            #     # print(name)
            #     mask_arr = load_image(file)
            #     mask_img = Image.fromarray(mask_arr)
            #     img = load_image(img_path) 
            #     rgb_image = Image.fromarray(img)
            #     target_size = rgb_image.size
            #     resampled_mask = mask_img.resize(target_size, resample=Image.NEAREST)
            #     mask_arr = np.asarray(resampled_mask)
            #     # save to tiff   
            #     save_dir = f"{save_dir1}/sub-{id}/sam-{id}-{region}/HE/{name}"
            #     os.makedirs(save_dir, exist_ok=True)
            #     mask_tgt = rf"{save_dir}/{name}-post-seg.tiff"
            #     imwrite(str(mask_tgt), mask_arr, compression ='zlib')
            #     # shutil.copy(img_path, rf"{save_dir}/{name}.ome.tiff")
        except:
            pass


