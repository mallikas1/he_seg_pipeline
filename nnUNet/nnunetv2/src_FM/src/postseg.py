#!/usr/bin/env python3
"""
remove_specks.py

Small utility to remove tiny specks (small connected components)
from segmentation masks (2D or 3D). Handles binary and labeled
segmentation images.

Usage:
  python remove_specks.py --input mask.tif --output cleaned.tif --min-size 64

Options:
  --input       Input segmentation image (tif)
  --output      Output path for cleaned image
  --min-size    Minimum component size (in pixels/voxels) to keep
  --connectivity  Connectivity for labeling (1 or 2 for 2D)

The script will detect whether the image is binary (0/1) or labeled
(multiple integer labels). For labeled images, it removes small connected
components per label (preserving label ids for surviving components).
"""

from __future__ import annotations
import argparse
import sys, os
import numpy as np
import cv2
from tqdm import tqdm
from tifffile import TiffFile, imread, imwrite
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
import zarr
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_dilation, binary_erosion, remove_small_holes
from scipy.ndimage import binary_fill_holes, median_filter



LABEL_FASCICLE = 1
LABEL_PERI = 2
LABEL_EPI = 3

MIN_PERI_SIZE = 100

def read_image(path):
    image = imread(path) #, aszarr=True)
    return np.array(image)


def write_image(path: str, img: np.ndarray):
    imwrite(path, img, compression='zlib')



def clean_peri_and_remove_epi_inside_peri(seg):
    seg = np.asarray(seg).copy()
    asc = seg == LABEL_FASCICLE
    fasc_cc = label(fasc, connectivity=2)
    # remvove tiny fascicles first
    for region in regionprops(fasc_cc, cache=False):
        if region.area < 200:
            seg[fasc_cc == region.label] = LABEL_EPI
    fasc = seg == LABEL_FASCICLE
    peri = seg == LABEL_PERI
    epi = seg == LABEL_EPI
    peri_cc = label(peri, connectivity=2)
    keep_peri = np.zeros_like(peri, dtype=bool)
    inside_any_peri = np.zeros_like(peri, dtype=bool)
    for region in regionprops(peri_cc):
        if region.area < MIN_PERI_SIZE:
            continue
        comp = peri_cc == region.label
        # fill enclosed region
        filled = binary_fill_holes(comp)
        inside = filled & (~comp)
        # keep only peri enclosing fascicles
        if np.any(inside & fasc):
            keep_peri |= comp
        
        inside_any_peri |= inside
    cleaned = seg.copy()
    # remove small/floating peri
    cleaned[peri] = 1
    cleaned[keep_peri] = LABEL_PERI
    # remove epi that lies inside peri
    cleaned[epi & inside_any_peri] = 0
    return cleaned


def clean_peri_and_remove_epi_inside_peri_fast(seg, pad=10):
    seg = np.asarray(seg).copy()
    fasc = seg == LABEL_FASCICLE
    fasc_cc = label(fasc, connectivity=2)
    # remvove tiny fascicles first
    for region in regionprops(fasc_cc, cache=False):
        if region.area < 200:
            seg[fasc_cc == region.label] = LABEL_EPI
    fasc = seg == LABEL_FASCICLE
    peri = seg == LABEL_PERI
    epi  = seg == LABEL_EPI
    peri_cc = label(peri, connectivity=2)
    keep_peri = np.zeros_like(peri, dtype=bool)
    inside_any_peri = np.zeros_like(peri, dtype=bool)
    H, W = seg.shape

    for region in regionprops(peri_cc, cache=False):
        if region.area < MIN_PERI_SIZE:
            continue
        minr, minc, maxr, maxc = region.bbox
        r0 = max(minr - pad, 0)
        r1 = min(maxr + pad, H)
        c0 = max(minc - pad, 0)
        c1 = min(maxc + pad, W)
        peri_crop_cc = peri_cc[r0:r1, c0:c1]
        fasc_crop = fasc[r0:r1, c0:c1]
        comp_crop = peri_crop_cc == region.label
        filled_crop = binary_fill_holes(comp_crop)
        inside_crop = filled_crop & (~comp_crop)
        if np.any(inside_crop & fasc_crop):
            keep_peri[r0:r1, c0:c1] |= comp_crop

        inside_any_peri[r0:r1, c0:c1] |= inside_crop
    cleaned = seg.copy()
    # remove floating peri
    cleaned[peri] = 0
    cleaned[keep_peri] = LABEL_PERI
    # remove epi inside peri
    cleaned[epi & inside_any_peri] = 0
    return cleaned



def remove_bad_fascicle_peri_pairs(seg, peri_dilate_px=1):
    seg = np.asarray(seg).copy()

    fasc_cc = label(seg == LABEL_FASCICLE, connectivity=2)
    peri_cc = label(seg == LABEL_PERI,     connectivity=2)

    # Precompute filled peri interiors once
    filled_peri = {
        r.label: binary_fill_holes(
            binary_dilation(peri_cc == r.label, footprint=disk(peri_dilate_px))
            if peri_dilate_px > 0 else peri_cc == r.label
        )
        for r in regionprops(peri_cc)
    }

    # Stack into (N, H, W) array for fast batch lookup
    peri_labels  = np.array(list(filled_peri.keys()),   dtype=np.int32)   # (N,)
    filled_stack = np.stack(list(filled_peri.values()))                    # (N, H, W)

    bad_fasc = np.zeros_like(fasc_cc, dtype=bool)
    bad_peri = np.zeros(len(peri_labels) + 1, dtype=bool)  # indexed by peri label

    for r in regionprops(fasc_cc):
        mask = fasc_cc == r.label

        # Which filled peri regions contain ANY pixel of this fascicle?
        hits = filled_stack[:, mask].any(axis=1)          # (N,) bool

        # Is the fascicle FULLY inside at least one of those?
        enclosed = hits & filled_stack[:, mask].all(axis=1)

        if not enclosed.any():
            bad_fasc |= mask
            bad_peri[peri_labels[hits]] = True

    cleaned = seg.copy()
    cleaned[bad_fasc] = LABEL_EPI
    cleaned[np.isin(peri_cc, peri_labels[bad_peri[peri_labels]])] = LABEL_EPI

    # Second pass: drop fascicles orphaned after peri removal
    peri_cc2   = label(cleaned == LABEL_PERI, connectivity=2)
    union_fill = np.zeros(cleaned.shape, dtype=bool)
    for r in regionprops(peri_cc2):
        union_fill |= binary_fill_holes(
            binary_dilation(peri_cc2 == r.label, footprint=disk(1))
        )

    fasc_cc2 = label(cleaned == LABEL_FASCICLE, connectivity=2)
    for r in regionprops(fasc_cc2):
        mask = fasc_cc2 == r.label
        if not union_fill[mask].all():
            cleaned[mask] = LABEL_EPI

    return cleaned


def remove_bad_fascicle_peri_pairs_fast(seg, peri_dilate_px=1):
    seg = np.asarray(seg).copy()
    H, W = seg.shape
    cleaned = seg.copy()

    fasc_cc = label(seg == LABEL_FASCICLE, connectivity=2)
    peri_cc = label(seg == LABEL_PERI,     connectivity=2)

    pad = peri_dilate_px + 2

    # --- Precompute filled peri crops (bbox only, not full image) ---
    peri_info = {}
    for r in regionprops(peri_cc):
        r0, c0, r1, c1 = r.bbox
        r0p, r1p = max(0, r0 - pad), min(H, r1 + pad)
        c0p, c1p = max(0, c0 - pad), min(W, c1 + pad)
        crop = peri_cc[r0p:r1p, c0p:c1p] == r.label
        if peri_dilate_px > 0:
            crop = binary_dilation(crop, footprint=disk(peri_dilate_px))
        peri_info[r.label] = (r0p, c0p, binary_fill_holes(crop))

    bad_fasc = set()
    bad_peri = set()

    # --- Check each fascicle using bbox overlap + crop-space indexing ---
    for fr in regionprops(fasc_cc):
        rows, cols = fr.coords[:, 0], fr.coords[:, 1]  # no np.where scan
        fr0, fc0, fr1, fc1 = fr.bbox

        nearby   = []
        enclosed = False

        for plbl, (r0p, c0p, filled) in peri_info.items():
            ph, pw = filled.shape
            # fast bbox rejection
            if r0p + ph <= fr0 or r0p >= fr1 or c0p + pw <= fc0 or c0p >= fc1:
                continue

            lr, lc = rows - r0p, cols - c0p
            in_bounds = (lr >= 0) & (lr < ph) & (lc >= 0) & (lc < pw)

            if not in_bounds.all():
                if in_bounds.any() and filled[lr[in_bounds], lc[in_bounds]].any():
                    nearby.append(plbl)
                continue

            vals = filled[lr, lc]
            if vals.any():
                nearby.append(plbl)
            if vals.all():
                enclosed = True
                break

        if not enclosed:
            bad_fasc.add(fr.label)
            bad_peri.update(nearby)

    # --- Apply removals in two vectorised calls ---
    if bad_fasc:
        cleaned[np.isin(fasc_cc, list(bad_fasc))] = LABEL_EPI
    if bad_peri:
        cleaned[np.isin(peri_cc, list(bad_peri))] = LABEL_EPI

    # --- Second pass: orphaned fascicles (bbox crops again) ---
    peri_cc2  = label(cleaned == LABEL_PERI,     connectivity=2)
    fasc_cc2  = label(cleaned == LABEL_FASCICLE, connectivity=2)
    union_fill = np.zeros((H, W), dtype=bool)

    for r in regionprops(peri_cc2):
        r0, c0, r1, c1 = r.bbox
        r0p, r1p = max(0, r0 - 2), min(H, r1 + 2)
        c0p, c1p = max(0, c0 - 2), min(W, c1 + 2)
        crop = peri_cc2[r0p:r1p, c0p:c1p] == r.label
        union_fill[r0p:r1p, c0p:c1p] |= binary_fill_holes(
            binary_dilation(crop, footprint=disk(1))
        )

    for r in regionprops(fasc_cc2):
        rows, cols = r.coords[:, 0], r.coords[:, 1]
        if not union_fill[rows, cols].all():
            cleaned[rows, cols] = LABEL_EPI

    return cleaned


def main(argv=None):
    parser = argparse.ArgumentParser(description="Remove small connected components from segmentation masks")
    parser.add_argument("--IN_DIR", "-i", required=True, help="Input segmentation path")
    parser.add_argument("--OUT_DIR", "-o", required=True, help="Output path")
    parser.add_argument("--cleaning", "-c", required=True, default='basic', help="Cleaning version to run: basic or pairs")
    args = parser.parse_args(argv)

    IN_DIR = Path(args.IN_DIR)
    OUT_DIR = Path(args.OUT_DIR)
    if args.cleaning == 'basic':
        os.makedirs(f"{OUT_DIR}/basic", exist_ok=True)
    elif args.cleaning == 'pairs':
        os.makedirs(f"{OUT_DIR}/clean_fas_peri_pairs", exist_ok=True)

    files = sorted(list(IN_DIR.glob('*.tiff')))
    for f in files:
        # if '001' in f.name:

        print(f"File Detected: {f}")
        img = read_image(f)
        # print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        a = f.name
        # print(name)
        if img is None:
            print(f"Failed to read image: {f}")
            sys.exit(2)

        if args.cleaning == 'basic':
            # out = clean_peri_and_remove_epi_inside_peri(img)
            out = clean_peri_and_remove_epi_inside_peri_fast(img)
            print("Basic cleaning done - saving it")
            write_image(f"{OUT_DIR}/basic/{a}", out)

        elif args.cleaning == 'pairs':
            # cleaned = remove_bad_fascicle_peri_pairs(img)
            cleaned = remove_bad_fascicle_peri_pairs_fast(img)
            # cleaned = remove_bad_fascicle_peri_pairs_v2(img)
            write_image(f"{OUT_DIR}/clean_fas_peri_pairs/{a}", cleaned)
            print("Removed fascicles - perineurium pairs that have issues")

if __name__ == '__main__':
    main()
