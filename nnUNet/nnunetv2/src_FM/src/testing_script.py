import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tiff
import scipy.ndimage as sp_nd
from skimage.measure import label, regionprops
from medpy.metric.binary import hd95, assd


# -----------------------------
# Simple region metrics
# -----------------------------
def dice_score(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else (2.0 * inter / s)

def iou_score(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return 1.0 if uni == 0 else (inter / uni)

def safe_hd95(a, b):
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    return float(hd95(a.astype(np.uint8), b.astype(np.uint8)))

def safe_assd(a, b):
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    return float(assd(a.astype(np.uint8), b.astype(np.uint8)))


# -----------------------------
# Connected component count
# -----------------------------
def count_components(bin_mask: np.ndarray) -> int:
    lab = label(bin_mask.astype(bool))
    return int(lab.max())  # labels are 0..N


# -------------------------------------------------------------
# Thickness code (inner-boundary referenced), refactored
# -------------------------------------------------------------
def effective_diameter(area_um2: float) -> float:
    return float(np.sqrt(4.0 * area_um2 / np.pi))

def analyse_slice_inner_boundary_thickness(
    seg: np.ndarray,
    *,
    mpp: float,
    fasc_label: int = 1,
    peri_label: int = 2,
    bbox: bool = True,
    pad: int = 64,
) -> pd.DataFrame:
    peri_mask = (seg == peri_label)
    fasc_mask = (seg == fasc_label)

    if not peri_mask.any():
        return pd.DataFrame()

    fasc_lab = label(fasc_mask)

    not_peri = ~peri_mask

    border = np.zeros_like(peri_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    outside_seed = not_peri & border

    outside_bg = np.zeros_like(peri_mask, dtype=bool)
    if outside_seed.any():
        cc = label(not_peri)
        outside_ids = np.unique(cc[outside_seed])
        outside_bg = np.isin(cc, outside_ids)
    else:
        outside_bg[:] = False

    inside_cavity = not_peri & (~outside_bg)
    if not inside_cavity.any():
        return pd.DataFrame()

    touch_inside = sp_nd.binary_dilation(inside_cavity, iterations=1)
    touch_outside = sp_nd.binary_dilation(outside_bg, iterations=1)

    inner_bnd = peri_mask & touch_inside
    outer_bnd = peri_mask & touch_outside
    if not inner_bnd.any() or not outer_bnd.any():
        return pd.DataFrame()

    if bbox:
        r, c = np.where(peri_mask)
        rr = slice(max(int(r.min()) - pad, 0), min(int(r.max()) + pad + 1, seg.shape[0]))
        cc = slice(max(int(c.min()) - pad, 0), min(int(c.max()) + pad + 1, seg.shape[1]))
        bb = (rr, cc)
    else:
        bb = (slice(0, seg.shape[0]), slice(0, seg.shape[1]))

    inner_bb = inner_bnd[bb]
    outer_bb = outer_bnd[bb]
    fasc_bb = fasc_mask[bb]
    fasc_lab_bb = fasc_lab[bb]

    dist_to_outer = sp_nd.distance_transform_edt(~outer_bb).astype(np.float32)

    ir, ic = np.where(inner_bb)
    thk_um = dist_to_outer[ir, ic] * float(mpp)

    # Assign each inner-boundary sample to nearest fascicle (handles airgap)
    _, (ri, ci) = sp_nd.distance_transform_edt(~fasc_bb, return_indices=True)
    nearest_fasc_lbl = fasc_lab_bb[ri, ci]
    sample_lbl = nearest_fasc_lbl[ir, ic]

    valid = sample_lbl > 0
    if not np.any(valid):
        return pd.DataFrame()

    thk_um = thk_um[valid]
    sample_lbl = sample_lbl[valid].astype(int)

    props = regionprops(fasc_lab)
    area_um2 = {}
    centroid_rc = {}
    for p in props:
        area_um2[int(p.label)] = float(p.area) * (mpp ** 2)
        centroid_rc[int(p.label)] = (float(p.centroid[0]), float(p.centroid[1]))

    rows = []
    for lbl in np.unique(sample_lbl):
        vals = thk_um[sample_lbl == lbl]
        area = area_um2.get(int(lbl), float("nan"))
        cent = centroid_rc.get(int(lbl), (float("nan"), float("nan")))

        if area >= 300:  # filter out tiny fascicles (diameter < ~20µm) that are unlikely to have reliable thickness measurements
            rows.append({
                "Fascicle Label": int(lbl),
                "Inner-Referenced Perineurium Median Thickness (µm)": float(np.median(vals)),
                "Inner-Referenced Minimum Thickness (µm)": float(np.min(vals)),
                "Inner-Referenced Maximum Thickness (µm)": float(np.max(vals)),
                "Fascicle Area (µm²)": float(area),
                "Fascicle Effective Diameter (µm)": float(effective_diameter(area)) if np.isfinite(area) else float("nan"),
                "N inner-boundary samples": int(vals.size),
                "Centroid Row (px)": float(cent[0]),
                "Centroid Col (px)": float(cent[1]),
            })

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# Matching GT fascicles to Pred fascicles by nearest centroid
# -------------------------------------------------------------
def match_by_nearest_centroid(gt_df, pr_df, max_dist_px=200.0):
    if gt_df.empty:
        return pd.DataFrame()

    if pr_df.empty:
        out = gt_df.copy()
        out["matched_pred_label"] = np.nan
        out["match_dist_px"] = np.nan
        for col in [
            "Inner-Referenced Perineurium Median Thickness (µm)",
            "Inner-Referenced Minimum Thickness (µm)",
            "Inner-Referenced Maximum Thickness (µm)",
        ]:
            out[col + " (pred)"] = np.nan
        return out

    pr_cent = pr_df[["Fascicle Label", "Centroid Row (px)", "Centroid Col (px)"]].to_numpy()
    pr_used = set()

    rows = []
    for _, g in gt_df.iterrows():
        gr, gc = g["Centroid Row (px)"], g["Centroid Col (px)"]
        d = np.sqrt((pr_cent[:, 1] - gr) ** 2 + (pr_cent[:, 2] - gc) ** 2)

        order = np.argsort(d)
        best = None
        for idx in order:
            lbl = int(pr_cent[idx, 0])
            if lbl in pr_used:
                continue
            best = idx
            break

        out = g.to_dict()
        if best is None or float(d[best]) > max_dist_px:
            out["matched_pred_label"] = np.nan
            out["match_dist_px"] = np.nan
            out["Inner-Referenced Perineurium Median Thickness (µm) (pred)"] = np.nan
            out["Inner-Referenced Minimum Thickness (µm) (pred)"] = np.nan
            out["Inner-Referenced Maximum Thickness (µm) (pred)"] = np.nan
        else:
            pr_used.add(int(pr_cent[best, 0]))
            pr_row = pr_df.loc[pr_df["Fascicle Label"] == int(pr_cent[best, 0])].iloc[0]
            out["matched_pred_label"] = int(pr_cent[best, 0])
            out["match_dist_px"] = float(d[best])
            out["Inner-Referenced Perineurium Median Thickness (µm) (pred)"] = float(
                pr_row["Inner-Referenced Perineurium Median Thickness (µm)"]
            )
            out["Inner-Referenced Minimum Thickness (µm) (pred)"] = float(
                pr_row["Inner-Referenced Minimum Thickness (µm)"]
            )
            out["Inner-Referenced Maximum Thickness (µm) (pred)"] = float(
                pr_row["Inner-Referenced Maximum Thickness (µm)"]
            )

        rows.append(out)

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# Evaluate folder
# -------------------------------------------------------------
def evaluate_folder_with_thickness_and_fascicles(
    pred_dir,
    gt_dir,
    *,
    mpp=0.172,
    fasc_label=1,
    peri_label=2,
    max_match_dist_px=200.0,
    output_csv="evaluation_with_thickness_and_fascicles.csv",
):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif*")))
    all_rows = []

    for pred_path in tqdm(pred_files):
        name = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            print(f"Missing GT for {name}")
            continue

        pred = tiff.imread(pred_path)
        gt = tiff.imread(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {name} pred {pred.shape} vs gt {gt.shape}")

        # --- binary masks
        pred_peri = (pred == peri_label)
        gt_peri = (gt == peri_label)
        pred_fasc = (pred == fasc_label)
        gt_fasc = (gt == fasc_label)

        # --- segmentation metrics
        row_img = {
            "image": name,

            # Perineurium metrics
            "peri_dice": dice_score(gt_peri, pred_peri),
            "peri_iou": iou_score(gt_peri, pred_peri),
            "peri_hd95": safe_hd95(gt_peri, pred_peri),
            "peri_assd": safe_assd(gt_peri, pred_peri),

            # Fascicle metrics
            "fasc_dice": dice_score(gt_fasc, pred_fasc),
            "fasc_iou": iou_score(gt_fasc, pred_fasc),
            "fasc_hd95": safe_hd95(gt_fasc, pred_fasc),
            "fasc_assd": safe_assd(gt_fasc, pred_fasc),
        }

        # --- fascicle counts (connected components)
        n_gt_cc = count_components(gt_fasc)
        n_pr_cc = count_components(pred_fasc)
        row_img.update({
            "n_gt_fascicles_cc": int(n_gt_cc),
            "n_pred_fascicles_cc": int(n_pr_cc),
            "delta_fascicles_cc": int(n_pr_cc - n_gt_cc),
        })

        # --- thickness per fascicle (GT and Pred)
        gt_th = analyse_slice_inner_boundary_thickness(
            gt, mpp=mpp, fasc_label=fasc_label, peri_label=peri_label, bbox=True
        )
        pr_th = analyse_slice_inner_boundary_thickness(
            pred, mpp=mpp, fasc_label=fasc_label, peri_label=peri_label, bbox=True
        )

        matched = match_by_nearest_centroid(gt_th, pr_th, max_dist_px=max_match_dist_px)
        matched["image"] = name

        # thickness errors per fascicle
        matched["err_median_um"] = (
            matched["Inner-Referenced Perineurium Median Thickness (µm) (pred)"]
            - matched["Inner-Referenced Perineurium Median Thickness (µm)"]
        )
        matched["abs_err_median_um"] = matched["err_median_um"].abs()

        # aggregate thickness errors per image
        mae_med = float(matched["abs_err_median_um"].mean(skipna=True)) if len(matched) else np.nan
        rmse_med = float(np.sqrt(np.nanmean(matched["err_median_um"] ** 2))) if len(matched) else np.nan

        # area-weighted MAE (prevents tiny fascicles dominating)
        w = matched["Fascicle Area (µm²)"].to_numpy(dtype=float)
        e = matched["abs_err_median_um"].to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(e) & (w > 0)
        wmae_med = float(np.average(e[ok], weights=w[ok])) if np.any(ok) else np.nan

        row_img.update({
            # thickness-table counts (note: these are the fascicles that have measurable inner boundary samples)
            "n_gt_fascicles_thickness": int(len(gt_th)) if gt_th is not None else 0,
            "n_matched_fascicles_thickness": int(matched["matched_pred_label"].notna().sum()) if len(matched) else 0,

            "thickness_mae_median_um": mae_med,
            "thickness_rmse_median_um": rmse_med,
            "thickness_wmae_median_um": wmae_med,
        })

        all_rows.append(row_img)

        # Optional: save fascicle-level table (append)
        # fasc_csv = f"{os.path.splitext(output_csv)[0]}__fascicle_level.csv"
        # matched.to_csv(
        #     fasc_csv,
        #     mode="a",
        #     header=not os.path.exists(fasc_csv),
        #     index=False,
        # )

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)

    print("\n===== IMAGE-LEVEL SUMMARY (mean) =====")
    cols = [
        "peri_dice", "peri_iou", "peri_hd95", "peri_assd",
        "fasc_dice", "fasc_iou", "fasc_hd95", "fasc_assd",
        "n_gt_fascicles_cc", "n_pred_fascicles_cc", "delta_fascicles_cc",
        "thickness_mae_median_um", "thickness_rmse_median_um", "thickness_wmae_median_um",
    ]
    print(df[cols].mean(numeric_only=True))

    return df


if __name__ == "__main__":
    pred_dir = "/home/fm/Documents/he_seg_pipeline/nnUNet/nnUNet_raw/output_952boundary"
    gt_dir = "/home/fm/Documents/he_seg_pipeline/nnUNet/nnUNet_raw/labelsTs"

    evaluate_folder_with_thickness_and_fascicles(
        pred_dir,
        gt_dir,
        # mpp=0.172,
        mpp = 0.688, # cause i ran this on downsampled images
        fasc_label=1,
        peri_label=2,
        max_match_dist_px=200.0,
        output_csv="/home/fm/Documents/he_seg_pipeline/nnUNet/nnUNet_raw/boundary_evaluation.csv",
    )

