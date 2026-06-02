# Import necessary libraries
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from tifffile import imread
import dask.array as da
from skimage import measure, morphology
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd 
import os
import zarr
import cupy as cp
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
import math


# Load TIFF image
def load_image(path):
    img_store = imread(path, aszarr=True)
    image = zarr.open(img_store, mode='r')
    image = image.astype(np.uint8)
    return np.array(image)



def compute_distance_transform_dask(mask):
    mask_da = da.from_array(mask, chunks=(mask.shape[0] // 4, mask.shape[1] // 4))
    distance_map = mask_da.map_blocks(distance_transform_edt, dtype=float)
    return distance_map.compute()



def compute_distance_transform_gpu(mask):
    # Convert to GPU array
    mask_gpu = cp.asarray(mask)
    # Compute distance transform
    distance_map_gpu = distance_transform_edt(mask_gpu)
    # Convert back to CPU array
    distance_map = cp.asnumpy(distance_map_gpu)
    return distance_map

## approximate
import cv2

def compute_distance_transform_opencv(mask):
    # Convert to uint8 if not already
    mask = mask.astype(np.uint8)
    # Compute distance transform (use DIST_L2 for Euclidean distance)
    distance_map = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
    return distance_map


import cv2

def skeletonize_opencv(mask):
    # Convert to uint8 if not already
    mask = mask.astype(np.uint8) * 255
    # Perform morphological thinning (similar to skeletonization)
    skeleton = cv2.ximgproc.thinning(mask)
    return skeleton > 0

from skimage.morphology import medial_axis

def skeletonize_medial_axis(mask):
    # Use medial axis transform as an alternative to skeletonization
    skeleton, _ = medial_axis(mask, return_distance=True)
    return skeleton


# Function to calculate effective diameter
def calculate_effective_diameter(area, microns_per_pixel=0.172):
    """
    Calculate effective diameter as the diameter of a circle with equivalent area.
    """
    diameter_pixels = (4 * area / np.pi) ** 0.5
    return diameter_pixels * microns_per_pixel


# Function to analyze perineurium properties
def analyze_perineurium(perineurium_labels, path, microns_per_pixel=0.172):
    results = []
    peri_masks = []
    peri_thickness = []
    i = 0
    for region in regionprops(perineurium_labels):
        area = region.area * microns_per_pixel**2
        # area >  20microns
        if area > 20:
            i += 1
            perineurium_label = region.label

            # Create binary mask for current perineurium
            current_perineurium_mask = perineurium_labels == perineurium_label

            # Compute distance transform
            # distance_map = distance_transform_edt(current_perineurium_mask)
            distance_map = compute_distance_transform_opencv(current_perineurium_mask)

            # Compute skeleton
            # skeleton = skeletonize(current_perineurium_mask)
            skeleton = skeletonize_medial_axis(current_perineurium_mask)

            # Extract thickness values
            thickness_values = distance_map[skeleton]
            avg_thickness = np.mean(thickness_values) * microns_per_pixel
            min_thickness = np.min(thickness_values) * microns_per_pixel
            max_thickness = np.max(thickness_values) * microns_per_pixel

            # Compute perimeter
            contours = measure.find_contours(current_perineurium_mask, level=0.5)
            outer_contour = max(contours, key=lambda x: len(x))
            outer_perimeter = np.sum(np.sqrt(np.sum(np.diff(outer_contour, axis=0) ** 2, axis=1))) * microns_per_pixel
            effective_diameter = calculate_effective_diameter(area, microns_per_pixel)
            peri_masks.append(current_perineurium_mask)
            print('done')
            
            results.append({
                "Perineurium Label": i,
                "Area (microns²)": area,
                "Effective Diameter (microns)": effective_diameter,
                "Average Thickness (microns)": avg_thickness,
                "Minimum Thickness (microns)": min_thickness,
                "Maximum Thickness (microns)": max_thickness,
                "Perimeter (microns)": outer_perimeter
            })
            plt.figure
            plt.subplot(1, 2, 1)
            plt.title(f"Perineurium Component {i}")
            plt.imshow(distance_map, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title("Thickness Map")
            plt.imshow(distance_map, cmap="hot")
            plt.colorbar(label="Thickness (pixels)")
            # plt.title("Thickness Map")
            # plt.imshow(distance_map, cmap="hot")
            # plt.colorbar(label="Thickness (pixels)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(rf"{path}/perineuirum_{i}.png")
            plt.close()

    # # Add results
    # # Save the perineurium component as an image
    # # Create subplots for all perineurium
    # num_peri = len(peri_masks)
    # cols = 4  # Number of columns in the subplot grid
    # rows = math.ceil(num_peri / cols)

    # fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # axes = axes.flatten()

    # for idx, (mask, ax) in enumerate(zip(peri_masks, axes)):
    #     ax.imshow(mask, cmap='gray')
    #     ax.set_title(f"Perinuerium {idx + 1}")
    #     ax.axis("off")

    # # Remove unused subplots if any
    # for ax in axes[num_peri:]:
    #     ax.axis("off")

    # # Save the subplot figure
    # output_path = os.path.join(path, "perineurium_overview.png")
    # plt.tight_layout()
    # plt.savefig(output_path)
    plt.close()
    print('perineurium analysis done')
    return pd.DataFrame(results)



# Function to analyze fascicle properties
def analyze_fascicles(fascicle_labels, path, microns_per_pixel=0.172):
    results = []
    fascicle_masks = []
    i = 0
    for region in regionprops(fascicle_labels):
        area = region.area * microns_per_pixel**2
        current_fascicle_mask = fascicle_labels == region.label
        # if area > 20 micron
        if area > 20:
            i +=1
            results.append({
                "Fascicle Label": i,
                "Area (microns²)": area,
            })
            fascicle_masks.append(current_fascicle_mask)
    
    # Create subplots for all fascicles
    num_fascicles = len(fascicle_masks)
    cols = 4  # Number of columns in the subplot grid
    rows = math.ceil(num_fascicles / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (mask, ax) in enumerate(zip(fascicle_masks, axes)):
        ax.imshow(mask, cmap='gray')
        ax.set_title(f"Fascicle {idx + 1}")
        ax.axis("off")

    # Remove unused subplots if any
    for ax in axes[num_fascicles:]:
        ax.axis("off")

    # Save the subplot figure
    output_path = os.path.join(path, "fascicles_overview.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print('Fascicle analysis done')
    return pd.DataFrame(results)


# Plotting function for fascicles
def plot_fascicle_results(fascicle_results, path):
    plt.figure(figsize=(10, 6))
    plt.bar(fascicle_results["Fascicle Label"], fascicle_results["Area (microns²)"])
    plt.xlabel("Fascicle Label")
    plt.ylabel("Area (microns²)")
    plt.title("Fascicle Areas")
    plt.tight_layout()
    plt.savefig(rf'{path}/fascicle_areas_plot.png')
    plt.close()
    # plt.show()

# Plotting function for perineurium
def plot_perineurium_results(perineurium_results, path):
    # Plot average thickness
    plt.figure(figsize=(10, 6))
    plt.bar(perineurium_results["Perineurium Label"], perineurium_results["Average Thickness (microns)"])
    plt.xlabel("Perineurium Label")
    plt.ylabel("Average Thickness (microns)")
    plt.title("Perineurium Average Thickness")
    plt.tight_layout()
    plt.savefig(rf'{path}/perineurium_thickness_plot.png')
    plt.close()
    # plt.show()

    # Plot perineurium areas
    plt.figure(figsize=(10, 6))
    plt.bar(perineurium_results["Perineurium Label"], perineurium_results["Area (microns²)"])
    plt.xlabel("Perineurium Label")
    plt.ylabel("Area (microns²)")
    plt.title("Perineurium Areas")
    plt.tight_layout()
    plt.savefig(rf'{path}/perineurium_areas_plot.png')
    plt.close()
    # plt.show()

    # Plot perineurium perimeter
    plt.figure(figsize=(10, 6))
    plt.bar(perineurium_results["Perineurium Label"], perineurium_results["Perimeter (microns)"])
    plt.xlabel("Perineurium Label")
    plt.ylabel("Perimeter (microns)")
    plt.title("Perineurium Perimeter")
    plt.tight_layout()
    plt.savefig(rf'{path}/perineurium_perimeter_plot.png')
    plt.close()
    # plt.show() 


from sklearn.linear_model import LinearRegression
import numpy as np

def plot_thickness_vs_diameter_with_fit(perineurium_results, path):
    """
    Plot average perineurium thickness vs effective diameter with a best-fit line.
    """
    # Extract data for the plot
    diameters = perineurium_results["Effective Diameter (microns)"].values.reshape(-1, 1)
    avg_thickness = perineurium_results["Average Thickness (microns)"].values
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(diameters, avg_thickness)
    thickness_pred = model.predict(diameters)
    # Plot data points
    plt.figure(figsize=(8, 6))
    plt.scatter(diameters, avg_thickness, color="blue", alpha=0.7, edgecolor="black", label="Data")
    # Plot best-fit line
    plt.plot(diameters, thickness_pred, color="red", linewidth=2, label=f"Best Fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
    # Customize plot
    plt.title("Perineurium Thickness vs Diameter")
    plt.xlabel("Effective Diameter (microns)")
    plt.ylabel("Average Thickness (microns)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(rf'{path}/thickness_vs_diameter_with_fit.png')  # Save the plot
    # plt.show()



# Load image

src_dir = rf"/media/fmallika/10TB/Mallika/autoseg/for_analysis"
img_name = rf"SR031-CR1-05fside2-140um-HE-20241029-s1"
image_path = rf'{src_dir}/{img_name}/{img_name}_SEG.tiff'

segmentation_mask = load_image(image_path)

# Separate labels for fascicles and perineurium
fascicle_labels = label(segmentation_mask == 1)
perineurium_labels = label(segmentation_mask == 2)

# # Count the number of components
num_fascicles = len(regionprops(fascicle_labels))
num_perineurium = len(regionprops(perineurium_labels))

print(f"Number of Fascicles: {num_fascicles}")
print(f"Number of Perineurium Components: {num_perineurium}")

# Analyze properties
fascicle_results = analyze_fascicles(fascicle_labels, rf'{src_dir}/{img_name}')

plot_fascicle_results(fascicle_results, rf'{src_dir}/{img_name}')

# Save results to CSV
fascicle_results.to_csv(rf'{src_dir}/{img_name}/fascicle_results.csv', index=False)

perineurium_results = analyze_perineurium(perineurium_labels, rf'{src_dir}/{img_name}')

# Generate plots
perineurium_results.to_csv(rf'{src_dir}/{img_name}/perineurium_results.csv', index=False)
plot_perineurium_results(perineurium_results, rf'{src_dir}/{img_name}')

# Assuming `perineurium_results` contains the required columns:
# "Effective Diameter (microns)" and "Average Thickness (microns)"
plot_thickness_vs_diameter_with_fit(perineurium_results, rf"{src_dir}/{img_name}")

# Display results
print("\nFascicle Results:")
print(fascicle_results)
print("\nPerineurium Results:")
print(perineurium_results)

print('DONE! MOVE ON!')






# import matplotlib.pyplot as plt
# import numpy as np

# # Example Data: Replace with your actual data
# sub_data = {
#     "C1": {"fascicle_count": 10, "total_area": 2000},
#     "T1": {"fascicle_count": 8, "total_area": 2500},
# }

# # Extract data
# subs = list(sub_data.keys())
# fascicle_counts = [sub_data[sub]["fascicle_count"] for sub in subs]
# total_areas = [sub_data[sub]["total_area"] for sub in subs]

# # Fascicle count figure
# plt.figure(figsize=(8, 6))
# plt.bar(subs, fascicle_counts, color='blue', alpha=0.8)
# plt.xlabel('Subregions')
# plt.ylabel('Fascicle Count')
# plt.title('Fascicle Count by Subregion')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig('/path/to/fascicle_count_plot.png')  # Save the fascicle count plot
# plt.show()

# # Total area figure
# plt.figure(figsize=(8, 6))
# plt.bar(subs, total_areas, color='orange', alpha=0.8)
# plt.xlabel('Subregions')
# plt.ylabel('Total Area (microns²)')
# plt.title('Total Fascicle Area by Subregion')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# # Save and show the plot
# plt.tight_layout()
# plt.savefig('/media/fmallika/10TB/Mallika/autoseg/for_analysis/fascicle_count_and_area_plot.png')
# plt.show()





