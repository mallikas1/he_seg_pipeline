import numpy as np
import cv2
from skimage import measure, morphology
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd 
import os
import zarr
from pathlib import Path
import numpy as np
from tifffile import TiffFile, imread, imwrite
import ome_types
import dask.array as da
import shutil
# use nnunet env

# Load TIFF image
def load_image(path):
    # if 'ome' in path:
    img_store = imread(path, aszarr=True)
    image = zarr.open(img_store, mode='r')
    # else:
    #     image = Image.open(path)
    image = image.astype(np.uint8)
    return np.array(image)

# Extract connected components (fascicles)
def get_fascicles(binary_image):
    # Label connected components connectivity = 2 ==> 8-connectivity (8-neighboring pixels)
    labeled_image, num_fascicles = measure.label(binary_image, connectivity=2, return_num=True)
    properties = measure.regionprops(labeled_image)
    return labeled_image, properties, num_fascicles

# Analyze fascicle properties with bounding box shape fits
def analyze_fascicles(properties, binary_image):
    fascicle_data = []
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        label = prop.label
        roi_area = np.count_nonzero(binary_image == label)
        # Minimum enclosing circle radius
        min_enclosing_radius = np.sqrt(area / np.pi)
        
        # Fit ellipse if major and minor axes are available
        if prop.minor_axis_length > 0:
            ellipse_major = prop.major_axis_length
            ellipse_minor = prop.minor_axis_length
            ellipse_area = np.pi * (ellipse_major / 2) * (ellipse_minor / 2)
            
            # Determine if region fits better as circle or ellipse
            shape_fit = "Circle" if np.isclose(aspect_ratio, 1, atol=0.2) else "Ellipse"
        else:
            ellipse_major, ellipse_minor, ellipse_area, shape_fit = None, None, None, "Unknown"
        
        fascicle_data.append({
            "centroid": prop.centroid,  # Add centroid for plotting
            "Area": area,
            "Area (non-zero pixels)": roi_area,
            "Perimeter": perimeter,
            "Aspect Ratio": aspect_ratio,
            "Roundness": roundness,
            "Min Enclosing Circle Radius": min_enclosing_radius,
            "Ellipse Major Axis": ellipse_major,
            "Ellipse Minor Axis": ellipse_minor,
            "Ellipse Area": ellipse_area,
            "Best Fit Shape": shape_fit
        })
    
    df = pd.DataFrame(fascicle_data)
    return df

# Save DataFrame to JSON
def save_to_json(df, filename="fascicle_properties.json"):
    df.to_json(filename, orient="records", indent=4)


# Plot best-fit shapes on the original image
def plot_shapes(he_image, image, df, save_path):
    # Create a copy of the image to draw on
    image_copy = image.copy()
    # Iterate over each region and plot both the circle and ellipse
    for idx, row in df.iterrows():
        if row['Best Fit Shape'] !='Unknown':
            y, x = row["centroid"]  # Centroid coordinates
            # Draw minimum enclosing circle
            radius = int(row["Min Enclosing Circle Radius"])
            cv2.circle(image_copy, (int(x), int(y)), radius, (255, 0, 255), 6)  # Blue circle  
            # Draw ellipse if ellipse axes are available
            if row["Ellipse Major Axis"] and row["Ellipse Minor Axis"]:
                axes = (int(row["Ellipse Major Axis"] / 2), int(row["Ellipse Minor Axis"] / 2))
                angle = 0  # Assuming the ellipse orientation is along the major axis (adjust if needed)
                cv2.ellipse(image_copy, (int(x), int(y)), axes, angle, 0, 360, (0, 255, 0), 6)  # Green ellipse
    # Display the original and annotated images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(he_image)
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.title("Segmentation")
    plt.subplot(1, 3, 3)
    plt.imshow(image_copy)
    plt.title("Circle and Ellipse Fits")
    plt.savefig(rf"{save_path}/fascicle_analysis_cor.png")
    # plt.show()



# Main function to load, process, analyze, and save the results
def main(he_image_path, image_path, save_path):
    print(he_image_path)
    he_image = load_image(he_image_path)
    image = load_image(image_path)
    binary_image = np.zeros_like(image)
    binary_image[image == 2] = 1
    labeled_image, properties, num_fascicles = get_fascicles(binary_image)
    fascicle_df = analyze_fascicles(properties, labeled_image)
    
    print(f"Number of Fascicles: {num_fascicles}")
    print(fascicle_df)
    # Save the properties to JSON
    save_to_json(fascicle_df, rf"{save_path}/fascicle_properties_cor.json")
    plot_shapes(he_image, image, fascicle_df, save_path)
    
    
# Example usage
# src_dir = rf"/home/fmallika/Downloads"
# img_file = rf"SR018-TR1-04c06c-356um-HE-20240819-s2_SEG_SP-reviewed.tiff"
# he_img_file = rf"SR018-TR1-04c06c-356um-HE-20240819-s2.ome.tiff"
# main(rf"{src_dir}/{he_img_file}", rf"{src_dir}/{img_file}", src_dir)


he_img_file = rf"/media/fmallika/10TB/Mallika/autoseg/for_analysis/SR004-TL1-18cside-s1-48um-HE/SR004-TL1-18cside-s1-48um-HE.ome.tiff"

src_dir = rf"/media/fmallika/10TB/Mallika/autoseg/for_analysis/SR004-TL1-18cside-s1-48um-HE"
img_file = rf"SR004-TL1-18cside-s1-48um-HE-SEG.tiff"
main(rf"{he_img_file}", rf"{src_dir}/{img_file}", src_dir)


