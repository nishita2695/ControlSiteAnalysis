# Import required libraries
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Define file paths
raster_paths = [
    'path/to/your/raster_1.tif',
    'path/to/your/raster_2.tif',
    'path/to/your/raster_3.tif'
]
shapefile_path = 'path/to/your/shapefile.shp'
output_raster_path = 'path/to/your/multiband/raster.tif'
output_clustering_path = 'path/to/your/clustered/raster.tif'
output_plot_path = "path/to/your/plot/image.jpeg"

# Load the shapefile
shapefile = gpd.read_file(shapefile_path)
shapes = [feature["geometry"] for feature in shapefile.iterfeatures()]


# Function to read, mask, and resample raster
def read_mask_resample(raster_path, target_meta):
    with rasterio.open(raster_path) as src:
        # Read the data
        data = src.read(1, masked=True).astype(np.float32)

        # Mask with shapefile
        out_image, out_transform = mask(src, shapes, crop=True)
        masked_raster_array = out_image[0].astype(np.float32)

        # Replace invalid values with NaN and filter out zeros
        invalid_value = -3.402823e+38
        masked_raster_array[(masked_raster_array == invalid_value) | (masked_raster_array == 0)] = np.nan

        # Resample to match the target metadata
        resampled_raster = np.empty(shape=(target_meta['height'], target_meta['width']), dtype=np.float32)
        reproject(
            source=masked_raster_array,
            destination=resampled_raster,
            src_transform=out_transform,
            src_crs=src.crs,
            dst_transform=target_meta['transform'],
            dst_crs=target_meta['crs'],
            resampling=Resampling.bilinear
        )

        return resampled_raster, out_transform


# Read and process the first raster to get its metadata
with rasterio.open(raster_paths[0]) as src:
    target_meta = src.meta.copy()
    target_image, target_transform = mask(src, shapes, crop=True)
    target_meta.update({
        'height': target_image.shape[1],
        'width': target_image.shape[2],
        'transform': src.transform
    })

# Read, mask, and resample all rasters
masked_rasters = [read_mask_resample(path, target_meta) for path in raster_paths]

# Unpack raster data and transformations
masked_rasters, out_transforms = zip(*masked_rasters)

# Stack the bands into a 2D array where each row is a pixel and each column is a band
combined_bands = np.stack([raster.flatten() for raster in masked_rasters], axis=-1)

# Remove NaN values
valid_pixel_mask = ~np.isnan(combined_bands).any(axis=1)
combined_bands_clean = combined_bands[valid_pixel_mask]

# Standardize the data
scaler = StandardScaler()
combined_bands_standardized = scaler.fit_transform(combined_bands_clean)

# Define weights for each variable (currently set to 1, can be adjusted later)
weights = np.ones(combined_bands_standardized.shape[1])

# Apply the weights by multiplying the standardized values
weighted_combined_bands = combined_bands_standardized * weights

# Perform k-means clustering on the weighted data
kmeans_1 = KMeans(n_clusters=6, n_init=10, random_state=0).fit(weighted_combined_bands)
clustered_values_1 = kmeans_1.labels_

# Sort clusters based on proximity to reference pixels (first run)
reference_pixels_1 = np.median(weighted_combined_bands, axis=0).reshape(1, -1)
sorted_centroids_indices_1 = np.argsort(np.linalg.norm(kmeans_1.cluster_centers_ - reference_pixels_1, axis=1))

# Sort the clusters
clustered_values_sorted_1 = np.zeros_like(clustered_values_1)
for i, index in enumerate(sorted_centroids_indices_1):
    clustered_values_sorted_1[clustered_values_1 == index] = i + 1  # Shift to start from 1

# Extend the process to include additional variables
raster_paths_extended = raster_paths + ['path/to/your/additional_raster1.tif', 'path/to/your/additional_raster2.tif']

# Read, mask, and resample all rasters including the new variables
masked_rasters_extended = [read_mask_resample(path, target_meta) for path in raster_paths_extended]

# Unpack extended raster data and transformations
masked_rasters_extended, out_transforms_extended = zip(*masked_rasters_extended)

# Stack the extended bands into a 2D array
combined_bands_extended = np.stack([raster.flatten() for raster in masked_rasters_extended], axis=-1)

# Remove NaN values
valid_pixel_mask_extended = ~np.isnan(combined_bands_extended).any(axis=1)
combined_bands_clean_extended = combined_bands_extended[valid_pixel_mask_extended]

# Standardize the data
combined_bands_standardized_extended = scaler.fit_transform(combined_bands_clean_extended)

# Apply the same weights to the extended data
weighted_combined_bands_extended = combined_bands_standardized_extended * weights

# Perform k-means clustering on the weighted extended data
kmeans_2 = KMeans(n_clusters=6, n_init=10, random_state=0).fit(weighted_combined_bands_extended)
clustered_values_2 = kmeans_2.labels_

# Sort clusters based on proximity to reference pixels (second run)
reference_pixels_2 = np.median(weighted_combined_bands_extended, axis=0).reshape(1, -1)
sorted_centroids_indices_2 = np.argsort(np.linalg.norm(kmeans_2.cluster_centers_ - reference_pixels_2, axis=1))

# Sort the clusters for the second run
clustered_values_sorted_2 = np.zeros_like(clustered_values_2)
for i, index in enumerate(sorted_centroids_indices_2):
    clustered_values_sorted_2[clustered_values_2 == index] = i + 1

# Initialize an empty array for the clustered raster (first run)
clustered_raster_flat_1 = np.full(masked_rasters[0].size, np.nan)
clustered_raster_flat_1[valid_pixel_mask] = clustered_values_sorted_1
clustered_raster_1 = clustered_raster_flat_1.reshape(masked_rasters[0].shape)

# Initialize an empty array for the clustered raster (second run)
clustered_raster_flat_2 = np.full(masked_rasters_extended[0].size, np.nan)
clustered_raster_flat_2[valid_pixel_mask_extended] = clustered_values_sorted_2
clustered_raster_2 = clustered_raster_flat_2.reshape(masked_rasters_extended[0].shape)

# Define a colormap with fixed colors for clusters
colors = ['#DAE8FC', '#B4C7E7', '#FFD966', '#F4B084', '#A9D08E', '#D9D9D9']
cmap = ListedColormap(colors)


# Calculate the extent for plotting
def calculate_extent(transform, width, height):
    return (
        transform[2],  # xmin
        transform[2] + width * transform[0],  # xmax
        transform[5] + height * transform[4],  # ymin
        transform[5]  # ymax
    )


extent_1 = calculate_extent(out_transforms[0], target_meta['width'], target_meta['height'])
extent_2 = calculate_extent(out_transforms_extended[0], target_meta['width'], target_meta['height'])

# Plot the clustered rasters side by side with latitude and longitude as axes
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# First raster
im1 = axs[0].imshow(clustered_raster_1, cmap=cmap, extent=extent_1)
axs[0].set_title('Clustered Raster (First Run)')
axs[0].set_xlabel('Easting')
axs[0].set_ylabel('Northing')

# Second raster
im2 = axs[1].imshow(clustered_raster_2, cmap=cmap, extent=extent_2)
axs[1].set_title('Clustered Raster (Second Run with Additional Variables)')
axs[1].set_xlabel('Easting')
axs[1].set_ylabel('Northing')

# Add a legend that applies to both subplots
cluster_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i + 1}') for i in range(6)]
fig.legend(handles=cluster_patches, loc='center right', bbox_to_anchor=(1.15, 0.5))

# Adjust layout
plt.tight_layout()

# Save the plot as a high-resolution JPEG with the legend
plt.savefig(output_plot_path, format='jpeg', dpi=600, bbox_inches='tight')

# Display the plot
plt.show()