import operator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.spatial.distance import euclidean
from skimage.color import rgb2hsv
from skimage.morphology import binary_dilation, binary_erosion, binary_closing, disk, remove_small_objects
from skimage.transform import hough_circle, hough_circle_peaks

def binary_filtering(patch):
    # Convert patch to HSV
    image_hsv = rgb2hsv(patch)
    
    # Get H and S channels and fix H channel (red color)
    H = image_hsv[:,:,0]
    S = image_hsv[:,:,1]
    H[H > 0.8] = 0
    
    # Determine filter size based on image size
    image_size = max(patch.shape[0], patch.shape[1])
    filter_size = int(image_size / 15)
    
    # Get red segments
    binary_red = np.logical_and(H <= 0.05, S >= 0.3)
    red_segments = remove_small_objects(binary_red, 64)
    red_segments = binary_closing(red_segments, disk(filter_size))
    red_segments = np.logical_xor(binary_dilation(red_segments), binary_erosion(red_segments))
    
    # Get blue segments
    binary_blue = np.logical_and(np.logical_and(H >= 0.55, H <= 0.65), S >= 0.4)
    blue_segments = remove_small_objects(binary_blue, 64)
    blue_segments = binary_closing(blue_segments, disk(filter_size))
    blue_segments = np.logical_xor(binary_dilation(blue_segments), binary_erosion(blue_segments))
    
    return red_segments, blue_segments

def find_circles(binary_patch):
    # Get size of the patch
    min_image_size = min(binary_patch.shape[0], binary_patch.shape[1])
    max_image_size = max(binary_patch.shape[0], binary_patch.shape[1])
    
    # Determine min and max radius based on patch size
    min_radius = max(10, int(min_image_size / 4))
    max_radius = int(max_image_size / 2)
    hough_radii = np.arange(min_radius, max_radius)
    
    # Apply Hough circle transform
    hough_res = hough_circle(binary_patch, hough_radii)
    
    # Set parameters for hough_circle_peaks
    min_distance = int(max_image_size / 3)
    threshold = 0.55
    num_peaks = np.inf
    total_num_peaks = 5
    normalize = True
    
    # Filter result of Hough circle transform based on parameters
    _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_distance, min_distance, threshold, num_peaks, total_num_peaks, normalize)
    detected_circles = zip(cx, cy, radii)
    
    # Return found circles
    return detected_circles
    
def apply_clustering(circles, cluster_size = 20):
    clusters = []
    
    for circle in circles:
        distance = None
        
        # Add current circle to an existing cluster if it is nearby this cluster
        for cluster in clusters:
            cluster_mean = np.mean(np.array(cluster)[:,:2], axis=0)
            distance = euclidean(circle[:2], cluster_mean)
            
            if (distance <= cluster_size):
                cluster.append(circle)
                break
        
        # Create new cluster if circle is not close to an existing cluster
        if distance is None or distance > cluster_size:
            clusters.append([circle])
    
    # Select circle with largest radius (per cluster)
    circles = []
    
    for cluster in clusters:
        largest_circle = max(cluster, key=operator.itemgetter(2))
        circles.append(largest_circle)
    
    # Return clustered circles
    return circles

def detect_circles(image, patch_coords):
    result = []
    
    for coords in patch_coords:
        # Extract patch from original image
        patch = image[coords[0]:coords[1], coords[2]:coords[3], :]
        
        # Apply red and blue segmentation on patch
        red_patch, blue_patch = binary_filtering(patch)
        
        # Find circles in red and blue segmented images
        red_circles = find_circles(red_patch)
        blue_circles = find_circles(blue_patch)
        
        # Divide detected circles in clusters and select circle with largest radius (per cluster)
        detected_circles = apply_clustering(red_circles + blue_circles)
        
        # Add found circles to the result
        for circle_y, circle_x, radius in detected_circles:
            circle = (circle_y + coords[0], circle_x + coords[2], radius)
            result.append(circle)
    
    return result

def draw_circles(image, circles, output):
    # Initialize figure
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Add image to figure
    ax.imshow(image, aspect='normal')
    
    # Add circles to figure
    i = 1
    
    for center_y, center_x, radius in circles:
        x = center_x - radius
        y = center_y - radius
        width = radius * 2
        height = radius * 2
        
        rectangle = mpatches.Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rectangle)
        ax.text(x - 5, y, i, color='red', horizontalalignment='right', verticalalignment='top')
        
        i = i + 1
    
    # Write figure to output bufer
    fig.savefig(output, format='jpeg', dpi=80)
