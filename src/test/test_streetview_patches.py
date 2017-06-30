import glob
import matplotlib.pyplot as plt
import os

from skimage.draw import circle_perimeter
from skimage.io import imread
from find_circles import binary_filtering, find_circles, apply_clustering
    
if __name__ == "__main__":
    for filename in glob.glob('../../data/streetview_patches/positive/*.jpg'):
        # Read patch
        image = imread(filename)
        
        # Perform circle detection
        red_patch, blue_patch = binary_filtering(image)
        red_circles = find_circles(red_patch)
        blue_circles = find_circles(blue_patch)
        detected_circles = apply_clustering(red_circles + blue_circles)
        
        # Draw circles
        result_image = image
        for center_y, center_x, radius in detected_circles:
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
            result_image[circy, circx] = (220, 20, 20)
        
        # Show results
        print detected_circles
        
        plt.imshow(result_image)
        plt.gcf().canvas.set_window_title(os.path.basename(filename))
        plt.show()
