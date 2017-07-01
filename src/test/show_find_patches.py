import glob
import matplotlib.pyplot as plt
import os
import numpy as np

from skimage.io import imread
from find_patches import binary_filtering, extract_ROIs

if __name__ == "__main__":
    for filename in glob.glob('../../data/streetview_images/*.jpg'):
        # Read image and perform segmentation
        image = imread(filename)
        segmented_image = binary_filtering(image)
        patches = extract_ROIs(segmented_image)
        
        # Generate image with extracted patches
        mask = np.ones(image.shape[:2], dtype=bool)
        
        for patch in patches:
            mask[patch[0]:patch[1], patch[2]:patch[3]] = False
        
        patches_image = image.copy()
        patches_image[mask,:] = (0, 0, 0)
        
        # Display results
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax = axes.ravel()
        
        ax[0].imshow(image)
        ax[0].set_title('Original image')
        
        ax[1].imshow(segmented_image, cmap='gray')
        ax[1].set_title('Segmented image')
        
        ax[2].imshow(patches_image)
        ax[2].set_title('Extracted patches')
        
        # Disable axes
        for i in range(0, 3):
            ax[i].axis('off')
            
        # Show results
        fig.tight_layout()
        plt.gcf().canvas.set_window_title(os.path.basename(filename))
        plt.show()
