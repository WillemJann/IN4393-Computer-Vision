import glob
import matplotlib.pyplot as plt
import os

from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from skimage.io import imread

from train_classifier_clean import HOG_HIST_NR_OF_BINS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM

if __name__ == "__main__":
    # Initialize figures
    fig1, axes1 = plt.subplots(nrows=4, ncols=6, figsize=(18, 12), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    fig2, axes2 = plt.subplots(nrows=4, ncols=6, figsize=(18, 12), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax1 = axes1.ravel()
    ax2 = axes2.ravel()
    i = 0
    
    for filename in glob.glob('../../data/dataset_clean/*.png'):
        # Extract label from filename
        label = os.path.splitext(os.path.basename(filename))[0]
        
        # Read image
        image = imread(filename)
        image = rgba2rgb(image, background=(1,1,1))
        
        # Generate HOG image
        _, hog_image = hog(rgb2gray(image), HOG_HIST_NR_OF_BINS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, transform_sqrt=True, visualise=True)
        
        # Add both images to figures
        ax1[i].imshow(image)
        ax1[i].set_title(label)
        #ax1[i].axis('off')
        
        ax2[i].imshow(hog_image, cmap='gray')
        ax2[i].set_title(label)
        #ax2[i].axis('off')
        
        i = i + 1
    
    # Show results
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
