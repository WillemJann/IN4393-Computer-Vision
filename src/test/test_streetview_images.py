import glob
import matplotlib.pyplot as plt
import os

from skimage.io import imread
from sign_detection import detect_signs

if __name__ == "__main__":
    for filename in glob.glob('../../data/streetview_images/*.jpg'):
        # Read image and perform sign detection
        image = imread(filename)
        output, recognized_signs = detect_signs(image)
        
        # Show results
        print recognized_signs
        
        result_image = imread(output)
        plt.imshow(result_image)
        plt.gcf().canvas.set_window_title(os.path.basename(filename))
        plt.show()
