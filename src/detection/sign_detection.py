import io
import glob
import matplotlib.pyplot as plt

from skimage.io import imread

from find_patches import binary_filtering, extract_ROIs
from find_circles import detect_circles, draw_circles
from classify_sign import classify_circles

def detect_signs(image):
    # Find patches
    segmented_image = binary_filtering(image)
    patches = extract_ROIs(segmented_image)
    
    # Detect circles
    circles = detect_circles(image, patches)
    
    # Classify signs
    recognized_signs = classify_circles(image, circles)
    
    # Generate output image
    output = io.BytesIO()
    draw_circles(image, circles, output)
    output.seek(0)
    
    # Return output image and classification result
    return output, recognized_signs

# Test on images
if __name__ == "__main__":
    for filename in glob.glob('../../data/streetview_images/*.jpg'):
        # Read image and perform sign detection
        image = imread(filename)
        output, recognized_signs = detect_signs(image)
        
        # Show results
        print recognized_signs
        
        result_image = imread(output)
        plt.imshow(result_image)
        plt.show()
