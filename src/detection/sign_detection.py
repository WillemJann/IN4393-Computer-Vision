from io import BytesIO

from find_patches import binary_filtering, extract_ROIs
from find_circles import detect_circles, draw_circles

def detect_signs(image):
    # Find patches
    segmented_image = binary_filtering(image)
    patches = extract_ROIs(segmented_image)
    
    # Detect circles
    circles = detect_circles(image, patches)
    
    # Classify signs
    recognized_signs = {}
    
    # Generate output image
    output = BytesIO()
    draw_circles(image, circles, output)
    output.seek(0)
    
    # Return output image and classification result
    return output, recognized_signs
