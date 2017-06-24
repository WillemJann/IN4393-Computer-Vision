import numpy as np

from skimage import draw
from skimage.color import rgb2gray
from skimage.transform import resize

from train_classifier import get_classifier, get_hog_features

TARGET_SIZE = (75,75)

# Returns list with classification result for each circle in given image
def classify_circles(image, circles):
    results = []
    
    # Load classifier
    classifier = get_classifier()
    
    for circle in circles:
        # Generate feature vector for current circle
        circle_image = crop_circle(image, circle)
        circle_image = resize(circle_image, TARGET_SIZE)
        sample = get_hog_features(circle_image)
        
        # Perform classification
        label = classifier.predict(sample.reshape(1,-1))
        
        results.append((label[0], circle))
    
    return results

# Returns cropped circle from image, with white background as mask
def crop_circle(image, circle):
    cropped_image = rgb2gray(image)

    center_y, center_x, radius = circle
    x1 = center_x - radius
    x2 = center_x + radius
    y1 = center_y - radius
    y2 = center_y + radius
    
    circy, circx = draw.circle(center_y, center_x, radius, shape=image.shape)
    mask = np.ones(cropped_image.shape, dtype=bool)
    mask[circy, circx] = False
    
    cropped_image[mask] = 0
    cropped_image = cropped_image[y1:y2, x1:x2]
    
    return cropped_image
