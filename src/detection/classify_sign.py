import numpy as np

from skimage import draw
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.externals import joblib

HOG_HIST_NR_OF_BINS = 9
HOG_PIXELS_PER_CELL = (8,8)
HOG_CELLS_PER_BLOCK = (3,3)
HOG_BLOCK_NORM = 'L2-Hys'

TARGET_SIZE = (75,75)

CLASSIFIER_FILE = '../../data/classifiers/linear_svc_dataset_training.pkl'

# Returns list with classification result for each circle in given image
def classify_circles(image, circles):
    results = []
    
    # Load classifier
    classifier = joblib.load(CLASSIFIER_FILE)
    
    for circle in circles:
        # Generate feature vector for current circle
        circle_image = rgb2gray(image)
        circle_image = crop_bounding_box(circle_image, circle)
        circle_image = resize(circle_image, TARGET_SIZE)
        sample = get_hog_features(circle_image)
        
        # Perform classification
        label = classifier.predict(sample.reshape(1,-1))
        
        # Add to results
        x1, y1, x2, y2 = calculate_bounding_box(circle)
        result = {'label': label[0], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        results.append(result)
    
    return results

# Returns HOG features for a single image
def get_hog_features(image):
    return hog(image, HOG_HIST_NR_OF_BINS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, transform_sqrt=True)

# Returns cropped circle from image, with white background as mask
def crop_circle(image, circle):    
    circy, circx = draw.circle(circle[0], circle[1], circle[2], shape=image.shape)
    mask = np.ones(image.shape, dtype=bool)
    mask[circy, circx] = False
    
    image[mask] = 0
    
    return crop_bounding_box(image, circle)

def crop_bounding_box(image, circle):
    x1, y1, x2, y2 = calculate_bounding_box(circle)
    return image[y1:y2, x1:x2]

def calculate_bounding_box(circle):
    center_y, center_x, radius = circle
    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius
    y2 = center_y + radius
    return x1, y1, x2, y2
