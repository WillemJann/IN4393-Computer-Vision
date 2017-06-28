import numpy as np

from skimage import draw
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.externals import joblib

from train_classifier import get_hog_features

TARGET_SIZE = (75,75)
CLASSIFIER_FILE = '../../data/classifiers/linear_svc_dataset_training.pkl'

# Returns list with classification result for each circle in given image
def classify_circles(image, circles):
    results = []
    
    # Load classifier
    classifier = joblib.load(CLASSIFIER_FILE)
    
    for circle in circles:
        # Generate feature vector for current circle
        circle_image = crop_circle(image, circle)
        circle_image = resize(circle_image, TARGET_SIZE)
        sample = get_hog_features(circle_image)
        
        # Perform classification
        label = classifier.predict(sample.reshape(1,-1))
        
        # Add to results
        x1, y1, x2, y2 = calculate_bounding_box(circle)
        result = {'label': label[0], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        results.append(result)
    
    return results

# Returns cropped circle from image, with white background as mask
def crop_circle(image, circle):
    cropped_image = rgb2gray(image)
    
    circy, circx = draw.circle(circle[0], circle[1], circle[2], shape=image.shape)
    mask = np.ones(cropped_image.shape, dtype=bool)
    mask[circy, circx] = False
    
    cropped_image[mask] = 0
    
    x1, y1, x2, y2 = calculate_bounding_box(circle)
    cropped_image = cropped_image[y1:y2, x1:x2]
    
    return cropped_image

def calculate_bounding_box(circle):
    center_y, center_x, radius = circle
    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius
    y2 = center_y + radius
    return x1, y1, x2, y2
