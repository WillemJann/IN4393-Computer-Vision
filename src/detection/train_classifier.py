import glob
import os

from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from sklearn.externals import joblib
from skimage.io import imread
from sklearn import svm

HOG_HIST_NR_OF_BINS = 9
HOG_PIXELS_PER_CELL = (8,8)
HOG_CELLS_PER_BLOCK = (3,3)
HOG_BLOCK_NORM = 'L2-Hys'

TRAINING_FOLDER = '../../data/dataset_clean/'
CLASSIFIER_FOLDER = '../../data/classifiers/'
CLASSIFIER_FILE = 'svc_dataset_clean.pkl'

# Returns dictonary with all labeled images from the training folder
def load_training_data():
    files = glob.glob(TRAINING_FOLDER + '*.png')
    data = {}
    
    for file in files:
        filename = os.path.basename(file)
        label = os.path.splitext(filename)[0]
        
        data[label] = imread(file)
    
    return data

# Returns dictonary with feature vector for each training image
def extract_hog_features(training_images):
    training_set = {}
    
    for label, image in training_images.iteritems():
        # Fill alpha channel with white and convert image to grayscale
        image = rgba2rgb(image, background=(0,0,0))
        image = rgb2gray(image)
        
        # Extract HOG features
        features = get_hog_features(image)
        
        # Add features to training set
        training_set[label] = features
    
    return training_set

# Returns HOG features for a single image
def get_hog_features(image):
    return hog(image, HOG_HIST_NR_OF_BINS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, transform_sqrt=True)

# Returns a trained SVM classifier based on training set
def train_classifier(training_set):
    training_samples = training_set.values();
    labels = training_set.keys();
    
    classifier = svm.SVC()
    classifier.fit(training_samples, labels)
    
    return classifier

# Returns the trained classifier
def get_classifier():
    return joblib.load(CLASSIFIER_FOLDER + CLASSIFIER_FILE)

# Train classifier and save to disk
if __name__ == "__main__":
    training_set = load_training_data()
    training_set = extract_hog_features(training_set)
    classifier = train_classifier(training_set)
    joblib.dump(classifier, CLASSIFIER_FOLDER + CLASSIFIER_FILE)
