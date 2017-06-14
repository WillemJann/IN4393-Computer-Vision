import collections
import glob
import os
import pickle

from skimage import color, feature, io
from sklearn import svm

TRAINING_FOLDER = '../../data/circles_normalized/'
CLASSIFIER_FILE = '../../data/classifiers/svm_circles_normalized.pickle'

HOG_ORIENTATIONS = 8
HOG_CELL_SIZE = (3,3)
HOG_BLOCK_SIZE = (1,1)
HOG_BLOCK_NORM = 'L2-Hys'

# Returns dictonary with all labeled images from the training folder
def load_training_data():
    files = glob.glob(TRAINING_FOLDER + '*.png')
    data = collections.OrderedDict()
    
    for file in files:
        filename = os.path.basename(file)
        label = os.path.splitext(filename)[0]
        
        data[label] = io.imread(file)
    
    return data

# Returns dictonary with feature vector for each training image
def extract_hog_features(training_images):
    training_set = collections.OrderedDict()
    
    for label, image in training_images.iteritems():
        # Fill alpha channel with white and convert image to grayscale
        image = color.rgba2rgb(image, background=(0,0,0))
        image = color.rgb2gray(image)
        
        # Extract HOG features
        features = feature.hog(image, HOG_ORIENTATIONS, HOG_CELL_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_NORM)
        
        # Show results
        #_, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #ax1.imshow(image, cmap='gray')
        #ax2.imshow(hog_image, cmap='gray')
        #plt.show()
        
        # Add features to training set
        training_set[label] = features
    
    return training_set

# Returns a trained SVM classifier based on training set
def train_classifier(training_set):
    training_samples = training_set.values();
    labels = training_set.keys();
    
    classifier = svm.SVC()
    classifier.fit(training_samples, labels)
    
    return classifier

if __name__ == "__main__":
    # Load labeled images
    training_set = load_training_data()
    
    # Convert to feature vectors
    training_set = extract_hog_features(training_set)
    
    # Train classifier
    classifier = train_classifier(training_set)
    
    # Save classifier to disk (for later use)
    pickle.dump(classifier, open(CLASSIFIER_FILE, 'wb'), pickle.HIGHEST_PROTOCOL)
