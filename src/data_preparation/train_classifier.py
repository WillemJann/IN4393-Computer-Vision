import numpy as np
import os.path
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import prepare_data

# Only extract hog features from images if features and labels files are not present on disk.
if not os.path.isfile('hog_features.npy') or not os.path.isfile('truth_labels.npy'):
    prepare_data.process_images()
# Load the dataset
X = np.load('hog_features.npy')
y = np.load('truth_labels.npy')

# Split dataset in test/train part
# split train: 0.8 test: 0.2
# stratify: y (makes sure the set is balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)

# Lazy initialization:
#  if exists, load classifier from disk.
#  else, train classifier
classifier_filename = 'svc_clf.pkl'
if os.path.isfile(classifier_filename.pkl):
    clf = joblib.load(classifier_filename)
else:
    # Train Support Vector Machine classifier
    clf = SVC(verbose=True)
    clf.fit(X_train, y_train)

    # Save classifier to disk
    joblib.dump(clf, classifier_filename)

# Evaluate classifier
