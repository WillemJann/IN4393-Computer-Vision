import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC

import prepare_data

import inspect
print os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Utility code to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    if normalize:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '%.2f' % cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Only extract hog features from images if features and labels files are not present on disk.
if (not os.path.isfile('hog_features.npy')) or (not os.path.isfile('truth_labels.npy')):
    prepare_data.process_images()
# Load the dataset
X = np.load('hog_features.npy')
y = np.load('truth_labels.npy')
class_names = np.unique(y)

# Standardize dataset. Both RBF kernel approach of SVC as linear SVM classifiers expect
# standardized data: http://scikit-learn.org/stable/modules/preprocessing.html
standardize = True
if standardize:
    X = preprocessing.scale(X)

# Split dataset in test/train part
# split train: 0.8 test: 0.2
# stratify: y (makes sure the set is balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)

# Lazy initialization:
#  if exists, load classifier from disk.
#  else, train classifier
classifier_filename = 'linear_svc_clf.pkl'
if os.path.isfile(classifier_filename):
    clf = joblib.load(classifier_filename)
else:
    print("Train data size: %d elements of size %d" % X_train.shape)
    # Train Support Vector Machine classifier
    #clf = SVC(cache_size=7000, verbose=True)
    clf = LinearSVC(verbose=2, max_iter=100000)
    clf.fit(X_train, y_train)

    # Save classifier to disk
    joblib.dump(clf, classifier_filename)

# Evaluate classifier
y_pred = clf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(num=None, figsize=(8,6), dpi=64)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(num=None, figsize=(8,6), dpi=64)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()