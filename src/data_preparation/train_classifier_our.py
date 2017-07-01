import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC

import prepare_data

#
def get_skimage_features():
    # Only extract hog features from images if features and labels files are not present on disk.
    if (not os.path.isfile('hog_features.npy')) or (not os.path.isfile('truth_labels.npy')):
        prepare_data.process_images()
    # Load the dataset
    X = np.load('hog_features.npy')
    y = np.load('truth_labels.npy')

    return X, y

def get_our_features():
    # Only extract hog features from images if features and labels files are not present on disk.
    path_to_features = 'our_hog_features.npy'
    path_to_labels = 'our_truth_labels.npy'
    if (not os.path.isfile(path_to_features)) or (not os.path.isfile(path_to_labels)):
        prepare_data.process_images(feature_source='our')
    # Load the dataset
    X = np.load(path_to_features)
    y = np.load(path_to_labels)

    return X, y

def standardize_dataset(X):
    # Standardize dataset. Both RBF kernel approach of SVC as linear SVM classifiers expect
    # standardized data: http://scikit-learn.org/stable/modules/preprocessing.html
    return preprocessing.scale(X)

def create_linear_svc_classifier():
    clf = LinearSVC(verbose=2, max_iter=100000)
    return clf

def create_random_forest_classifier(n_estimators=10):
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=2)
    return clf

# Evaluate classifier by performing stratified k-fold crossvalidation.
def evaluate_classifier(clf, name, X, y, nr_folds=5):
    class_names = np.unique(y)
    # stratified k-fold crossvalidation
    skf = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=13)
    print('%d-Fold Cross Validation:' % nr_folds)
    iteration = 1
    for train, test in skf.split(X,y):
        print_data_information(X[train], y[train], iteration, which_set='Train')
        print_data_information(X[test], y[test], iteration, which_set='Test')

        y_pred = clf.fit(X[train,:], y[train]).predict(X[test])

        # Save predictions to disk
        np.save(name+'/'+name+'cross_val_predictions_fold-%d' % iteration, y_pred)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y[test], y_pred)
        # Save confusion matrix to disk
        np.save(name+'/'+'cnf_mat_fold-%d' % iteration, cnf_matrix)

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure(num=None, figsize=(8,6), dpi=64)
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix - Fold: %d' % iteration)
        plt.savefig(name+'/'+name+'cnf_mat fold-%d.png' % iteration, bbox_inches='tight', dpi=64)
        # Plot normalized confusion matrix
        plt.figure(num=None, figsize=(8,6), dpi=64)
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix - Fold: %d' % iteration)
        plt.savefig(name+'/'+name + 'norm_cnf_mat fold-%d.png' % iteration, bbox_inches='tight', dpi=64)
        iteration += 1
    plt.show()

# Print helper function that prints information about the train or test data of a given iteration
# in K-fold cross validation
def print_data_information(X, y, fold_iter, which_set='Train'):
    print('Fold nr: %d' % fold_iter)
    print('  - '+which_set+'ing samples: %d' % X.shape[0])
    print('  - Samples per class:')
    labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(labels, counts):
        print('    * %s: %d' % (label, count))

# Utility code to plot confusion matrix
# Code found at: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# Adjusted it a bit to get correct colorbar range when normalized confusion matrix is chosen.
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

def create_evaluate_train_classifier(classifier='LinearSVC', hog_features='SKImage'):
    if classifier == 'LinearSVC':
        name = 'Linear_SVC_Classifier_Features_'+hog_features
        if not os.path.exists(name):
            os.makedirs(name)
        clf = create_linear_svc_classifier()
    elif classifier == 'RandomForest':
        name = 'Random_Forest_Classifier_Features_'+hog_features
        if not os.path.exists(name):
            os.makedirs(name)
        clf = create_random_forest_classifier(n_estimators=100)
    else:
        name = 'Random_Forest_Classifier_Features_'+hog_features
        if not os.path.exists(name):
            os.makedirs(name)
        clf = create_random_forest_classifier(n_estimators=100)

    # load data
    if hog_features == 'SKImage':
        X, y = get_skimage_features()
    else:
        X, y = get_our_features()

    # Standardize dataset
    X = standardize_dataset(X)

    # Evaluate classifier
    # evaluate_classifier(clf, 'Linear SVC', X, y, nr_folds=5)
    evaluate_classifier(clf, name, X, y, nr_folds=5)

    # Train classifier on full set
    clf.fit(X, y)

    # Save classifier to disk
    # joblib.dump(clf, 'Linear_SVC_Classifier')
    joblib.dump(clf, name + '/' + name)

if __name__ == '__main__':
    create_evaluate_train_classifier(classifier='LinearSVC', hog_features='Our')