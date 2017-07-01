import itertools
import matplotlib.pyplot as plt
import numpy as np

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
            plt.text(j, i,'%.0f' % cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def average_cnf_matrix(path, target):
    class_names = np.unique(np.load(path + target +
                                    'Random_Forest_Classifier_Features_SKImagecross_val_predictions_fold-1.npy'))
    cnf_mat_fold_1 = np.load(path + target + 'cnf_mat_fold-1.npy')
    average_cnf_mat = np.zeros(cnf_mat_fold_1.shape, dtype=np.double)
    for i in range(1, 6):
        cnf_mat = np.load(path + target + 'cnf_mat_fold-%d.npy' % i)
        average_cnf_mat += 0.2 * cnf_mat
    np.save(path + target + '_avg_cnf_mat', average_cnf_mat)
    # Plot non-normalized confusion matrix
    plt.figure(num=None, figsize=(8, 6), dpi=64)
    plot_confusion_matrix(average_cnf_mat, classes=class_names,
                          title='Average Confusion matrix')
    plt.savefig(path + target + 'avg_cnf_mat.png', bbox_inches='tight', dpi=64)
    # Plot normalized confusion matrix
    plt.figure(num=None, figsize=(8, 6), dpi=64)
    plot_confusion_matrix(average_cnf_mat, classes=class_names, normalize=True,
                          title='Average Normalized confusion matrix')
    plt.savefig(path + target + 'avg_norm_cnf_mat.png', bbox_inches='tight', dpi=64)
    plt.show()

if __name__ == '__main__':
    path = '../../results/'
    target = 'Random_Forest_Classifier_300_estimators_Features_SKImage/'

    # Calculate average confusion matrix
    average_cnf_matrix(path, target)

    # Calculate average accuracy
    avg_accuracy = 0.
    for i in range(1, 6):
        acc = np.load(path + target +
                      'Random_Forest_Classifier_Features_SKImagecross_val_predictions_fold-%d-accuracy.npy' % i)
        avg_accuracy += 0.2 * acc
    np.save(path + target + 'cross_val_predictions_fold-avg_accuracy.npy', avg_accuracy)

    # Calculate average precision
    avg_precision = np.zeros(np.load(path + target +
                             'Random_Forest_Classifier_Features_SKImagecross_val_predictions_fold-1-precision.npy').shape,
                             dtype=np.double)
    for i in range(1, 6):
        precision = np.load(path + target + 'Random_Forest_Classifier_Features_SKImagecross_val_predictions_fold-%d-precision.npy' % i)
        avg_precision += 0.2 * precision
    np.save(path + target + 'cross_val_predictions_fold-avg_precision.npy', avg_precision)

    # Calculate average recall
    avg_recall = np.zeros(np.load(path + target +
                          'Random_Forest_Classifier_Features_SKImagecross_val_predictions_fold-1-recall.npy').shape,
                          dtype=np.double)
    for i in range(1, 6):
        recall = np.load(path + target + 'Random_Forest_Classifier_Features_SKImagecross_val_predictions_fold-%d-recall.npy' % i)
        avg_recall += 0.2 * recall
    np.save(path + target + 'cross_val_predictions_fold-avg_recall.npy', avg_recall)

