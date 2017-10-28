from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

import argparse, os, sys

def get_input_data(filename):
    """
    Function to read the input data from the letter recognition data file.

    Parameters
    ----------
    filename: The path to input data file

    Returns
    -------
    X: The input for the SVM classifier of the shape [n_samples, n_features].
       n_samples is the number of data points (or samples) that are to be loaded.
       n_features is the length of feature vector for each data point (or sample).
    Y: The labels for each of the input data point (or sample). Shape is [n_samples,].

    """

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    """
    An important part is missing here. Corresponding to point (1) in "Remember".
    ===========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ===========================================================================
    """

    return X, Y

def calculate_metrics(predictions, labels):
    """
    Function to calculate the precision, recall and F-1 score.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    precision: true_positives / (true_positives + false_positives)
    recall: true_positives / (true_positives + false_negatives)
    f1: 2 * (precision * recall) / (precision + recall)
    ===========================================================================
    """

    # YOUR CODE GOES HERE
    avg = 'weighted'
    precision = precision_score(labels, predictions, average = avg)
    recall = recall_score(labels, predictions, average = avg)
    f1 = f1_score(labels, predictions, average = avg)

    """
    ===========================================================================
    """

    return precision, recall, f1

def calculate_accuracy(predictions, labels):
    """
    Function to calculate the accuracy for a given set of predictions and
    corresponding labels.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    accuracy: Fraction of total samples that have correct predictions (same as
    true label)

    """
    return accuracy_score(labels, predictions)

def SVM(train_data,
        train_labels,
        test_data,
        test_labels,
        kernel='linear'):

    # YOUR CODE GOES HERE
    clf = svm.SVC(kernel = kernel,C = 100, gamma = 0.1, degree = 4)
    clf.fit(train_data, train_labels)
    test_predictions = clf.predict(test_data)

    test_accuracy = calculate_accuracy(test_predictions, test_labels)
    # print "Testing Accuracy: %.4f" % (test_accuracy)

    """
    Use the trained model to perform testing. Using the output of the testing
    prodecure, get the top prediction for each sample and calculate the accuracy
    on test data using the function given (as shown above for train accuracy).

    Also, complete the function given above for metrics using scikit-learn and
    return their values in this function.
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """
    accuracy = calculate_accuracy(test_predictions, test_labels)
    precision, recall, f1 = calculate_metrics(test_predictions, test_labels)

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
            help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        print "Usage: python letter_classification_svm.py --data_dir='<dataset dir path>'"
        sys.exit()
    else:
        filename = os.path.join(args.data_dir, 'letter_classification_train.data')
        try:
            if os.path.exists(filename):
                print "Using %s as the dataset file" % filename
        except:
            print "%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir)
            sys.exit()

    # Set the value for svm_kernel as required.
    svm_kernel = 'rbf'

    """
    Get the input data using the provided function. Store the X and Y returned
    as X_data and Y_data. Use filename found above as the input to the function.
    ==========================================================================
    """

    # YOUR CODE GOES HERE
    X_data, Y_data = get_input_data(filename)
    """
    ==========================================================================
    """

    """
    We use 5-fold cross validation for reporting the final scores and metrics.
    Code to generate the 5-folds (You can change the split function to something
    else that you like as well) and then calling the SVM to classify the present
    split.
    ==========================================================================
    """
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.125)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        accumulated_metrics.append(
            SVM(train_data, train_labels, test_data, test_labels,
                svm_kernel))
        fold += 1

    """
    Print out the accumulated metrics in a good format.
    ==========================================================================
    """
    # YOUR CODE GOES HERE
    final_metrics = [0 for i in accumulated_metrics[0]]
    for row in accumulated_metrics:
        for i in xrange(len(row)):
            final_metrics[i] += row[i]
    final_metrics = [x/5. for x in final_metrics]

    print str(final_metrics[0])+', '+str(final_metrics[1])+', '+str(final_metrics[2])+', ',str(final_metrics[3])
    """
    ==========================================================================
    """