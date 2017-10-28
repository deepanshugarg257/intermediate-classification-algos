"""
--------
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet
--------
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np
import sys

def train_data(filename):
    X = np.genfromtxt(filename, delimiter = ',', dtype = float)
    Y = np.array(X[:, len(X[0])-1])
    X = X[:, 0: len(X[0])-1]
    return X, Y

def test_data(filename):
    X = np.genfromtxt(filename, delimiter = ',', dtype = float)
    return X

def map_predictions(predicted_values):
    predicted_labels = [1 if x > 0.5 else 0 for x in predicted_values]
    return np.asarray(predicted_labels, dtype = float)

def calculate_accuracy(predictions, labels):
    return accuracy_score(labels, predictions)    

def classifier(X_Train, Y_Train, X_Test):
    reg = LinearRegression()
    reg.fit(X_Train, Y_Train)
    predicted_test = reg.predict(X_Test)
    Y_pred = map_predictions(predicted_test)
    for label in Y_pred:
        print int(label)
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python q4_a.py [relative/path/to/train/file] [relative/path/to/test/file]"
        exit()

    X_Train, Y_Train = train_data(sys.argv[1])
    X_Test= test_data(sys.argv[2])
    classifier(X_Train, Y_Train, X_Test)