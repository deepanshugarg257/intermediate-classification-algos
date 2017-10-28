# from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
import keras
import random
import numpy as np
import sys
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot as PLT
from keras.datasets import mnist
from copy import deepcopy
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from sklearn import svm
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.metrics import accuracy_score
from six.moves import cPickle
from keras.models import Model

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def convert_mat(inp):
	inp = inp.astype('float32')
	inp /= 255
	return inp.reshape((len(inp), 32, 32, 3), order = 'F')

if "__name__" != "__main__":
	arg1 = sys.argv[1]
	arg2 = sys.argv[2]
	fl = 0
	for i in xrange(1, 6):
		a = unpickle(arg1+"/data_batch_"+str(i))
		data = a["data"]
		labels = a["labels"]
		if fl == 0:
			fl = 1
			train_data = data
			train_labels = labels
		else:
			train_data = np.concatenate((train_data, data))
			train_labels = np.concatenate((train_labels, labels))

	a = unpickle(arg2)
	test_data = a["data"]
	test_data = convert_mat(test_data)

	train_y = keras.utils.to_categorical(train_labels, num_classes=None)
	train_data = convert_mat(train_data)
	y_train_svm = deepcopy(train_labels)


	b = unpickle(arg1+"/batches.meta")
	label_names = b["label_names"]

	batch_size = 32
	num_classes = 10
	epochs = 60


	clf = Sequential()

	#layer - 1
	clf.add(Conv2D(32, (3, 3), padding = 'same', input_shape = train_data.shape[1:]))
	clf.add(BatchNormalization())
	clf.add(LeakyReLU(alpha = .05))
	#layer - 2
	clf.add(Conv2D(32, (3, 3)))
	clf.add(BatchNormalization())
	clf.add(LeakyReLU(alpha = .05))
	#layer - 3 (dimension shrinking)
	clf.add(MaxPooling2D(pool_size = (2, 2)))
	#(remove nodes from the last layer)
	clf.add(Dropout(0.25))

	#layer - 4
	clf.add(Conv2D(64, (3, 3), padding = 'same'))
	clf.add(BatchNormalization())
	clf.add(LeakyReLU(alpha = .05))

	#layer - 5
	clf.add(Conv2D(64, (3, 3)))
	clf.add(BatchNormalization())
	clf.add(LeakyReLU(alpha=.05))

	#layer - 6
	clf.add(MaxPooling2D(pool_size=(2, 2)))
	clf.add(Dropout(0.25))

	# n d array - > 1 d array
	clf.add(Flatten())
	# layer - 7: 
	clf.add(Dense(512))
	clf.add(BatchNormalization())
	clf.add(LeakyReLU(alpha = .05))
	clf.add(Dropout(0.5))

	#layer - 8
	clf.add(Dense(num_classes))
	clf.add(BatchNormalization())
	clf.add(Activation('softmax'))

	opt = keras.optimizers.rmsprop(lr = 0.0005, decay = 1e-6)
	clf.compile(loss = keras.losses.categorical_crossentropy,
	              optimizer=opt,
	              metrics=['accuracy'])
	
	clf.fit(train_data, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split = .2, shuffle=True)

	# score = clf.evaluate(test_data, test_y, verbose=0)
	# print('Test loss CNN:', score[0])
	# print('Test accuracy CNN:', score[1])

	intermediate_layer_clf = Model(inputs=clf.input,
                                 outputs=clf.layers[19].output)
	intermediate_output_train = intermediate_layer_clf.predict(train_data)
	intermediate_output_test = intermediate_layer_clf.predict(test_data)
	clf2 = svm.SVC(decision_function_shape = 'ovo')
	clf2.fit(intermediate_output_train, y_train_svm)
	test_predictions=clf2.predict(intermediate_output_test)
	# print ("Test Accuracy SVM:", test_accuracy)

	outfile = open('q2_c_output.txt', 'w')
	for yp in test_predictions :
		outfile.write(label_names[int(yp)] + '\n')
