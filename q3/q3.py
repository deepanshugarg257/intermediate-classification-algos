
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')


# In[2]:


from numpy import genfromtxt
import numpy as np
from random import randint
import PIL.Image
from cStringIO import StringIO
import IPython.display
import sklearn
import sklearn.linear_model as lm
import copy


# In[21]:


X_train = genfromtxt('notMNIST_train_data.csv', delimiter=',')
y_train = genfromtxt('notMNIST_train_labels.csv', delimiter=',')
X_test = genfromtxt('notMNIST_test_data.csv', delimiter=',')
y_test = genfromtxt('notMNIST_test_labels.csv', delimiter=',')
print len(X_test)


# In[4]:


def showarray(a, fmt='png'):
    a = np.uint8(a)
#     print a
    new = []
    new.append([])
    k=0
    for i in range(1,85):
        for j in range(1,85):
            x = (i-1)*28
            y = (j-1)*28
            x = int(x/84)
            y = int(y/84)
            new[k].append(a[x][y])

        new.append([])
        k+=1
    new.remove([])
#                 print new
    new = np.uint8(new)
    f = StringIO()
    PIL.Image.fromarray(new).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


# In[18]:


finc = []
x = 0.0001
while (x <= 100.0):
    print "l1"
    l2clf = lm.LogisticRegression(penalty = 'l2', C=x)
    l2clf.fit(X_train, y_train)
    l2_y = l2clf.predict(X_test)
    l2_y = [1 if i > 0.0 else 0 for i in l2_y]
    print "lambda = ", 1.0/float(x)
    print "accuracy = ", sklearn.metrics.accuracy_score(y_test, l2_y)
    coef = l2clf.coef_.reshape(784,1)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,255), copy=False)
    scaler.fit(coef)
    scaler.transform(coef)
#     print coef
    showarray(coef.reshape(28,28))
#     finc.append(copy.deepcopy(coef))
#     print "l1"
#     l2clf = lm.LogisticRegression(penalty = 'l1', C=x)
#     l2clf.fit(X_train, y_train)
#     l2_y = l2clf.predict(X_test)
#     l2_y = [1 if i > 0.0 else 0 for i in l2_y]
#     print "lambda = ", 1.0/float(x)
#     print "accuracy = ", sklearn.metrics.accuracy_score(y_test, l2_y)
#     coef = l2clf.coef_.reshape(784,1)
#     scaler.fit(coef)
#     scaler.transform(coef)
# #     print coef
#     showarray(coef.reshape(28,28))
#     finc.append(copy.deepcopy(coef))
    x *= 1.5
# print finc[4]-finc[0]

