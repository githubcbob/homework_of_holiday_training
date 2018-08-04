# you can change the following hyper-parameter

learning_rate = 0.0005
max_epoch = 10
test_size = 0.40
C = 0.5
# ------------------------------------------------------------------


# download the dataset
import requests

# load the dataset
from sklearn.datasets import load_svmlight_file
from io import BytesIO

# X, y = load_svmlight_file(f=BytesIO(r.content), n_features=123)
X, y = load_svmlight_file(f='./a9a.t')
#X_val, y_val = load_svmlight_file(f=BytesIO(r_val.content), n_features=123)

# preprocess the dataset
import numpy as np
import random

X = X.toarray()
n_samples, n_features = X.shape
y = y.reshape((-1, 1))

# permutation = np.random.permutation(y.shape[0])
# X = X[permutation, :, :]
# y = y[permutation]


# devide the dataset into traning set and validation set
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

X_train_n_samples, X_train_n_features = X_train.shape
X_val_n_samples, X_val_n_features = X_val.shape

# initialize the loss array
losses_train = []
losses_val = []
# loss_train = 0.0
# loss_val = 0.0

# select different initializing method
w = np.zeros((n_features, 1))  # initialize with zeros
b = 0.0
# w = numpy.random.random((n_features + 1, 1))  # initialize with random numbers
# w = numpy.random.normal(1, 1, size=(n_features + 1, 1))  # initialize with zero normal distribution
def _sign(x,y,w,b):
    s = 1-y*(np.dot(x, w)+b)
    return (np.sign(s)+1)/2

def hinge(x,y,w,b):
    s = 1-y*(np.dot(x, w)+b)
    return s*(np.sign(s)+1)/2

def decision_funtion(X,w,b):
    s = np.dot(X, w)+b
    return np.sign(s)

# core code of gradient descent

for epoch in range(max_epoch):

#    v_train = y_train * np.dot(X_train, w)

    loss_train = 0.0
    loss_val = 0.0
    G = np.zeros((n_features, 1))
#   b = 0.0
    for k in range(X_train_n_samples - 1):
        slice_indices = random.sample(list(range(X_train.shape[0])), 1)
        s = _sign(X_train[slice_indices], y_train[slice_indices], w, b)
        G_w = w + C * X_train[slice_indices].transpose() * y_train[slice_indices] * s
        G_w = -G_w
        w += learning_rate * G_w

        G_b = -C * y_train[slice_indices] * s
        G_b = -G_b
        b += learning_rate * G_b
        loss_train += hinge(X_train[slice_indices], y_train[slice_indices], w, b)

    loss_train = np.sum(w * w) * 0.5 + C * loss_train / X_train_n_samples
#    print(loss_train)
    losses_train.append(loss_train)
    print(losses_train)


    loss_val = np.sum(w * w) * 0.5 + C * np.mean(hinge(X_val, y_val, w, b))
    losses_val.append(loss_val)


preds = decision_funtion(X_val,w,b)
report = classification_report(y_val,preds)
print(report)

    # loss_train = np.zeros((X_train_n_samples, 1))
    # for j in range(X_train_n_samples - 1):
    #     loss_train[j] = max(0, 1-v_train[j])
    #
    # loss_train = np.average(loss_train)
    # losses_train.append(loss_train)

    # v_val = y_val * np.dot(X_val, w)
    # loss_val = np.zeros((X_val_n_samples, 1))
    #
    # for i in range(X_val_n_samples - 1):
    #     loss_val[i] = max(0, 1-v_val[i])
    #
    # loss_val = np.average(loss_val)
    # losses_val.append(loss_val)


#draw the figure
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
plt.plot(losses_train, "-", color="r", label="train loss")
plt.plot(losses_val, "-", color="b", label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations.")
plt.show()
plt.savefig("result.png")

print('finish')