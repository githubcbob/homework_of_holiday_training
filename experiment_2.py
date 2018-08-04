C = 0.5
learning_rate = 0.0005
batch_size = 50
max_epoch = 2
test_size = 0.25


# load the dataset
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import classification_report
data_dir = '../data/'
X, y = load_svmlight_file(f='./a9a.t')

# preprocess the dataset
import numpy
import numpy as np
import random
X = X.toarray()
n_samples, n_features = X.shape
n_features = n_features+1
X = numpy.column_stack((X, numpy.ones((n_samples, 1))))
y = y.reshape((-1, 1))


# devide the dataset into traning set and validation set
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)


# initialize the loss array
losses_train = []
losses_val = []

max_iter = X_train.shape[0]//batch_size*max_epoch

# select different initializing method
w = numpy.zeros((n_features, 1))  # initialize with zeros
b = numpy.zeros((1))

#define some functions
def _sign(X,y,w,b):
    s = 1-y*(np.dot(X, w)+b)
    return (np.sign(s)+1)/2
def hinge(X,y,w,b):
    s = 1-y*(np.dot(X, w)+b)
    return s*(np.sign(s)+1)/2
def decision_funtion(X,w,b):
    s = np.dot(X, w)+b
    return np.sign(s)


# core code of gradient descent
for _ in range(max_iter):
    slice_indices = random.sample(list(range(X_train.shape[0])), batch_size)
    s = _sign(X_train[slice_indices],y_train[slice_indices],w,b)
    G_w = w+ C*np.dot(X_train[slice_indices].transpose(), y_train[slice_indices]*s)  # calculate the gradient(contain -G)
    w += learning_rate * G_w  # update the parameters

    G_b = -C*np.mean(y_train[slice_indices]*s)  # calculate the gradient
    G_b = -G_b
    b += learning_rate * G_b  # update the parameters

    hinge_loss = hinge(X_train,y_train,w,b)
    loss_train = np.mean(w*w)/2+C*np.mean(hinge_loss)
    losses_train.append(loss_train)

    loss_val = np.mean(w*w)/2+C*np.mean(hinge(X_val,y_val,w,b))
    losses_val.append(loss_val)

preds = decision_funtion(X_val, w, b)
report = classification_report(y_val, preds)
print(report)

# draw the figure
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
plt.plot(losses_train, "-", color="r", label="train loss")
plt.plot(losses_val, "-", color="b", label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations.")
plt.show()
# plt.savefig("result.png")