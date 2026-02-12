import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

train_data = data[0:int(0.8 * m), :] # data[rows, columns]
test_data = data[int(0.8 * m):m, :]

X_train = train_data[:, 1:].T
X_train = X_train / 255.0 # scaling to [0, 255] -> [0,1]
Y_train = train_data[:, 0]

X_test = test_data[:, 1:].T
X_test = X_test / 255.0 # scaling to [0, 255] -> [0,1]
Y_test = test_data[:, 0]

def initialize_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def Relu(X):
    return np.maximum(X, 0)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot_encoding(Y):
    oneHot_Y = np.zeros((Y.size, Y.max() + 1))
    oneHot_Y[np.arange(Y.size), Y] = 1
    return oneHot_Y.T

def backward_propagation(W1, b1, W2, b2, Z1, A1, Z2, A2, X, Y):
    m = Y.size
    oneHot_Y = one_hot_encoding(Y)
    dZ2 = A2 - oneHot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2 , axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1 , axis=1, keepdims=True)
    return dW1, dB1, dW2, dB2, dZ1, dZ2

def update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * dB2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initialize_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, dB1, dW2, dB2, dZ1, dZ2 = backward_propagation(W1, b1, W2, b2, Z1, A1, Z2, A2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha)

        if (i%20) == 0:
            print("Iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 100)

val_index = 0 
Z1val, A1val, Z2val, A2val = forward_propagation(W1, b1, W2, b2, X_test[:, val_index, None])
print("Predicted label:", get_predictions(A2val))
print("True label:", Y_test[val_index])

image_array = X_test[:, val_index].reshape((28, 28))
plt.imshow(image_array, cmap='gray')
plt.show()