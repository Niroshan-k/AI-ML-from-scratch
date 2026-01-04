import numpy as np
import matplotlib.pyplot as plt

class BayesianLinearRegression:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.w = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        I = np.eye(n_features)
        
        S_inv = self.beta * np.dot(X.T, X) + self.alpha * I
        S = np.linalg.inv(S_inv)
        
        self.w = self.beta * np.dot(S, np.dot(X.T, y))

    def predict(self, X):
        return np.dot(X, self.w)

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_bias = np.c_[np.ones((100, 1)), X]

alpha = 1.0
beta = 1.0

model = BayesianLinearRegression(alpha, beta)
model.fit(X_bias, y)

X_new = np.array([[0], [2]])
X_new_bias = np.c_[np.ones((2, 1)), X_new]
y_pred = model.predict(X_new_bias)

print(f"Calculated Weights: {model.w.flatten()}")

plt.scatter(X, y, color='blue')
plt.plot(X_new, y_pred, color='red', linewidth=2)
plt.show()