import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ElasticNet:
    def __init__(self, l1_ratio, l2_ratio, learning_rate, n_iterations):
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.Y = Y

        for i in range(self.n_iterations):
            self.update_weights()
        return self
    
    def update_weights(self):
        Y_pred = self.predict(self.X)
        dw = np.zeros(self.n)
        for j in range(self.n):
            l1_grad = self.l1_ratio if self.weights[j] > 0 else -self.l1_ratio
            dw[j] = (-2 * (self.X[:, j].dot(self.Y - Y_pred)) + l1_grad + 2 * self.l2_ratio * self.weights[j] ) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db 

    def predict(self, X):
        return X.dot(self.weights) + self.bias

df = pd.read_csv('salary_data.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=42)

model = ElasticNet(l1_ratio=0.5, l2_ratio=0.5, learning_rate=0.01, n_iterations=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("Predicted salaries:", np.round(Y_pred[:3], 2))
print("Real salaries:", Y_test[:3])
print("Trained weights:", np.round(model.weights[0], 2))
print("Trained bias:", np.round(model.bias, 2))

plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, Y_pred, color='red')
plt.title('Elastic Net Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()