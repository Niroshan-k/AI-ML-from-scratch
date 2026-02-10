import numpy as np
import matplotlib.pyplot as plt

X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_or = np.array([0, 1, 1, 1])

class perceptron:
    def __init__(self, learning_rate = 0.1, epochs = 20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors_per_epoch = []

    def activation_function(self, x):
        # Step function: Return 1 if x >= 0, else 0
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            errors = 0
            
            for idx, x_i in enumerate(X):
                combination = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(combination)
                update = self.learning_rate * (y[idx] - y_pred)

                self.weights += update * x_i
                self.bias += update

                if update != 0: errors += 1
        
            self.errors_per_epoch.append(errors)

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        return self.activation_function(output)
    
# --- Test ---
model = perceptron(learning_rate=0.1, epochs=10)
model.fit(X_or, y_or)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", model.predict(X_or))

plt.figure(figsize=(6, 4))
plt.plot(model.errors_per_epoch, marker='o')
plt.title("Misclassifications per Epoch (OR)")
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.grid(True)
plt.show()