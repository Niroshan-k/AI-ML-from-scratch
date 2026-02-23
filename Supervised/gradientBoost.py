import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoost:
   def __init__(self, n_estimators, learning_rate):
       self.n_estimators = n_estimators
       self.learning_rate = learning_rate
       self.trees = []
       self.initial_prediction = 0
   
   def fit(self, X, y):
       self.initial_prediction = np.mean(y)
       predictions = np.full(X.shape[0], self.initial_prediction)

       for i in range(self.n_estimators):
           negative_grad = y - predictions

           tree = DecisionTreeRegressor(max_depth=3)
           tree.fit(X, negative_grad)

           self.trees.append(tree)
           update = tree.predict(X)
           predictions = predictions + (self.learning_rate * update)
   
   def predict(self, X):
       final_predictions = np.full(X.shape[0], self.initial_prediction)
       for tree in self.trees:
           final_predictions += self.learning_rate * tree.predict(X)
       
       return final_predictions
           
       
import matplotlib.pyplot as plt

# 1. Create Synthetic Data (A simple curve)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X.flatten()**2 + np.random.normal(0, 2, 100) # y = x^2 + noise

# 2. Train our Model
gbm = GradientBoost(n_estimators=50, learning_rate=0.1)
gbm.fit(X, y)

# 3. Predict
y_pred = gbm.predict(X)

# 4. Visualize
plt.scatter(X, y, color="black", label="Data")
plt.plot(X, y_pred, color="red", linewidth=3, label="GBM Prediction")
plt.legend()
plt.title("Gradient Boosting from Scratch")
plt.show()     