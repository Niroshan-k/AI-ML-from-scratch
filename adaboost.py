import numpy as np

class DecisionStump:
  def __init__(self, polarity, feature_idx, threshold, alpha):
    self.polarity = polarity
    self.feature_idx = feature_idx
    self.threshold = threshold
    self.alpha = alpha

  def predict(self, X):
    n_samples = X.shape[0]
    column_values = X[:, self.feature_idx]
    
    predictions = np.ones(n_samples)
    mask = column_values < self.threshold
    
    # 3. Mark those specific cases as -1
    predictions[mask] = -1

    # If polarity is 1, keep the answer (Small is -1, Big is 1)
    # If polarity is -1, flip the answer (Small is 1, Big is -1)
    return self.polarity * predictions 
  
class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = [] 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Step A: Initialize Weights (1/N for everyone)
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = [] 
        
        # Loop over the number of stumps we want to find
        for _ in range(self.n_clf):
            
            # Variables to track the winner of this round
            min_error = float('inf')
            best_stump = None 
            
            # --- 1. THE GREEDY SEARCH ---
            # Find the single best cut that minimizes weighted error
            
            for feature_i in range(n_features):
                column_values = X[:, feature_i]
                thresholds = np.unique(column_values)
                
                for thresh in thresholds:
                    # Check both "Greater than" (1) and "Less than" (-1)
                    for p in [1, -1]:
                        # Create a temporary stump to test
                        stump = DecisionStump(polarity=p, feature_idx=feature_i, threshold=thresh, alpha=0)
                        
                        # Predict
                        predictions = stump.predict(X)
                        
                        # Calculate Weighted Error
                        # error = Sum of weights where prediction was wrong
                        error = sum(w[y != predictions])
                        
                        # Save it if it's the best so far
                        if error < min_error:
                            min_error = error
                            best_stump = stump

            # --- 2. CALCULATE ALPHA (Amount of Say) ---
            # Formula: 0.5 * ln((1-error) / error)
            # 1e-10 to avoid dividing by zero
            EPS = 1e-10
            best_stump.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            # --- 3. UPDATE WEIGHTS ---
            # Formula: w_new = w_old * exp(-alpha * y * prediction)
            # If prediction is correct: y*pred is positive -> exp(-alpha) -> weight goes down
            # If prediction is wrong:   y*pred is negative -> exp(alpha)  -> weight goes up
            predictions = best_stump.predict(X)
            
            w = w * np.exp(-best_stump.alpha * y * predictions)
            
            # Normalize weights so they sum to 1 again
            w = w / np.sum(w)
            
            # Save the winner
            self.clfs.append(best_stump)

    def predict(self, X):
        # Start with 0
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        
        # Sum them up
        y_pred = np.sum(clf_preds, axis=0)
        
        # Return sign (-1 or 1)
        return np.sign(y_pred)
            

# 1. Create Data (XOR problem is impossible for 1 cut, but easy for AdaBoost)
X = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
y = np.array([-1, 1, -1, 1])

# 2. Train
clf = AdaBoost(n_clf=5)
clf.fit(X, y)

# 3. Predict
print(f"Predictions: {clf.predict(X)}")
# Should match [-1, 1, -1, 1]

# AdaBoost has a strict rule: If a weak learner has 50% error (random guessing),
# it gets 0 votes. alpha = 0.5 * ln((1 - 0.5) / 0.5) = ln(1) = 0


# second
X_easy = np.array([[1, 1], [1, 2], [8, 8], [9, 9]])
y_easy = np.array([-1, -1, 1, 1])

# Train
clf = AdaBoost(n_clf=5)
clf.fit(X_easy, y_easy)

# Predict
print(f"Predictions: {clf.predict(X_easy)}")
# Should match [-1, -1, 1, 1]