import numpy as np
print(np.full(10, (1/10)))
class DecisionStumpp:
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
    self.clf = [] # for trained stumps
  
  def fit(self, X, y):
    n_sample, n_features = X.shape
    w = np.full(n_sample, (1 / n_sample))
    self.clf = []
    for _ in range(self.n_clf):
      min_error = float('inf')
      best_stump = None
      for feature_i in range(n_features):
        thresholds = np.unique(X[:, feature_i])
        for thresh in thresholds:
          for p in [1, -1]:
            #to do
            pass
            