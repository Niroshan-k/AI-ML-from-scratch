import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy import stats # Helper for calculating "Mode" (Majority Vote)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = [] # This will store our 10 trained trees

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = [] # Reset
        
        # Loop to build the forest
        for _ in range(self.n_trees):
            
            # Step 1: Create a "Parallel Universe" (Bootstrap)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            # Step 2: Create a tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            
            # Step 3: Train on the random sample
            tree.fit(X_sample, y_sample)
            
            # Step 4: Save
            self.trees.append(tree)
            
        print(f"Forest Trained with {self.n_trees} trees!")

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        predictions, _ = stats.mode(tree_preds, axis=1, keepdims=False)
        
        return predictions

X_train = np.array([[2, 2], [2, 1], [2, 3], [10, 10], [10, 9], [10, 11]])
y_train = np.array([0, 0, 0, 1, 1, 1])

rf = RandomForest(n_trees=5, max_depth=3)
rf.fit(X_train, y_train)

test_points = np.array([[2.1, 2.1], [9.9, 9.9]])
print(f"Predictions: {rf.predict(test_points)}")