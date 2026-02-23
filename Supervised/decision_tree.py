import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        
        # Leaf Node part
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)
   
    def calculate_entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    def split_data(self, X, y, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        return X[mask], X[~mask], y[mask], y[~mask]

    def get_best_split(self, X, y):
        best_split = {}
        max_gain = -float("inf")
        n_features = X.shape[1]
        
        # Loop through features
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            
            # Loop through unique values
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split_data(X, y, feature_index, threshold)
                
                if len(X_left) == 0 or len(X_right) == 0:
                    continue
                
                # Calculate Gain
                parent_entropy = self.calculate_entropy(y)
                n = len(y)
                n_l, n_r = len(y_left), len(y_right)
                e_l, e_r = self.calculate_entropy(y_left), self.calculate_entropy(y_right)
                child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
                
                gain = parent_entropy - child_entropy
                
                if gain > max_gain:
                    max_gain = gain
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "X_left": X_left, "y_left": y_left,
                        "X_right": X_right, "y_right": y_right,
                        "gain": gain
                    }
        return best_split

    def calculate_leaf_value(self, y):
        val, counts = np.unique(y, return_counts=True)
        return val[np.argmax(counts)]

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # STOPPING CRITERIA (Leaf)
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split):
            return Node(value=self.calculate_leaf_value(y))

        # FIND BEST SPLIT
        best_split = self.get_best_split(X, y)
        
        # If no valid split found (gain <= 0), make leaf
        if best_split.get("gain", 0) <= 0:
            return Node(value=self.calculate_leaf_value(y))

        # RECURSION
        left_node = self.build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
        right_node = self.build_tree(best_split["X_right"], best_split["y_right"], depth + 1)

        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_node,
            right=right_node
        )
    
    def make_prediction(self, x, tree):
        # If this node is a leaf (has a value), return it
        if tree.value is not None:
            return tree.value
        
        # Else, traverse
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
 
        
# Create Dummy Data
X_train = np.array([[2, 2], [2, 1], [2, 3], [10, 10], [10, 9], [10, 11]])
y_train = np.array([0, 0, 0, 1, 1, 1]) # 0 = Small numbers, 1 = Big numbers

# Initialize and Train
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Predict
test_point = np.array([[2.5, 2.5], [9, 9]])
preds = tree.predict(test_point)

print("Predictions:", preds) 
# Should print [0, 1]