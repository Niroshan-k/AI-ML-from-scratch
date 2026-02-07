import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, gradients, hessians, lambda_reg=1, gamma_reg=0):
        self.g = gradients
        self.h = hessians
        self.lambda_reg = lambda_reg
        self.gamma_reg = gamma_reg

        #sum of gradient and hessian
        self.G = np.sum(gradients)
        self.H = np.sum(hessians)

        #weights : w = -G/ (H+ lambda)
        self.weight = -self.G / ( self.H + self.lambda_reg )

        self.best_feature = None
        self.best_threshold = None
        self.left_child = None
        self.right_child = None
    
    def calculate_score(self, G, H):
        # G^2 / ( H + lambda)
        return (G**2) / (H + self.lambda_reg)
    
    def find_best_split(self, X):
        if len(X) <= 1: return

        n_samples, n_features = X.shape
        best_gain = 0

        score_parent = self.calculate_score(self.G, self.H)

        for feature_idx in range(n_features):
            #sort data
            sorted_indices = np.argsort(X[:, feature_idx])
            G_left, H_left = 0,0

            #loop through every possible cut
            for i in range(n_samples -1):
                idx = sorted_indices[i]

                G_left += self.g[idx]
                H_left += self.h[idx]

                G_right = self.G - G_left
                H_right = self.H - H_left

                #gain Formula: 0.5 * [Score_L + Score_R - Score_Parent] - Gamma
                score_left = self.calculate_score(G_left, H_left)
                score_right = self.calculate_score(G_right, H_right)
                gain = 0.5 * (score_left + score_right -score_parent) - self.gamma_reg

                if gain > best_gain:
                    best_gain = gain
                    self.best_feature = feature_idx
                    self.best_threshold = X[idx, feature_idx] #cut
                
        # if found a good split, create children
        if best_gain > 0:
            #split the data indices
            mask = X[:, self.best_feature] <= self.best_threshold

            # Recursion: Build Left and Right Nodes
            # Note: We pass the specific gradients/hessians for the children
            self.left_child = Node(self.g[mask], self.h[mask], self.lambda_reg, self.gamma_reg)
            self.right_child = Node(self.g[~mask], self.h[~mask], self.lambda_reg, self.gamma_reg)

    def predict(self, x):
        # Recursively walk down the tree
        if self.left_child is None or self.right_child is None:
            return self.weight
            
        if x[self.best_feature] <= self.best_threshold:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)
            

class XGBoost:
    def __init__(self, n_estimators=5, lambda_reg=1, gamma_reg=0, max_depth=3):
        self.n_estimators = n_estimators
        self.lambda_reg = lambda_reg
        self.gamma_reg = gamma_reg
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        # 1. Initial Prediction (0.5 is standard, or mean)
        self.base_pred = np.mean(y)
        preds = np.full(y.shape, self.base_pred)
        
        for _ in range(self.n_estimators):
            # 2. Calculate Gradients & Hessians
            # For MSE: Grad = (Pred - y), Hess = 1
            grads = preds - y
            hessians = np.ones(len(y))
            
            # 3. Build the Custom Tree (Root Node)
            root = Node(grads, hessians, self.lambda_reg, self.gamma_reg)
            
            # (We cheat slightly and just build 1 layer of depth here for simplicity,
            # but you can loop this to build deeper trees)
            root.find_best_split(X)
            
            # If we split, go one level deeper
            if root.left_child:
                root.left_child.find_best_split(X[X[:, root.best_feature] <= root.best_threshold])
                root.right_child.find_best_split(X[X[:, root.best_feature] > root.best_threshold])
            
            self.trees.append(root)
            
            # 4. Update Predictions
            # XGBoost learning rate is often called "eta", applied here
            learning_rate = 0.3
            update_preds = np.array([root.predict(row) for row in X])
            preds += learning_rate * update_preds

    def predict(self, X):
        preds = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            preds += 0.3 * np.array([tree.predict(row) for row in X])
        return preds
    

# 1. Load Data (Mini-MNIST)
digits = load_digits()

# 2. Filter: Keep only '0' and '1' to make it binary
# (Our simple regression tree works best on 0 vs 1)
mask = (digits.target == 0) | (digits.target == 1)
X = digits.data[mask]
y = digits.target[mask]

# 3. Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} images with {X_train.shape[1]} pixels each...")

# 4. Train YOUR Model
# We use 10 trees. 
# Since y is 0 or 1, the model will output probabilities (like 0.1 or 0.9)
model = XGBoost(n_estimators=10, lambda_reg=1, gamma_reg=0)
model.fit(X_train, y_train)

# 5. Predict
raw_preds = model.predict(X_test)

# 6. Convert to Binary (Classification)
# If prediction > 0.5, call it a '1'. Else '0'.
final_preds = [1 if p > 0.5 else 0 for p in raw_preds]

# 7. Check Score
acc = accuracy_score(y_test, final_preds)
print(f"\nAccuracy: {acc * 100:.2f}%")

# Let's see some actual examples
print("\n--- Sample Predictions ---")
for i in range(5):
    print(f"True: {y_test[i]} | Predicted Raw: {raw_preds[i]:.4f} | Final: {final_preds[i]}")