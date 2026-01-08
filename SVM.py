import numpy as np
from cvxopt import matrix, solvers

class KernelSVM:
    def __init__(self, C=10.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        self.alphas_sv = None
        self.X_sv = None
        self.y_sv = None
        self.b = 0

    def fit(self, X, y):
        # 1. Save data shape
        m, n = X.shape
        
        # 2. Compute Gram Matrix (k)
        k = np.zeros([m, m])
        t = np.outer(y, y)
        
        # Gram Matrix
        for i in range(m):
            for j in range(m):
                diff = X[i] - X[j]
                dist_sq = np.sum(diff**2)
                k[i][j] = np.exp(-self.gamma * dist_sq)
        
        # 3. Compute P Matrix
        p = t * k

        # 4. CVXOPT Setup
        P_cvx = matrix(p)
        q_cvx = matrix(-np.ones((m, 1)))
        G_cvx = matrix(np.vstack((-np.eye(m), np.eye(m)))) # Handles the "Box" limits. Top is −1 (for ≥0), Bottom is +1 (for ≤C).
        h_cvx = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C))) #The actual limits. Top is 0, Bottom is C.
        A_cvx = matrix(y.reshape(1, -1).astype('float')) # Ensures α values balance the classes (∑αy=0).
        b_cvx = matrix(np.zeros(1)) #The target sum for the labels.

        # 5. Run Solver
        solvers.options['show_progress'] = False
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
        alphas = np.array(solution['x']).flatten()

        # 6. Extract Support Vectors
        sv_indices = np.where(alphas > 1e-5)[0]
        
        self.alphas_sv = alphas[sv_indices]
        self.X_sv = X[sv_indices]
        self.y_sv = y[sv_indices]

        # 7. Calculate Bias (b)
        b_list = []
        for s in sv_indices:
            k_s = k[s, sv_indices] 
            sum_term = np.sum(self.alphas_sv * self.y_sv * k_s)
            b_s = y[s] - sum_term
            b_list.append(b_s)
            
        self.b = np.mean(b_list)
        print(f"Training Complete. Bias: {self.b}")

    def predict(self, points):
        # Ensure input is a list/array of points
        points = np.array(points)
        predictions = []
        
        for point in points:
            # prediction
            diff = self.X_sv - point
            dist_sq = np.sum(diff**2, axis=1)
            k_new = np.exp(-self.gamma * dist_sq)
            
            score = np.sum(self.alphas_sv * self.y_sv * k_new) + self.b
            predictions.append(np.sign(score))
            
        return np.array(predictions)

# 1. Define Data
X_train = np.array([[1,0], [-1,0], [0,1], [0,-1], [3,0], [-3,0], [0,3], [0,-3]])
y_train = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# 2. Create Model
model = KernelSVM(C=10.0, gamma=1.0)

# 3. Train
model.fit(X_train, y_train)

# 4. Predict
new_points = [[0, 0], [5, 5]]
print(f"Predictions: {model.predict(new_points)}")