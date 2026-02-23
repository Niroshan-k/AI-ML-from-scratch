import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles

# 1. Create the "Impossible" Dataset (Doughnut)
# factor=0.5 means the inner circle is half the size of the outer
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1)

# 2. The "Linear" SVM (Fails)
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X, y)

# 3. The "RBF" Kernel SVM (Succeeds)
# This uses the "Lift" trick math
clf_rbf = svm.SVC(kernel='rbf', C=1.0)
clf_rbf.fit(X, y)

# --- PLOTTING LOGIC (Skip reading this, just run it) ---
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM with RBF Kernel (The 'Doughnut' Solved)")
plt.show()