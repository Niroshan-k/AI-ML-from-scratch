import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

x = np.random.rand(50, 1) * 100  
y = 3.5 * x + np.random.randn(50, 1) * 20


m,b = 0,0
alpha = 0.0001 # learning rate

def gradient_descent(x, y, m, b, alpha, iters):
    N = float(len(y))
    for _ in range(iters):
        y_pred = m * x + b
        D_m = (2/N) * sum(x * (y_pred - y))
        D_b = (2/N) * sum(y_pred - y)
        m = m - alpha * D_m
        b = b - alpha * D_b
    return m, b


m, b = gradient_descent(x, y, m, b, alpha, 10)
yP = m * x + b

plt.scatter(x, y)
plt.plot(x, yP, color='red')
plt.show()