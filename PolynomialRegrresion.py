import numpy as np
import matplotlib.pyplot as plt

x = [0, 20, 40, 60, 80, 100]
y = [0.0002, 0.0012, 0.0060, 0.0300, 0.0900, 0.2700]

newX = np.array(x)/100
b0 = 0
b1 = 0
b2 = 0
b3 = 0
b4 = 0


def gradientDescent(x, y, b0, b1, b2, b3, b4, alpha, iters):
    N = float(len(y))
    for _ in range(iters):
        y_pred = b0 + b1*np.array(x) + b2*np.power(np.array(x), 2) + b3*np.power(np.array(x), 3) + b4*np.power(np.array(x), 4)
        D_b0 = (2/N) * sum(y_pred - y)
        D_b1 = (2/N) * sum((y_pred - y) * np.array(x))
        D_b2 = (2/N) * sum((y_pred - y) * np.power(np.array(x), 2))
        D_b3 = (2/N) * sum((y_pred - y) * np.power(np.array(x), 3))
        D_b4 = (2/N) * sum((y_pred - y) * np.power(np.array(x), 4))
        
        b0 = b0 - alpha * D_b0
        b1 = b1 - alpha * D_b1
        b2 = b2 - alpha * D_b2
        b3 = b3 - alpha * D_b3
        b4 = b4 - alpha * D_b4
    return b0, b1, b2, b3, b4

b0, b1, b2, b3, b4 = gradientDescent(newX, y, b0, b1, b2, b3, b4, 0.01, 100000)
yP = b0 + b1*np.array(newX) + b2*np.power(np.array(newX), 2) + b3*np.power(np.array(newX), 3) + b4*np.power(np.array(newX), 4)

plt.scatter(x, y)
plt.plot(x,yP, color = 'red')
plt.show()