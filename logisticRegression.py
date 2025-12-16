import numpy as np
import matplotlib.pyplot as plt

# 0 = Fail, 1 = Pass
x = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,  4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0])
y = np.array([0,   0,    0,   0,    0,   0,    0,    1,   1,    1,   1,    1,   1,    1])

w = 0
b = 0

def gradientDescent(x, y, w, b, alpha, iters):
    N = float(len(y))
    for _ in range(iters):
        z = w * x + b
        #sigmoid
        y_pred = 1 / (1 + np.exp(-z))
        
        D_w = (1/N) * sum((y_pred - y) * x)
        D_b = (1/N) * sum(y_pred - y)
        
        w = w - alpha * D_w
        b = b - alpha * D_b
    return w, b

def predict(x, w, b):
    z = w * x + b
    probability = 1 / (1 + np.exp(-z))
    
    # If prob >= 0.5 return 1, else return 0
    if probability >= 0.5:
        return 1
    else:
        return 0

w, b = gradientDescent(x, y, w, b, 0.1, 5000)

x_smooth = np.linspace(0, 7, 100)
z_smooth = w * x_smooth + b
y_smooth = 1 / (1 + np.exp(-z_smooth))

plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x_smooth, y_smooth, color='red', label='Sigmoid Model')
plt.legend()
plt.show()

# Example predictions
test_values = [1.0, 3.0, 5.0]
for val in test_values:
    pred = predict(val, w, b)
    print(f"Prediction for {val}: {pred}")