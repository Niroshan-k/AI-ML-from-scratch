import numpy as np
X = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
y = np.array([-2, 1, -1, 5]) # Note: Using -1 instead of 0!

n = X.shape
#w = np.full(n, (1/n))
print(n)

prediction = np.zeros(X.shape)
for i in range(len(prediction)):
 prediction[i] = np.mean(y)


print(np.mean(y))
print(prediction)
print(y[1])