# K Nearest Neighbor
import numpy as np
import matplotlib.pyplot as plt

training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

def euclidean_distance(point1, point2):
  return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

#calculate the distances
distances = []
for i in range(len(training_data)):
  distance = euclidean_distance(test_point, training_data[i])
  distances.append((distance, training_labels[i]))

# sort it to find the smallest(k=3) 3 values
distances.sort(key=lambda x: x[0])
k_nearest_labels = [label for _, label in distances[:k]]

unique_labels = list(set(k_nearest_labels))

votes = []
for label in unique_labels:
    count = k_nearest_labels.count(label)
    votes.append((label, count))

predict = max(votes, key=lambda x: x[1])
print(f"Predicted Class: {predict[0]}")