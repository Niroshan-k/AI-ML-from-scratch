import random

class Brain:
    def __init__(self):
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.bias = random.uniform(-1,1)

    def think(self, inputs):
        sum_value = 0
        for i in range(3):
            sum_value += inputs[i] *self.weights[i]
        
        sum_value += self.bias
        return sum_value
    
    def mutate(self):

        mutation_rate = 0.1 # 10% change

        #tweak the wweights
        for i in range(3):
            self.weights[i] += random.uniform(-mutation_rate, mutation_rate)

        # tweak the bias
        self.bias += random.uniform(-mutation_rate, mutation_rate)