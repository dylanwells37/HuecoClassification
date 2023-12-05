import numpy as np
import network as ga_nn

class GA:
    
    
    def __init__(self, n_population, n_generations, n_crossover, n_mutate, n_layers):
        self.n_population = n_population
        self.n_generations = n_generations
        self.n_crossover = n_crossover
        self.n_mutate = n_mutate
    
        # define the possible values for each parameter
        self.parameters = {
            'n_layers': [1, 2, 3, 4, 5, 6, 7, 8],
            'n_neurons': [8, 16, 32, 64, 128, 256, 512],
            'activation': ['relu', 'sigmoid', 'tanh', 'softmax'],
            'optimizer': ['adam', 'rmsprop', 'sgd'],
            'loss': ['binary_crossentropy', 'categorical_crossentropy']
        }     

        
    def initialize_population(self):
        for i in range(self.n_population):
            