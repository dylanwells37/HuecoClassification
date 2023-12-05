import numpy as np
import keras

class gaNetwork:
    
    def __init__(self, n_layers, n_neurons, activation, optimizer, loss):
        self.n_layers = n_layers # number of layers
        self.n_neurons = n_neurons # array of number of neurons in each layer
        self.activation = activation # array of activation functions for each layer
        self.optimizer = optimizer # optimizer
        self.loss = loss # loss function
        self.model = None # keras model
        
    def build_model(self, input_shape, output_shape):
        model = keras.models.Sequential()
        for i in range(self.n_layers):
            if i == 0:
                model.add(keras.layers.Dense(self.n_neurons[i], activation=self.activation[i], input_shape=input_shape))
            else:
                model.add(keras.layers.Dense(self.n_neurons[i], activation=self.activation[i]))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model = model
    
        
    def train_model(self, x_train, y_train, epochs, batch_size):
        
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size
                          , verbose=0)
        
    