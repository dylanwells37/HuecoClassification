import numpy as np
import keras

class gaNetwork:
    
    def __init__(self, n_layers, n_neurons, activation, optimizer, loss, id,
                 input_shape, output_shape):
        self.id = id 
        self.n_layers = n_layers 
        self.n_neurons = n_neurons 
        self.activation = activation 
        self.optimizer = optimizer 
        self.loss = loss 
        self.model = None 
        self.history = None
        self.best_val_acc = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        
    def build_model(self):
        '''Build the model'''
        model = keras.models.Sequential()
        for i in range(self.n_layers):
            if i == 0:
                model.add(keras.layers.Dense(self.n_neurons[i],
                                             activation=self.activation[i], 
                                             input_shape=self.input_shape))
            else:
                model.add(keras.layers.Dense(self.n_neurons[i],
                                             activation=self.activation[i]))
        model.add(keras.layers.Dense(self.output_shape, activation='softmax'))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model = model
    
        
    def train_model(self, x_train, y_train, 
                    epochs, batch_size, 
                    validation_data, out_path,
                    verbose=0):
        '''Train the model'''
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
                    keras.callbacks.ModelCheckpoint(filepath=f'{out_path}/model_{self.id}.h5', 
                                                     monitor='val_accuracy', 
                                                     save_best_only=True)]
        
        self.model.fit(x_train, y_train, epochs=epochs, 
                       batch_size=batch_size, verbose=verbose,
                       validation_data=validation_data, 
                       callbacks=callbacks)

        self.history = self.model.history.history
        self.best_val_acc = np.max(self.history['val_accuracy'])

    
    def write_to_csv(self, filepath, open_mode='a'):
        '''Write the model's genes to a csv file'''
        with open(filepath, open_mode) as f:
            genes = self.get_genes()
            genes = [str(gene) for gene in genes]
            for gene in genes:
                f.write(f'{gene},')
            f.write(f'{self.best_val_acc}\n')
            
    
    def get_model(self):
        return self.model
    
    
    def get_history(self):
        return self.history
    
    
    def get_best_val_acc(self):
        return self.best_val_acc


    def get_gene(self, index):
        '''Get the gene at a specific index'''
        if index < self.n_layers:
            return self.n_neurons[index]
        elif index < self.n_layers * 2:
            return self.activation[index - self.n_layers]
        elif index == self.n_layers * 2:
            return self.optimizer
        elif index == self.n_layers * 2 + 1:
            return self.loss

        print("Error: index out of range")
        return None
    
    
    def set_gene(self, index, value):
        '''Set the gene at a specific index'''
        if index < self.n_layers:
            self.n_neurons[index] = value
        elif index < self.n_layers * 2:
            self.activation[index - self.n_layers] = value
        elif index == self.n_layers * 2:
            self.optimizer = value
        elif index == self.n_layers * 2 + 1:
            self.loss = value
        else:
            print("Error: index out of range")
            return None
        
    
    def get_genes(self):
        '''Get a list of all the model's genes'''
        return self.n_neurons + self.activation + [self.optimizer, self.loss]
    