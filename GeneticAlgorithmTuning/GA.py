import numpy as np
import pandas as pd
import pickle
import time
import argparse

from copy import deepcopy
from pathlib import Path
from network import gaNetwork

class GA:
    def __init__(self, n_population, n_crossover, n_mutate,
                 input_shape, output_shape, run_id):
        self.run = run_id
        # make the run directory
        Path(f"run_data/{self.run}/networks").mkdir(parents=True, exist_ok=True)
        self.n_pop = n_population
        self.n_crossover = n_crossover
        self.n_mutate = n_mutate
        self.n_layers = 8
        self.input_shape = input_shape
        self.output_shape = output_shape
    
        # define the possible values for each parameter
        self.parameters = {
            #'n_layers': [1, 2, 3, 4, 5, 6, 7, 8],
            'n_neurons': [8, 16, 32, 64, 128, 256, 512],
            'activation': ['relu', 'sigmoid', 'tanh', 'softmax'],
            'optimizer': ['adam', 'rmsprop', 'sgd'],
            'loss': ['binary_crossentropy', 'categorical_crossentropy']
        }     
        self.population = []
        self.fitnesses = []
        self.best_fitness = 0
        self.best_network = None
        self.best_history = None
        self.best_model = None
        # count the number of genes in the genome
        self.gene_count = self.n_layers * 2 + 2
        self.genfitnesses = []
        
    def initialize_population(self):
        for i in range(self.n_pop):
            #n_layers = np.random.choice(self.parameters['n_layers'])
            n_layers = self.n_layers
            n_neurons = [np.random.choice(self.parameters['n_neurons']) 
                         for j in range(n_layers)]
            activation = [np.random.choice(self.parameters['activation']) 
                          for j in range(n_layers)]
            optimizer = np.random.choice(self.parameters['optimizer'])
            loss = np.random.choice(self.parameters['loss'])
            id = i
            
            self.population.append(gaNetwork(n_layers, n_neurons, 
                                             activation, optimizer, loss, id,
                                             self.input_shape, self.output_shape))
    
    
    def train_population(self, x_train, y_train, epochs, batch_size, 
                         validation_data, verbose=0):
        for network in self.population:
            if verbose == 1:
                time_start = time.time()
                print(f"Training network {network.id}...")
            network.build_model()
            network.train_model(x_train, y_train, epochs, 
                                batch_size, validation_data,
                                f"run_data/{self.run}/networks")
            if verbose == 1:
                time_end = time.time()
                print(f"Network {network.id} trained in {time_end - time_start} seconds")
    
    
    def evaluate_population(self):
        fit_list = []
        for network in self.population:
            fitness = network.get_best_val_acc()
            fit_list.append(fitness)
            network.write_to_csv(f"run_data/{self.run}/ga_out.csv")
        
        self.fitnesses = fit_list
        self.genfitnesses.append(np.max(fit_list))
        
        with open(f"run_data/{self.run}/tracker.txt", 'a') as f:
            f.write(f"{np.max(fit_list)}\n")

        if np.max(fit_list) > self.best_fitness:
            self.best_fitness = np.max(fit_list)
            self.best_network = deepcopy(self.population[np.argmax(fit_list)])
            self.best_history = self.best_network.get_history()
            self.best_model = self.best_network.get_model()
            # save the best model
            self.best_model.save(f"run_data/{self.run}/best_model.h5")
            self.best_network.write_to_csv(f"run_data/{self.run}/best_network.csv", "w")

    
    def select_parents(self):
        parent_no = self.n_crossover + self.n_mutate
        roulette_no = int(parent_no * (1/4))
        rank_no = int(parent_no * (1/2))
        tournament_no = parent_no - roulette_no - rank_no
        
        parent_ids = np.array([])
        
        parent_ids = np.append(parent_ids, self.roulette_selection(roulette_no))
        parent_ids = np.append(parent_ids, self.rank_selection(rank_no))
        parent_ids = np.append(parent_ids, self.tournament_selection(tournament_no))
        
        parent_ids = [int(id) for id in parent_ids]
        
        return parent_ids
        
    
    def tournament_selection(self, tournament_no):
        # randomly select 7 percent of the population and select the best
        # individual from that group
        parent_ids = []
        for i in range(tournament_no):
            if int(self.n_pop * 0.07) == 0:
                tournament = np.random.choice(self.population, 2)
            else:
                tournament = np.random.choice(self.population, int(self.n_pop * 0.07))
            fitnesses = [network.get_best_val_acc() for network in tournament]
            parent_ids.append(tournament[np.argmax(fitnesses)].id)
            
        return np.array(parent_ids)
        
    
    def roulette_selection(self, roulette_no):
        
        parent_ids = []
        fitnesses = np.array(self.fitnesses)
        fitnesses = fitnesses / np.sum(fitnesses)
        
        for i in range(roulette_no):
            parent_ids.append(np.random.choice(self.population, p=fitnesses).id)
        
        return np.array(parent_ids)
    
    
    def rank_selection(self, rank_no):
        parent_ids = []
        fitnesses = np.array(self.fitnesses)
        indices = np.argsort(fitnesses)
        p_array = np.arange(1, len(indices) + 1) / np.sum(np.arange(1, len(indices) + 1))
        
        for i in range(rank_no):
            index = np.random.choice(indices, p=p_array)
            parent_ids.append(self.population[index].id)
        
        return np.array(parent_ids)
            
    
    def crossover(self, parent_ids):
        # randomly select two parents and perform crossover
        # repeat until the desired number of offspring is reached
        offspring = []
        
        for i in range(int(self.n_crossover/2)):
            # check if the set of parent_ids has only one element,
            # then we can't perform crossover
            set_ids = set(parent_ids)
            if len(set_ids) == 1:
                return offspring, parent_ids, self.n_crossover - i
            
            parent1 = np.random.choice(parent_ids)
            parent2 = np.random.choice(parent_ids)
            
            while parent1 == parent2:
                parent2 = np.random.choice(parent_ids)
                
            # choose a gene to swap
            chosen_gene = np.random.randint(0, self.gene_count)
            gene1 = self.population[parent1].get_gene(chosen_gene)
            gene2 = self.population[parent2].get_gene(chosen_gene)
            child1 = deepcopy(self.population[parent1])
            child2 = deepcopy(self.population[parent2])
            child1.set_gene(chosen_gene, gene2)
            child2.set_gene(chosen_gene, gene1)
            
            offspring += [child1, child2]
                
            # remove the parents from the list of ids
            parent_ids.pop(parent_ids.index(parent1))
            parent_ids.pop(parent_ids.index(parent2))
        
        return offspring, parent_ids, 0
    
    
    def mutation(self, parent_ids, extra_parents):
        # randomly select a parent and mutate it
        # repeat until the desired number of offspring is reached
        offspring = []
        
        for i in range(self.n_mutate + extra_parents):
            parent = np.random.choice(parent_ids)
            
            child = deepcopy(self.population[parent])
            
            # choose a gene to mutate
            chosen_gene = np.random.randint(0, self.gene_count)
            if chosen_gene < self.n_layers:
                gene_set = set(self.parameters['n_neurons']) - {child.get_gene(chosen_gene)} 
            elif chosen_gene < self.n_layers * 2:
                gene_set = set(self.parameters['activation']) - {child.get_gene(chosen_gene)}
            elif chosen_gene == self.n_layers * 2:
                gene_set = set(self.parameters['optimizer']) - {child.get_gene(chosen_gene)}
            elif chosen_gene == self.n_layers * 2 + 1:
                gene_set = set(self.parameters['loss']) - {child.get_gene(chosen_gene)}
            else:
                print("Error: gene index out of range")
                
            gene = np.random.choice(list(gene_set))
            child.set_gene(chosen_gene, gene)
            
            offspring.append(child)
            parent_ids.pop(parent_ids.index(parent))
            
        return offspring, parent_ids
        
        
    def next_generation(self):
        parent_ids = self.select_parents()
        offspring = []
        print("Performing Crossover")
        c_offspring, parent_ids, extra_parents = self.crossover(parent_ids)
        print("Performing Mutation")
        m_offspring, parent_ids = self.mutation(parent_ids, extra_parents)
        
        # if the number of offspring is less than the desired number,
        # then we need to create more offspring
        print("Compiling offpsring")
        offspring = c_offspring + m_offspring
        
        print(len(offspring))
        
        for i in range(self.n_pop):
            offspring[i].id = i

        self.population = offspring
    
    
    def get_best_network(self):
        return self.best_network
    
    def get_best_history(self):
        return self.best_history
    
    def get_best_model(self):
        return self.best_model
    
    def get_best_fitness(self):
        return self.best_fitness
    
    def get_genfitnesses(self):
        return self.genfitnesses



def main(g):
    # load the dataset
    # Load data
    # Load data
    cwd = Path.cwd()
    moondf = pickle.load(open(cwd / '..' / 'raw_data' / 'moonGen_scrape_2016_with_labels', 'rb'))
    # change the grade column from a number 4 - 14 to a list of 11 binary values
    y_temp = moondf['grade'].values
    moondf['grade'] = moondf['grade'].apply(lambda x: [1 if i == x else 0 for i in range(4, 15)])
    # one hot encode the grade column
    grade_cols = ['V_' + str(i) for i in range(4, 15)]
    moondf[grade_cols] = pd.DataFrame(moondf['grade'].to_list(), index=moondf.index)
    y = moondf[grade_cols].values
    X = moondf.drop(columns=grade_cols, axis=1)
    X = X.drop(["is_benchmark", "repeats", "grade"], axis=1).values
    

    X_train = np.zeros((0, 141))
    y_train = np.zeros((0, 11))
    X_rest = np.zeros((0, 141))
    y_rest = np.zeros((0, 11))
    sample_size = 2400
    for i in range(4, 15):
        if len(X[y_temp == i])*0.8 < sample_size:
            temp_size = int(len(X[y_temp == i]) * 0.8)
        else:
            temp_size = sample_size
        X_train = np.concatenate((X_train, X[y_temp == i][:temp_size]))
        y_train = np.concatenate((y_train, y[y_temp == i][:temp_size]))
        # remove those from the X and y
        X_rest = np.concatenate((X_rest, X[y_temp == i][temp_size:]))
        y_rest = np.concatenate((y_rest, y[y_temp == i][temp_size:]))



    genetic_algorithm = GA(100, 80, 20, (141,), 11, g.run_name)
    
    generation = 0
    stop_gen = 1000
    print("Initializing population")
    genetic_algorithm.initialize_population()
    
    while generation < stop_gen:
        print(f"Generation {generation}")
        start = time.time()
        print("Training population")
        genetic_algorithm.train_population(X_train, y_train, 50, 256, 
                                           (X_rest, y_rest), verbose=1)
        print("Evaluating population")
        genetic_algorithm.evaluate_population()
        print("Creating next generation")
        genetic_algorithm.next_generation()
        
        generation += 1
        end = time.time()
        
        print("Best fitness: ", genetic_algorithm.get_best_fitness())
        print(f"Generation {generation} complete in {end - start} seconds")
        

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('run_name', type=str, default='test')
    args = argparser.parse_args()
    return args

    
# main
if __name__ == "__main__":
    g = parse_args()
    main(g)
    
