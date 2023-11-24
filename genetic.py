import numpy as np
from pid import *
# Define the objective function (fitness function)
def objective_function(x):
    return evaluate(1000, LunarController(x[:3], x[3:]))

# Initialize the population
def initialize_population(population_size, vector_size, origin_vector):
    v_random = origin_vector + np.random.rand(population_size-10, vector_size) * 2 - 1
    vo = np.array([origin_vector]*(population_size-10))
    population = np.concatenate((v_random, vo))
    return population

# Evaluate the fitness of each individual in the population
def calculate_fitness(population):
    return np.array([objective_function(individual) for individual in population])

# Select parents for crossover using tournament selection
def select_parents(population, fitness, tournament_size=3):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        selected_parent_index = tournament_indices[np.argmin(tournament_fitness)]
        selected_parents.append(population[selected_parent_index])
    return np.array(selected_parents)

# Perform crossover to create offspring
def crossover(parents, crossover_rate=0.8):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    return np.array(offspring)

# Perform mutation on the offspring
def mutate(offspring, mutation_rate=0.1):
    mutated_offspring = offspring.copy()
    for i in range(len(mutated_offspring)):
        for j in range(len(mutated_offspring[i])):
            if np.random.rand() < mutation_rate:
                mutated_offspring[i][j] = np.random.rand() * 2 - 1
    return mutated_offspring



