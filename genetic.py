import numpy as np

# Define the genetic algorithm parameters
population_size = 10
num_generations = 10
mutation_rate = 0.1


# Function to initialize a population of decision trees
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        max_depth = np.random.randint(1, 10)
        min_samples_split = np.random.randint(2, 10)
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        population.append(tree)
    return population


# Function to calculate the fitness of each individual in the population
def calculate_fitness(population, X_train, y_train):
    fitness_scores = []
    for individual in population:
        individual.fit(X_train, y_train)
        y_pred = individual.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        fitness_scores.append(accuracy)
    return np.array(fitness_scores)


# Function to select individuals for the next generation using tournament selection
def tournament_selection(population, fitness_scores, tournament_size):
    selected_indices = []
    for _ in range(population_size):
        tournament_indices = np.random.choice(range(population_size), size=tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices.append(winner_index)
    return selected_indices


# Function to perform crossover between two individuals
def crossover(parent1, parent2):
    # For simplicity, let's swap the subtrees at a random depth
    max_depth = min(parent1.get_depth(), parent2.get_depth())
    crossover_depth = np.random.randint(1, max_depth)

    # Clone the parents to avoid modifying them
    child1 = parent1.clone()
    child2 = parent2.clone()

    # Perform crossover by swapping subtrees
    child1.tree_.children_left[:crossover_depth], child2.tree_.children_left[:crossover_depth] = \
        child2.tree_.children_left[:crossover_depth], child1.tree_.children_left[:crossover_depth].copy()

    child1.tree_.children_right[:crossover_depth], child2.tree_.children_right[:crossover_depth] = \
        child2.tree_.children_right[:crossover_depth], child1.tree_.children_right[:crossover_depth].copy()

    return child1, child2


# Function to perform mutation on an individual
def mutate(individual, mutation_rate):
    # Randomly change the max_depth or min_samples_split with probability mutation_rate
    if np.random.rand() < mutation_rate:
        if np.random.rand() < 0.5:
            individual.tree_.max_depth = np.random.randint(1, 10)
        else:
            individual.tree_.min_samples_split = np.random.randint(2, 10)
    return individual


# Main genetic algorithm loop
population = initialize_population(population_size)

for generation in range(num_generations):
    # Calculate fitness scores
    fitness_scores = calculate_fitness(population, X_train, y_train)

    # Select individuals for the next generation using tournament selection
    selected_indices = tournament_selection(population, fitness_scores, tournament_size=3)

    # Create the next generation through crossover and mutation
    next_generation = []
    for i in range(0, population_size, 2):
        parent1 = population[selected_indices[i]]
        parent2 = population[selected_indices[i + 1]]

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)

        next_generation.extend([child1, child2])

    population = next_generation

# Select the best individual from the final population
final_fitness_scores = calculate_fitness(population, X_train, y_train)
best_individual_index = np.argmax(final_fitness_scores)
best_individual = population[best_individual_index]

# Evaluate the best individual on the test set
y_pred_test = best_individual.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Best Decision Tree Parameters:")
print("Max Depth:", best_individual.tree_.max_depth)
print("Min Samples Split:", best_individual.tree_.min_samples_split)
print("Accuracy on Test Set:", accuracy_test)
