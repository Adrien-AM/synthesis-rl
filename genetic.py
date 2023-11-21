import numpy as np
from controller import *

#
# # Main genetic algorithm loop
# population = initialize_population(population_size)
#
# for generation in range(num_generations):
#     # Calculate fitness scores
#     fitness_scores = calculate_fitness(population, env)
#
#     # Select individuals for the next generation using tournament selection
#     selected_indices = tournament_selection(population, fitness_scores, tournament_size=3)
#
#     # Create the next generation through crossover and mutation
#     next_generation = []
#     for i in range(0, population_size, 2):
#         parent1 = population[selected_indices[i]]
#         parent2 = population[selected_indices[i + 1]]
#
#         child1, child2 = crossover(parent1, parent2)
#
#         child1 = mutate(child1, mutation_rate)
#         child2 = mutate(child2, mutation_rate)
#
#         next_generation.extend([child1, child2])
#
#     population = next_generation
#
# # Select the best individual from the final population
# final_fitness_scores = calculate_fitness(population, X_train, y_train)
# best_individual_index = np.argmax(final_fitness_scores)
# best_individual = population[best_individual_index]
#
# # Evaluate the best individual on the test set
# y_pred_test = best_individual.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
#
# print("Best Decision Tree Parameters:")
# print("Max Depth:", best_individual.tree_.max_depth)
# print("Min Samples Split:", best_individual.tree_.min_samples_split)
# print("Accuracy on Test Set:", accuracy_test)
