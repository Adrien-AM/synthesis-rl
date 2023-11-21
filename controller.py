import random
import gymnasium as gym
import numpy as np

from genetic import *

# Define the genetic algorithm parameters
population_size = 4
num_generations = 10
mutation_rate = 0.1


# Function to initialize a population of decision trees
def initialize_population(population_size, max_depth = 7):
    population = []
    for _ in range(population_size):
        tree = build_rand_tree(max_depth)
        population.append(tree)
    return population


# Function to calculate the fitness of each individual in the population
def calculate_fitness(population, env):
    fitness_scores = []
    observation, _ = env.reset()
    reward = 0
    for individual in population:
        done = False
        terminated = False
        while not (done or terminated):
            move = individual.forward(observation)
            move = np.random.choice(3, 1, p=move)[0]
            observation, reward, terminated, info, done = env.step(move)
        fitness_scores.append(reward)
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
    # max_depth = min(parent1.get_depth(), parent2.get_depth())
    # crossover_depth = np.random.randint(1, max_depth)

    # Clone the parents to avoid modifying them
    child1 = parent1.clone()
    child2 = parent2.clone()

    # Perform crossover by swapping subtrees
    child1.child_true, child2.child_true = \
        child2.child_true, child1.child_true.copy()

    child1.child_false, child2.child_false = \
        child2.child_false, child1.child_false.copy()

    return child1, child2


# Function to perform mutation on an individual
def mutate(individual, mutation_rate):
    # Randomly change the leaf values of the tree
    if np.random.rand() < mutation_rate:
        individual = build_rand_tree(7)
    return individual


env = gym.make("LunarLander-v2", enable_wind=True, wind_power=10)
env_rend = gym.make("LunarLander-v2", enable_wind=True, wind_power=10, render_mode="human")
observation, info = env.reset()

observation_high = [1.5, 1.5, 5., 5., 3.14, 5., 1., 1.]
observation_low = [-1.5, -1.5, -5., -5., -3.14, -5., -0., -0.]

class Node:
    def __init__(self, pred):
        self.predicate = pred

    def add_child_false(self, child):
        self.child_false = child

    def add_child_true(self, child):
        self.child_true = child

    def forward(self, observation):
        if self.predicate(observation):
            return self.child_true.forward(observation)
        else:
            return self.child_false.forward(observation)

    def clone(self):
        clone = Node(self.predicate)
        clone.add_child_false(self.child_false)
        clone.add_child_true(self.child_true)
        return clone

    def __copy__(self):
        return self.clone()

def build_rand_tree(depth):
    if depth == 0:
        return Leaf(np.random.dirichlet(np.ones(4)))
    else:
        node = Node(lambda o : o[depth] > random.uniform(observation_low[depth], observation_high[depth]))
        node.add_child_false(build_rand_tree(depth - 1))
        node.add_child_true(build_rand_tree(depth - 1))
        return node

class Leaf(Node):
    def __init__(self, action):
        self.action = action
        
    def forward(self, observation):
        return self.action




if __name__ == "__main__":
    # x, y, vx, vy, a, va, l1, l2 = observation
    # TODO : Si vitesse angulaire trop élevée mais position OK, contrebalancer un peu la rotation
    # nothing = Leaf(0)
    # fire_right = Leaf(1)
    # fire_main = Leaf(2)
    # fire_left = Leaf(3)
    #
    # tree = Node(lambda o : o[0] > 0)
    #
    # angle_left = Node(lambda o : o[4] > 0)
    # angle_right = Node(lambda o : o[4] > 0)
    #
    # v1 = Node(lambda o : o[3] < -0.8)
    # v1.add_child_false(nothing)
    # v1.add_child_true(fire_main)
    # v2 = Node(lambda o : o[3] < -0.8)
    # v2.add_child_false(fire_left)
    # v2.add_child_true(fire_left)
    # v3 = Node(lambda o : o[3] < -0.8)
    # v3.add_child_false(fire_right)
    # v3.add_child_true(fire_right)
    # v4 = Node(lambda o : o[3] < -0.8)
    # v4.add_child_false(nothing)
    # v4.add_child_true(fire_main)
    #
    # angle_left.add_child_false(v1)
    # angle_left.add_child_true(v2)
    # angle_right.add_child_false(v3)
    # angle_right.add_child_true(v4)
    #
    # tree.add_child_false(angle_left)
    # tree.add_child_true(angle_right)


    print("Building tree")

    pop = initialize_population(population_size=100, max_depth=7)

    tree = pop[0]

    env_rend.reset()

    done = False
    terminated = False
    while not (done or terminated):
        move = tree.forward(observation)
        move = np.random.choice(4, 1, p=move)[0]
        observation, reward, terminated, info, done = env_rend.step(move)
        env_rend.render()

    env.reset()

    env_rend.close()

    for generation in range(num_generations):
        print(f"Generation {generation} started")
        # Calculate fitness scores
        fitness_scores = calculate_fitness(pop, env)

        # Select individuals for the next generation using tournament selection
        selected_indices = tournament_selection(pop, fitness_scores, tournament_size=3)

        # Create the next generation through crossover and mutation
        next_generation = []
        for i in range(0, population_size, 2):
            print(f"Generation {generation} - {i}/{population_size}")
            parent1 = pop[selected_indices[i]]
            parent2 = pop[selected_indices[i + 1]]

            print("Crossover")
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate=0.1)
            child2 = mutate(child2, mutation_rate=0.1)

            print("Appending")
            next_generation.append(child1)
            next_generation.append(child2)

        pop = next_generation
        print(f"Generation {generation} finished")

    print(fitness_scores)


    # keep the best individual
    best_index = np.argmax(fitness_scores)
    tree = pop[best_index]


    done = False
    while not done:
        move = tree.forward(observation)
        observation, reward, _, info, done = env_rend.step(move)
        env_rend.render()

    env_rend.close()