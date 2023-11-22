import gymnasium as gym
import math
import time
import random
import numpy as np

THRESHOLD = 1

# env = gym.make("LunarLander-v2", enable_wind=True, wind_power=10, render_mode="human")
env = gym.make("LunarLander-v2", enable_wind=True, wind_power=10)
observation, info = env.reset()

class Node:
    def __init__(self, pred):
        self.predicate = pred

    def add_child_false(self, child):
        self.child_false = child

    def add_child_true(self, child):
        self.child_true = child

    def forward(self, observation, t):
        if self.predicate(observation):
            return self.child_true.forward(observation, t)
        else:
            return self.child_false.forward(observation, t)


class Leaf(Node):
    def __init__(self, action, p=0.1, diffT=0.3, delta=0.1, speedT=0.3):
        self.action = action
        self.p = p
        self.time_diff = 0
        self.prev_time = 0
        self.diffT = diffT
        self.delta = delta
        self.speedT = speedT

    def forward(self, observation, t):
        self.time_diff += t - self.prev_time
        self.prev_time = t
        if self.time_diff > self.diffT:
            self.time_diff = 0
            self.p += self.delta
    
        if observation[3] < -self.speedT:
            if self.p > random.random():
                if self.p > self.delta:
                    self.p -= self.delta
                return random.choice([2, self.action])  # Random action
        return self.action


def randomizedOptimization(nb_iter=300, nb_test=30):
    ps = np.linspace(0.1, 0.9, 100)
    diffTs = np.linspace(0.1, 0.9, 100)
    deltas = np.linspace(0.01, 0.3, 100)
    speedTs = np.linspace(0.1, 0.6, 100)
    angle = 0
    speed = 0.3
    best = -200
    best_params = []

    seeds = np.random.randint(0, 100000, nb_test)
    
    for i in range(nb_iter):
        if (i+1) % 20 == 0:
            print("Episode : ", i+1)

        p = np.random.choice(ps)
        diffT = np.random.choice(diffTs)
        delta = np.random.choice(deltas)
        speedT = np.random.choice(speedTs)


        nothing = Leaf(0, p, diffT, delta, speedT)
        fire_right = Leaf(1, p, diffT, delta, speedT)
        fire_main = Leaf(2, p, diffT, delta, speedT)
        fire_left = Leaf(3, p, diffT, delta, speedT)

        tree = Node(lambda o : o[0] > 0)
        
        angle_left = Node(lambda o : o[4] > angle)
        angle_right = Node(lambda o : o[4] > angle)

        v1 = Node(lambda o : o[3] < -speed)
        v1.add_child_false(nothing)
        v1.add_child_true(fire_main)
        v2 = Node(lambda o : o[3] < -speed)
        v2.add_child_false(fire_left)
        v2.add_child_true(fire_left)
        v3 = Node(lambda o : o[3] < -speed)
        v3.add_child_false(fire_right)
        v3.add_child_true(fire_right)
        v4 = Node(lambda o : o[3] < -speed)
        v4.add_child_false(nothing)
        v4.add_child_true(fire_main)

        angle_left.add_child_false(v1)
        angle_left.add_child_true(v2)
        angle_right.add_child_false(v3)
        angle_right.add_child_true(v4)

        tree.add_child_false(angle_left)
        tree.add_child_true(angle_right)

        total_reward = 0
        for j in range(nb_test):
            observation, _ = env.reset(seed=int(seeds[j]))
            init_time = time.time()
            done = False
            while not done:
                t = time.time()
                t = t - init_time
                move = tree.forward(observation, t)
                observation, reward, truncated, _, done = env.step(move)
                done = done or truncated
                total_reward += reward
            
        mean_reward = total_reward / 20
        # print("Mean reward : ", mean_reward)
        if mean_reward > best:
            print("Mean reward : ", mean_reward)
            print("Parameters : ", [p, diffT, delta, speedT])
            best = mean_reward
            best_params = [p, diffT, delta, speedT]
    
    return best, best_params


def randomOptimizer(nb_iter=300, nb_test=30):
    init_p = np.random.uniform(0.1, 0.9)
    init_diffT = np.random.uniform(0.1, 0.9)
    init_delta = np.random.uniform(0.01, 0.3)
    init_speedT = np.random.uniform(0.1, 0.6)
    angle = 0
    speed = 0.3
    best = -200
    best_params = [init_p, init_diffT, init_delta, init_speedT]

    seeds = np.random.randint(0, 100000, nb_test)
    
    for i in range(nb_iter):
        if (i+1) % 20 == 0:
            print("Episode : ", i+1)

        p = init_p + np.random.uniform(-0.1, 0.1)
        diffT = init_diffT + np.random.uniform(-0.1, 0.1)
        delta = init_delta + np.random.uniform(-0.1, 0.1)
        speedT = init_speedT + np.random.uniform(-0.1, 0.1)

        nothing = Leaf(0, p, diffT, delta, speedT)
        fire_right = Leaf(1, p, diffT, delta, speedT)
        fire_main = Leaf(2, p, diffT, delta, speedT)
        fire_left = Leaf(3, p, diffT, delta, speedT)

        tree = Node(lambda o : o[0] > 0)
        
        angle_left = Node(lambda o : o[4] > angle)
        angle_right = Node(lambda o : o[4] > angle)

        v1 = Node(lambda o : o[3] < -speed)
        v1.add_child_false(nothing)
        v1.add_child_true(fire_main)
        v2 = Node(lambda o : o[3] < -speed)
        v2.add_child_false(fire_left)
        v2.add_child_true(fire_left)
        v3 = Node(lambda o : o[3] < -speed)
        v3.add_child_false(fire_right)
        v3.add_child_true(fire_right)
        v4 = Node(lambda o : o[3] < -speed)
        v4.add_child_false(nothing)
        v4.add_child_true(fire_main)

        angle_left.add_child_false(v1)
        angle_left.add_child_true(v2)
        angle_right.add_child_false(v3)
        angle_right.add_child_true(v4)

        tree.add_child_false(angle_left)
        tree.add_child_true(angle_right)

        total_reward = 0
        for j in range(nb_test):
            observation, _ = env.reset(seed=int(seeds[j]))
            init_time = time.time()
            done = False
            while not done:
                t = time.time()
                t = t - init_time
                move = tree.forward(observation, t)
                observation, reward, truncated, _, done = env.step(move)
                done = done or truncated
                total_reward += reward
            
        mean_reward = total_reward / 20
        if mean_reward > best:
            print("Mean reward : ", mean_reward)
            print("Parameters : ", [p, diffT, delta, speedT])
            best = mean_reward
            best_params = [p, diffT, delta, speedT]
        
        init_p, init_diffT, init_delta, init_speedT = best_params
    
    return best, best_params


if __name__ == "__main__":
    print("Randomized optimization")
    best_params_1 = randomizedOptimization(nb_iter=1500, nb_test=20)

    print("Random optimizer")
    best_params_2 = randomOptimizer(nb_iter=1500, nb_test=20)

    print("Best reward 1 : ", best_params_1[0])
    print("Best parameters 2 : ", best_params_1[1])
    print("Best reward 2 : ", best_params_2[0])
    print("Best parameters 2 : ", best_params_2[1])

# Best reward 2 :  -71.72182293387648
# Best parameters 2 :  [0.33652405083830694, 1.0099550012525433, 0.07304317731214606, 0.48982303070654243]