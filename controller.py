import gymnasium as gym
import math
import time

THRESHOLD = 1

env = gym.make("LunarLander-v2", enable_wind=True, wind_power=10, render_mode="human")
observation, info = env.reset()

class Node:
    def __init__(self, pred, timed=True):
        self.predicate = pred
        self.timed = timed

    def add_child_false(self, child):
        self.child_false = child

    def add_child_true(self, child):
        self.child_true = child

    def forward(self, observation, t):
        if t > THRESHOLD and self.timed:
            if observation[3] < -0.9:
                return 2
            else:
                return 0
        if self.predicate(observation):
            return self.child_true.forward(observation, t)
        else:
            return self.child_false.forward(observation, t)

class Leaf(Node):
    def __init__(self, action):
        self.action = action

    def forward(self, observation, t):
        print(self.action)
        return self.action
    


if __name__ == "__main__":
    # x, y, vx, vy, a, va, l1, l2 = observation
    # TODO : Si vitesse angulaire trop élevée mais position OK, contrebalancer un peu la rotation
    nothing = Leaf(0)
    fire_right = Leaf(1)
    fire_main = Leaf(2)
    fire_left = Leaf(3)

    tree = Node(lambda o : o[0] > 0)
    
    angle_left = Node(lambda o : o[4] > 0)
    angle_right = Node(lambda o : o[4] > 0)

    v1 = Node(lambda o : o[3] < -0.8)
    v1.add_child_false(nothing)
    v1.add_child_true(fire_main)
    v2 = Node(lambda o : o[3] < -0.8)
    v2.add_child_false(fire_left)
    v2.add_child_true(fire_left)
    v3 = Node(lambda o : o[3] < -0.8)
    v3.add_child_false(fire_right)
    v3.add_child_true(fire_right)
    v4 = Node(lambda o : o[3] < -0.8)
    v4.add_child_false(nothing)
    v4.add_child_true(fire_main)

    angle_left.add_child_false(v1)
    angle_left.add_child_true(v2)
    angle_right.add_child_false(v3)
    angle_right.add_child_true(v4)

    tree.add_child_false(angle_left)
    tree.add_child_true(angle_right)

    init_time = time.time()

    done = False
    i = 0
    while not done:
        t = time.time()
        t = t - init_time
        move = tree.forward(observation, t)
        if move == 0:
            print(i)
        observation, reward, _, info, done = env.step(move)
        env.render()
        i += 1

    env.close()