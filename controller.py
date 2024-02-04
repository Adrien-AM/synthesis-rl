import gymnasium as gym

env = gym.make("LunarLander-v2", wind_power=5, render_mode="human")
observation, info = env.reset()

def play(observation):
    x, y, vx, vy, a, va, l1, l2 = observation

    if l1 and l2:
        return 0

    if x > 0.2:
        return 1
    if x < -0.2:
        return 3
    
    if a > 0.1:
        return 3
    if a < -0.1:
        return 1
    
    if abs(vy) > 0.8:
        return 2

    return 0

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

class Leaf(Node):
    def __init__(self, action):
        self.action = action
        
    def forward(self, observation):
        return self.action

if __name__ == "__main__":
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

    done = False
    while not done:
        move = tree.forward(observation)
        observation, reward, _, info, done = env.step(move)
        env.render()

    env.close()