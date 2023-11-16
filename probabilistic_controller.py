import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

ACTION_NOTHING = 0
ACTION_LEFT = 1
ACTION_MAIN = 2
ACTION_RIGHT = 3

class Node():
    def __init__(self, predicat, child_true=None, child_false=None):
        self.predicat = predicat
        self.child_true = child_true
        self.child_false = child_false

    def add_child_true(self, child):
        self.child_true = child

    def add_child_false(self, child):
        self.child_false = child

    def forward(self, observation):
        if self.predicat(observation):
            return self.child_true.forward(observation)
        return self.child_false.forward(observation)

class Leaf(Node):
    def __init__(self, action):
        self.action = action

    def forward(self, observation):
        return self.action
    
class Probabilistic_Leaf(Node):
    def __init__(self, prob_actions):
        self.prob_actions = prob_actions

    def update(self):
        pass

    def forward(self, observation):
        action = np.random.choice(len(self.prob_actions), p=self.prob_actions)
        return action

def test_tree(tree: Node, env: gym.Env):
    """
    """
    observation, info = env.reset()
    done = False
    cum_reward = 0
    n_steps = 0
    while not done:
        next_action = tree.forward(observation)
        observation, reward, done, _, info = env.step(next_action)
        cum_reward += reward
        n_steps += 1
        env.render()
    return n_steps, cum_reward

def make_tree_from_nodes(list_nodes: list[Node], n_observations: int):
    """
    """
    n_nodes = len(list_nodes)
    n_steps = int((n_nodes+1)/2)-1
    child_index = 1
    
    for i in range(n_steps):
        list_nodes[i].add_child_true(list_nodes[child_index])
        list_nodes[i].add_child_false(list_nodes[child_index+1])
        child_index += 2
    
    for i in range(n_steps, n_nodes):
        list_nodes[i].add_child_true(Probabilistic_Leaf([0.25, 0.25, 0.25, 0.25]))
        list_nodes[i].add_child_false(Probabilistic_Leaf([0.25, 0.25, 0.25, 0.25]))
        child_index += 2
    return list_nodes[0]

if __name__ == "__main__":
    n_observations = len(observation) - 2
    paramaters = [0 for i in range(n_observations)]

    # Naive tree
    tree_nodes = [Node(lambda o: o[i] > paramaters[i]) for i in range(n_observations) for _ in range(2**i)]
    tree = make_tree_from_nodes(tree_nodes, n_observations)

    env.close()
