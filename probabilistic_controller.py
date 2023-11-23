import gymnasium as gym
import numpy as np

import time
import hill_climbing



env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

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

    def forward(self, observation):
        if np.sum(self.prob_actions) == 0:
            self.prob_actions = [0.25, 0.25, 0.25, 0.25]
        if any(x < 0 for x in self.prob_actions):
            self.prob_actions = [0.25 if x < 0 else x for x in self.prob_actions]
        action = np.random.choice(len(self.prob_actions), p=self.prob_actions/np.sum(self.prob_actions)) # We normalize to have always sum(p)=1
        return action

def test_tree(tree: Node, env: gym.Env, render: bool=False):
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
        if render:
            env.render()
    print(f'Total reward: {np.round(cum_reward, 3)}')
    return np.round(cum_reward, 3)
    # return n_steps, cum_reward

def make_tree_from_nodes(list_nodes: list[Node], init_child: bool=True):
    """
    """
    n_nodes = len(list_nodes)
    n_steps = int((n_nodes+1)/2)-1
    child_index = 1
    
    for i in range(n_steps):
        list_nodes[i].add_child_true(list_nodes[child_index])
        list_nodes[i].add_child_false(list_nodes[child_index+1])
        child_index += 2
    if init_child:
        for i in range(n_steps, n_nodes):
            list_nodes[i].add_child_true(Probabilistic_Leaf([0.25, 0.25, 0.25, 0.25]))
            list_nodes[i].add_child_false(Probabilistic_Leaf([0.25, 0.25, 0.25, 0.25]))
    return list_nodes[0]

def eval_func(parameters):
        print("Parameters to evaluate:", parameters)
        theta_condition_1.add_child_true(Probabilistic_Leaf(parameters[0]))
        theta_condition_1.add_child_false(Probabilistic_Leaf(parameters[1]))

        theta_condition_2.add_child_true(Probabilistic_Leaf(parameters[2]))
        theta_condition_2.add_child_false(Probabilistic_Leaf(parameters[3]))

        return test_tree(tree_balance, env)

def save_parameters(parameter_list):
    np.save("weights.pt", np.array(parameter_list))
    return True

def load_parameters():
    return np.load("weights.pt")

if __name__ == "__main__":
    ACTION_NOTHING = Leaf(0)
    ACTION_LEFT = Leaf(1)
    ACTION_MAIN = Leaf(2)
    ACTION_RIGHT = Leaf(3)

    n_observations = len(observation) - 2
    obs_paramaters = [0 for i in range(n_observations)]

    # Naive tree
    # tree_nodes = [Node(lambda o: o[i] > obs_paramaters[i]) for i in range(n_observations) for _ in range(2**i)]
    # tree = make_tree_from_nodes(tree_nodes)
    
    # observation, info = env.reset()
    # test_tree(tree=tree, env=env)

    # Simple tree: balance the lander
    tree_balance = Node(lambda o: o[0] > obs_paramaters[0])
    
    theta_condition_1 = Node(lambda o: o[5] > obs_paramaters[5])
    theta_condition_2 = Node(lambda o: o[5] > obs_paramaters[5])

    tree_balance.add_child_true(theta_condition_1)
    tree_balance.add_child_false(theta_condition_2)

    init_parameter_leaves = [
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.7, 0.1, 0.1]
    ]
    
    theta_condition_1.add_child_true(Probabilistic_Leaf(init_parameter_leaves[0]))
    theta_condition_1.add_child_false(Probabilistic_Leaf(init_parameter_leaves[1]))

    theta_condition_2.add_child_true(Probabilistic_Leaf(init_parameter_leaves[2]))
    theta_condition_2.add_child_false(Probabilistic_Leaf(init_parameter_leaves[3]))

    # test_tree(tree_balance, env)

    hill_climbing_algo = hill_climbing.Hill_Climbing(parameters=init_parameter_leaves,\
                                                     eval_func=eval_func)
    
    EPISODES = 10
    start_time = time.time()
    for episode in range(EPISODES):
        print("EPISODE: ", episode)
        new_parameters = hill_climbing_algo.run_hill_climbing(10)
        print()
        print("--------------------------------")
        print("New parameters:", new_parameters)
        print("--------------------------------")
        print()
        eval_func(new_parameters)
        hill_climbing_algo.parameters = new_parameters
    total_time = time.time() - start_time
    print(f"Total time: {total_time}")
    
    save_parameters(new_parameters)

    env.close()
