from dsl import DSL
from typing import Callable, List, Optional, Tuple
from program import Constant, Program

import gymnasium as gym
import numpy as np

def make_env() -> Tuple[gym.Env, float]:
    return gym.make("LunarLander-v2"), 200 

def get_returns(episodes: List[List[Tuple[np.ndarray, int, float]]]) -> List[float]:
    return [sum([trans[2] for trans in episode]) for episode in episodes]

def __state2env__(state: np.ndarray) -> Tuple:
    return state.tolist()

def eval_function(env: gym.Env, evaluator: DSLEvaluator) -> Callable[[Program], Tuple[bool, float]]:
    def func(program: Program) -> Tuple[bool, float]:
        int, episodes = eval_program(env, program, evaluator, 1)
        return int, get_returns(episodes)[0] if int else 0
    return func

def eval_program(env: gym.Env, program, evaluator: DSLEvaluator, n: int) -> Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]:
    # assumes that the program does not include constants
    episodes = []
    try:
        for _ in range(n):
            episode = []
            state = env.reset()
            done = False
            while not done:
                input = __state2env__(state)
                action = evaluator.eval(program, input)
                if action not in env.action_space:
                    return False, None
                next_state, reward, done, truncated, _ = env.step(action)
                done |= truncated
                episode.append((state.copy(), action, reward))
                state = next_state
            episodes.append(episode)
    except OverflowError:
        return False, None
    return True, episodes

# enumerate all constants in a program
# for const in prog.constants:
# !!if prog does not have constants, yields None!!
# !!after the loop you MUST reset all constants!!
# assign a value
# const.assign(value)
# resets
# const.reset()

# possible_values = {
#     INT: [0,1,2,3]
#     STRING: ["", "a"]
# }
# for instantiated_prog in prog.all_constants_instantiation(possible_values):

# alternative: new_pcfg = pcfg.instantiate_constants(possible_values)
# yields programs with constants but already assigned
