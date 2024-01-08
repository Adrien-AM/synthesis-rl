# from dsl import DSL
from typing import Callable, List, Optional, Tuple
from synth.syntax.program import Constant, Program
from synth.semantic import DSLEvaluator
from pid import PIDController

import gymnasium as gym
import numpy as np

def make_env() -> Tuple[gym.Env, float]:
    return gym.make("LunarLander-v2"), 200 

def get_returns(episodes: List[List[Tuple[np.ndarray, int, float]]]) -> List[float]:
    return [sum([trans[2] for trans in episode]) for episode in episodes]

def __state2env__(state: np.ndarray) -> Tuple:
    return state.tolist()

def eval_function(env: gym.Env, evaluator: DSLEvaluator) -> Callable[[Program], Tuple[bool, float]]:
    def func(program: Program, n : int=1) -> Tuple[bool, float]:
        int, episodes = eval_program(env, program, evaluator, n)
        return int, get_returns(episodes)[0] if int else 0
    return func


def update_pids(pids: List[PIDController], program: Program, input: np.ndarray) -> List:
    new_input = [0] * len(input)
    for idx, val in enumerate(program.used_variables()):
        result = pids[idx].update(input[val])
        new_input[val] = result
    return new_input


def eval_program(env: gym.Env, program, evaluator: DSLEvaluator, n: int) -> Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]:
    # assumes that the program does not include constants
    episodes = []
    try:
        for _ in range(n):
            episode = []
            state, _ = env.reset()
            done = False
            # Create a pid controller for each variable
            pids = [PIDController(0.5, 0, 25, setpoint=0) for _ in program.used_variables()]

            while not done:
                input = __state2env__(state)
                # update all pids and get new input
                input = update_pids(pids, program, input)

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
