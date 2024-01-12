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


# def update_pids(pids: List[PIDController], program: Program, input: np.ndarray) -> List:
#     new_input = [0] * len(input)
#     for idx, val in enumerate(program.used_variables()):
#         result = pids[idx].update(input[val])
#         new_input[val] = result
#     return new_input

def update_pids(pids: List[PIDController], program: Program, input: np.ndarray) -> List:
    new_input = [0] * len(input)
    idx = [0,1,4]
    for i, val in enumerate(idx):
        result = pids[i].update(input[val])
        new_input[val] = result
    return new_input

def get_nb_variables():
    # Save the previous sub_prog, if the actual is instance of Variable, count++.
    ...

def get_variables_index():
    # Save the previous sub_prog, if the actual is instance of Variable, save the index of the previous one.
    ...



def eval_program(env: gym.Env, program, evaluator: DSLEvaluator, n: int) -> Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]:
    # assumes that the program does not include constants
    episodes = []
    try:
        for _ in range(n):
            episode = []
            state, _ = env.reset()
            done = False
            # Create a pid controller for each variable
            pids = [PIDController(np.random.random(), 0, np.random.randint(0,50), setpoint=0)] * 3

            while not done:
                input = __state2env__(state)
                # update all pids and get new input
                input = update_pids(pids, program, input)
                input = [input]

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

# def optimisation(programs, eval_func, reward_min, n=100):
#     probas = [1 / len(programs)] * len(programs)
#     best_constants = [c for c in programs[0].constants()]
#     best_result = eval_func(programs[0])[1]
#     best_program = programs[0]
#     for i in range(n):
#         # select program
#         program_idx = np.random.choice(np.arange(len(programs)), p=probas)
#         program = programs[program_idx]
#         # evaluate
#         int, reward = eval_func(program)
#         # update probas
#         if int:
#             probas[program_idx] *= 1.1
#         else:
#             probas[program_idx] *= 0.9
#         probas = [proba / sum(probas) for proba in probas]
#     return best_program


# if __name__ == '__main__':
#     optimisation([1,2,3,4,5])

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
