from typing import Callable, List, Optional, Tuple
from synth.syntax.program import Program
from synth.semantic import DSLEvaluator
from synth.syntax.program import Program, Variable
from pid import PIDController

import gymnasium as gym
import numpy as np

def make_env() -> Tuple[gym.Env, float]:
    """
    """
    return gym.make("LunarLander-v2"), 200 

def get_returns(episodes: List[List[Tuple[np.ndarray, int, float]]]) -> List[float]:
    """
    """
    return [sum([trans[2] for trans in episode]) for episode in episodes]

def __state2env__(state: np.ndarray) -> Tuple:
    """
    """
    return state.tolist()

def eval_function(env: gym.Env, evaluator: DSLEvaluator) -> Callable[[Program], Tuple[bool, float]]:
    """
    """
    def func(program: Program, n : int=1) -> Tuple[bool, float]:
        int, episodes = eval_program(env, program, evaluator, n)
        return int, np.mean(get_returns(episodes)) if int else 0
    return func

def get_nb_variables(program: Program):
    """
    """
    count = 0
    for sub_prog in program.depth_first_iter():
        if isinstance(sub_prog, Variable):
            count += 1
    return count

def get_variables_index(program: Program):
    """
    """
    pre_prog_list = []
    pre_prog = None
    for sub_prog in program.depth_first_iter():
        if isinstance(sub_prog, Variable):
            pre_prog_list.append(int(pre_prog.primitive))
        pre_prog = sub_prog
    return pre_prog_list

def create_pids(n_vars: int):
    """
    """
    return [PIDController(np.random.random(), 0, np.random.randint(0,50), setpoint=0)]*n_vars

def update_pids(pids: List[PIDController], program: Program, input: np.ndarray, indices: List[int]) -> List:
    """
    """
    new_input = [0]*len(input)
    for i, val in enumerate(indices):
        result = pids[i].update(input[val])
        new_input[val] = result
    return new_input

def eval_program(env: gym.Env, program, evaluator: DSLEvaluator, n: int) -> Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]:
    """
    """
    # assumes that the program does not include constants
    episodes = []
    try:
        indices = get_variables_index(program)
        n_vars = get_nb_variables(program)
        for _ in range(n):
            episode = []
            state, _ = env.reset()
            done = False
            # Create a pid controller for each variable
            pids = create_pids(n_vars)
            while not done:
                input = __state2env__(state)
                # update all pids and get new input
                input = update_pids(pids, program, input, indices)
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
