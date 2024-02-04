from typing import Callable, List, Optional, Tuple
from synth.syntax.program import Program
from synth.semantic import DSLEvaluator
from synth.syntax.program import Program, Variable
from pid import PIDController

import gymnasium as gym
import numpy as np

def make_env() -> Tuple[gym.Env, float]:
    """
    Creates a Gym environment and sets a default goal score.

    Returns:
    - Tuple[gym.Env, float]: Tuple containing the Gym environment and the goal score.
    """
    return gym.make("CartPole-v1"), 400 

def get_returns(episodes: List[List[Tuple[np.ndarray, int, float]]]) -> List[float]:
    """
    Computes the total returns for each episode.

    Parameters:
    - episodes (List[List[Tuple[np.ndarray, int, float]]]): List of episodes.

    Returns:
    - List[float]: List containing the total returns for each episode.
    """
    return [sum([trans[2] for trans in episode]) for episode in episodes]

def __state2env__(state: np.ndarray) -> Tuple:
    """
    Converts a NumPy array state to a tuple.

    Parameters:
    - state (np.ndarray): NumPy array representing the state.

    Returns:
    - Tuple: Tuple representation of the state.
    """
    return state.tolist()

def eval_function(env: gym.Env, pids: List[PIDController], evaluator: DSLEvaluator) -> Callable[[Program], Tuple[bool, float]]:
    """
    Creates a function for evaluating a program's performance in the given environment.

    Parameters:
    - env (gym.Env): Gym environment for evaluation.
    - pids (List[PIDController]): List of PID controllers for the program.
    - evaluator (DSLEvaluator): Evaluator for the Domain-Specific Language (DSL).

    Returns:
    - Callable[[Program], Tuple[bool, float]]: Evaluation function taking a program and returning a tuple of success status and mean reward.
    """
    def func(program: Program, n : int=1) -> Tuple[bool, float]:
        int, episodes = eval_program(env, program, evaluator, pids, n)
        return int, np.mean(get_returns(episodes)) if int else 0
    return func

def get_nb_variables(program: Program):
    """
    Counts the number of variables in a program.

    Parameters:
    - program (Program): The program to be analyzed.

    Returns:
    - int: Number of variables in the program.
    """
    count = 0
    for sub_prog in program.depth_first_iter():
        if isinstance(sub_prog, Variable):
            count += 1
    return count

def get_variables_index(program: Program):
    """
    """    """
    Gets the indices of variables in a program.

    Parameters:
    - program (Program): The program to be analyzed.

    Returns:
    - List[int]: List of indices of variables in the program.
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
    Creates a list of PID controllers.

    Parameters:
    - n_vars (int): Number of PID controllers to create.

    Returns:
    - List[PIDController]: List of PID controllers with random parameters.
    """
    return [PIDController(np.random.random(), 0, np.random.randint(0,50), setpoint=0)]*n_vars

def update_pids(pids: List[PIDController], input: np.ndarray, indices: List[int]) -> List:
    """
    Updates PID controllers and returns the new input based on the specified indices.

    Parameters:
    - pids (List[PIDController]): List of PID controllers.
    - program (Program): The program to update PID controllers.
    - input (np.ndarray): The input to the program.
    - indices (List[int]): List of indices specifying which variables to update.

    Returns:
    - List: The new input after updating PID controllers.
    """
    new_input = [0]*len(input)
    for i, val in enumerate(indices):
        result = pids[i].update(input[val])
        new_input[val] = result
    return new_input

def eval_program(
        env: gym.Env,
        program: Program,
        evaluator: DSLEvaluator,
        pids: List[PIDController],
        n: int
    ) -> Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]:
    """
    Evaluates a program in the given environment for a specified number of episodes.

    Parameters:
    - env (gym.Env): Gym environment for evaluation.
    - program (Program): The program to be evaluated.
    - evaluator (DSLEvaluator): Evaluator for the Domain-Specific Language (DSL).
    - pids (List[PIDController]): List of PID controllers.
    - n (int): Number of episodes to run.

    Returns:
    - Tuple[bool, Optional[List[List[Tuple[np.ndarray, int, float]]]]]: A tuple indicating success and the list of episodes.
    """
    # assumes that the program does not include constants
    episodes = []
    try:
        indices = get_variables_index(program)
        # n_vars = get_nb_variables(program)
        for _ in range(n):
            episode = []
            state, _ = env.reset()
            done = False
            # # Create a pid controller for each variable
            # pids = create_pids(n_vars)
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
