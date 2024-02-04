from synth.semantic import DSLEvaluator
from synth.syntax import auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax import bps_enumerate_prob_grammar
from synth.syntax.grammars import ProbDetGrammar
from synth.syntax.type_system import Type
from synth.syntax.program import Program, Variable, Primitive
from prog_eval import eval_function, create_pids, get_nb_variables
from optim.hill_climbing import hill_climbing, get_pid_parameters
from utils import save_with_pickle
import gymnasium as gym
import time
import math

def create_semantics(observation_dimension, action_space):
    """
    Generate a generic semantic for the DSL with observation_dimension input variables.

    Parameters:
    - observation_dimension (int): Number of input variables.
    - action_space (int): Number of actions in the environment.

    Returns:
    - dict: Semantic dictionary mapping DSL primitives to lambda functions.
    """
    semantics = {
        # primitives
        "ite": lambda cond: lambda if_block: lambda else_block: if_block if cond > 0 else else_block,
        "scalar": lambda const: lambda inp: const * inp,
        "+": lambda float1: lambda float2: float1 + float2,
    }
    # Actions
    for i in range(action_space):
        semantics["act_" + str(i)] = i
    # Inputs
    for i in range(observation_dimension):
        semantics[str(i)] = lambda s: s[i]

    return semantics

def create_syntax(observation_dimension, action_space):
    """
    Generate a generic syntax for the DSL with observation_dimension input variables.

    Parameters:
    - observation_dimension (int): Number of input variables.
    - action_space (int): Number of actions in the environment.

    Returns:
    - dict: Syntax dictionary mapping DSL primitives to type specifications.
    """
    syntax = {
        # primitives
        "ite": "FLOAT -> ACTION -> ACTION -> ACTION",
        "scalar": "CONSTANT -> INPUT -> FLOAT",
        "+": "FLOAT -> FLOAT -> FLOAT",
    }
    # Actions
    for i in range(action_space):
        syntax["act_" + str(i)] = "ACTION"
    # Inputs
    for i in range(observation_dimension):
        syntax[str(i)] = "STATE -> INPUT"

    return auto_type(syntax)


def check_all_action_possible(program: Program, action_space: int):
    """
    Check if the program contains all possible actions.

    Parameters:
    - program (Program): The program to be checked.
    - action_space (int): Number of actions in the environment.

    Returns:
    - bool: True if all actions are present, False otherwise.
    """
    actions = [False]*action_space
    for sub_prog in program.depth_first_iter():
        if isinstance(sub_prog, Primitive):
            if "act_" in sub_prog.primitive:
                actions[int(sub_prog.primitive.split("_")[1])] = True
    return actions.count(False) == 0

def is_lunar_lander(env: gym.Env):
    """
    Check if the environment is the Lunar Lander.

    Parameters:
    - env (gym.Env): The environment.

    Returns:
    - bool: True if the environment is the Lunar Lander, False otherwise.
    """
    return env.spec.id == "LunarLander-v2"

def contains_bool_observations(program: Program, env: gym.Env):
    """
    Check if the program contains boolean observations.

    Parameters:
    - program (Program): The program to be checked.
    - env (gym.Env): The environment.

    Returns:
    - bool: True if the program contains boolean observations, False otherwise.
    """
    iterator = None
    observation_dim = env.observation_space.shape[0]
    for sub_prog in program.depth_first_iter():
        if isinstance(sub_prog, Variable):
            if int(iterator.primitive) >= observation_dim-2:
                return True
        iterator = sub_prog
    return False

def synthesis(
    env: gym.Env,
    cfg: CFG,
    evaluator: DSLEvaluator,
    possible_values: dict[Type, list[float]],
    time_out: float,
    threshold: float,
    delete_program_threshold: float,
    save_programs: bool=False,
    save_path: str="potential_programs.pkl",
    optimize: bool=True,
    optim_algo=hill_climbing
):
    """
    Perform program synthesis using a probabilistic context-free grammar.

    Parameters:
    - env (gym.Env): Gym environment for evaluation.
    - cfg (CFG): Context-free grammar for program generation.
    - evaluator (DSLEvaluator): Evaluator for the Domain-Specific Language (DSL).
    - possible_values (dict[Type, list[float]]): Dictionary mapping types to possible values.
    - time_out (float): Maximum time allowed for synthesis.
    - threshold (float): Reward threshold for selecting potential programs.
    - delete_program_threshold (float): Threshold to delete programs from enumeration seen set.
    - save_programs (bool): Flag to save potential programs.
    - save_path (str): Path to save potential programs if save_programs is True.
    - optimize (bool): Flag to enable or disable optimization.
    - optim_algo (Callable): Optimization algorithm function, default is hill_climbing.

    Returns:
    - Tuple[int, Program, float, List[Tuple[Program, float]]]: A tuple containing:
        - Number of programs generated.
        - Best program found.
        - Best program reward.
        - List of potential programs with their rewards.
    """
    print(f"-----------------------------------------------")
    print(f"Program synthesis starts...")
    if save_programs and save_path is None:
        raise ValueError("save_path cannot be None when save_programs is True")
    start_time = time.time()
    pcfg = ProbDetGrammar.uniform(cfg)
    n_iters = 0
    best_program = None
    best_reward = -math.inf
    potential_programs = []
    
    enumerator = bps_enumerate_prob_grammar(pcfg)

    for i, program in enumerate(enumerator):
        if time.time() - start_time > time_out:
            print("Time out reached")
            break
        if not check_all_action_possible(program, env.action_space.n):
            continue
        if is_lunar_lander(env) and contains_bool_observations(program, env):
            continue
        for instantiated_prog in program.all_constants_instantiation(possible_values):
            n_vars = get_nb_variables(instantiated_prog)
            pids = create_pids(n_vars)
            _, returns =  eval_function(env, create_pids(n_vars), evaluator)(instantiated_prog, 10)
            if returns > best_reward:
                best_reward = returns
                best_program = instantiated_prog
                if best_reward > threshold and optimize:
                    optim_algo(env, instantiated_prog, pids, evaluator)
                    _, best_reward =  eval_function(env, pids, evaluator)(instantiated_prog, 10)
                    returns = best_reward
                print(f"Program: {instantiated_prog}")
                print(f"Best reward: {best_reward}")
                print(f"--------------------------------------------")
            if threshold <= returns:
                potential_programs.append((instantiated_prog, returns))
            n_iters += 1
    if save_programs:
        save_with_pickle(save_path, potential_programs)
    
    n_selected_programs = len(potential_programs)
    print(f"Number of programs generated is {n_iters}")
    print(f"Number of selected programs is {n_selected_programs}")
    print(f"Best program found: {best_program}")
    print(f"Best program reward: {best_reward}")
    print(f"-----------------------------------------------")
    return n_iters, best_program, best_reward, potential_programs
