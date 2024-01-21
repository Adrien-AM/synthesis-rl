from synth.semantic import DSLEvaluator
from synth.syntax import auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax import bps_enumerate_prob_grammar
from synth.syntax.grammars import ProbDetGrammar
from synth.syntax.type_system import Type
from synth.syntax.program import Program, Variable, Function, Primitive, Constant
from prog_eval import eval_function
from utils import save_with_pickle
import gymnasium as gym
import time
import math

def create_semantics(observation_dimension,action_space):
    """
    Generate a generic semantic for the DSL with observation_dimension input variables
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
    Generate a generic syntax for the DSL with observation_dimension input variables
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
    Check if the program contains all possible actions
    """
    actions = [False]*action_space
    for sub_prog in program.depth_first_iter():
        if isinstance(sub_prog, Primitive):
            if "act_" in sub_prog.primitive:
                actions[int(sub_prog.primitive.split("_")[1])] = True
    return actions.count(False) == 0


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
):
    """
    The objective collect programs that have potential
    """
    print(f"-----------------------------------------------")
    print(f"Program synthesis starts...")
    if save_programs and save_path is None:
        raise ValueError("save_path cannot be None when save_programs is True")
    start_time = time.time()
    pcfg = ProbDetGrammar.uniform(cfg)
    eval_func = eval_function(env, evaluator)
    n_iters = 0
    best_program = None
    best_reward = -math.inf
    potential_programs = []
    
    enumerator = bps_enumerate_prob_grammar(pcfg)

    for program in enumerator:
        if len(enumerator._seen) > delete_program_threshold:
            enumerator._seen = set()
        if time.time() - start_time > time_out:
            print("Time out reached")
            break
        if not check_all_action_possible(program, env.action_space.n):
            continue
        for instantiated_prog in program.all_constants_instantiation(possible_values):
            _, returns =  eval_func(instantiated_prog, 15)
            if returns > best_reward:
                best_reward = returns
                best_program = instantiated_prog
                print(f"Program: {instantiated_prog}")
                print(f"Best reward: {best_reward}")
                print(f"--------------------------------------------")
            if  threshold <= returns:
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