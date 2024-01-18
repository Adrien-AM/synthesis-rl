from synth.semantic import DSLEvaluator
from synth.syntax import auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax import bps_enumerate_prob_grammar
from synth.syntax.grammars import ProbDetGrammar
from synth.syntax.type_system import Type
from prog_eval import eval_function
from utils import save_with_pickle
import gymnasium as gym
import time
import math

def create_semantics(observation_dimension):
    """
    Generate a generic semantic for the DSL with observation_dimension input variables
    """
    semantics = {
        # actions
        "nothing": 0,
        "right": 1,
        "main": 2,
        "left": 3,
        # primitives
        "ite": lambda cond: lambda if_block: lambda else_block: if_block if cond > 0 else else_block,
        "scalar": lambda const: lambda inp: const * inp,
        "+": lambda float1: lambda float2: float1 + float2,
    }
    for i in range(observation_dimension):
        semantics[str(i)] = lambda s: s[i]
    return semantics

def create_syntax(observation_dimension):
    """
    Generate a generic syntax for the DSL with observation_dimension input variables
    """
    syntax = {
        # actions
        "nothing": "ACTION",
        "right": "ACTION",
        "main": "ACTION",
        "left": "ACTION",
        # primitives
        "ite": "FLOAT -> ACTION -> ACTION -> ACTION",
        "scalar": "CONSTANT -> INPUT -> FLOAT",
        "+": "FLOAT -> FLOAT -> FLOAT",
    }
    for i in range(observation_dimension):
        syntax[str(i)] = "STATE -> INPUT"
    return auto_type(syntax)

def synthesis(
    env: gym.Env,
    cfg: CFG,
    evaluator: DSLEvaluator,
    possible_values: dict[Type, list[float]],
    time_out: float,
    threshold: float,
    save_programs: bool=False,
    save_path: str="potential_programs.pkl",
):
    """
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
    
    for program in bps_enumerate_prob_grammar(pcfg):
        if time.time() - start_time > time_out:
            print("Time out reached")
            break
        for instantiated_prog in program.all_constants_instantiation(possible_values):
            _, returns =  eval_func(instantiated_prog, 5)
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

# Program: (ite (scalar -1.0 (0 var0)) (ite (scalar 1.0 (1 var0)) right main) nothing)
# Best reward: 248.83281240733658

# for program in bps_enumerate_prob_grammar(pcfg):
#     # programs = []
#     for instantiated_prog in program.all_constants_instantiation(possible_constantes):
#         for c in instantiated_prog.constants():
#             print(c)
#         # programs.append(instantiated_prog)
#     # optimisation(programs, eval_fun, reward_min)

# print(f"Best score : {best_score}")
# print(best_program.pretty_print())

# # ['x0: CONSTANT = 1.0', 'x2: INPUT = x(var0)', 'x3: FLOAT = scalar(x0, x2)', 'x4: FLOAT = +(x3, x3)', 'x5: CONSTANT = -1.0', 'x6: FLOAT = scalar(x5, x2)', 'x7: ACTION = left', 'x8: ACTION = right', 'x9: ACTION = ite(x6, x7, x8)', 'x10: ACTION = nothing', 'x11: ACTION = ite(x4, x9, x10)']  262.19
