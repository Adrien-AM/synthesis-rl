# from synth import *

from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax.type_helper import FunctionType
from dataclasses import dataclass
from synth.syntax.type_system import PrimitiveType
import numpy as np
from synth.syntax import bps_enumerate_prob_grammar
from synth.syntax.grammars import ProbDetGrammar
from prog_eval import make_env, eval_function, eval_program
from synth.syntax.program import Constant, Function, Primitive, Program, Variable



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

env, reward_min = make_env()
# observation space
observation_dimension = env.observation_space.shape[0]

__semantics = create_semantics(observation_dimension)
__syntax = create_syntax(observation_dimension)
# __syntax = auto_type(
#         {
#         # actions
#         "nothing": "ACTION",
#         "right": "ACTION",
#         "main": "ACTION",
#         "left": "ACTION",
#         "0" : "STATE -> INPUT",
#         # primitives
#         "ite": "FLOAT -> ACTION -> ACTION -> ACTION",
#         "scalar": "CONSTANT -> INPUT -> FLOAT",
#         "+": "FLOAT -> FLOAT -> FLOAT",
#     })

# print(__syntax)


__forbidden_patterns = {}

dsl = DSL(__syntax, __forbidden_patterns)

cfg = CFG.depth_constraint(dsl, auto_type("STATE -> ACTION"), 5, constant_types = {auto_type("CONSTANT")})
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))




pcfg = ProbDetGrammar.uniform(cfg)

possible_constantes = {
    auto_type("CONSTANT"): [-1.0, 1.0]
}


eval_fun = eval_function(env, evaluator)
best_program = None
best_score = -200

can_break = False
for program in bps_enumerate_prob_grammar(pcfg):
    for instantiated_prog in program.all_constants_instantiation(possible_constantes):
        for sub_prog in instantiated_prog.depth_first_iter():
            # Constant, Function, Primitive, Program, Variable
            if isinstance(sub_prog, Function):
                print("Function : ", sub_prog)
            if isinstance(sub_prog, Constant):
                print("Constant : ", sub_prog)
            if isinstance(sub_prog, Primitive):
                print("Primitive : ", sub_prog)
            if isinstance(sub_prog, Variable):
                print("Variable : ", sub_prog)
                can_break = True
            if isinstance(sub_prog, Program):
                print("Program : ", sub_prog)
        print("-----")
    if can_break:
        break
        # print(instantiated_prog)
        # print(instantiated_prog.pretty_print())
        # print(eval_fun(instantiated_prog))
        # print("---")
#         score = eval_fun(instantiated_prog, 5)[1]
#         if score > best_score:
#             best_score = score
#             best_program = instantiated_prog
#             print(f"New best score : {best_score}")
#             print(best_program.pretty_print())
#             print("-----")
    

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
