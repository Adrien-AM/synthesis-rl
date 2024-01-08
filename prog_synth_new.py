# from synth import *

from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax.type_helper import FunctionType
from dataclasses import dataclass
from synth.syntax.type_system import PrimitiveType
import numpy as np

__semantics = {
    # actions
    "nothing": 0,
    "right": 1,
    "main": 2,
    "left": 3,
    # input variables (we keep only the relevant ones)
    "x": lambda s: s,                   # var0 is s0, var1 is s1, ...
    "y": lambda s: s,
    "a": lambda s: s,
    # primitives
    "ite": lambda cond: lambda if_block: lambda else_block: if_block if cond > 0 else else_block,
    "scalar": lambda const: lambda inp: const * inp,
    "+": lambda float1: lambda float2: float1 + float2,
}

__syntax = auto_type(
    {
    # actions
    "nothing": "ACTION",
    "right": "ACTION",
    "main": "ACTION",
    "left": "ACTION",
    # input variables (we keep only the relevant ones)
    "x": "STATE -> INPUT",
    "y": "STATE -> INPUT",
    "a": "STATE -> INPUT",
    # primitives
    "ite": "FLOAT -> ACTION -> ACTION -> ACTION",
    "scalar": "CONSTANT -> INPUT -> FLOAT",
    "+": "FLOAT -> FLOAT -> FLOAT",

})

__forbidden_patterns = {}

dsl = DSL(__syntax, __forbidden_patterns)

cfg = CFG.depth_constraint(dsl, auto_type("STATE -> ACTION"), 5, constant_types = {auto_type("CONSTANT")})
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))

from synth.syntax import bps_enumerate_prob_grammar
from synth.syntax.grammars import ProbDetGrammar
from prog_eval import make_env, eval_function, eval_program
from synth.syntax.program import Constant, Function, Primitive, Program, Variable


pcfg = ProbDetGrammar.uniform(cfg)

possible_constantes = {
    # auto_type("CONSTANT"): np.arange(-1, 1, 0.1)
    auto_type("CONSTANT"): [-1.0, 0.0, 1.0]
}


env, reward_min = make_env()
eval_fun = eval_function(env, evaluator)
best_program = None
best_score = -200

for program in bps_enumerate_prob_grammar(pcfg):
    for instantiated_prog in program.all_constants_instantiation(possible_constantes):
        # print(instantiated_prog)
        # print(instantiated_prog.pretty_print())
        # print(eval_fun(instantiated_prog))
        # print("---")
        score = eval_fun(instantiated_prog, 5)[1]
        if score > best_score:
            best_score = score
            best_program = instantiated_prog
            print(f"New best score : {best_score}")
            print(best_program.pretty_print())
            print("-----")
    
print(f"Best score : {best_score}")
print(best_program.pretty_print())