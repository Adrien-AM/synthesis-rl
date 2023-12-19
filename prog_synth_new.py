# from synth import *

from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax.type_helper import FunctionType
from dataclasses import dataclass
from synth.syntax.type_system import PrimitiveType

__semantics = {
    # actions
    "nothing": 0,
    "right": 1,
    "main": 2,
    "left": 3,
    # input variables (we keep only the relevant ones)
    "x": lambda s: s[0],
    "y": lambda s: s[1],
    "a": lambda s: s[4],
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
    "ite": "FLOAT -> ACTION -> ACTION",
    "scalar": "CONSTANT -> INPUT -> FLOAT",
    "+": "FLOAT -> FLOAT -> FLOAT",

})

__forbidden_patterns = {}

dsl = DSL(__syntax, __forbidden_patterns)

cfg = CFG.depth_constraint(dsl, auto_type("STATE -> ACTION"), 4, constant_types = {auto_type("CONSTANT")})
