# from synth import *

from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax.type_helper import FunctionType
from dataclasses import dataclass
from synth.syntax.type_system import PrimitiveType

FLOAT = PrimitiveType("float")

from synth.syntax.type_system import (
    INT,
    STRING,
)

@dataclass(frozen=True)
class PROG:
    def run(self, ite:"ITE", action:"ACTION") -> "PROG":
        if action == None:
            return ITE(ite)
        else:
            return ACTION(action)

@dataclass(frozen=True)
class ACTION(PROG):
    action: INT


@dataclass(frozen=True)
class EXPR(PROG):
    const1: FLOAT
    var1: STRING

    def sub_expr(self):
        if self.const1 == 0.0:
            return 0.0
        return self

@dataclass(frozen=True)
class ITE(PROG):
    expr: EXPR
    then: PROG
    els: PROG

__semantics = {
    # Primitive
    "nothing": ACTION(0),
    "right": ACTION(1),
    "main": ACTION(2),
    "left": ACTION(3),

    # Non primitive
    "prog": lambda ite: lambda action: PROG.run(ite, action),
    "if": lambda expr: lambda then: lambda els: ITE(expr > 0.0, then, els),
    
    "expr": lambda const1: lambda var1: lambda expr: const1 * var1 + expr.sub_expr(), 
}

__syntax = auto_type(
    {
    # Primitive
    "nothing": "ACTION",
    "right": "ACTION",
    "main": "ACTION",
    "left": "ACTION",

    # Non primitive
    "prog": "ITE -> ACTION -> PROG",
    "if": "EXPR -> PROG -> PROG",
    "expr": "FLOAT -> STRING -> EXPR -> EXPR",
})

__forbidden_patterns = {}

dsl = DSL(__syntax, __forbidden_patterns)

evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))

cfg = CFG.depth_constraint(dsl, FunctionType(PROG, ACTION), 4)
# evaluator.eval("if 1.0 + 2.0 - 3.0 + 4.0 > 0 then nothing else right", )