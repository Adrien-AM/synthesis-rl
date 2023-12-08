from synth.syntax import auto_type, DSL
from dataclasses import dataclass


@dataclass(frozen=True)
class PROG:
    def then(self, s2: "PROG") -> "PROG":
        return THEN(self, s2)

@dataclass(frozen=True)
class THEN(PROG):
    s1: PROG
    s2: PROG

# Almost sure that these classes are correct
@dataclass(frozen=True)
class EXPR(PROG):
    is1: IS
    const1: float
    var1: str
    op1: OP
    is2: IS
    const2: float
    var2: str
    op2: OP
    is3: IS
    const3: float
    var3: str
    op3: OP
    is4: IS
    const4: float
    var4: str

@dataclass(frozen=True)
class COND(PROG):
    cond: EXPR
    comp: COMP

@dataclass(frozen=True)
class ACTION(PROG):
    action: int

@dataclass(frozen=True)
class ITE(PROG):
    cond: COND
    then: ACTION
    els: ACTION

@dataclass(frozen=True)
class COMP(PROG):
    comp: str

@dataclass(frozen=True)
class OP(PROG):
    op: str

@dataclass(frozen=True)
class IS(PROG):
    is_used: int