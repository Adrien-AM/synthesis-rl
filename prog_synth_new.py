from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from synth.syntax import bps_enumerate_prob_grammar
from synth.syntax.grammars import ProbDetGrammar
from synth.syntax.type_system import Type
from prog_eval import make_env, eval_function
import codecarbon
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
        semantics["obs" + str(i)] = lambda s: s[i]
        
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
        syntax["obs_" + str(i)] = "STATE -> INPUT"

    return auto_type(syntax)

def synthesis(
    cfg: CFG,
    evaluator: DSLEvaluator,
    possible_values: dict[Type, list[float]],
    reward_min: float
):
    """
    """
    pcfg = ProbDetGrammar.uniform(cfg)
    eval_func = eval_function(env, evaluator)
    n_iters = 0
    best_program = None
    best_reward = -math.inf
    for program in bps_enumerate_prob_grammar(pcfg):
        for instantiated_prog in program.all_constants_instantiation(possible_values):
            _, returns =  eval_func(instantiated_prog, 15)
            if returns > best_reward:
                if returns > reward_min:
                    return n_iters, instantiated_prog, returns
                best_reward = returns
                best_program = instantiated_prog
                print(f"Program: {instantiated_prog}")
                print(f"Best reward: {best_reward}")
                print(f"--------------------------------------------")
            n_iters += 1
    return n_iters, best_program, best_reward


if __name__ == "__main__":
    # tracker = codecarbon.EmissionsTracker()
    # tracker.start()
    env, reward_min = make_env()
    # observation space
    observation_dimension = env.observation_space.shape[0]
    action_space = env.action_space.n

    __semantics = create_semantics(observation_dimension, action_space)
    __syntax = create_syntax(observation_dimension, action_space)
    __forbidden_patterns = {}

    dsl = DSL(__syntax, __forbidden_patterns)

    cfg = CFG.depth_constraint(dsl, auto_type("STATE -> ACTION"), 4, constant_types = {auto_type("CONSTANT")})
    evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
    possible_constants = {
            auto_type("CONSTANT"): [-1.0, 1.0]
        }
    
    n_iters, best_program, best_reward = synthesis(cfg, evaluator, possible_constants, reward_min)
    print(f"Number of programs generated is {n_iters}")
    print(f"Best program found: {best_program} with reward: {best_reward}")

    # tracker.stop()

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
