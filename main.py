from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from prog_eval import make_env
from prog_synth import create_semantics, create_syntax, synthesis
from ucb import ucb_selection
from optim.hill_climbing import hill_climbing
import codecarbon

# Global variables
NB_MINUTES = 30
TIME_OUT = 60*NB_MINUTES
DELETE_PROGRAM_THRESHOLD = 1e9

# UCB parameters
NB_ITERATIONS = 100
NB_PROGRAM_EVALUATIONS = 100

# Environment parameters
env, reward_min = make_env()
REWARD_THRESHOLD = reward_min
OBS_DIM = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

OPTIM_ALGO = hill_climbing

if __name__ == '__main__':
    tracker = codecarbon.EmissionsTracker()
    tracker.start()

    __semantics = create_semantics(OBS_DIM, ACTION_SPACE)
    __syntax = create_syntax(OBS_DIM, ACTION_SPACE)
    __forbidden_patterns = {}

    dsl = DSL(__syntax, __forbidden_patterns)

    cfg = CFG.depth_constraint(dsl, auto_type("STATE -> ACTION"), 6, constant_types = {auto_type("CONSTANT")})
    evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
    possible_constants = {
        auto_type("CONSTANT"): [-5.0, -2.0, -1.0, 0.5, 1.0, 2.0, 5.0]
    }

    n_iters, best_program, best_reward, potential_programs = synthesis(env,
                                                                    cfg,
                                                                    evaluator,
                                                                    possible_constants,
                                                                    TIME_OUT,
                                                                    REWARD_THRESHOLD,
                                                                    DELETE_PROGRAM_THRESHOLD,
                                                                    save_programs=True,
                                                                    optimize=True,
                                                                    optim_algo=OPTIM_ALGO)
    
    best_ucb_program, best_ucb_avg_rewards = ucb_selection(env,
                                                           NB_ITERATIONS,
                                                           NB_PROGRAM_EVALUATIONS,
                                                           evaluator,
                                                           potential_programs)

    tracker.stop()
