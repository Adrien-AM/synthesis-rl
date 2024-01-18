from synth.semantic import DSLEvaluator
from synth.syntax import DSL, auto_type
from synth.syntax.grammars.cfg import CFG
from prog_eval import make_env
from prog_synth_new import create_semantics, create_syntax, synthesis
from ucb import ucb_selection

# Global variables
NB_MINUTES = 15
TIME_OUT = 60*NB_MINUTES
REWARD_THRESHOLD = 150

# UCB parameters
NB_ITERATIONS = 100
NB_PROGRAM_EVALUATIONS = 100

if __name__ == '__main__':
    env, reward_min = make_env()

    # observation space
    observation_dimension = env.observation_space.shape[0]

    __semantics = create_semantics(observation_dimension)
    __syntax = create_syntax(observation_dimension)
    __forbidden_patterns = {}

    dsl = DSL(__syntax, __forbidden_patterns)

    cfg = CFG.depth_constraint(dsl, auto_type("STATE -> ACTION"), 5, constant_types = {auto_type("CONSTANT")})
    evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
    possible_constants = {
        auto_type("CONSTANT"): [-1.0, 1.0]
    }

    n_iters, best_program, best_reward, potential_programs = synthesis(env,
                                                                    cfg,
                                                                    evaluator,
                                                                    possible_constants,
                                                                    TIME_OUT,
                                                                    REWARD_THRESHOLD,
                                                                    save_programs=True)
    
    best_ucb_program, best_ucb_avg_rewards = ucb_selection(env,
                                                           NB_ITERATIONS,
                                                           NB_PROGRAM_EVALUATIONS,
                                                           evaluator,
                                                           potential_programs)
