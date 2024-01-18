from typing import List, Tuple
from synth.syntax.program import Program
from synth.semantic import DSLEvaluator
from prog_eval import eval_function
import gymnasium as gym
import math

def ucb_selection(
        env: gym.Env,
        n_iterations: int,
        n_program_eval: int,
        evaluator: DSLEvaluator,
        potential_programs: List[Tuple[Program, float]],
        UCB_parameter: float= math.sqrt(2)
) -> List[Tuple[Program, float]]:
    """
    """
    print(f"---------------------------------------------------------")
    eval_func = eval_function(env, evaluator)
    n_programs = len(potential_programs)
    print(f"UCB program selection starts on {n_programs} programs...")

    n_selections = [0]*n_programs
    average_rewards = [0.0] * n_programs

    # Eleminate basic cases that does not need UCB
    if n_programs == 0:
        raise ValueError("You provided empty potential programs list.")
    if n_programs == 1:
        program, _ = potential_programs[0]
        _, returns = eval_func(program, n_program_eval)
        print(f"After UCB:")
        print(f"You gave one program: {best_program}")
        print(f"The program average reward: {returns}")
        print(f"---------------------------------------------------------")
        print()
        return program, returns
    
    # First UCB iteration on all programs
    for i in range(n_programs):            
        _, returns =  eval_func(potential_programs[i][0], n_program_eval)
        n_selections[i] += 1
        average_rewards[i] += returns

    # UCB algorithm iterations
    for t in range(2, n_iterations+1):
        ucb_values = [0.0]*n_programs
        for i in range(n_programs):
            confidence_interval = math.sqrt((UCB_parameter*math.log(t))/n_selections[i])
            ucb_values[i] = average_rewards[i] + confidence_interval

        selected_index = ucb_values.index(max(ucb_values))
        selected_program, _ = potential_programs[selected_index]
        _, returns = eval_func(selected_program, n_program_eval)
        n_selections[selected_index] += 1

        step_size = 1/n_selections[selected_index]
        average_rewards[selected_index] += step_size*(returns - average_rewards[selected_index])

    # Get the best program and its average reward
    best_program_index = average_rewards.index(max(average_rewards))
    best_program = potential_programs[best_program_index][0]
    best_average_reward = average_rewards[best_program_index]

    print(f"After UCB:")
    print(f"Program selected: {best_program}")
    print(f"Program selected average rewards: {best_average_reward}")
    print(f"---------------------------------------------------------")
    print()

    return best_program, best_average_reward
