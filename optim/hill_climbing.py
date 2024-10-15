import numpy as np
import gymnasium as gym
from typing import List
from pid import PIDController
from synth.semantic import DSLEvaluator
from synth.syntax.program import Program
from prog_eval import eval_function

def get_pid_parameters(pids: List[PIDController]):
    """
    Gets the proportional (Kp), integral (Ki), and derivative (Kd) parameters 
    of a list of PID controllers.

    Parameters:
    - pids (List[PIDController]): List of PID controllers.

    Returns:
    - List[List[float]]: List containing the Kp, Ki, and Kd parameters for each PID controller.
    """
    return [[pid.Kp, pid.Ki, pid.Kd] for pid in pids]

def set_pid_parameters(pids: List[PIDController], parameters: list):
    """
    Sets the proportional (Kp), integral (Ki), and derivative (Kd) parameters 
    for a list of PID controllers.

    Parameters:
    - pids (List[PIDController]): List of PID controllers.
    - parameters (List[List[float]]): List containing the Kp, Ki, and Kd parameters for each PID controller.
    """
    for pid, (Kp, Ki, Kd) in zip(pids, parameters):
        pid.Kp = Kp
        pid.Ki = Ki
        pid.Kd = Kd

def hill_climbing(
        env: gym.Env,
        program: Program,
        pids: List[PIDController],
        evaluator: DSLEvaluator,
        n_iters: int=10,
        acceleration: float=1.2*np.sqrt(2)
    ):
    """
    Optimizes PID controller parameters using the hill climbing algorithm.

    Parameters:
    - env (gym.Env): Gym environment for evaluation.
    - program (Program): The program to be evaluated by the PID controllers.
    - pids (List[PIDController]): List of PID controllers to be optimized.
    - evaluator (DSLEvaluator): Evaluator for the Domain-Specific Language (DSL).
    - n_iters (int): Number of iterations for the optimization process.
    - acceleration (float): Acceleration factor for adjusting step sizes in each iteration.

    Returns:
    - List[List[float]]: Final optimized PID parameters (Kp, Ki, Kd).
    """
    print("Starts optimization with hill climbing algorithm...")
    candidates = np.array([-acceleration, -1/acceleration, 1/acceleration, acceleration])
    current_point = get_pid_parameters(pids)
    step_size = np.array([[1]*len(current_point[0])]*len(current_point))
    _, current_score = eval_function(env, pids, evaluator)(program, 10)
    n_evaluations = 1

    for iteration in range(n_iters):
        # print(f"Iteration: {iteration}", end=' ')
        improved = False
        for i, parameter in enumerate(current_point):
            for j, _ in enumerate(parameter):
                best_step = 0
                best_score = current_score

                for candidate in candidates:
                    step = np.round(step_size[i][j]*candidate, 3)
                    current_point[i][j] += step
                    set_pid_parameters(pids, current_point)
                    _, score = eval_function(env, pids, evaluator)(program, 10)

                    n_evaluations += 1
                    if score > best_score:
                        best_score = score
                        best_step = step
                        improved = True

                    # Revert to previous value
                    current_point[i][j] -= step
                    set_pid_parameters(pids, current_point)

                if best_step != 0:
                    current_point[i][j] += best_step
                    set_pid_parameters(pids, current_point)
                    step_size[i][j] *= acceleration
                else:
                    step_size[i][j] *= -acceleration

                # print(f"Parameter updated: PID {i}, Param {j}, New value: {current_point[i][j]}, Step size: {step_size[i][j]}")
        if improved:
            current_score = best_score
            print(f"Improved score: {current_score} at iteration {iteration}")
        # else:
        #     print("No improvement in this iteration.")
        # print("----------------------------------------------------")
    # print(f"Final PID parameters: {current_point}")
    # print(f"Final score: {current_score}")
    # print(f"Total evaluations: {n_evaluations}")
    return current_point
