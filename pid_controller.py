import gymnasium as gym
import hill_climbing
import numpy as np

ACTION_NOTHING = 0
ACTION_LEFT = 1
ACTION_MAIN = 2
ACTION_RIGHT = 3

class PIDController:
    def __init__(self, P, I, D, setpoint):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.integral = 0
        self.previous_error = 0
        self.setpoint = setpoint

    def update(self, measured_value):
        dt = 1
        # First term : proportional
        error = self.setpoint - measured_value

        # Second term : integral (accumulated over time)
        self.integral += error * dt

        # Third term : derivative (using last error)
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        return output

class LunarController:
    def __init__(self, parameters):
        # Setpoint is 0 for all 3 of them since we want to get the robot to point (0,0) with angle 0
        self.xcontroller = PIDController(P=parameters[0][0], I=0, D=parameters[0][1], setpoint=0)
        self.ycontroller = PIDController(P=parameters[1][0], I=0, D=parameters[1][1], setpoint=0)
        self.acontroller = PIDController(P=parameters[2][0], I=-0, D=parameters[2][1], setpoint=0)

    def play(self, obs):
        x, y, vx, vy, a, va, l1, l2 = obs
        errx = self.xcontroller.update(x)
        erry = self.ycontroller.update(y)
        erra = self.acontroller.update(a)

        if abs(errx + erra) > abs(erry):
            if (errx + erra) < -.1:
                return ACTION_LEFT
            elif (errx + erra) > .1:
                return ACTION_RIGHT
        elif erry > .1:
            return ACTION_MAIN
        
        return ACTION_NOTHING

def display(parameters):
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()
    controller = LunarController(parameters)

    done = False
    truncated = False
    total_reward = 0
    while True:
        move = controller.play(observation)
        observation, reward, terminated, truncated, info = env.step(move)
        total_reward += reward
        if (terminated or truncated):
            env.reset()
            break
        env.render()

    print(f'Total reward : {total_reward:.2f}')
    env.close()

def evaluate(parameters, nb_epochs = 1000, verbose=False):
    env = gym.make("LunarLander-v2")
    observation, info = env.reset()
    controller = LunarController(parameters)

    i = 0
    s = 0
    while i < nb_epochs:
        if verbose:
            if i % (nb_epochs//10) == 0:
                print(f'Iteration {i}/{nb_epochs}')
        i += 1
        done = False
        truncated = False
        total_reward = 0
        while True:
            move = controller.play(observation)
            observation, reward, terminated, truncated, info = env.step(move)
            total_reward += reward
            if (terminated or truncated):
                observation, info = env.reset()
                break
        s += total_reward
    env.close()
    return s / nb_epochs


def save_parameters(parameter_list, file_name):
    np.save(file_name, np.array(parameter_list))
    return True

def load_parameters(parameters_path):
    return np.load(parameters_path)

if __name__ == "__main__":
    parameters_init = [[0.5, 4.828], [0.1, 25.707], [-1.507, -2.828]]
    
    saver_init = parameters_init.copy()

    optimizer = hill_climbing.Hill_Climbing(parameters=parameters_init,\
                                                eval_func=evaluate)

    optimizer.run_hill_climbing(20)

    new_parameters = optimizer.get_parameters
    print(f"Old parameters: {saver_init}")
    print(f"New parameters: {new_parameters}")
    save_parameters(new_parameters, "pid_hill_climbing2")

    display(new_parameters)
    nb_epochs = 1000
    avg_reward = evaluate(new_parameters, nb_epochs, True)
    print(f'Average reward on {nb_epochs} iterations : {avg_reward:.2f}')

# [[0.5, 4.828], [0.1, 25.707], [-1.507, -2.828]]: Average reward on 1000 iterations : 210.56