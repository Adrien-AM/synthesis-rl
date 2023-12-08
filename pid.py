import gymnasium as gym
import random
import math

ACTION_NOTHING = 0
ACTION_LEFT = 1
ACTION_MAIN = 2
ACTION_RIGHT = 3

def temperature(t):
    return .99 * t

def neighbours(params):
    # i = random.randrange(len(params))
    for i in range(len(params)):
        p, d = params[i]
        p += random.uniform(-.1, .1)
        d += random.uniform(-.1, .1)
        params[i] = (p, d)
    return params

def proba(e, e_new, t):
    de = (e - e_new)
    if de < 0:
        return 1
    return math.exp(-de / t)

def optimize(params, n, E):
    '''
    params : p and d for all controllers
    n : number of iterations
    E : energy  function
    '''
    c = LunarController(params)
    current_energy = E(c, 1000)
    t = 10
    for k in range(n):
        if k % (n // 10) == 0:
            current_energy = E(LunarController(params), 1000)
            print(f'Simulated Annealing iteration {k} : energy = {current_energy}')
        t = temperature(t)
        new_params = neighbours(params)
        c = LunarController(new_params)
        new_energy = E(c, 50)
        # print(f'Temperature : {t}, proba : {proba(current_energy, new_energy, t)}')
        if proba(current_energy, new_energy, t) >= random.random():
            params = new_params
            current_energy = new_energy

    return params

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
    def __init__(self, params):
        # Setpoint is 0 for all 3 of them since we want to get the robot to point (0,0) with angle 0
        (px, dx), (py, dy), (pa, da) = params
        self.xcontroller = PIDController(P=px, I=0, D=dx, setpoint=0)
        self.ycontroller = PIDController(P=py, I=0, D=dy, setpoint=0)
        self.acontroller = PIDController(P=pa, I=0, D=da, setpoint=0)

    def play(self, obs):
        x, y, vx, vy, a, va, l1, l2 = obs
        errx = self.xcontroller.update(x)
        erry = self.ycontroller.update(y)
        erra = self.acontroller.update(a)

        if abs(errx + erra) > abs(erry):
            if (errx + erra) < -1:
                return ACTION_LEFT
            elif (errx + erra) > 1:
                return ACTION_RIGHT
        elif erry > 1:
            return ACTION_MAIN
        
        return ACTION_NOTHING

def display(controller):
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

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
    env.close()

def evaluate(controller, nb_tests = 50):
    env = gym.make("LunarLander-v2", wind_power=20)
    observation, info = env.reset()

    total_reward = 0
    for j in range(nb_tests):
        done = False
        while not done:
            move = controller.play(observation)
            observation, reward, terminated, truncated, info = env.step(move)
            total_reward += reward
            done = terminated or truncated
            if done:
                observation, info = env.reset()
                break
            
    avg_reward = total_reward / nb_tests
    return avg_reward


if __name__ == "__main__":
    # [(0.7738700711164033, 2.45110950874502), (0.2587640480344262, 24.860236061123114), (-1.387602756853892, 0.7242691704651993)]
    params = [(.5, 2), (.1, 25), (-.8, 0)]
    # params = [(0, 0), (0, 0), (0, 0)]
    params = optimize(params, 2000, evaluate)
    print(f'Found parameters : {params}')
    controller = LunarController(params)
    display(controller)
    nb_epochs = 1000
    avg_reward = evaluate(controller, nb_epochs)
    print(f'Average reward on {nb_epochs} iterations : {avg_reward:.2f}')