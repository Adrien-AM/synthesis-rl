import gymnasium as gym
import numpy as np
import time

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
    def __init__(self, Ps=[.5, .1, -.8], Is=[0, 0, 0], Ds=[2, 25, 0]):
        # Setpoint is 0 for all 3 of them since we want to get the robot to point (0,0) with angle 0
        self.xcontroller = PIDController(P=Ps[0], I=Is[0], D=Ds[0], setpoint=0)
        self.ycontroller = PIDController(P=Ps[1], I=Is[1], D=Ds[1], setpoint=0)
        self.acontroller = PIDController(P=Ps[2], I=Is[2], D=Ds[2], setpoint=0)

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

def display(Ps, Ds):
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()
    controller = LunarController(Ps=Ps, Ds=Ds)

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

def evaluate(Ps, Ds, nb_tests = 50):
    env = gym.make("LunarLander-v2")
    observation, info = env.reset()

    controller = LunarController(Ps=Ps, Ds=Ds)

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

def optimizer(nb_episodes = 1000, nb_tests = 50):
    init_Ps=[.5, .1, -.8]
    init_Ds=[2, 25, 0]

    iteration_of_last_change = 0
    best = -200
    best_params = [init_Ps, init_Ds]

    final_best_param = []
    final_best_reward = []
    for i in range(nb_episodes):
        if i % (nb_episodes//10) == 0:
            print(f'Iteration {i}/{nb_episodes}')
        
        if i - iteration_of_last_change > nb_episodes//4:
            init_Ps=[.5, .1, -.8]
            init_Ds=[2, 25, 0]
            print("Resetting parameters")
            print(f"New values : Ps={init_Ps}, Ds={init_Ds}")
            iteration_of_last_change = i
            final_best_param.append(best_params)
            final_best_reward.append(best)
            best = -200

        Ps = init_Ps + np.random.uniform(low=-1, high=1, size=(3,))
        Ds = init_Ds + np.random.uniform(low=-1, high=1, size=(3,))

        avg_reward = evaluate(Ps, Ds, nb_tests)

        if avg_reward > best:
            iteration_of_last_change = i
            best = avg_reward
            best_params = [Ps, Ds]
            print(f'New best : {best:.2f} with Ps={Ps}, Ds={Ds}')

    final_best_param.append(best_params)
    final_best_reward.append(best)
    best = max(final_best_reward)
    best_params = final_best_param[final_best_reward.index(best)]
    return best, best_params



if __name__ == "__main__":
    t = time.time()
    best_reward, best_params = optimizer(1000, 50)
    nb_epochs = 1000
    print(best_params)
    avg_reward = evaluate(best_params[0], best_params[1], nb_epochs)
    print(f'Average reward on {nb_epochs} iterations : {avg_reward:.2f}')
    print(f'Total time : {time.time() - t:.2f}s')

    display(best_params[0], best_params[1])

    # Ps = [ 1.06198339,  0.22837168, -1.65524936]
    # Ds = [ 1.7604426 , 25.82791385,  0.34561309]