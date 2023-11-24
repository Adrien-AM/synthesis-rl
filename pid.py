import gymnasium as gym
from genetic import *

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
    def __init__(self, param1=(.5, 0, 2), param3=(-.8, 0, 0)):
        # Setpoint is 0 for all 3 of them since we want to get the robot to point (0,0) with angle 0
        self.xcontroller = PIDController(P=param1[0], I=0, D=param3[0], setpoint=0)
        self.ycontroller = PIDController(P=param1[1], I=0, D=param3[1], setpoint=0)
        self.acontroller = PIDController(P=param1[2], I=0, D=param3[2], setpoint=0)

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

def display(controller = LunarController()):
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

    print(f'Total reward : {total_reward:.2f}')
    env.close()

def evaluate(nb_epochs = 1000, controller = LunarController()):
    env = gym.make("LunarLander-v2")
    observation, info = env.reset()

    i = 0
    s = 0
    while i < nb_epochs:
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



population_size = 25
vector_size = 6
generations = 10

if __name__ == "__main__":
    # display()
    v_init = [.5, .1, -.8, 2, 25, 0]
    c = LunarController(v_init[:3], v_init[3:])
    population = initialize_population(population_size=population_size, vector_size=vector_size, origin_vector=v_init)

    best_fitness = 0
    goat = LunarController()

    for gen in range(generations):
        fitness = calculate_fitness(population)
        parents = select_parents(population, fitness)
        offspring = crossover(parents)
        mutated_offspring = mutate(offspring)
        population = mutated_offspring
        best_individual_index = np.argmax(fitness)
        best_individual = population[best_individual_index]
        if fitness[best_individual_index] > best_fitness:
            best_fitness = fitness[best_individual_index]
            goat = LunarController(best_individual[:3], best_individual[3:])
        print(f"Generation {gen + 1}, Best Individual: {best_individual}, Fitness: {fitness[best_individual_index]}")


    display(goat)
    nb_epochs = 1000
    avg_reward = evaluate(nb_epochs)
    print(f'Average reward on {nb_epochs} iterations : {avg_reward:.2f}')