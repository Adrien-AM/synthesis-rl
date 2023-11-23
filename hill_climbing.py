import numpy as np

"""
Hill climbing for continuous space
"""

class Hill_Climbing:
    def __init__(self, parameters, eval_func):
        """
        We mean by parameters all leaf probabilities
        """
        self.parameters = parameters
        self.nb_parameters = len(self.parameters)
        self.eval_func = eval_func 

        self.step_size = np.array([1]*self.nb_parameters)

        self.acceleration = 1.2
        self.candidates = np.array([-self.acceleration, -1/self.acceleration, 1/self.acceleration, self.acceleration])

        self.current_point = self.parameters
        self.current_score = self.eval_func(self.current_point)

    @property
    def get_parameters(self):
        return self.parameters

    def run_hill_climbing(self, epsilon):
        while True:
            before_score = self.current_score
            for i, parameter in enumerate(self.current_point):
                for j, element in enumerate(parameter):
                    before_point = self.current_point[i][j]
                    best_step = 0
                    for k in range(len(self.candidates)):
                        step = self.step_size[i]*self.candidates[k]
                        print("Step: ", step)
                        self.current_point[i][j] = before_point*step
                        print("New value: ", self.current_point[i][j])
                        score = self.eval_func(self.current_point)
                        if score > before_score:
                            self.current_score = score
                            best_step = step

                    if best_step == 0:
                        self.current_point[i][j] = before_point
                        self.step_size[i] = step
                    else:
                        print("Update parameter: ", i + 1)
                        self.current_point[i][j] = before_point+best_step
                        self.step_size[i] = best_step//self.acceleration
                if (self.current_score - before_score) < epsilon:
                    return self.current_point
