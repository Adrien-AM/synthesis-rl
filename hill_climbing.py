import numpy as np

"""
Hill climbing for continuous space
"""

class Hill_Climbing:
    def __init__(self, parameters, eval_func):
        """
        We mean by parameters all leaf probabilities
        """
        self.parameters = parameters.copy()
        self.nb_parameters = len(self.parameters)
        self.nb_parameters_inside = len(self.parameters[0])
        self.eval_func = eval_func 

        self.step_size = np.array([[1]*self.nb_parameters_inside]*self.nb_parameters)

        # self.acceleration = np.round(np.sqrt(2), 3)

        self.acceleration = np.round(np.sqrt(2), 3)
        self.candidates = np.array([-self.acceleration, -1/self.acceleration, 1/self.acceleration, self.acceleration])

        self.current_point = self.parameters
        self.current_score = self.eval_func(self.current_point)

    @property
    def get_parameters(self):
        return self.parameters

    def run_hill_climbing(self, epsilon):
        n_evaluations = 0
        while True:
            before_score = self.current_score
            for i, parameter in enumerate(self.current_point):
                for j, element in enumerate(parameter):
                    if all(x == 0. for x in self.step_size[i]):
                        print("BREAK!!!!!!!!!!!!")
                        break
                    before_point = self.current_point[i][j]
                    best_step = 0
                    for k in range(len(self.candidates)):
                        step = np.round(self.step_size[i][j]*self.candidates[k], 3)
                        print("Step: ", step)
                        self.current_point[i][j] = np.round(before_point+step, 3)
                        print("New value: ", self.current_point[i][j])
                        score = self.eval_func(self.current_point)
                        if n_evaluations%50==0:
                            print(f"Average reward with current parameters: {score:.3}")
                        n_evaluations += 1

                        if score > before_score:
                            self.current_score = score
                            best_step = step
                    print("NB_evaluations: ", n_evaluations)
                    print("----------------------------------------------------")
                    if best_step == 0:
                        self.current_point[i][j] = before_point
                        self.step_size[i][j] = step
                    else:
                        print("Update parameter: ", i + 1)
                        self.current_point[i][j] = np.round(before_point+best_step, 3)
                        self.step_size[i][j] = best_step//self.acceleration
            if self.current_score > before_score:
                return self.current_point
            return before_point
