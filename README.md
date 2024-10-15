# Programmatic RL

The goal of this project is to train an agent to generate programs solving the Lunar Lander problem (from Gym).
We tested multiple approaches, including writing PID Controllers and Decision Trees. Then, we wrote a Domain Specific Language to generate these components automatically.

## How to use

First, install all the packages using `pip install -r requirement.txt`.\
To launch the program generation, use the `main.py` file.\
To change the test environment, update the function `make_env` in file `prog_eval.py`.
