import gymnasium as gym
import numpy as np
import pygame

pygame.init()
env = gym.make("CarRacing-v2", render_mode="human", domain_randomize=True, continuous=False)

env.reset(options={"randomize": True})

proba = [0.23, 0.19, 0.25, 0.33]

img_to_save = []
i = 0
while True:
    env.render()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_z]:
        move = 3
    elif keys[pygame.K_q]:
        move = 2
    elif keys[pygame.K_d]:
        move = 1
    elif keys[pygame.K_s]:
        move = 4
    else:
        move = 0
    
    if(keys[pygame.K_ESCAPE]):
        break

    observation, reward, _, info, done = env.step(move)

    if i % 5 == 0:
        img_to_save.append(observation)
        print(i//5)

    i += 1

env.close()

np.save("data.npy", np.array(img_to_save))
