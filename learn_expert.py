import gymnasium as gym 
import numpy as np

import time 

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

model = None 

def learn_model():
    global model 
    env = DummyVecEnv([lambda: gym.make('InvertedPendulum-v5')])

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    st_time = time.time() 

    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100000, log_interval=50)
    model.save("InvertedPendulum_models/td3_invpendulum-100k")

    en_time = time.time() 
    print("Training took {}s".format(en_time - st_time)) 

# del model # remove to demonstrate saving and loading

# model = TD3.load("InvertedPendulum_models/td3_invpendulum-5k")

learn_model()

# vec_env = model.get_env()
# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")

visual_env = gym.make('InvertedPendulum-v5', render_mode = "human", reset_noise_scale=0.1) 

obs, info = visual_env.reset()
terminated = truncated = False 
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = visual_env.step(action)

    if terminated or truncated:
        obs, info = visual_env.reset()
    # visual_env.render()