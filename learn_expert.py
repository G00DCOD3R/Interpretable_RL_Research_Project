import gymnasium as gym 
import numpy as np

import time 

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3.common.monitor import Monitor

model = None 

def learn_model():
    global model 
    env = DummyVecEnv([lambda: gym.make('Swimmer-v5')])

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    eval_env = Monitor(gym.make('Swimmer-v5', render_mode = None))
    eval_callback = EvalCallback(
        eval_env,
        log_path="./logs/",
        eval_freq=5000,
        n_eval_episodes=6,
        deterministic=True,
        best_model_save_path = "Swimmer_models/eval_best_1kk-v2",
        render=False,
    )

    st_time = time.time() 

    model = TD3(
        MlpPolicy,
        env, 
        action_noise=action_noise, 
        verbose=1, 
        learning_rate = 1e-3, 
        buffer_size = 500000)
    model.learn(total_timesteps=1000000, log_interval=5, callback = eval_callback)
    model.save("Swimmer_models/td3_swimmer-1kk-v2")

    en_time = time.time() 
    print("Training took {}s".format(en_time - st_time)) 

# del model # remove to demonstrate saving and loading

model = TD3.load("logs/td3/Swimmer-v4_2/Swimmer-v4.zip")

# learn_model()

# vec_env = model.get_env()
# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")

visual_env = gym.make('Swimmer-v5', render_mode = None) 

obs, info = visual_env.reset()
terminated = truncated = False 
cur_reward, fwd_rewards, ctrl_cost = 0, 0, 0


while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = visual_env.step(action)

    cur_reward += rewards 
    # fwd_rewards += info['forward_reward']
    fwd_rewards += info['reward_forward']
    ctrl_cost += info['reward_ctrl']

    if terminated or truncated:
        print("reward = {}\tfwd_reward = {}\tctrl_reward = {}\n".format(cur_reward, fwd_rewards, ctrl_cost))
        cur_reward, fwd_rewards, ctrl_cost = 0, 0, 0
        obs, info = visual_env.reset()
    # visual_env.render()