import gym
import torch
from simple_xarm.envs.xarm_env import XarmEnv
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from simple_xarm.resources.wrapper import ProcessFrame84,ImageToPyTorch

# from simple_xarm.resources.wrapper import ProcessFrame84,ImageToPyTorch
#tensorboard --logdir ./Mlp_log/
def main():
  log_dir = "./Mlp_log"
  env = XarmEnv()
  env = img_obs(env)

  SAC_result = "/test_env_custom_policy_rew_1000000_steps"
  PPO_result = "/test_env_CNN_100000_steps"
  New_start_pos = "/Xarm_SoftBody_grab_50000_steps"

  observation = env.reset()
  model = PPO.load(log_dir + PPO_result)

  while True:
    env.render()
    action, _state = model.predict(observation,deterministic=True)  
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()
  env.close()

def img_obs(env):
  env = ProcessFrame84(env)  #for image
  env = DummyVecEnv([lambda: env]) 
  env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
  return env

if __name__ == '__main__':
  main()