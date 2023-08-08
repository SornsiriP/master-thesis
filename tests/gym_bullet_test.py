import gym
import torch
from simple_xarm.envs.xarm_env import XarmEnv
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from simple_xarm.resources.wrapper import ProcessFrame84,ImageToPyTorch
import pybullet as pb

# from simple_xarm.resources.wrapper import ProcessFrame84,ImageToPyTorch
#tensorboard --logdir ./Mlp_log/
def main():
  log_dir = "./Model"
  env = XarmEnv()

  #To test image observation model
  # env = img_obs(env)

  # SAC_result = "/SAC_img/SAC_img_random_300_250000_steps"
  PPO_result = "/PPO_direct/PPO_normal_random_300_no_randorien_400000_steps"

  rKey = ord('r')

  observation = env.reset()
  model = PPO.load(log_dir + PPO_result)
  # model = SAC.load(log_dir + SAC_result)

  while True:
    env.render()
    action, _state = model.predict(observation,deterministic=True)  
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()

    #Press r to reset environment
    events = pb.getKeyboardEvents()
    if rKey in events and events[rKey] & pb.KEY_IS_DOWN:
      env.reset()  
  env.close()

def img_obs(env):
  env = ProcessFrame84(env)  #for image
  env = DummyVecEnv([lambda: env]) 
  env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
  return env

if __name__ == '__main__':
  main()