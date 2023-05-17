import gym
import torch
from simple_xarm.envs.xarm_env import XarmEnv
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from simple_xarm.resources.wrapper import ProcessFrame84,ImageToPyTorch
from custom_policy import custom_policy

#tensorboard --logdir ./Mlp_log/
def main():
  log_dir = "./Mlp_log"
  env = XarmEnv()
  # n_envs = 4 # Number of copies of the environment
  # env_list = [XarmEnv() for _ in range(4)]
  # env = SubprocVecEnv([lambda: env] * n_envs)
  # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
  # env = DummyVecEnv([lambda: env])
  # env = make_vec_env(XarmEnv,n_envs=2)

  env = Monitor(env,log_dir)
  # env = img_obs(env)

  prefix_first = "test_env_simple8_no_img"
  prefix_cont  = prefix_first + "_grab"
  timestep = 2000000

  zip_name = "/test_env_simple8_no_img_800000_steps.zip"


  # model = first_train(env,log_dir,prefix_first,timestep)
  # model = cont_train(env,log_dir,prefix_first,zip_name,timestep)
  model = cont_train_no_reset_timestep(env,log_dir,prefix_cont,zip_name,timestep)

  #while True:
  for _ in range(30000):
    env.render()
    action, _state = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()
  env.close()

def first_train(env,log_dir,prefix,timestep):
  # policy_kwargs = dict(net_arch=[dict(pi=[64, 32, 32], vf=[64, 32, 32])])
  checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=prefix)
  model = PPO('MlpPolicy', env, verbose=1,learning_rate = 0.00025,batch_size=10,gamma=0.995,tensorboard_log=log_dir,n_steps = 1000)
  # model = PPO('CnnPolicy', env, verbose=1,learning_rate = 0.00025,batch_size=8,gamma=0.999,tensorboard_log=log_dir,n_steps = 1000,policy_kwargs=dict(normalize_images=False))

  # model = SAC('CnnPolicy', env, verbose=1,learning_rate = 0.00025,batch_size=8,gamma=0.999,tensorboard_log=log_dir,train_freq = 1)
  model.learn(total_timesteps=timestep,callback=[checkpoint_callback],log_interval=1)
  model.save("SAC_grab")

  return model

def cont_train(env,log_dir,prefix_cont,zip_name,timestep):
  checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=prefix_cont)
  model = PPO.load(log_dir + zip_name)
  # model = SAC.load(log_dir + zip_name)
  model.set_env(env)
  model.learn(total_timesteps=timestep,callback=[checkpoint_callback],reset_num_timesteps=False)
  model.save("SAC_grab")
  return model

def cont_train_no_reset_timestep(env,log_dir,prefix_cont,zip_name,timestep):
  checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=prefix_cont)
  model = PPO.load(log_dir + zip_name)
  # model = SAC.load(log_dir + zip_name)
  model.set_env(env)
  model.learn(total_timesteps=timestep,callback=[checkpoint_callback],reset_num_timesteps=True)
  model.save("SAC_grab")
  return model

def img_obs(env):
  env = ProcessFrame84(env)  #for image
  env = DummyVecEnv([lambda: env]) 
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
  return env
  #img obs change to map action 0.2->0.3 to make it faster training


if __name__ == '__main__':
  main()

