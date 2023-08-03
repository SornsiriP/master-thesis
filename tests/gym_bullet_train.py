import gym
import torch
from simple_xarm.envs.xarm_env import XarmEnv
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from simple_xarm.resources.wrapper import ProcessFrame84
from stable_baselines3.common.policies import  NatureCNN
import torch as th

#To open tensorboard
#tensorboard --logdir ./Mlp_log/

def main():
  log_dir = "./Mlp_log"
  env = XarmEnv()
  env = Monitor(env,log_dir)



  prefix_first = "Test"
  prefix_cont  = prefix_first + "_grab"
  timestep = 2000000

  zip_name = "/SAC_img_random_300_1_450000_steps.zip"

  policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[512, 512]) 
  policy = 'MlpPolicy'

  # If want to use image as observation
  # env,policy_kwargs,policy = img_obs(env)


  model = first_train(env,log_dir,prefix_first,timestep,policy_kwargs,policy)
  # model = cont_train(env,log_dir,prefix_first,zip_name,timestep)
  # model = cont_train_no_reset_timestep(env,log_dir,prefix_cont,zip_name,timestep)

# To run environment after finished training
  # for _ in range(30000):
  #   env.render()
  #   action, _state = model.predict(observation, deterministic=True)
  #   observation, reward, done, info = env.step(action)
  #   if done:
  #     observation = env.reset()
  # env.close()

def first_train(env,log_dir,prefix,timestep,policy_kwargs,policy,): 
  checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=prefix)
  model = PPO(policy, env, verbose=1,learning_rate = 0.0001,batch_size=100,gamma=0.995,tensorboard_log=log_dir,n_steps = 3000,policy_kwargs=policy_kwargs, )
  # model = SAC(policy, env, verbose=1,learning_rate = 0.0001,batch_size=100,gamma=0.995,tensorboard_log=log_dir,policy_kwargs=policy_kwargs, )  
  model.learn(total_timesteps=timestep,callback=[checkpoint_callback],log_interval=1)
  model.save("SAC_grab")
  return model

def cont_train(env,log_dir,prefix_cont,zip_name,timestep):
  checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=prefix_cont)
  # model = PPO.load(log_dir + zip_name)
  model = SAC.load(log_dir + zip_name)
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
  # env = FlattenObservation(env)
  env = DummyVecEnv([lambda: env]) 
  env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
  policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=128),
  ) 
  policy ='CnnPolicy'
  return env,policy_kwargs,policy

if __name__ == '__main__':
  main()

