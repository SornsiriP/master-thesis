import gym
import torch
from simple_xarm.envs.xarm_env import XarmEnv
from simple_xarm.envs.xarm_env_img import XarmEnv_img
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from simple_xarm.resources.wrapper import ProcessFrame84,ImageToPyTorch
from custom_policy import custom_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th

#tensorboard --logdir ./Mlp_log/
def main():
  log_dir = "./Mlp_log"
  env = XarmEnv()
  # env = XarmEnv_img()
  # n_envs = 4 # Number of copies of the environment
  # env_list = [XarmEnv() for _ in range(4)]
  
  # env = SubprocVecEnv([lambda: env] * n_envs)
  # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
  # env = DummyVecEnv([lambda: env])
  # env = make_vec_env(XarmEnv,n_envs=2)

  env = Monitor(env,log_dir)
  # env = img_obs(env)

  prefix_first = "PPO_normal_random_300"
  prefix_cont  = prefix_first + "_grab"
  timestep = 2000000

  zip_name = "/PPO_normal_random_start_tanh_700000_steps.zip"


  model = first_train(env,log_dir,prefix_first,timestep)
  # model = cont_train(env,log_dir,prefix_first,zip_name,timestep)
  # model = cont_train_no_reset_timestep(env,log_dir,prefix_cont,zip_name,timestep)

  #while True:
  for _ in range(30000):
    env.render()
    action, _state = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()
  env.close()

def first_train(env,log_dir,prefix,timestep): 
  policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[512, 512])   #policy_kwargs=policy_kwargs,
  # policy_kwargs = dict(
  #   features_extractor_class=CustomCNN,
  #   features_extractor_kwargs=dict(features_dim=128),
  # ) 
  checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix=prefix)
  model = PPO('MlpPolicy', env, verbose=1,learning_rate = 0.0001,batch_size=100,gamma=0.995,tensorboard_log=log_dir,n_steps = 3000,policy_kwargs=policy_kwargs, )
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
  env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
  # env = VecNormalize(env)
  return env
  #img obs change to map action 0.2->0.3 to make it faster training

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = th.nn.Sequential(th.nn.Linear(n_flatten, features_dim), th.nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == '__main__':
  main()

