from gym.envs.registration import register

register(
    id='xarm-v0',
    entry_point='simple_xarm.envs:XarmEnv',
)
