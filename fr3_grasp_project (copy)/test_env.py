from envs.grasp_env import GraspFeedbackEnv
import numpy as np

env = GraspFeedbackEnv(material='can')
obs, _ = env.reset()
print('obs shape:', obs.shape)
print(f'기본 파지력 (캔): {env.base_force}N')
print()

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f'reward: {reward:.3f} | '
          f'기본력: {info["base_force"]}N | '
          f'delta: {info["delta"]:+.3f}N | '
          f'최종력: {info["final_force"]:.3f}N | '
          f'slip: {info["slip_velocity"]:.4f}')
