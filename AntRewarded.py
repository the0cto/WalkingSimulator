from EnvRecompense import AntGoalEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

env = AntGoalEnv()


obs, _ = env.reset()

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99
)

model.learn(total_timesteps=40_000, callback=CheckpointCallback, progress_bar=True)
model.save("AimAnt_40k")

env = AntGoalEnv(render_mode = "human")

model = SAC.load("AimAnt_40k.zip", env=env)

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
