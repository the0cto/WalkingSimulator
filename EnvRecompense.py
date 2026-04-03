import gymnasium as gym
import numpy as np


class AntGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.env = gym.make("Ant-v5", render_mode=render_mode)
        self.action_space = self.env.action_space

        # Observation = obs original + goal (x, y)
        obs_dim = self.env.observation_space.shape[0] + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.goal = None

    def render(self):
        viewer = self.env.unwrapped.mujoco_renderer.viewer
        if viewer is not None and self.goal is not None:
            viewer.add_marker(
                pos=np.array([self.goal[0], self.goal[1],0.5]),
                size=np.array([0.2, 0.2, 0.2]),
                rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                type=2  # sphere
            )
        return self.env.render()

    def _get_rew(self, obs: float, action):

        # position de l'ant
        ant_pos = self.env.unwrapped.data.qpos[:2]
        # distance à la cible
        dist = np.linalg.norm(ant_pos - self.goal)

        reward = -25*dist

        #bonus si proche de la cible (inversement proportionnel à la distance)
        if dist < 1:
            reward += 10/dist
            print(reward)

        # pénalité d'énergie (optionnel mais recommandé)
        # ctrl_cost = 0.05 * np.square(action).sum()
        # reward -= ctrl_cost


        reward_info = {
            "distance to target": dist
        }

        return reward, reward_info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        # cible aléatoire
        self.goal = np.random.uniform(low=-5.0, high=5.0, size=2)
        return self._get_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward, reward_info = self._get_rew(obs, action)
        if self.render_mode == "human":
            self.render()
            # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(obs), reward, terminated, truncated, info


    def close(self):
        self.env.close()

    def _get_obs(self, obs):
        return np.concatenate([obs, self.goal])