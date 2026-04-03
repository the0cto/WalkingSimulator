import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv


class JumpingAntEnv(AntEnv):
    """Custom Environment that follows gym interface."""

    #metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            render_mode = "human",
            xml_file = "ant.xml",
            frame_skip = 5,
            **kwargs
    ):
        # You can override default kwargs before passing to parent
        super().__init__(
            xml_file = "ant.xml",
            healthy_z_range = (0.2, 3),
            terminate_when_unhealthy=True,

            **kwargs
        )
        # Optionally modify observation/action spaces here
        self.render_mode = render_mode

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        torso_xmat = self.data.xmat[1].reshape(3, 3)
        torso_normal = torso_xmat[:, 2]
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z and torso_normal[2] > 0
        return is_healthy

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(observation, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


    def _get_rew(self, obs: float, action):

        # Customize the reward
        torso_height_reward = 5 * obs[0]
        torso_z_vel_reward = 10 * max(0, obs[15])
        is_in_air = (self.data.ncon == 0)
        in_air_reward = 10 * (is_in_air - 0.3)

        #ctrl_cost = 0.5 * np.square(action).sum()

        reward = torso_height_reward + torso_z_vel_reward + in_air_reward + self.healthy_reward

        reward_info = {
            "torso_height_reward": torso_height_reward,
            "torso_z_vel_reward": torso_z_vel_reward,
            "in_air_reward": in_air_reward,
            "reward_survive": self.healthy_reward,
        }

        return reward, reward_info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Custom reset logic here
        return obs, info

