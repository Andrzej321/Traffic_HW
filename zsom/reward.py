import gymnasium as gym
import traci
import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Discrete, Box
import gym as gym_old  # For compatibility with Stable-Baselines3 wrapper


class SumoEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "sumo_env_v0"}

    def __init__(self):
        self.agents = ["tls_H2", "tls_H3"]
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: Discrete(2) for agent in self.agents}  # 0: GGG, 1: rGG
        self.observation_spaces = {agent: Box(low=0, high=np.inf, shape=(4,)) for agent in
                                   self.agents}  # [speed, density, queue, waiting_time]
        self.sumo_cmd = ["sumo", "-c", "../sumo_files/onramp.sumo.cfg"]  # Update path if needed
        self.min_phase_duration = 5
        self.last_phase_change = {"H2": 0, "H3": 0}
        self.current_phase = {"H2": "GGG", "H3": "GGG"}

    def reset(self, seed=None, options=None):
        traci.close()
        traci.start(self.sumo_cmd)
        self.last_phase_change = {"H2": 0, "H3": 0}
        self.current_phase = {"H2": "GGG", "H3": "GGG"}
        self.agents = self.possible_agents[:]
        obs = {f"tls_H2": self._get_obs("H2"), f"tls_H3": self._get_obs("H3")}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        current_time = traci.simulation.getTime()
        for agent, action in actions.items():
            tls_id = agent[4:]  # Extract H2 or H3
            if current_time - self.last_phase_change[tls_id] >= self.min_phase_duration:
                state = "GGG" if action == 0 else "rGG"
                if state != self.current_phase[tls_id]:
                    traci.trafficlight.setRedYellowGreenState(tls_id, state)
                    self.last_phase_change[tls_id] = current_time
                    self.current_phase[tls_id] = state

        traci.simulationStep()

        obs = {f"tls_H2": self._get_obs("H2"), f"tls_H3": self._get_obs("H3")}
        reward = self._compute_reward()
        rewards = {agent: reward for agent in self.agents}  # Shared reward
        terminated = {agent: traci.simulation.getTime() >= 1000 for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if terminated["tls_H2"]:
            self.agents = []
        return obs, rewards, terminated, truncated, infos

    def _get_obs(self, tls):
        if tls == "H2":
            lanes = ["h1_0", "h1_1", "r1_0"]
        else:
            lanes = ["h2_0", "h2_1", "r2_0"]
        active_lanes = [lane for lane in lanes[:2] if traci.lane.getLastStepVehicleNumber(lane) > 0]
        speed = sum(traci.lane.getLastStepMeanSpeed(lane) for lane in active_lanes) / (len(active_lanes) or 1)
        density = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes[:2]) / 2
        queue = traci.lane.getLastStepHaltingNumber(lanes[2])
        waiting_time = traci.lane.getWaitingTime(lanes[2])
        return np.array([speed, density, queue, waiting_time], dtype=np.float32)

    def _compute_reward(self):
        all_lanes = ["h1_0", "h1_1", "h2_0", "h2_1", "r1_0", "r2_0"]
        speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
                  traci.lane.getLastStepVehicleNumber(lane) > 0]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        queue_r1 = traci.lane.getLastStepHaltingNumber("r1_0")
        queue_r2 = traci.lane.getLastStepHaltingNumber("r2_0")
        speed_reward = avg_speed / 33.33 if avg_speed > 0 else 0.0
        queue_penalty = -0.1 * (queue_r1 + queue_r2)
        return speed_reward + queue_penalty

    def close(self):
        # Ensure the SUMO connection is properly closed
        traci.close()


# Wrapper to make PettingZoo env compatible with Stable-Baselines3
class SB3Wrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Combine action spaces into a single Discrete space for both agents (0: GGG/GGG, 1: GGG/rGG, 2: rGG/GGG, 3: rGG/rGG)
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0, high=np.inf, shape=(8,), dtype=np.float32  # Concatenate observations for both agents
        )

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        combined_obs = np.concatenate([obs["tls_H2"], obs["tls_H3"]])
        return combined_obs, infos["tls_H2"]

    def close(self):
        # Ensure the inner environment closes properly
        self.env.close()


    def step(self, action):
        # Map single action to multi-agent actions
        action_map = {
            0: {"tls_H2": 0, "tls_H3": 0},  # GGG/GGG
            1: {"tls_H2": 0, "tls_H3": 1},  # GGG/rGG
            2: {"tls_H2": 1, "tls_H3": 0},  # rGG/GGG
            3: {"tls_H2": 1, "tls_H3": 1}  # rGG/rGG
        }
        actions = action_map[action]
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        combined_obs = np.concatenate([obs["tls_H2"], obs["tls_H3"]])
        reward = rewards["tls_H2"]  # Shared reward
        done = terminated["tls_H2"]
        return combined_obs, reward, done, truncated["tls_H2"], infos["tls_H2"]


# Training
def train():
    env = SumoEnv()
    wrapped_env = SB3Wrapper(env)
    check_env(wrapped_env, warn=True)  # Validate environment
    model = PPO("MlpPolicy", wrapped_env, verbose=1, learning_rate=0.0001)
    print("Starting training...")
    model.learn(total_timesteps=10000)  # Adjust based on convergence
    model.save("sumo_ppo_model")
    print("Training complete. Model saved at sumo_ppo_model.zip")
    wrapped_env.close()
    return model


# Deployment
def deploy(model_path):
    env = SumoEnv()
    env.sumo_cmd = ["sumo-gui", "-c", "../sumo_files/onramp.sumo.cfg"]
    wrapped_env = SB3Wrapper(env)
    model = PPO.load(model_path)
    obs, _ = wrapped_env.reset()
    terminated = False

    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, _, _ = wrapped_env.step(action)
        print(f"Time: {traci.simulation.getTime():.1f}, Reward: {reward:.2f}")

    wrapped_env.close()


if __name__ == "__main__":
    print("Starting training...")
    model = train()
    print("Deploying trained model in SUMO-GUI...")
    deploy("sumo_ppo_model")