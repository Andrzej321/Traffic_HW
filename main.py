import traci
import os
import sys
from stable_baselines3 import DQN  # or any other RL algorithm

# Set up SUMO environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME")

# Create and train the agent
env = TrafficLightEnv()
model = DQN('MlpPolicy', env)
model.learn(total_timesteps=100000)