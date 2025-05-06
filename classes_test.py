from typing import Tuple, Dict
import numpy as np
from gymnasium import spaces
import os
import sys


class TrafficLightEnvTest:
    def __init__(self, sumocfg_file: str):
        # SUMO initialization code here...
        # SUMO configuration
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise ImportError("Please declare environment variable 'SUMO_HOME'")

        self.sumocfg = sumocfg_file
        self.sumo_seed = 42
        self.step_length = 1.0  # seconds per step

        self.tl_id = "your_traffic_light_id"  # Set your traffic light ID

        # Define action space: 0 = red, 1 = green
        self.action_space = spaces.Discrete(2)

        # State/observation space definition will depend on what state
        # information you want to use from induction loops
        self.observation_space = self._define_observation_space()

        # Track current phase
        self.current_phase = 0  # 0 = red, 1 = green

        # Maximum steps per episode
        self.max_steps = 3600  # or whatever limit you want to set

        # Step counter
        self.steps = 0

    def _define_observation_space(self) -> spaces.Box:
        """
        Define observation space based on detector data we want to use
        """
        # Example: if we use vehicle count, mean speed, and occupancy for each detector
        num_detectors = len(self.sumo.inductionloop.getIDList())
        num_features = 3  # count, speed, occupancy

        return spaces.Box(
            low=np.zeros(num_detectors * num_features),
            high=np.array([50, 30, 100] * num_detectors),  # max values for each feature
            dtype=np.float32
        )

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        # Reset SUMO simulation
        if self.sumo.simulation.getMinExpectedNumber() > 0:
            self.sumo.close()
        self.sumo.start([self.sumocfg_file])

        # Reset internal variables
        self.steps = 0
        self.current_phase = 0  # Start with red light
        self.sumo.trafficlight.setPhase(self.tl_id, 0)  # Set initial phase to red

        # Get initial state
        state = self._get_state()

        return state, {}

    def _get_state(self) -> np.ndarray:
        """
        Get current state from induction loop detectors
        """
        detector_states = []

        for detector_id in self.sumo.inductionloop.getIDList():
            # Get detector data
            count = self.sumo.inductionloop.getLastStepVehicleNumber(detector_id)
            speed = self.sumo.inductionloop.getLastStepMeanSpeed(detector_id)
            occupancy = self.sumo.inductionloop.getLastStepOccupancy(detector_id)

            detector_states.extend([count, speed, occupancy])

        return np.array(detector_states, dtype=np.float32)

    def _compute_reward(self) -> float:
        """
        Compute reward based on detector data
        """
        detector_states = self._get_inductionloop_state()
        reward = 0.0

        # Weights for different components
        THROUGHPUT_WEIGHT = 1.0
        WAITING_PENALTY = -0.3
        SPEED_WEIGHT = 0.5

        for detector_id, data in detector_states.items():
            # Reward for vehicles passing through
            vehicles_passed = len(data['interval_vehicle_data'])
            reward += THROUGHPUT_WEIGHT * vehicles_passed

            # Penalty for waiting vehicles
            waiting_vehicles = sum(1 for vid in data['last_step_vehicle_ids']
                                   if self.sumo.vehicle.getSpeed(vid) < 0.1)
            reward += WAITING_PENALTY * waiting_vehicles

            # Reward for maintaining speed
            mean_speed = data['last_step_mean_speed']
            max_speed = self.sumo.lane.getMaxSpeed(data['lane_id'])
            speed_ratio = mean_speed / max_speed if max_speed > 0 else 0
            reward += SPEED_WEIGHT * speed_ratio

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment

        Args:
            action (int): 0 for red, 1 for green
        """
        self.steps += 1

        # Only change the phase if it's different from current phase
        if action != self.current_phase:
            self.sumo.trafficlight.setPhase(self.tl_id, action)
            self.current_phase = action

        # Execute one simulation step
        self.sumo.simulationStep()

        # Get new state
        new_state = self._get_state()

        # Calculate reward
        reward = self._compute_reward()

        # Check if episode is done
        done = self.steps >= self.max_steps

        # Additional info
        info = {
            'step': self.steps,
            'phase': self.current_phase,
            'metrics': self._get_current_metrics()
        }

        return new_state, reward, done, False, info

    def close(self):
        """Close the environment"""
        self.sumo.close()

    def render(self):
        """
        Render the environment - in SUMO this is typically handled by the GUI version
        If you're using SUMO-GUI, visualization is automatic
        """
        pass
