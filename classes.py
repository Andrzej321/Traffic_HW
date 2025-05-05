import os
import sys
import traci
import numpy as np
from gymnasium import Env
from gymnasium import spaces
from typing import Tuple


class TrafficLightEnv(Env):
    def __init__(self,
                 sumocfg_file: str,
                 num_seconds: int = 3600,  # 1 hour simulation
                 max_steps: int = 1000):
        super().__init__()

        # SUMO configuration
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise ImportError("Please declare environment variable 'SUMO_HOME'")

        self.sumocfg = sumocfg_file
        self.num_seconds = num_seconds
        self.max_steps = max_steps
        self.sumo_seed = 42
        self.step_length = 1.0  # seconds per step

        # Traffic light settings
        self.tl_id = "intersection1"  # ID of the traffic light (from your SUMO network)
        self.phases = traci.trafficlight.getAllProgramLogics(self.tl_id)[0].phases
        self.num_green_phases = len([p for p in self.phases if 'G' in p.state])

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_green_phases)

        # Define observation space based on what state information you want
        # Example: queue length for each lane (4 approaches)
        num_lanes = 4  # Modify based on your intersection
        self.observation_space = spaces.Box(
            low=0,
            high=100,  # Maximum number of vehicles you expect
            shape=(num_lanes,),
            dtype=np.float32
        )

        # Initialize connection to SUMO
        self.sumo = None
        self._start_simulation()

    def _start_simulation(self):
        """Starts or restarts the SUMO simulation"""
        if self.sumo is None:
            sumo_cmd = ["sumo", "-c", self.sumocfg,
                        "--waiting-time-memory", str(self.num_seconds),
                        "--random", "--seed", str(self.sumo_seed)]
            traci.start(sumo_cmd)
            self.sumo = traci

    def _get_state(self) -> np.ndarray:
        """
        Returns the current state of the environment.
        State consists of queue lengths for each incoming lane.
        """
        state = []

        # Get all incoming lanes
        lanes = self.sumo.trafficlight.getControlledLanes(self.tl_id)

        for lane in lanes:
            # Get number of halting vehicles in the lane
            queue_length = self.sumo.lane.getLastStepHaltingNumber(lane)
            state.append(queue_length)

        return np.array(state, dtype=np.float32)

    def _compute_reward(self, state: np.ndarray) -> float:
        """
        Compute the reward based on the current state.
        Here we use negative sum of queue lengths as reward.
        """
        # Simple reward: negative sum of queue lengths
        reward = -np.sum(state)

        return float(reward)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action in the environment.
        Returns: (next_state, reward, terminated, truncated, info)
        """
        # Set the new traffic light phase
        self.sumo.trafficlight.setPhase(self.tl_id, action)

        # Run simulation for one step
        self.sumo.simulationStep()

        # Get new state and reward
        next_state = self._get_state()
        reward = self._compute_reward(next_state)

        # Check if simulation should end
        terminated = self.sumo.simulation.getTime() >= self.num_seconds
        truncated = False

        info = {
            'time': self.sumo.simulation.getTime(),
            'total_waiting_time': self.sumo.lane.getWaitingTime(self.tl_id)
        }

        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        """
        super().reset(seed=seed)

        # Close existing SUMO connection if any
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None

        # Start new simulation
        self._start_simulation()

        # Get initial state
        initial_state = self._get_state()

        return initial_state, {}

    def close(self):
        """
        Close the SUMO connection.
        """
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None