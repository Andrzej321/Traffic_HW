import os
import sys
import traci
import numpy as np
from gymnasium import Env
from gymnasium import spaces
from typing import Tuple, List


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
        num_detectors = 4  # One per approach
        features_per_detector = 3  # [count, speed, occupancy]

        self.observation_space = spaces.Box(
            low=np.zeros(num_detectors * features_per_detector),
            high=np.array([50, 30, 100] * num_detectors),  # [max_count, max_speed, max_occupancy]
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

    def _get_inductionloop_state(self):
        """
        Get information from induction loop detectors
        """
        # Get all induction loops in the network
        detector_ids = self.sumo.inductionloop.getIDList()
        detector_data = {}

        for detector_id in detector_ids:
            detector_data[detector_id] = {
                # Current state
                'last_step_vehicle_number': self.sumo.inductionloop.getLastStepVehicleNumber(detector_id),
                'last_step_mean_speed': self.sumo.inductionloop.getLastStepMeanSpeed(detector_id),
                'last_step_occupancy': self.sumo.inductionloop.getLastStepOccupancy(detector_id),  # in %

                # Vehicle information
                'last_step_vehicle_ids': self.sumo.inductionloop.getLastStepVehicleIDs(detector_id),

                # Time since last detection
                'time_since_detection': self.sumo.inductionloop.getTimeSinceDetection(detector_id),

                # Position information
                'lane_id': self.sumo.inductionloop.getLaneID(detector_id),
                'position': self.sumo.inductionloop.getPosition(detector_id),

                # Collected vehicle data
                'interval_vehicle_data': self._get_vehicle_data(detector_id)
            }

        return detector_data

    def _get_vehicle_data(self, detector_id: str):
        """
        Get detailed vehicle data from an induction loop
        """
        # Get vehicle data for the last simulation step
        vehicle_data = self.sumo.inductionloop.getVehicleData(detector_id)

        # Process the vehicle data
        processed_data = []
        for veh_id, entry_time, leave_time, type_id in vehicle_data:
            processed_data.append({
                'vehicle_id': veh_id,
                'entry_time': entry_time,
                'leave_time': leave_time,
                'vehicle_type': type_id,
                'occupancy_time': leave_time - entry_time if leave_time > 0 else 0
            })

        return processed_data

    def _get_specific_detector_state(self, detector_id: str):
        """
        Get comprehensive state for a specific detector
        """
        if not self.sumo.inductionloop.exists(detector_id):
            raise ValueError(f"Detector {detector_id} does not exist")

        return {
            'current': {
                'vehicle_count': self.sumo.inductionloop.getLastStepVehicleNumber(detector_id),
                'mean_speed': self.sumo.inductionloop.getLastStepMeanSpeed(detector_id),
                'occupancy': self.sumo.inductionloop.getLastStepOccupancy(detector_id),
                'present_vehicles': self.sumo.inductionloop.getLastStepVehicleIDs(detector_id)
            },
            'position_info': {
                'lane': self.sumo.inductionloop.getLaneID(detector_id),
                'position': self.sumo.inductionloop.getPosition(detector_id)
            },
            'vehicle_data': self._get_vehicle_data(detector_id)
        }

    def _get_approach_detectors_state(self):
        """
        Get state from detectors grouped by approach direction
        """
        # Define detector IDs for each approach (modify according to your setup)
        approach_detectors = {
            'north': ['detector_north_1', 'detector_north_2'],
            'south': ['detector_south_1', 'detector_south_2'],
            'east': ['detector_east_1', 'detector_east_2'],
            'west': ['detector_west_1', 'detector_west_2']
        }

        approach_states = {}

        for direction, detectors in approach_detectors.items():
            direction_data = {
                'vehicle_count': 0,
                'mean_speed': 0,
                'occupancy': 0,
                'active_detectors': 0
            }

            for detector_id in detectors:
                if self.sumo.inductionloop.exists(detector_id):
                    direction_data['vehicle_count'] += self.sumo.inductionloop.getLastStepVehicleNumber(detector_id)

                    speed = self.sumo.inductionloop.getLastStepMeanSpeed(detector_id)
                    occupancy = self.sumo.inductionloop.getLastStepOccupancy(detector_id)

                    if speed > 0:  # Only count active detectors for averaging
                        direction_data['mean_speed'] += speed
                        direction_data['occupancy'] += occupancy
                        direction_data['active_detectors'] += 1

            # Calculate averages
            if direction_data['active_detectors'] > 0:
                direction_data['mean_speed'] /= direction_data['active_detectors']
                direction_data['occupancy'] /= direction_data['active_detectors']

            approach_states[direction] = direction_data

        return approach_states

    def _get_detector_state_vector(self, detector_ids: List[str]):
        """
        Create a fixed-size state vector from multiple detectors
        Useful for RL input
        """
        state_vector = []

        for detector_id in detector_ids:
            if self.sumo.inductionloop.exists(detector_id):
                # Get basic measurements
                count = self.sumo.inductionloop.getLastStepVehicleNumber(detector_id)
                speed = self.sumo.inductionloop.getLastStepMeanSpeed(detector_id)
                occupancy = self.sumo.inductionloop.getLastStepOccupancy(detector_id)

                # Add to state vector
                state_vector.extend([count, speed, occupancy])
            else:
                # Add zeros for non-existent detectors to maintain fixed size
                state_vector.extend([0, 0, 0])

        return np.array(state_vector, dtype=np.float32)

    def _compute_reward(self) -> float:
        """
        Compute reward based on induction loop detector data
        """
        reward = 0.0
        detector_states = self._get_inductionloop_state()

        # Weights for different reward components
        WAITING_PENALTY = -0.5  # Penalty for vehicles waiting
        SPEED_REWARD = 1.0  # Reward for maintaining good speed
        THROUGHPUT_REWARD = 2.0  # Reward for throughput
        OCCUPANCY_PENALTY = -0.3  # Penalty for high occupancy

        for detector_id, data in detector_states.items():
            # Reward based on speed (higher speeds are better, up to speed limit)
            mean_speed = data['last_step_mean_speed']
            speed_limit = self.sumo.lane.getMaxSpeed(data['lane_id'])
            speed_ratio = mean_speed / speed_limit if speed_limit > 0 else 0
            reward += SPEED_REWARD * speed_ratio

            # Penalty for stopped vehicles (detected by the loop)
            vehicles = data['last_step_vehicle_ids']
            waiting_vehicles = sum(1 for vid in vehicles
                                   if self.sumo.vehicle.getSpeed(vid) < 0.1)
            reward += WAITING_PENALTY * waiting_vehicles

            # Reward for throughput (vehicles successfully passing the detector)
            vehicle_data = data['interval_vehicle_data']
            vehicles_passed = len([v for v in vehicle_data
                                   if v['leave_time'] > v['entry_time']])
            reward += THROUGHPUT_REWARD * vehicles_passed

            # Penalty for high occupancy (indicates congestion)
            occupancy = data['last_step_occupancy']
            if occupancy > 80:  # High occupancy threshold
                reward += OCCUPANCY_PENALTY * (occupancy / 100)

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute step with updated reward calculation
        """
        # Store initial state metrics
        initial_metrics = self._get_current_metrics()

        # Apply action
        self.sumo.trafficlight.setPhase(self.tl_id, action)

        # Simulation steps
        for _ in range(self.yellow_time + self.green_time):
            self.sumo.simulationStep()

        # Get new state
        next_state = self._get_state()

        # Calculate reward using one of the reward functions
        reward = self._compute_balanced_reward()

        # Check termination conditions
        done = self.sumo.simulation.getTime() >= self.max_steps

        # Additional info
        info = {
            'throughput': self._get_total_throughput(),
            'average_speed': self._get_average_speed(),
            'total_waiting_time': self._get_total_waiting_time(),
            'average_occupancy': self._get_average_occupancy()
        }

        return next_state, reward, done, False, info

    def _get_current_metrics(self):
        """
        Helper function to get current traffic metrics
        """
        detector_states = self._get_inductionloop_state()

        return {
            'throughput': sum(len(d['interval_vehicle_data'])
                              for d in detector_states.values()),
            'waiting_time': sum(self.sumo.vehicle.getWaitingTime(vid)
                                for d in detector_states.values()
                                for vid in d['last_step_vehicle_ids']),
            'average_speed': np.mean([d['last_step_mean_speed']
                                      for d in detector_states.values()]),
            'occupancy': np.mean([d['last_step_occupancy']
                                  for d in detector_states.values()])
        }

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