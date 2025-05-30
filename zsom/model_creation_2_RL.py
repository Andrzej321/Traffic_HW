# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

CONFIG_DIR = "../sumo_files"

# Get list of .sumocfg files and randomly select one
def get_random_config():
    try:
        # List all files in the config directory
        files = os.listdir(CONFIG_DIR)
        # Filter for .sumocfg files
        config_files = [f for f in files if f.endswith('.sumocfg')]
        if not config_files:
            sys.exit(f"No .sumocfg files found in {CONFIG_DIR}")
        # Randomly choose one config file
        chosen_config = random.choice(config_files)
        # Return the full path to the chosen file
        config_path = os.path.join(CONFIG_DIR, chosen_config)
        print(f"Selected SUMO config file: {config_path}")
        return config_path
    except Exception as e:
        sys.exit(f"Error accessing config files: {e}")

# Define SUMO configuration with random choice

# traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)
loop0 = 0
loop1 = 0
loop2 = 0
loop3 = 0
loop4 = 0
loop5 = 0
current_phase_1 = 0
current_phase_2 = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 6000  # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1  # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
# If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9  # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
# If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
EPSILON = 0.1  # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

ACTIONS = [0, 1]  # The discrete action space (0 all red, 1 1st ramp green, 2nd ramp green, 3rd both green)

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 10

# Number of training runs
NUM_RUNS = 5

# Lists to store results for all runs
all_step_histories_1 = []
all_step_histories_2 = []
all_reward_histories_1 = []
all_reward_histories_2 = []
all_queue_histories_1 = []
all_queue_histories_2 = []

all_queue_histories_overall = []


# -------------------------
# Step 7: Define Functions
# -------------------------

def build_model(state_size, action_size):
    """
    Build a simple feedforward neural network that approximates Q-values.
    """
    model = keras.Sequential()  # Feedforward neural network
    model.add(layers.Input(shape=(state_size,)))  # Input layer
    model.add(layers.Dense(32, activation='relu'))  # First hidden layer
    model.add(layers.Dense(32, activation='relu'))  # Second hidden layer
    model.add(layers.Dense(action_size, activation='linear'))  # Output layer
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return model


def to_array(state_tuple):
    """
    Convert the state tuple into a NumPy array for neural network input.
    """
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

    # Create the DQN model


def get_max_Q_value_of_state_1(s):  # 1. Objective Function
    state_array = to_array(s)
    Q_values = dqn_model_1.predict(state_array, verbose=0)[0]  # shape: (action_size,)
    return np.max(Q_values)

def get_max_Q_value_of_state_2(s):  # 1. Objective Function
    state_array = to_array(s)
    Q_values = dqn_model_2.predict(state_array, verbose=0)[0]  # shape: (action_size,)
    return np.max(Q_values)


def get_reward_1(state):  # 2. Constraint 2
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    all_lanes = ["h1_0", "h1_1", "r1_0"]
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
              traci.lane.getLastStepVehicleNumber(lane) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    queue = traci.lane.getLastStepHaltingNumber("r1_0")

    emergency_brakes = 0
    for veh_id in traci.vehicle.getIDList():
        # Get current deceleration (positive value, m/s^2)
        decel = traci.vehicle.getDecel(veh_id)
        # Get emergency deceleration threshold (m/s^2)
        emergency_decel = traci.vehicle.getEmergencyDecel(veh_id)
        # Count as emergency braking if deceleration exceeds or equals emergency threshold
        if decel >= emergency_decel:
            emergency_brakes += 1

    speed_reward = avg_speed / 33.33 if avg_speed > 0 else 0.0
    queue_penalty = -0.1 * queue
    brake_penalty = -0.9 * emergency_brakes

    return speed_reward + queue_penalty + brake_penalty


def get_reward_2(state):  # 2. Constraint 2
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    all_lanes = ["h2_0", "h2_1", "r2_0"]
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
              traci.lane.getLastStepVehicleNumber(lane) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    queue = traci.lane.getLastStepHaltingNumber("r2_0")

    emergency_brakes = 0
    for veh_id in traci.vehicle.getIDList():
        # Get current deceleration (positive value, m/s^2)
        decel = traci.vehicle.getDecel(veh_id)
        # Get emergency deceleration threshold (m/s^2)
        emergency_decel = traci.vehicle.getEmergencyDecel(veh_id)
        # Count as emergency braking if deceleration exceeds or equals emergency threshold
        if decel >= emergency_decel:
            emergency_brakes += 1

    speed_reward = avg_speed / 33.33 if avg_speed > 0 else 0.0
    queue_penalty = -0.1 * queue
    brake_penalty = -0.9 * emergency_brakes

    return speed_reward + queue_penalty + brake_penalty


def get_reward(state):  # 2. Constraint 2
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    all_lanes = ["h1_0", "h1_1", "h2_0", "h2_1", "r1_0", "r2_0"]
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
              traci.lane.getLastStepVehicleNumber(lane) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    queue_1 = traci.lane.getLastStepHaltingNumber("r1_0")
    queue_2 = traci.lane.getLastStepHaltingNumber("r2_0")

    emergency_brakes = 0
    for veh_id in traci.vehicle.getIDList():
        # Get current deceleration (positive value, m/s^2)
        decel = traci.vehicle.getDecel(veh_id)
        # Get emergency deceleration threshold (m/s^2)
        emergency_decel = traci.vehicle.getEmergencyDecel(veh_id)
        # Count as emergency braking if deceleration exceeds or equals emergency threshold
        if decel >= emergency_decel:
            emergency_brakes += 1

    speed_reward = avg_speed / 33.33 if avg_speed > 0 else 0.0
    queue_penalty = -0.1 * (queue_1 + queue_2)
    brake_penalty = -0.9 * emergency_brakes

    return speed_reward + queue_penalty + brake_penalty


def get_state_1():  # 3&4. Constraint 3 & 4
    global loop0, loop1, loop2, current_phase_1

    # Traffic light ID
    det0 = "loop0"
    det1 = "loop1"
    det2 = "loop2"

    # Get queue lengths from each detector
    loop0 = traci.lanearea.getLastStepVehicleNumber(det0)
    loop1 = traci.lanearea.getLastStepVehicleNumber(det1)
    loop2 = traci.lanearea.getLastStepVehicleNumber(det2)

    traffic_light_id_1 = "H2"

    # Get current phase index
    current_phase_1 = traci.trafficlight.getPhase(traffic_light_id_1)

    return (loop0, loop1, loop2, current_phase_1)


def get_state_2():  # 3&4. Constraint 3 & 4
    global loop3, loop4, loop5, current_phase_2

    # Traffic light ID
    det3 = "loop3"
    det4 = "loop4"
    det5 = "loop5"



    # Get queue lengths from each detector
    loop3 = traci.lanearea.getLastStepVehicleNumber(det3)
    loop4 = traci.lanearea.getLastStepVehicleNumber(det4)
    loop5 = traci.lanearea.getLastStepVehicleNumber(det5)

    traffic_light_id_2 = "H3"

    # Get current phase index
    current_phase_2 = traci.trafficlight.getPhase(traffic_light_id_2)

    return (loop3, loop4, loop5, current_phase_2)


def get_state():  # 3&4. Constraint 3 & 4
    global loop0, loop1, loop2, loop3, loop4, loop5, current_phase_1, current_phase_2

    # Traffic light ID
    det0 = "loop0"
    det1 = "loop1"
    det2 = "loop2"
    det3 = "loop3"
    det4 = "loop4"
    det5 = "loop5"



    # Get queue lengths from each detector
    loop0 = traci.lanearea.getLastStepVehicleNumber(det0)
    loop1 = traci.lanearea.getLastStepVehicleNumber(det1)
    loop2 = traci.lanearea.getLastStepVehicleNumber(det2)
    loop3 = traci.lanearea.getLastStepVehicleNumber(det3)
    loop4 = traci.lanearea.getLastStepVehicleNumber(det4)
    loop5 = traci.lanearea.getLastStepVehicleNumber(det5)

    traffic_light_id_1 = "H2"
    traffic_light_id_2 = "H3"

    # Get current phase index
    current_phase_1 = traci.trafficlight.getPhase(traffic_light_id_1)
    current_phase_2 = traci.trafficlight.getPhase(traffic_light_id_2)

    return (loop0, loop1, loop2, loop3, loop4, loop5, current_phase_1, current_phase_2)


def apply_action_1(action):  # 5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step

    if action == 0:
        traci.trafficlight.setRedYellowGreenState("H2","rGG")
        return
    elif action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            traci.trafficlight.setRedYellowGreenState("H2","GGG")
            last_switch_step = current_simulation_step
        return


def apply_action_2(action):
    # 5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step

    if action == 0:
        traci.trafficlight.setRedYellowGreenState("H3","rGG")
        return
    elif action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            traci.trafficlight.setRedYellowGreenState("H3","GGG")
            last_switch_step = current_simulation_step
        return


def update_Q_table_1(old_state, action, reward, new_state):  # 6. Constraint 6
    """
    In DQN, we do a single-step gradient update instead of a table update.
    """
    # 1) Predict current Q-values from old_state (current state)
    old_state_array = to_array(old_state)
    Q_values_old = dqn_model_1.predict(old_state_array, verbose=0)[0]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    new_state_array = to_array(new_state)
    Q_values_new = dqn_model_1.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)

    # 3) Incorporate ALPHA to partially update the Q-value
    Q_values_old[action] = Q_values_old[action] + ALPHA * (reward + GAMMA * best_future_q - Q_values_old[action])

    # 4) Train (fit) the DQN on this single sample
    dqn_model_1.fit(old_state_array, np.array([Q_values_old]), verbose=0)


def update_Q_table_2(old_state, action, reward, new_state):  # 6. Constraint 6
    """
    In DQN, we do a single-step gradient update instead of a table update.
    """
    # 1) Predict current Q-values from old_state (current state)
    old_state_array = to_array(old_state)
    Q_values_old = dqn_model_2.predict(old_state_array, verbose=0)[0]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    new_state_array = to_array(new_state)
    Q_values_new = dqn_model_2.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)

    # 3) Incorporate ALPHA to partially update the Q-value
    Q_values_old[action] = Q_values_old[action] + ALPHA * (reward + GAMMA * best_future_q - Q_values_old[action])

    # 4) Train (fit) the DQN on this single sample
    dqn_model_2.fit(old_state_array, np.array([Q_values_old]), verbose=0)


def get_action_from_policy_1(state):  # 7. Constraint 7
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model_1.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))


def get_action_from_policy_2(state):  # 7. Constraint 7
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model_2.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))


# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting

state_size = 8  # (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)
action_size = len(ACTIONS)

dqn_model_1 = build_model(state_size, action_size)
dqn_model_2 = build_model(state_size, action_size)

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for run in range(NUM_RUNS):

    last_switch_step = -MIN_GREEN_STEPS
    cumulative_reward_1 = 0.0
    cumulative_reward_2 = 0.0
    step_history_1 = []
    step_history_2 = []
    reward_history_1 = []
    reward_history_2 = []
    queue_history_1 = []
    queue_history_2 = []

    chosen_config_file = get_random_config()

    # Step 4: Define Sumo configuration
    Sumo_config = [
        'sumo',
        '-c', chosen_config_file,
        '--step-length', '0.2',
        '--delay', '10',
        '--lateral-resolution', '0'
    ]

    # Step 5: Open connection between SUMO and Traci
    traci.start(Sumo_config)
    print(f"\nRun {run + 1}/{NUM_RUNS}: Training Started")

    for step in range(TOTAL_STEPS):
        current_simulation_step = step  # keep this variable for apply_action usage

        state_1 = get_state()
        state_2 = get_state()

        action_1 = get_action_from_policy_1(state_1)
        action_2 = get_action_from_policy_2(state_2)

        apply_action_1(action_1)
        apply_action_2(action_2)

        traci.simulationStep()  # Advance simulation by one step

        new_state_1 = get_state()
        new_state_2 = get_state()

        reward_1 = get_reward(new_state_1)
        reward_2 = get_reward(new_state_2)

        cumulative_reward_1 += reward_1
        cumulative_reward_2 += reward_2

        update_Q_table_1(state_1, action_1, reward_1, new_state_1)
        update_Q_table_2(state_2, action_2, reward_2, new_state_2)

        # Print Q-values for the old_state right after update
        updated_q_vals_1 = dqn_model_1.predict(to_array(state_1), verbose=0)[0]
        updated_q_vals_2 = dqn_model_2.predict(to_array(state_2), verbose=0)[0]

        # Record data every 100 steps
        if step % 1 == 0:
            updated_q_vals_1 = dqn_model_1.predict(to_array(state_1), verbose=0)[0]
            updated_q_vals_2 = dqn_model_2.predict(to_array(state_2), verbose=0)[0]
            print(
                f"Step {step}, Current_State: {state_1}, Action: {action_1}, New_State: {new_state_1}, Reward: {reward_1:.2f}, Cumulative Reward: {cumulative_reward_1:.2f}, Q-values(current_state): {updated_q_vals_1}")
            print(
                f"Step {step}, Current_State: {state_2}, Action: {action_2}, New_State: {new_state_2}, Reward: {reward_2:.2f}, Cumulative Reward: {cumulative_reward_2:.2f}, Q-values(current_state): {updated_q_vals_2}")
        step_history_1.append(step)
        step_history_2.append(step)

        reward_history_1.append(cumulative_reward_1)
        reward_history_2.append(cumulative_reward_1)

        queue_history_1.append(sum(new_state_1[:-1]))  # sum of queue lengths
        queue_history_2.append(sum(new_state_2[:-1]))  # sum of queue lengths

    os.makedirs("models_2", exist_ok=True)
    model_path_1 = f"../zsom/models_2/run_1_{run + 1}"
    model_path_2 = f"../zsom/models_2/run_2_{run + 1}"
    dqn_model_1.export(model_path_1)
    dqn_model_2.export(model_path_2)
    print(f"Run {run + 1}: Trained model saved to '{model_path_1}' and '{model_path_2}'")

    # Store results for this run
    all_step_histories_1.append(step_history_1)
    all_reward_histories_1.append(reward_history_1)
    all_queue_histories_1.append(queue_history_1)

    all_step_histories_2.append(step_history_2)
    all_reward_histories_2.append(reward_history_2)
    all_queue_histories_2.append(queue_history_2)

    all_queue_histories_overall.append(queue_history_1 + queue_history_2)

    # Close SUMO connection for this run
    traci.close()
    print(f"Run {run + 1}: Training Completed")



# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model_1.summary()
dqn_model_2.summary()

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
for run in range(NUM_RUNS):
    plt.plot(all_step_histories_1[run], all_reward_histories_1[run], marker='o', linestyle='-', label=f"Run {run + 1}")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL1 Training (DQN): Cumulative Reward over Steps for 10 Runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
for run in range(NUM_RUNS):
    plt.plot(all_step_histories_1[run], all_queue_histories_1[run], marker='o', linestyle='-', label=f"Run {run + 1}")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL1 Training (DQN): Queue Length over Steps for 10 Runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
for run in range(NUM_RUNS):
    plt.plot(all_step_histories_2[run], all_reward_histories_2[run], marker='o', linestyle='-', label=f"Run {run + 1}")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL2 Training (DQN): Cumulative Reward over Steps for 10 Runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
for run in range(NUM_RUNS):
    plt.plot(all_step_histories_2[run], all_queue_histories_2[run], marker='o', linestyle='-', label=f"Run {run + 1}")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL2 Training (DQN): Queue Length over Steps for 10 Runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
for run in range(NUM_RUNS):
    plt.plot(all_step_histories_2[run], all_queue_histories_overall[run], marker='o', linestyle='-', label=f"Run {run + 1}")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training (DQN): Overall queue Length over Steps for 10 Runs")
plt.legend()
plt.grid(True)
plt.show()