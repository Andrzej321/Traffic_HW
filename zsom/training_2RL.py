# ---------------------------------------------------------------------------
# Step 1: Add modules to provide access to specific libraries and functions
# ---------------------------------------------------------------------------
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import traci

# -------------------------------------------
# Step 2: Establish path to SUMO (SUMO_HOME)
# -------------------------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# -------------------------
# Step 3: Define Variables
# -------------------------

# Get directory of sumo config files
CONFIG_DIR = "../sumo_files"

# Variables for RL State (queue lengths from detectors and current phase)
loop0 = 0
loop1 = 0
loop2 = 0
loop3 = 0
loop4 = 0
loop5 = 0
current_phase_1 = 0
current_phase_2 = 0

# Reinforcement Learning Hyperparameters
TOTAL_STEPS = 5000  # The total number of simulation steps for continuous (online) training.
ALPHA = 0.1  # Learning rate (α) between[0, 1]
GAMMA = 0.9  # Discount factor (γ) between[0, 1]
EPSILON = 0.1  # Exploration rate (ε) between[0, 1]

# Discrete action space
ACTIONS = [0, 1]  # (0 = keep phase, 1 = switch phase)

# Additional Stability Parameter
MIN_GREEN_STEPS = 10

# Number of training runs
NUM_RUNS = 10

# Lists to store results for all runs
all_step_histories_1 = []
all_step_histories_2 = []
all_reward_histories_1 = []
all_reward_histories_2 = []
all_queue_histories_1 = []
all_queue_histories_2 = []

all_queue_histories_overall = []


# -------------------------
# Step 4: Define Functions
# -------------------------

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

# Feedforward neural network that approximates Q-values
def build_model(state_size, action_size):
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

# Convert the state tuple into a NumPy array for neural network input
def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))


def get_max_Q_value_of_state_1(s):
    state_array = to_array(s)
    Q_values = dqn_model_1.predict(state_array, verbose=0)[0]
    return np.max(Q_values)

def get_max_Q_value_of_state_2(s):
    state_array = to_array(s)
    Q_values = dqn_model_2.predict(state_array, verbose=0)[0]  # shape: (action_size,)
    return np.max(Q_values)

# Reward function for agents
def get_reward(state):

    # Reward for higher average speed in the simulation enviroment. Punishing stationary vehicles behind red lights.
    all_lanes = ["h1_0", "h1_1", "h2_0", "h2_1", "r1_0", "r2_0"]
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
              traci.lane.getLastStepVehicleNumber(lane) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    # Penalizing waiting on the on-ramps
    queue_1 = traci.lane.getLastStepHaltingNumber("r1_0")
    queue_2 = traci.lane.getLastStepHaltingNumber("r2_0")

    # Penalizing dangerous on-ramp green lights if it causes emergency breaking
    emergency_brakes = 0
    for veh_id in traci.vehicle.getIDList():
        # Get current deceleration
        decel = traci.vehicle.getDecel(veh_id)
        # Get emergency deceleration threshold
        emergency_decel = traci.vehicle.getEmergencyDecel(veh_id)
        if decel >= emergency_decel:
            emergency_brakes += 1

    speed_reward = avg_speed / 33.33 if avg_speed > 0 else 0.0
    queue_penalty = -0.3 * (queue_1 + queue_2)
    brake_penalty = -0.9 * emergency_brakes

    return speed_reward + queue_penalty + brake_penalty

# Get states for 1st on-ramp traffic light
def get_state_1():
    global loop0, loop1, loop2, current_phase_1

    det0 = "loop0"
    det1 = "loop1"
    det2 = "loop2"

    loop0 = traci.lanearea.getLastStepVehicleNumber(det0)
    loop1 = traci.lanearea.getLastStepVehicleNumber(det1)
    loop2 = traci.lanearea.getLastStepVehicleNumber(det2)

    traffic_light_id_1 = "H2"

    current_phase_1 = traci.trafficlight.getPhase(traffic_light_id_1)

    return (loop0, loop1, loop2, current_phase_1)

# Get states for 2nd on-ramp traffic light
def get_state_2():
    global loop3, loop4, loop5, current_phase_2

    det3 = "loop3"
    det4 = "loop4"
    det5 = "loop5"

    loop3 = traci.lanearea.getLastStepVehicleNumber(det3)
    loop4 = traci.lanearea.getLastStepVehicleNumber(det4)
    loop5 = traci.lanearea.getLastStepVehicleNumber(det5)

    traffic_light_id_2 = "H3"

    current_phase_2 = traci.trafficlight.getPhase(traffic_light_id_2)

    return (loop3, loop4, loop5, current_phase_2)

# Apply action for 1st on-ramp traffic light
def apply_action_1(action, tls_id = "H2"):

    global last_switch_step_1

    if action == 0:
        # Keep current phase
        return
    elif action == 1:
        # Switch phase
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step_1 >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (traci.trafficlight.getPhase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            # Record when the switch happened
            last_switch_step_1 = current_simulation_step

# Apply action for 2nd on-ramp traffic light
def apply_action_2(action, tls_id = "H3"):

    global last_switch_step_2

    if action == 0:
        # Keep current phase
        return
    elif action == 1:
        # Switch phase
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step_2 >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (traci.trafficlight.getPhase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            # Record when the switch happened
            last_switch_step_2 = current_simulation_step

# Update Q values for 1st on-ramp traffic light
def update_Q_table_1(old_state, action, reward, new_state):

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

# Update Q values for 2nd on-ramp traffic light
def update_Q_table_2(old_state, action, reward, new_state):

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

# Get action from DQN's predicted Q-values for 1st on-ramp traffic light
def get_action_from_policy_1(state):

    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model_1.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))

# Get action from DQN's predicted Q-values for 2nd on-ramp traffic light
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


# ------------------------------------------
# Step 5: Online Continuous Learning Loop
# ------------------------------------------

state_size = 4  # (2 highway detector, 1 on-ramp detector, current traffic light phase)
action_size = len(ACTIONS)

# Build DQN model
dqn_model_1 = build_model(state_size, action_size)
dqn_model_2 = build_model(state_size, action_size)

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for run in range(NUM_RUNS):

    # Initialize parameters
    last_switch_step_1 = -MIN_GREEN_STEPS
    last_switch_step_2 = -MIN_GREEN_STEPS
    cumulative_reward_1 = 0.0
    cumulative_reward_2 = 0.0


    # Choose a random config file (predefined set of config files with varying lane density) to train on
    chosen_config_file = get_random_config()

    # Define Sumo configuration with the chosen config file
    Sumo_config = [
        'sumo',
        '-c', chosen_config_file,
        '--step-length', '0.2',
        '--delay', '10',
        '--lateral-resolution', '0'
    ]

    # Open connection between SUMO and Traci
    traci.start(Sumo_config)
    print(f"\nRun {run + 1}/{NUM_RUNS}: Training Started")

    # Run training steps
    for step in range(TOTAL_STEPS):
        current_simulation_step = step

        state_1 = get_state_1()
        state_2 = get_state_2()

        action_1 = get_action_from_policy_1(state_1)
        action_2 = get_action_from_policy_2(state_2)

        apply_action_1(action_1)
        apply_action_2(action_2)

        traci.simulationStep()  # Advance simulation by one step

        new_state_1 = get_state_1()
        new_state_2 = get_state_2()

        reward_1 = get_reward(new_state_1)
        reward_2 = get_reward(new_state_2)

        cumulative_reward_1 += reward_1
        cumulative_reward_2 += reward_2

        update_Q_table_1(state_1, action_1, reward_1, new_state_1)
        update_Q_table_2(state_2, action_2, reward_2, new_state_2)

        if step % 50 == 0:
            print(
                f"Step {step}, Current_State: {state_1}, Action: {action_1}, New_State: {new_state_1}, Reward: {reward_1:.2f}, Cumulative Reward: {cumulative_reward_1:.2f}")
            print(
                f"Step {step}, Current_State: {state_2}, Action: {action_2}, New_State: {new_state_2}, Reward: {reward_2:.2f}, Cumulative Reward: {cumulative_reward_2:.2f}")


    # Export trained model
    os.makedirs("models_3_Zs", exist_ok=True)
    model_path_1 = f"../zsom/models_3_Zs/run_{run + 1}_1"
    model_path_2 = f"../zsom/models_3_Zs/run_{run + 1}_2"
    dqn_model_1.export(model_path_1)
    dqn_model_2.export(model_path_2)
    print(f"Run {run + 1}: Trained model saved to '{model_path_1}' and '{model_path_2}'")

    # Close SUMO connection for this run
    traci.close()
    print(f"Run {run + 1}: Training Completed")


# ------------------------------------------
# Step 6: Print final model summary (replacing Q-table info)
# ------------------------------------------
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model_1.summary()
dqn_model_2.summary()