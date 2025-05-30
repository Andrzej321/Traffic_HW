import os
import sys
import numpy as np
import traci
from tensorflow import keras
import tensorflow as tf

# Set up SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Define paths
CONFIG_FILE = "../sumo_files/onramp_test.sumocfg"  # Unseen scenario config
MODEL_DIR = "../zsom/models_3_Zs"  # Directory with trained models
RUN_NUMBER = 1  # Choose which run's models to load (1 to NUM_RUNS)

# Variables for state
loop0 = loop1 = loop2 = loop3 = loop4 = loop5 = 0
current_phase_1 = current_phase_2 = 0

# Parameters
MIN_GREEN_STEPS = 10  # Minimum green time
TOTAL_STEPS = 1000  # Simulation steps for testing
ACTIONS = [0, 1]  # 0: keep phase, 1: switch phase
last_switch_step_1 = -MIN_GREEN_STEPS
last_switch_step_2 = -MIN_GREEN_STEPS

# Lists to store metrics
emergency_brakes_history = []
co2_emission_history = []
waiting_time_history = []
avg_speed_history = []
reward_history = []
queue_1_history = []
queue_2_history = []

# Functions from training code
def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

def get_reward(state):
    all_lanes = ["h1_0", "h1_1", "h2_0", "h2_1", "r1_0", "r2_0"]
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
              traci.lane.getLastStepVehicleNumber(lane) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    queue_1 = traci.lane.getLastStepHaltingNumber("r1_0")
    queue_2 = traci.lane.getLastStepHaltingNumber("r2_0")
    emergency_brakes = 0
    for veh_id in traci.vehicle.getIDList():
        decel = traci.vehicle.getDecel(veh_id)
        emergency_decel = traci.vehicle.getEmergencyDecel(veh_id)
        if decel >= emergency_decel:
            emergency_brakes += 1
    speed_reward = avg_speed / 33.33 if avg_speed > 0 else 0.0
    queue_penalty = -0.3 * (queue_1 + queue_2)
    brake_penalty = -0.9 * emergency_brakes
    return speed_reward + queue_penalty + brake_penalty

def get_state_1():
    global loop0, loop1, loop2, current_phase_1
    det0, det1, det2 = "loop0", "loop1", "loop2"
    loop0 = traci.lanearea.getLastStepVehicleNumber(det0)
    loop1 = traci.lanearea.getLastStepVehicleNumber(det1)
    loop2 = traci.lanearea.getLastStepVehicleNumber(det2)
    traffic_light_id_1 = "H2"
    current_phase_1 = traci.trafficlight.getPhase(traffic_light_id_1)
    return (loop0, loop1, loop2, current_phase_1)

def get_state_2():
    global loop3, loop4, loop5, current_phase_2
    det3, det4, det5 = "loop3", "loop4", "loop5"
    loop3 = traci.lanearea.getLastStepVehicleNumber(det3)
    loop4 = traci.lanearea.getLastStepVehicleNumber(det4)
    loop5 = traci.lanearea.getLastStepVehicleNumber(det5)
    traffic_light_id_2 = "H3"
    current_phase_2 = traci.trafficlight.getPhase(traffic_light_id_2)
    return (loop3, loop4, loop5, current_phase_2)

def apply_action_1(action, tls_id="H2"):
    global last_switch_step_1
    if action == 0:
        return
    elif action == 1:
        if current_simulation_step - last_switch_step_1 >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (traci.trafficlight.getPhase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step_1 = current_simulation_step

def apply_action_2(action, tls_id="H3"):
    global last_switch_step_2
    if action == 0:
        return
    elif action == 1:
        if current_simulation_step - last_switch_step_2 >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (traci.trafficlight.getPhase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step_2 = current_simulation_step

def get_action_from_policy_1(state, model):
    state_array = to_array(state)
    Q_values = dqn_model_1.predict(state_array, verbose=0)[0]
    return int(np.argmax(Q_values))

def get_action_from_policy_2(state, model):
    state_array = to_array(state)
    Q_values = dqn_model_1.predict(state_array, verbose=0)[0]
    return int(np.argmax(Q_values))

def collect_metrics():
    # Emergency brakes
    emergency_brakes = 0
    for veh_id in traci.vehicle.getIDList():
        decel = traci.vehicle.getDecel(veh_id)
        emergency_decel = traci.vehicle.getEmergencyDecel(veh_id)
        if decel >= emergency_decel:
            emergency_brakes += 1

    # CO2 emission (mg)
    co2_emission = sum(traci.vehicle.getCO2Emission(veh_id) for veh_id in traci.vehicle.getIDList())

    # Waiting time (seconds)
    waiting_time = sum(traci.vehicle.getAccumulatedWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList())

    # Average speed (m/s)
    all_lanes = ["h1_0", "h1_1", "h2_0", "h2_1", "r1_0", "r2_0"]
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in all_lanes if
              traci.lane.getLastStepVehicleNumber(lane) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    return emergency_brakes, co2_emission, waiting_time, avg_speed


# Load trained models
model_path_1 = os.path.join(MODEL_DIR, f"run_{RUN_NUMBER}_1")
model_path_2 = os.path.join(MODEL_DIR, f"run_{RUN_NUMBER}_2")

try:
    dqn_model_1 = tf.keras.models.load_model('Model')
    dqn_model_2 = tf.keras.models.load_model('Model')
    print(f"Loaded models from {model_path_1} and {model_path_2}")
except Exception as e:
    sys.exit(f"Error loading models: {e}")

# SUMO configuration for testing
sumo_cmd = [
    "sumo-gui",
    "-c", CONFIG_FILE,
    "--step-length", "0.2",
    "--delay", "10",
    "--lateral-resolution", "0",
]

# Initialize simulation
print("\n=== Starting SUMO-GUI Testing with Trained DQN Models ===")
traci.start(sumo_cmd)
cumulative_reward = 0.0

# Run simulation
for step in range(TOTAL_STEPS):

    current_simulation_step = step

    # Get states
    state_1 = get_state_1()
    state_2 = get_state_2()

    # Get actions from trained models (no exploration, pure exploitation)
    action_1 = get_action_from_policy_1(state_1, dqn_model_1)
    action_2 = get_action_from_policy_2(state_2, dqn_model_2)

    # Apply actions
    apply_action_1(action_1)
    apply_action_2(action_2)

    # Advance simulation
    traci.simulationStep()

    # Compute reward
    reward = get_reward(state_1)  # Same reward for both agents
    emergency_brakes, co2_emission, waiting_time, avg_speed = collect_metrics()
    queue_1 = traci.lane.getLastStepHaltingNumber("r1_0")
    queue_2 = traci.lane.getLastStepHaltingNumber("r2_0")
    cumulative_reward += reward

    # Store metrics
    emergency_brakes_history.append(emergency_brakes)
    co2_emission_history.append(co2_emission)
    waiting_time_history.append(waiting_time)
    avg_speed_history.append(avg_speed)
    reward_history.append(reward)
    queue_1_history.append(queue_1)
    queue_2_history.append(queue_2)

    # Log performance every step
    if step % 1 == 0:

        print(f"Step {step}:")
        print(f"  TLS H2 - State: {state_1}, Action: {action_1}, Queue: {queue_1}, Reward: {reward:.2f}")
        print(f"  TLS H3 - State: {state_2}, Action: {action_2}, Queue: {queue_2}, Reward: {reward:.2f}")
        print(f"  Cumulative Reward: {cumulative_reward:.2f}")

# Close simulation
traci.close()

total_emergency_brakes = sum(emergency_brakes_history)
total_co2_emission = sum(co2_emission_history)
total_waiting_time = waiting_time_history[-1] if waiting_time_history else 0  # Last cumulative value
avg_speed_overall = np.mean(avg_speed_history) if avg_speed_history else 0.0

print("\n=== Testing Completed ===")
print(f"Final Cumulative Reward: {cumulative_reward:.2f}")
print("Performance Metrics Summary:")
print(f"  Total Emergency Brakes: {total_emergency_brakes}")
print(f"  Total CO2 Emission: {total_co2_emission:.2f} mg")
print(f"  Total Waiting Time: {total_waiting_time:.2f} seconds")
print(f"  Average Speed (Overall): {avg_speed_overall:.2f} m/s")

# Plot metrics
plt.figure(figsize=(12, 10))

# Plot 1: Emergency Brakes
plt.subplot(2, 2, 1)
plt.plot(emergency_brakes_history, label='Emergency Brakes')
plt.xlabel('Step')
plt.ylabel('Number of Emergency Brakes')
plt.title('Emergency Brakes Over Time')
plt.legend()
plt.grid(True)

# Plot 2: CO2 Emission
plt.subplot(2, 2, 2)
plt.plot(co2_emission_history, label='CO2 Emission', color='green')
plt.xlabel('Step')
plt.ylabel('CO2 Emission (mg)')
plt.title('CO2 Emission Over Time')
plt.legend()
plt.grid(True)

# Plot 3: Waiting Time
plt.subplot(2, 2, 3)
plt.plot(waiting_time_history, label='Waiting Time', color='red')
plt.xlabel('Step')
plt.ylabel('Waiting Time (seconds)')
plt.title('Waiting Time Over Time')
plt.legend()
plt.grid(True)

# Plot 4: Average Speed
plt.subplot(2, 2, 4)
plt.plot(avg_speed_history, label='Average Speed', color='blue')
plt.xlabel('Step')
plt.ylabel('Average Speed (m/s)')
plt.title('Average Speed Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('performance_metrics.png')
print("\nPerformance metrics plot saved as 'performance_metrics.png'")

# Additional plot for reward and queues
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label='Reward', color='purple')
plt.plot(queue_1_history, label='Queue r1_0', color='orange')
plt.plot(queue_2_history, label='Queue r2_0', color='brown')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Reward and Queue Lengths Over Time')
plt.legend()
plt.grid(True)
plt.savefig('reward_and_queues.png')
print("Reward and queue plot saved as 'reward_and_queues.png'")