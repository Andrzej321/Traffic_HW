import traci
import numpy as np

# Path to your SUMO configuration file
test_config_path = r"../sumo_files/onramp_test.sumocfg"


sumo_config = [
    "sumo-gui",  # Switch to "sumo-gui" to visually debug the simulation
    '-c', test_config_path,
    '--step-length', '1.0'
]

# Path to your trained RL agent models
model_1_path = r"..\zsom\models_2\final\run_1_5"
model_2_path = r"..\zsom\models_2\final\run_2_5"

# Load the trained models using TensorFlow
import tensorflow as tf
dqn_model_1 = tf.keras.layers.TFSMLayer(model_1_path, call_endpoint='serving_default')
dqn_model_2 = tf.keras.layers.TFSMLayer(model_2_path, call_endpoint='serving_default')



# Function to get the agent's state
def get_state(intersection_id, detectors):
    state = [traci.lanearea.getLastStepVehicleNumber(det) for det in detectors]
    current_phase = traci.trafficlight.getPhase(intersection_id)
    state.append(current_phase)
    return state

# Function to apply actions to traffic lights
def apply_action(intersection_id, action):
    if action == 0:
        traci.trafficlight.setRedYellowGreenState(intersection_id, "rGG")  # Example action
    elif action == 1:
        traci.trafficlight.setRedYellowGreenState(intersection_id, "GGG")  # Example action

# Function to predict the action using the RL model
def get_action(state, model):
    state_array = np.array(state, dtype=np.float32).reshape((1, -1))
    Q_values = model.predict(state_array, verbose=0)[0]
    return int(np.argmax(Q_values))

# Metrics for evaluation
total_emissions = 0  # Track total CO2 emissions
total_waiting_time = 0  # Sum of the waiting times of all vehicles
total_emergency_brakes = 0  # Count emergency stops
sum_speeds = 0  # Accumulate vehicle speeds
num_vehicles = 0  # Count total vehicles

# Intersections and detectors
intersection_1 = "H2"
intersection_2 = "H3"

detectors_1 = ["loop0", "loop1", "loop2"]  # Loop detectors near intersection_1
detectors_2 = ["loop3", "loop4", "loop5"]  # Loop detectors near intersection_2

# Start the simulation
traci.start(sumo_config)

try:
    total_steps = 6000  # Adjust as needed
    for step in range(total_steps):
        # Get states of two intersections
        state_1 = get_state(intersection_1, detectors_1)
        state_2 = get_state(intersection_2, detectors_2)

        # Get actions from the RL models
        action_1 = get_action(state_1, dqn_model_1)
        action_2 = get_action(state_2, dqn_model_2)

        # Apply actions
        apply_action(intersection_1, action_1)
        apply_action(intersection_2, action_2)

        # Advance simulation
        traci.simulationStep()

        # Update performance metrics
        total_emissions += traci.simulation.getCO2Emission()  # Grabs total CO2 emitted
        total_waiting_time += traci.simulation.getWaitingTime()  # Total waiting time
        total_emergency_brakes += sum([traci.vehicle.getEmergencyStop(i) for i in traci.vehicle.getIDList()])

        vehicle_speeds = [traci.vehicle.getSpeed(vehicle_id) for vehicle_id in traci.vehicle.getIDList()]
        sum_speeds += sum(vehicle_speeds)  # Add the speeds of all vehicles at this step
        num_vehicles += len(vehicle_speeds)  # Total vehicles encountered

    # Calculate average speed
    avg_speed = sum_speeds / num_vehicles if num_vehicles > 0 else 0

    # Display metrics at the end of the simulation
    print("Simulation Evaluation Results:")
    print(f"- Total CO2 Emissions: {total_emissions:.2f} g")
    print(f"- Average Speed: {avg_speed:.2f} m/s")
    print(f"- Total Waiting Time: {total_waiting_time:.2f} s")
    print(f"- Total Emergency Brakes: {total_emergency_brakes}")

finally:
    # Close the SUMO simulation
    traci.close()
