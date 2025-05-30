from tensorflow.keras.models import load_model
import numpy as np

model_path_1 = r"..\zsom\models_2\final\run_1_5\saved_model.pb"
model_path_2 = r"..\zsom\models_2\final\run_2_5\saved_model.pb"

dqn_model_1 = load_model(model_path_1)
dqn_model_2 = load_model(model_path_2)

def get_state_1():
    det0, det1, det2 = "loop0", "loop1", "loop2"  # Loop detector IDs
    loop0 = traci.lanearea.getLastStepVehicleNumber(det0)
    loop1 = traci.lanearea.getLastStepVehicleNumber(det1)
    loop2 = traci.lanearea.getLastStepVehicleNumber(det2)
    current_phase_1 = traci.trafficlight.getPhase("H2")
    return [loop0, loop1, loop2, current_phase_1]


def get_state_2():
    det3, det4, det5 = "loop3", "loop4", "loop5"
    loop3 = traci.lanearea.getLastStepVehicleNumber(det3)
    loop4 = traci.lanearea.getLastStepVehicleNumber(det4)
    loop5 = traci.lanearea.getLastStepVehicleNumber(det5)
    current_phase_2 = traci.trafficlight.getPhase("H3")
    return [loop3, loop4, loop5, current_phase_2]

def get_state():
    det0, det1, det2, det3, det4, det5 = "loop0", "loop1", "loop2", "loop3", "loop4", "loop5"
    loop0 = traci.lanearea.getLastStepVehicleNumber(det0)
    loop1 = traci.lanearea.getLastStepVehicleNumber(det1)
    loop2 = traci.lanearea.getLastStepVehicleNumber(det2)
    loop3 = traci.lanearea.getLastStepVehicleNumber(det3)
    loop4 = traci.lanearea.getLastStepVehicleNumber(det4)
    loop5 = traci.lanearea.getLastStepVehicleNumber(det5)
    current_phase_1 = traci.trafficlight.getPhase("H2")
    current_phase_2 = traci.trafficlight.getPhase("H3")
    return [loop0, loop1, loop2, loop3, loop4, loop5, current_phase_2, current_phase_2]

def apply_action_1(action):
    if action == 0:
        traci.trafficlight.setRedYellowGreenState("H2", "rGG")  # Example: all red
    elif action == 1:
        traci.trafficlight.setRedYellowGreenState("H2", "GGG")  # Example: green
    # Add more actions as necessary for your logic


def apply_action_2(action):
    if action == 0:
        traci.trafficlight.setRedYellowGreenState("H3", "rGG")
    elif action == 1:
        traci.trafficlight.setRedYellowGreenState("H3", "GGG")


def get_action(state, model):
    state_array = np.array(state, dtype=np.float32).reshape((1, -1))
    Q_values = model.predict(state_array, verbose=0)[0]
    return int(np.argmax(Q_values))  # Choose the action with the highest Q-value


import traci

chosen_config_file = r"..\zsom\sumo_files\onramp5.sumocfg"

sumo_config = [
    'sumo-gui',
    '-c', chosen_config_file,
    '--step-length', '0.2',
    '--lateral-resolution', '0'
]

traci.start(sumo_config)  # Start SUMO

total_steps = 6000  # Example: number of simulation steps
for step in range(total_steps):
    # Get the current state of both agents
    state_1 = get_state()
    state_2 = get_state()

    # Get actions from the trained models
    action_1 = get_action(state_1, dqn_model_1)
    action_2 = get_action(state_2, dqn_model_2)

    # Apply the actions to the traffic lights
    apply_action_1(action_1)
    apply_action_2(action_2)

    # Advance the simulation by one step
    traci.simulationStep()

    # Optional: Log states, actions, rewards, etc., for analysis
    print(f"Step {step}: State_1: {state_1}, Action_1: {action_1}")
    print(f"Step {step}: State_2: {state_2}, Action_2: {action_2}")

# Close the SUMO simulation after completion
traci.close()
