import os
import sys
import traci
#comment
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\SUMO"  # Update this path

# Make sure SUMO_HOME environment variable is set correctly
if 'SUMO_HOME' not in os.environ:
    # This should point to the actual SUMO installation directory, not the config files
    raise EnvironmentError("Please set SUMO_HOME environment variable")

# Add SUMO tools to Python path
tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools_path)
cfg_file = "../sumo_files/onramp.sumocfg"

def run_simulation(sumocfg):

    # Set the SUMO command to start SUMO with GUI and dump emissions to the results folder
    sumocmd = ["sumo", "-c", sumocfg]
    # Start SUMO with the command
    traci.start(sumocmd)

    simulation_time = 0.00
    for i in range(1000):
        traci.simulationStep()
        simulation_time = i * 1


    # Close TraCI
    traci.close()

    print("Simulation ended.")
    return


def main():

    if cfg_file:
        if os.path.exists(cfg_file):
            run_simulation(cfg_file)
            return
        else:
            run_simulation(cfg_file)
            print("The given SUMO configuration file does not exist!")
            return
    else:
        print("Please provide a SUMO configuration file for the simulation.")
        return


if __name__ == "__main__":
    main()

    """
    env = TrafficLightEnvTest(cfg_file)

    # Create DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        verbose=1
    )

    # Training loop
    EPISODES = 100
    STEPS_PER_EPISODE = 1000

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(STEPS_PER_EPISODE):
            # Get action from model
            action = model.predict(state, deterministic=False)[0]

            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)

            # Store transition in replay buffer
            model.replay_buffer.add(
                state, next_state, action, reward, done, truncated, info
            )

            # Update current state and episode reward
            state = next_state
            episode_reward += reward

            # Train the model
            model.train()

            if done:
                break

        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {episode_reward:.2f}")

        # Save model every 10 episodes
        if (episode + 1) % 10 == 0:
            model.save(f"traffic_light_model_episode_{episode + 1}")

    env.close()
    """
"""
# Function to evaluate the trained agent
def evaluate_agent(model_path, num_episodes=5):
    env = TrafficLightEnvTest("simple.sumocfg")
    model = DQN.load(model_path)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = model.predict(state, deterministic=True)[0]
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Print metrics
            print(f"Step metrics: {info['metrics']}")

        print(f"Evaluation Episode {episode + 1}, Total Reward: {episode_reward:.2f}")

    env.close()


# Example usage with visualization
def run_visualization(model_path):
    # Use SUMO-GUI instead of SUMO for visualization
    env = TrafficLightEnvTest("simple.sumocfg", gui=True)
    model = DQN.load(model_path)

    state, _ = env.reset()
    done = False

    while not done:
        action = model.predict(state, deterministic=True)[0]
        state, reward, done, truncated, info = env.step(action)

        # Sleep to slow down visualization
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    # Training
    main()

    # Evaluation
    evaluate_agent("traffic_light_model_episode_100")

    # Visualization
    run_visualization("traffic_light_model_episode_100")
"""