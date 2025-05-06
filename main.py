import os
import sys
import traci
from stable_baselines3 import DQN
from classes_test import TrafficLightEnvTest

# Make sure SUMO_HOME environment variable is set correctly
if 'SUMO_HOME' not in os.environ:
    # This should point to the actual SUMO installation directory, not the config files
    raise EnvironmentError("Please set SUMO_HOME environment variable")

# Add SUMO tools to Python path
tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools_path)


def run_simulation():
    # Configuration
    sumoBinary = "sumo"  # Use "sumo" for headless mode
    cfg_file = 'sumo_files/onramp.sumocfg'

    # Check if the config file exists
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    # Start SUMO with the configuration
    sumoCmd = [sumoBinary,
               "-c", cfg_file,
               "--tripinfo-output", "tripinfo.xml",  # Output file for trip information
               "--summary-output", "summary.xml",  # Summary output
               "--queue-output", "queue.xml",  # Queue information
               "--statistic-output", "statistics.xml"  # Statistical output
               ]

    try:
        traci.start(sumoCmd)

        # Simulation loop
        step = 0
        while step < 1000:  # Simulate for 1000 steps
            traci.simulationStep()

            # You can add your simulation logic here
            # For example, get traffic light states, vehicle counts, etc.
            if step % 100 == 0:  # Print status every 100 steps
                print(f"Simulation step: {step}")

            step += 1

    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        traci.close()
        print("Simulation finished")


def main():
    # First run a basic simulation to verify everything works
    run_simulation()

    """
    # Your RL training code can go here
    env = TrafficLightEnvTest(cfg_file)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        verbose=1
    )

    # Rest of your training code...
    """


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