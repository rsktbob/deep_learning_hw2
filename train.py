import numpy as np
import pickle
from cliff_walking import CliffWalkingEnv
from agents import QLearningAgent, SARSAAgent


def train_q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Train a Q-Learning agent on the given environment.

    Returns:
        agent: trained QLearningAgent
        rewards: list of total rewards per episode
    """
    agent = QLearningAgent(env.get_all_states(), [0, 1, 2, 3], alpha, gamma, epsilon)
    rewards = []

    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        if (i + 1) % 100 == 0:
            print(f"Q-Learning Episode {i + 1}/{episodes}, Reward: {total_reward}")

    return agent, rewards


def train_sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Train a SARSA agent on the given environment.

    Returns:
        agent: trained SARSAAgent
        rewards: list of total rewards per episode
    """
    agent = SARSAAgent(env.get_all_states(), [0, 1, 2, 3], alpha, gamma, epsilon)
    rewards = []

    for i in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            total_reward += reward
        rewards.append(total_reward)
        if (i + 1) % 100 == 0:
            print(f"SARSA Episode {i + 1}/{episodes}, Reward: {total_reward}")

    return agent, rewards


def run_multiple_experiments(n_runs=30, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Run multiple independent experiments for statistical significance.

    Returns:
        dict with q_rewards_all, sarsa_rewards_all (n_runs x episodes arrays),
        and best Q-tables from the last run.
    """
    q_rewards_all = np.zeros((n_runs, episodes))
    sarsa_rewards_all = np.zeros((n_runs, episodes))
    q_table_best = None
    sarsa_table_best = None

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        env_q = CliffWalkingEnv()
        q_agent, q_rewards = train_q_learning(env_q, episodes, alpha, gamma, epsilon)
        q_rewards_all[run] = q_rewards

        env_s = CliffWalkingEnv()
        sarsa_agent, sarsa_rewards = train_sarsa(env_s, episodes, alpha, gamma, epsilon)
        sarsa_rewards_all[run] = sarsa_rewards

        # Keep the last run's Q-tables for path visualization
        q_table_best = q_agent.q_table
        sarsa_table_best = sarsa_agent.q_table

    return {
        "q_rewards_all": q_rewards_all,
        "sarsa_rewards_all": sarsa_rewards_all,
        "q_table": q_table_best,
        "sarsa_table": sarsa_table_best,
        "n_runs": n_runs,
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
    }


if __name__ == "__main__":
    print("Running multiple experiments for statistical analysis...")
    results = run_multiple_experiments(n_runs=30, episodes=500)

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nResults saved to results.pkl")

    # Print summary statistics
    q_final = results["q_rewards_all"][:, -50:]  # last 50 episodes
    s_final = results["sarsa_rewards_all"][:, -50:]

    print(f"\n--- Summary (last 50 episodes, averaged over {results['n_runs']} runs) ---")
    print(f"Q-Learning  : mean={q_final.mean():.2f}, std={q_final.std():.2f}")
    print(f"SARSA       : mean={s_final.mean():.2f}, std={s_final.std():.2f}")
