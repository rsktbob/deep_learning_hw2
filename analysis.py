import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from cliff_walking import CliffWalkingEnv

# Use a clean style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})


def moving_average(data, window=10):
    """Compute moving average with given window size."""
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def plot_rewards_comparison(q_rewards_all, sarsa_rewards_all, filename='reward_plot.png'):
    """
    Plot reward curves with mean ± std shading across multiple runs.
    Also includes a smoothed version.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    episodes = np.arange(q_rewards_all.shape[1])

    # --- Left: Raw mean ± std ---
    ax = axes[0]
    q_mean = q_rewards_all.mean(axis=0)
    q_std = q_rewards_all.std(axis=0)
    s_mean = sarsa_rewards_all.mean(axis=0)
    s_std = sarsa_rewards_all.std(axis=0)

    ax.plot(episodes, q_mean, label='Q-Learning', color='#2196F3', linewidth=1.5, alpha=0.9)
    ax.fill_between(episodes, q_mean - q_std, q_mean + q_std, color='#2196F3', alpha=0.15)
    ax.plot(episodes, s_mean, label='SARSA', color='#FF9800', linewidth=1.5, alpha=0.9)
    ax.fill_between(episodes, s_mean - s_std, s_mean + s_std, color='#FF9800', alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward per Episode')
    ax.set_title('Mean Reward ± 1 Std Dev (Raw)')
    ax.legend(loc='lower right')
    ax.set_ylim(-200, 0)
    ax.grid(True, alpha=0.3)

    # --- Right: Smoothed version ---
    ax = axes[1]
    window = 20

    q_smooth_all = np.array([moving_average(q_rewards_all[i], window)
                              for i in range(q_rewards_all.shape[0])])
    s_smooth_all = np.array([moving_average(sarsa_rewards_all[i], window)
                              for i in range(sarsa_rewards_all.shape[0])])

    q_sm_mean = q_smooth_all.mean(axis=0)
    q_sm_std = q_smooth_all.std(axis=0)
    s_sm_mean = s_smooth_all.mean(axis=0)
    s_sm_std = s_smooth_all.std(axis=0)

    sm_episodes = np.arange(len(q_sm_mean))

    ax.plot(sm_episodes, q_sm_mean, label='Q-Learning', color='#2196F3', linewidth=2)
    ax.fill_between(sm_episodes, q_sm_mean - q_sm_std, q_sm_mean + q_sm_std,
                     color='#2196F3', alpha=0.15)
    ax.plot(sm_episodes, s_sm_mean, label='SARSA', color='#FF9800', linewidth=2)
    ax.fill_between(sm_episodes, s_sm_mean - s_sm_std, s_sm_mean + s_sm_std,
                     color='#FF9800', alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward per Episode')
    ax.set_title(f'Mean Reward ± 1 Std Dev (Smoothed, window={window})')
    ax.legend(loc='lower right')
    ax.set_ylim(-200, 0)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Q-Learning vs SARSA: Reward Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Reward comparison plot saved as {filename}")


def get_greedy_path(env, q_table, max_steps=100):
    """Extract the greedy policy path from start to goal."""
    state = env.reset()
    path = [state]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = int(np.argmax(q_table[state]))
        state, _, done = env.step(action)
        path.append(state)
        steps += 1
    return path


def visualize_path(env, path, q_table, title, filename):
    """
    Visualize the learned path on the grid with arrows showing
    the greedy policy at each cell.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Draw the grid
    for r in range(env.height):
        for c in range(env.width):
            state = (r, c)
            if state in env.cliff:
                color = '#D32F2F'  # Red for cliff
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor=color, edgecolor='#333', linewidth=0.5))
                ax.text(c, r, '☠', ha='center', va='center', fontsize=14)
            elif state == env.start_state:
                color = '#4CAF50'
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor=color, edgecolor='#333', linewidth=0.5))
                ax.text(c, r, 'S', ha='center', va='center', fontsize=14,
                        fontweight='bold', color='white')
            elif state == env.goal_state:
                color = '#FFD700'
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor=color, edgecolor='#333', linewidth=0.5))
                ax.text(c, r, 'G', ha='center', va='center', fontsize=14,
                        fontweight='bold', color='#333')
            else:
                color = '#E8F5E9'
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor=color, edgecolor='#999', linewidth=0.5))
                # Draw greedy action arrow
                best_action = int(np.argmax(q_table[state]))
                dx, dy = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}[best_action]
                ax.annotate('', xy=(c + dx, r + dy), xytext=(c, r),
                           arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

    # Draw the path
    path_r = [p[0] for p in path]
    path_c = [p[1] for p in path]
    ax.plot(path_c, path_r, marker='o', color='#1565C0', markersize=7,
            linewidth=2.5, zorder=5, markeredgecolor='white', markeredgewidth=1)

    # Mark start and end of path with larger markers
    ax.plot(path_c[0], path_r[0], marker='s', color='#4CAF50', markersize=12,
            zorder=6, markeredgecolor='white', markeredgewidth=2)
    ax.plot(path_c[-1], path_r[-1], marker='*', color='#FFD700', markersize=16,
            zorder=6, markeredgecolor='#333', markeredgewidth=1)

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_aspect('equal')
    ax.set_title(f'{title} ({len(path) - 1} steps)', fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Start'),
        mpatches.Patch(facecolor='#FFD700', label='Goal'),
        mpatches.Patch(facecolor='#D32F2F', label='Cliff'),
        mpatches.Patch(facecolor='#E8F5E9', label='Normal Cell'),
        plt.Line2D([0], [0], marker='o', color='#1565C0', label='Agent Path',
                   markersize=7, linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Path visualization saved as {filename}")


def plot_q_value_heatmap(env, q_table, title, filename):
    """Plot a heatmap of max Q-values for each state."""
    grid = np.zeros((env.height, env.width))
    for r in range(env.height):
        for c in range(env.width):
            state = (r, c)
            if state in env.cliff:
                grid[r, c] = np.nan  # will be masked
            else:
                grid[r, c] = np.max(q_table[state])

    fig, ax = plt.subplots(figsize=(14, 4))
    masked = np.ma.array(grid, mask=np.isnan(grid))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='#D32F2F')

    im = ax.imshow(masked, cmap=cmap, interpolation='nearest', aspect='equal')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Max Q-value')

    # Annotate cells
    for r in range(env.height):
        for c in range(env.width):
            state = (r, c)
            if state in env.cliff:
                ax.text(c, r, '☠', ha='center', va='center', fontsize=12)
            elif state == env.start_state:
                ax.text(c, r, f'S\n{grid[r, c]:.1f}', ha='center', va='center',
                        fontsize=8, fontweight='bold')
            elif state == env.goal_state:
                ax.text(c, r, f'G\n{grid[r, c]:.1f}', ha='center', va='center',
                        fontsize=8, fontweight='bold')
            else:
                ax.text(c, r, f'{grid[r, c]:.1f}', ha='center', va='center', fontsize=7)

    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Q-value heatmap saved as {filename}")


def print_statistics(q_rewards_all, sarsa_rewards_all):
    """Print detailed statistical comparison."""
    print("\n" + "=" * 70)
    print("Statistical Analysis")
    print("=" * 70)

    n_runs = q_rewards_all.shape[0]
    episodes = q_rewards_all.shape[1]

    # Overall statistics
    print(f"\nNumber of runs: {n_runs}")
    print(f"Episodes per run: {episodes}")

    # Convergence analysis: last 50 episodes
    last_n = 50
    q_final = q_rewards_all[:, -last_n:]
    s_final = sarsa_rewards_all[:, -last_n:]

    q_final_mean_per_run = q_final.mean(axis=1)
    s_final_mean_per_run = s_final.mean(axis=1)

    print(f"\n--- Last {last_n} Episodes Statistics ---")
    print(f"Q-Learning:")
    print(f"  Mean reward   : {q_final_mean_per_run.mean():.2f} ± {q_final_mean_per_run.std():.2f}")
    print(f"  Min/Max       : {q_final_mean_per_run.min():.2f} / {q_final_mean_per_run.max():.2f}")
    print(f"  Avg Std (within run): {q_final.std(axis=1).mean():.2f}")
    print(f"SARSA:")
    print(f"  Mean reward   : {s_final_mean_per_run.mean():.2f} ± {s_final_mean_per_run.std():.2f}")
    print(f"  Min/Max       : {s_final_mean_per_run.min():.2f} / {s_final_mean_per_run.max():.2f}")
    print(f"  Avg Std (within run): {s_final.std(axis=1).mean():.2f}")

    # Stability: coefficient of variation in last 50 episodes
    q_cv = (q_final.std(axis=1) / np.abs(q_final.mean(axis=1))).mean()
    s_cv = (s_final.std(axis=1) / np.abs(s_final.mean(axis=1))).mean()
    print(f"\n--- Stability (Coefficient of Variation, last {last_n} eps) ---")
    print(f"Q-Learning CV: {q_cv:.4f}")
    print(f"SARSA      CV: {s_cv:.4f}")
    more_stable = "SARSA" if s_cv < q_cv else "Q-Learning"
    print(f"→ {more_stable} is more stable")

    # Convergence speed: episode where smoothed reward first exceeds threshold
    threshold = -30  # reasonable threshold for cliff walking
    q_mean = q_rewards_all.mean(axis=0)
    s_mean = sarsa_rewards_all.mean(axis=0)
    window = 20

    def first_above(arr, thresh, win):
        smooth = moving_average(arr, win)
        above = np.where(smooth > thresh)[0]
        return above[0] + win if len(above) > 0 else None

    q_conv = first_above(q_mean, threshold, window)
    s_conv = first_above(s_mean, threshold, window)
    print(f"\n--- Convergence Speed (avg reward first exceeds {threshold}) ---")
    print(f"Q-Learning: episode {q_conv}" if q_conv else "Q-Learning: did not converge")
    print(f"SARSA     : episode {s_conv}" if s_conv else "SARSA     : did not converge")

    return {
        "q_final_mean": q_final_mean_per_run.mean(),
        "q_final_std": q_final_mean_per_run.std(),
        "s_final_mean": s_final_mean_per_run.mean(),
        "s_final_std": s_final_mean_per_run.std(),
        "q_cv": q_cv,
        "s_cv": s_cv,
        "q_convergence_ep": q_conv,
        "s_convergence_ep": s_conv,
    }


if __name__ == "__main__":
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

    q_rewards_all = results["q_rewards_all"]
    sarsa_rewards_all = results["sarsa_rewards_all"]
    q_table = results["q_table"]
    sarsa_table = results["sarsa_table"]

    # 1. Plot reward curves
    plot_rewards_comparison(q_rewards_all, sarsa_rewards_all, 'reward_plot.png')

    # 2. Path visualization
    env = CliffWalkingEnv()

    q_path = get_greedy_path(env, q_table)
    env2 = CliffWalkingEnv()
    visualize_path(env2, q_path, q_table, "Q-Learning Greedy Path", "q_learning_path.png")

    env3 = CliffWalkingEnv()
    sarsa_path = get_greedy_path(env3, sarsa_table)
    env4 = CliffWalkingEnv()
    visualize_path(env4, sarsa_path, sarsa_table, "SARSA Greedy Path", "sarsa_path.png")

    # 3. Q-value heatmaps
    env5 = CliffWalkingEnv()
    plot_q_value_heatmap(env5, q_table, "Q-Learning: Max Q-values", "q_value_heatmap_qlearning.png")
    plot_q_value_heatmap(env5, sarsa_table, "SARSA: Max Q-values", "q_value_heatmap_sarsa.png")

    # 4. Statistical analysis
    stats = print_statistics(q_rewards_all, sarsa_rewards_all)

    # 5. Print path info
    print(f"\nQ-Learning Greedy Path: {len(q_path) - 1} steps")
    print(f"  Path: {' → '.join([str(s) for s in q_path])}")
    print(f"SARSA Greedy Path: {len(sarsa_path) - 1} steps")
    print(f"  Path: {' → '.join([str(s) for s in sarsa_path])}")
