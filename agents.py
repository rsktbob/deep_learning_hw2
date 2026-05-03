import numpy as np
import random


class RLAgent:
    """Base class for RL agents with ε-greedy action selection."""

    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Args:
            states: list of all possible states
            actions: list of action indices (e.g. [0,1,2,3])
            alpha: learning rate
            gamma: discount factor
            epsilon: exploration rate for ε-greedy
        """
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table with zeros
        self.q_table = {}
        for s in states:
            self.q_table[s] = np.zeros(len(actions))

    def choose_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # If multiple actions have the same max Q-value, pick one randomly
            max_q = np.max(self.q_table[state])
            actions_with_max_q = [a for a, q in enumerate(self.q_table[state]) if q == max_q]
            return random.choice(actions_with_max_q)

    def get_greedy_action(self, state):
        """Pure greedy action selection (no exploration)."""
        return int(np.argmax(self.q_table[state]))


class QLearningAgent(RLAgent):
    """
    Q-Learning Agent (Off-policy TD Control)

    Update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

    Uses the maximum Q-value of the next state for updates,
    regardless of the action actually taken.
    """

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])


class SARSAAgent(RLAgent):
    """
    SARSA Agent (On-policy TD Control)

    Update rule:
        Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]

    Uses the Q-value of the actual next state-action pair for updates,
    reflecting the behavior policy including exploration.
    """

    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]

        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])
