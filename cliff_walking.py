import numpy as np


class CliffWalkingEnv:
    """
    Cliff Walking Environment (4 x 12 Gridworld)

    Layout:
        Row 0: [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
        Row 1: [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
        Row 2: [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
        Row 3: [S][C][C][C][C][C][C][C][C][C][C][G]

    S = Start (3,0), G = Goal (3,11), C = Cliff
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Rewards: -1 per step, -100 for cliff (reset to start), episode ends at goal.
    """

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
    ACTION_SYMBOLS = ['↑', '↓', '←', '→']

    def __init__(self, height=4, width=12):
        self.height = height
        self.width = width
        self.start_state = (height - 1, 0)
        self.goal_state = (height - 1, width - 1)
        self.cliff = set((height - 1, i) for i in range(1, width - 1))
        self.n_states = height * width
        self.n_actions = 4
        self.reset()

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right

        Returns:
            next_state, reward, done
        """
        r, c = self.current_state

        if action == self.ACTION_UP:
            r = max(0, r - 1)
        elif action == self.ACTION_DOWN:
            r = min(self.height - 1, r + 1)
        elif action == self.ACTION_LEFT:
            c = max(0, c - 1)
        elif action == self.ACTION_RIGHT:
            c = min(self.width - 1, c + 1)

        new_state = (r, c)

        if new_state in self.cliff:
            reward = -100
            done = False
            self.current_state = self.start_state
        elif new_state == self.goal_state:
            reward = -1
            done = True
            self.current_state = new_state
        else:
            reward = -1
            done = False
            self.current_state = new_state

        return self.current_state, reward, done

    def get_all_states(self):
        states = []
        for r in range(self.height):
            for c in range(self.width):
                states.append((r, c))
        return states

    def is_cliff(self, state):
        return state in self.cliff

    def is_goal(self, state):
        return state == self.goal_state

    def is_start(self, state):
        return state == self.start_state
