import numpy as np
from envs import tetris


# State definition:
'''
Using Pierre Dellacherie's features
These are six hand-crafted features.

1. Landing height
    - The height where the next piece is going to land for a given action

2. Piece contribution
    - The number of rows eliminated because of piece * height contribution
      of the piece

3. Row transitions
    - The total number of row transitions. A row transition occurs when an
      empty cell is adjacent to a filled cell on the same row and vice versa.

4. Column transitions
    - The total number of column transitions. A column transition occurs 
      when an empty cell is adjacent to a filled cell on the same column and
      vice versa

5. Number of holes
    - A hole is an empty cell that has at least one filled cell above it in
      the same column.

6. Well sums
    - A well is a succession of empty cells such that their left cells and
      right cells are both filled.

'''

class state_features:

    def __init__(self):
        self.env = tetris.TetrisEnv()
        self.n_rows = self.env.n_rows
        self.n_cols = self.env.n_cols


    def landing_height(self, prev_env, action):
        _, slot = action
        landing_ht = prev_env.state.top[slot]
        return landing_ht


    def piece_con(self, prev_env, piece, action, env):
        orient, slot = action
        ht_before = prev_env.state.top[slot]
        ht_after = env.state.top[slot]
        piece_ht = env.piece_height[piece][orient]

        ht_con = ht_before + piece_ht - ht_after # height contribution
        # as per the Dellacherie definition, we need to multiply with
        # the number of lines cleared
        piece_con = env.cleared_current_turn * ht_con

        return piece_con


    def row_transitions(self, state):

        end_row = np.max(state.top)
        transitions = 0

        for i in range(end_row):
            diff = np.diff(state.field[i])
            transitions+= np.count_nonzero(diff)
            
            # considering the left and right walls as filled
            if state.field[i, 0] == 0:
                transitions+=1
            if state.field[i, self.n_cols-1] == 0:
                    transitions+=1
        
        return transitions


    def col_transitions(self, state):

        transitions = 0
        for i in range(self.n_cols):
            diff = np.diff(state.field[:, i])
            transitions+= np.count_nonzero(diff)

            # considering the top and bottom walls as filled
            # if state.field[0, i] == 0:
            #     transitions+=1
            # if state.field[self.n_rows-1, i] == 0:
            #     transitions+=1
        
        return transitions


    def num_holes(self, state):
        
        holes = 0
        for i in range(self.n_cols):
            for j in range(state.top[i]-1, -1, -1):
                if state.field[j, i]==0:
                    holes+=1
        return holes
    

    def well_sum(self, state):

        sum = 0
        for i in range(self.n_cols):
            if i-1<0: # left wall 
                a = state.top[i+1]
            else: a = state.top[i-1]

            if i+1>self.n_cols-1: # right wall
                b = state.top[i-1]
            else: b = state.top[i+1]

            if state.top[i] < a and state.top[i] < b:
                depth = min(a, b) - state.top[i]
                depth_sum = depth*(depth+1)/2 # arithmetic sum: n*(n+1)/2
                sum+=depth_sum
        
        return sum


    def features(self, prev_env, piece, action, env):
        
        '''
        All the above features return integer.
        So, we are trying to get the entire state description using these
        hand-crafted features.
        All the return values from above are placed in a numpy array
        '''
        state_input = np.array([self.landing_height(prev_env, action),
                                self.piece_con(prev_env, piece, action, env),
                                self.row_transitions(env.state),
                                self.col_transitions(env.state),
                                self.num_holes(env.state),
                                self.well_sum(env.state)])
        return state_input
