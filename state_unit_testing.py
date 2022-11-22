from envs import tetris
import state_def
import state_features
from train_utils import find_best_action
import numpy as np

env2 = tetris.TetrisEnv()
env2.reset()
piece = env2.state.next_piece

env2.render()
action = env2.legal_moves[piece][0]
print(action)

# setup environment
env_after = tetris.TetrisEnv()
env_after.set_state(env2.state)

env_after.step(action)
env_after.render()

features = state_def.state_features()
landing_height = features.landing_height(env2, action)
piece_con = features.piece_con(env2, piece, action, env_after)
r_t = features.row_transitions(env_after.state)
c_t = features.col_transitions(env_after.state)
holes = features.num_holes(env_after.state)
well_sum = features.well_sum(env_after.state)

features_2 = state_features.DellacherieFeatureMap()
landing_height2 = features_2.f1(env2, action)
piece_con2 = features_2.f2(env2, env_after, piece, action)
r_t2 = features_2.f3(env_after.state)
c_t2 = features_2.f4(env_after.state)
holes2 = features_2.f5(env_after.state)
well_sum2 = features_2.f6(env_after.state)


print("landing height: ", landing_height)
print("Piece contribution: ", piece_con)
print("Row transitions: ", r_t)
print("Column transitions: ", c_t)
print("Holes: ", holes)
print("Well sum: ", well_sum)

print("landing height: ", landing_height2)
print("Piece contribution: ", piece_con2)
print("Row transitions: ", r_t2)
print("Column transitions: ", c_t2)
print("Holes: ", holes2)
print("Well sum: ", well_sum2)
