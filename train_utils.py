import numpy as np
from envs import tetris
from state_def import state_features


def find_reward(weights, episodes):
    """
    Finds the average reward of a player given weights
        and number of episodes
        i.e., reward gained on each episode on avg. 
    Each player has six weights (because we defined six
        features)
    """
    
    env = tetris.TetrisEnv()
    cum_reward = 0.0
    assert episodes!=0, "Episodes should not be zero."

    for _ in range(episodes):
        env.reset()
        is_terminal = False
        while not is_terminal: # until game ends - is_terminal!=True
            best_action = find_best_action(weights, env)
            _, rew, is_terminal, _ = env.step(best_action)
            cum_reward += rew

    avg_reward = cum_reward/episodes
    return avg_reward


def find_best_action(weights, env):
    """
    Find the best action for the current piece and field
    """
    
    env_local = tetris.TetrisEnv()
    feature_fn = state_features()

    piece = env.state.next_piece
    max_val = float('-inf')
    best_action = None


    for action in env.legal_moves[piece]:
        env_local.set_state(env.state)
        env_local.step(action)

        feature_vector = feature_fn.features(env, piece, action, env_local)
        cur_val = np.dot(weights, feature_vector)
        if cur_val > max_val:
            best_action = action
        max_val = max(max_val, cur_val)
    
    return best_action



