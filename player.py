from envs import tetris
import numpy as np
from train_utils import find_best_action

# Load the agent
with open('elite_weights_best.npy', 'rb') as f:
    weights = np.load(f)

elite_player = weights[0]

episodes = 20
env = tetris.TetrisEnv()
assert episodes!=0, "Episodes should not be zero."
total_lines_cleared = 0
max_lines_cleared = 0

for episode in range(episodes):
    env.reset()
    is_terminal = False
    print("-------- Episode " + str(episode) + " --------")
    status = 0

    while is_terminal!=True: # until game ends
        best_action = find_best_action(elite_player, env)
        _, rew, is_terminal, _ = env.step(best_action)

        if (status%2000==0):
            print("Game ongoing...")
        status+=1
    
    lines_per_episode = env.state.cleared
    print("Lines cleared in episode " + str(episode) \
           + ": " + str(lines_per_episode) + "\n")

    total_lines_cleared += lines_per_episode
    max_lines_cleared = max(lines_per_episode, max_lines_cleared)
    

print("Average lines cleared: ", (int)(total_lines_cleared/episodes))
print("Maximum lines cleared among 20 episodes: ", max_lines_cleared)    