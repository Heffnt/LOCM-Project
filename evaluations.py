from gym_locm.envs import LOCMBattleSingleEnv
from gym_locm.agents import RandomBattleAgent, RandomDraftAgent, RLBattleAgent
from sb3_contrib import MaskablePPO
import numpy as np
from collections import deque

def evaluate_model(model_path, n_games=100, print_every=10):
    # Load the trained model
    model = MaskablePPO.load(model_path)
    
    # Create the RL agent
    rl_agent = RLBattleAgent(model, deterministic=True)
    
    # Create environment
    env = LOCMBattleSingleEnv(
        deck_building_agents=(RandomDraftAgent(), RandomDraftAgent()),
        battle_agent=RandomBattleAgent(),
        return_action_mask=True,
        seed=42,
        items=True,
        version="1.5",
        reward_functions=("win-loss",)
    )
    
    # Track results
    wins = 0
    total_rewards = []
    recent_wins = deque(maxlen=print_every)
    
    for game in range(n_games):
        done = False
        obs = env.reset()
        episode_reward = 0
        
        while not done:
            action_masks = env.action_masks()
            action = rl_agent.act(obs, action_masks)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        # Track wins
        game_won = int(info["winner"] == 0)
        wins += game_won
        recent_wins.append(game_won)
        total_rewards.append(episode_reward)
        
        if (game + 1) % print_every == 0:  # Print every 10 games
            cumulative_wr = 100 * wins/(game + 1)
            recent_wr = 100 * sum(recent_wins)/len(recent_wins)
            print(f"Game {game+1}/{n_games}: Cumulative: {cumulative_wr:.2f}% Recent: {recent_wr:.2f}%")
    
    # Print final statistics
    win_rate = 100 * wins/n_games
    avg_reward = np.mean(total_rewards)
    print(f"\nFinal Results:")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    
    return win_rate, avg_reward

if __name__ == "__main__":
    evaluate_model("battle_agent_v2.zip", n_games=200, print_every=20)
