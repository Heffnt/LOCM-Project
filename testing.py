import gym
import gym_locm
import warnings

def play_games(player1, player1_name, player2, player2_name, n_games=10, env=None, verbose=True):
    """
    Play multiple games between two agents and return their win statistics.
    
    Args:
        player1: First player agent object
        player1_name: Name of the first player
        player2: Second player agent object
        player2_name: Name of the second player
        n_games: Number of games to play (default: 10)
        env: Optional pre-configured environment (if None, creates new one)
        verbose: Whether to print game results (default: True)
        
    Returns:
        tuple: (player1_wins, player2_wins)
    """
    # Create the battle environment if not provided
    if env is None:
        env = gym.make('LOCM-battle-2p-v0', version="1.5")
    
    # Track wins
    p1_wins = 0
    p2_wins = 0
    
    # Play multiple games
    for game in range(n_games):
        obs = env.reset()
        done = False
        current_player = 0
        
        while not done:
            # Get action from current player
            if current_player == 0:
                action = player1.act(env.state)
                player_name = player1_name
            else:
                action = player2.act(env.state)
                player_name = player2_name
            
            if verbose:
                print(f"{player_name} chose action {action}")

            # Take step in environment
            obs, reward, done, _ = env.step(action)
            
            # Switch players
            current_player = 1 - current_player
            
            if done:
                winner = player1_name if reward == 1 else player2_name
                if verbose:
                    print(f"Game {game + 1} winner: {winner}")
                if reward == 1:
                    p1_wins += 1
                else:
                    p2_wins += 1
    
    if verbose:
        print(f"\nResults after {n_games} games:")
        print(f"{player1_name} wins: {p1_wins}")
        print(f"{player2_name} wins: {p2_wins}")
    else:
        print(f"Results after {n_games} games: {player1_name} wins: {p1_wins}, {player2_name} wins: {p2_wins}")
    return p1_wins, p2_wins

# Example usage:
if __name__ == "__main__":
    # Create the battle environment
    env = gym.make('LOCM-battle-2p-v0', version="1.5")

    # Create two rule-based battle agents
    player1 = gym_locm.agents.RuleBasedBattleAgent()
    player2 = gym_locm.agents.MaxAttackBattleAgent()
    player1_name = "RuleBased"
    player2_name = "MaxAttack"

    # Play the games
    play_games(player1, player1_name, player2, player2_name, n_games=1, env=env, verbose=True)
