from gym_locm.envs import LOCMBattleSingleEnv
from gym_locm.agents import RandomBattleAgent, RandomDraftAgent
from sb3_contrib import MaskablePPO
import gym

# 1. Create the environment
env = LOCMBattleSingleEnv(
    deck_building_agents=(RandomDraftAgent(), RandomDraftAgent()),  # Random draft for both players
    battle_agent=RandomBattleAgent(),  # Opponent for training
    return_action_mask=True,  # Important for MaskablePPO
    seed=42,
    items=True,  # Include item cards
    version="1.5",
    reward_functions=("win-loss",)  # Simple reward: +1 for win, -1 for loss
)

# 2. Create vectorized environment (required for stable-baselines3)
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# 3. Create the model
model = MaskablePPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048
)

# 4. Train the model
model.learn(total_timesteps=1000000, progress_bar=True)  

# 5. Save the model
model.save("battle_agent_v2")