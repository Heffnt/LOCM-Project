# import gym
# import gym_locm
# import numpy as np
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
# from gym.wrappers import Monitor
# from sb3_contrib import MaskablePPO
# from torch import nn
# import os
# import time
# import torch

# # Custom callback for intermediate logging
# class TrainingCallback(BaseCallback):
#     def __init__(self, check_freq, verbose=1):
#         super(TrainingCallback, self).__init__(verbose)
#         self.check_freq = check_freq

#     def _on_step(self) -> bool:
#         # Print done status and reward during training
#         if self.num_timesteps % self.check_freq == 0:
#             print(f"Step: {self.num_timesteps}")
            
#             # Access the current environment's status
#             done = self.locals.get("done", None)
#             if done is not None:
#                 print(f"Done status: {done}")
            
#             # Optionally, print reward and other information if you want
#             reward = self.locals.get("reward", None)
#             if reward is not None:
#                 print(f"Reward: {reward}")
        
#         return True

# # Custom callback for intermediate logging
# class CustomCallback(BaseCallback):
#     def __init__(self, check_freq, verbose=1):
#         super(CustomCallback, self).__init__(verbose)
#         self.check_freq = check_freq

#     def _on_step(self) -> bool:
#         if self.num_timesteps % self.check_freq == 0:
#             print(f"Step {self.num_timesteps}: Model training in progress...")
        
#         return True

# # Configuration Parameters
# TRAINING_EPISODES = 100_000
# EVAL_EPISODES = 500
# SAVE_INTERVAL = 20000
# EVAL_FREQ = SAVE_INTERVAL
# OPPONENT_UPDATE_FREQ = 10
# SEED = 42

# # Create the environment
# def create_env():
#     """Creates and returns the LOCM environment."""
#     env = gym.make("LOCM-battle-v0", version="1.5")  # Use the LOCM battle phase environment with version 1.5
#     return env

# # Create vectorized environment for parallel simulation
# train_env = make_vec_env(create_env, n_envs=1, seed=SEED)

# # Create evaluation environment
# log_dir = f"./logs/{int(time.time())}"
# eval_env = Monitor(create_env(), directory=log_dir)

# # Use a unique directory for the logs to prevent conflicts from previous runs
# # log_dir = f"./logs/{int(time.time())}"
# # eval_env = Monitor(eval_env, directory=log_dir)

# # Callback to evaluate the agent and stop training if no improvement
# stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, verbose=1)
# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path="./logs/",
#     log_path="./logs/",
#     eval_freq=EVAL_FREQ,
#     deterministic=True,
#     render=False,
#     n_eval_episodes=EVAL_EPISODES,
#     callback_after_eval=stop_callback,
#     verbose=1
# )

# # Custom intermediate log callback
# log_callback = CustomCallback(check_freq=5000, verbose=1)

# # Define the policy (MLP with 7 hidden layers of 455 neurons)
# policy_kwargs = {
#     "net_arch": {
#         "pi": [455] * 7,  # 7 hidden layers for the policy network
#         "vf": [455] * 7   # 7 hidden layers for the value function network
#     },
#     "activation_fn": nn.ReLU,  # Activation function for the network
# }

# # Instantiate the MaskablePPO model
# model = MaskablePPO(
#     policy="MlpPolicy",
#     env=train_env,
#     learning_rate=4.114e-4,  # Adjust learning rate if necessary
#     batch_size=256,  # Set batch size
#     n_steps=10,  # Steps per update (adjust as necessary)
#     ent_coef=0.005,  # Entropy coefficient for exploration
#     vf_coef=1.0,  # Value function coefficient
#     max_grad_norm=0.5,
#     n_epochs=1,  # Number of epochs per update
#     clip_range=0.2,  # PPO clip range for policy optimization
#     seed=SEED,
#     verbose=1,
#     tensorboard_log="./ppo_locm_tensorboard/",
#     policy_kwargs=policy_kwargs,
#     device="cuda",  # Use GPU for faster training (if available)
# )

# print("Model initialized")
# training_callback = TrainingCallback(check_freq=1000, verbose=1)  # Print every 1000 steps

# # Train the model
# model.learn(
#     # total_timesteps=TRAINING_EPISODES*train_env.num_envs,  # Adjust number of timesteps for training
#     total_timesteps=100000,
#     callback=[eval_callback, log_callback, training_callback],  # Combine multiple callbacks
# )

# print("Model training complete")

# # Save the model
# model.save("ppo_locm_agent")
# # torch.save(model.state_dict(), 'ppo_locm_agent.pth', pickle_protocol=4)
# print("Model saved")

# # Evaluate the trained agent
# print("Starting final evaluation...")
# obs = eval_env.reset()
# for i in range(10):  # Play 10 evaluation episodes
#     done = False
#     total_reward = 0
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = eval_env.step(action)
#         total_reward += reward
#         eval_env.render()
#     print(f"Episode {i + 1}: Total Reward: {total_reward}")


##--------------------------------------------------------------
# import gym
# import gym_locm
# import numpy as np
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import BaseCallback
# from gym.wrappers import Monitor
# from sb3_contrib import MaskablePPO
# from torch import nn
# import os
# import time

# # Custom callback for intermediate logging
# class TrainingCallback(BaseCallback):
#     def __init__(self, check_freq, verbose=1):
#         super(TrainingCallback, self).__init__(verbose)
#         self.check_freq = check_freq

#     def _on_step(self) -> bool:
#         if self.num_timesteps % self.check_freq == 0:
#             print(f"Step: {self.num_timesteps}")
#         return True

# # Configuration Parameters
# TRAINING_EPISODES = 1_000_000
# EVAL_EPISODES = 10
# SAVE_INTERVAL = 20000
# SEED = 42

# # Create the environment
# def create_env():
#     """Creates and returns the LOCM environment."""
#     env = gym.make("LOCM-battle-v0", version="1.5")  # Use the LOCM battle phase environment
#     return env

# # Create training environment
# train_env = make_vec_env(create_env, n_envs=1, seed=SEED)

# # obs = train_env.reset()
# # print("observation---------", obs)
# # print("observation shape---------", obs.shape)

# # action_mask = train_env.action_space
# # print("Action Mask:", action_mask)
# # exit()



# # Define the policy (MLP with 7 hidden layers of 455 neurons)
# policy_kwargs = {
#     "net_arch": {
#         "pi": [455] * 7,  # 7 hidden layers for the policy network
#         "vf": [455] * 7   # 7 hidden layers for the value function network
#     },
#     "activation_fn": nn.ReLU,  # Activation function for the network
# }

# # Instantiate the MaskablePPO model
# model = MaskablePPO(
#     policy="MlpPolicy",
#     env=train_env,
#     learning_rate=4.114e-4,
#     batch_size=256,
#     n_steps=10,
#     ent_coef=0.005,
#     vf_coef=1.0,
#     max_grad_norm=0.5,
#     n_epochs=1,
#     clip_range=0.2,
#     seed=SEED,
#     verbose=1,
#     tensorboard_log="./ppo_locm_tensorboard/",
#     policy_kwargs=policy_kwargs,
#     device="cuda",  # Use GPU for faster training
# )

# print("Model initialized")
# training_callback = TrainingCallback(check_freq=1000, verbose=1)

# # Train the model
# model.learn(
#     total_timesteps=TRAINING_EPISODES*train_env.num_envs,  # Adjust number of timesteps for training
#     callback=[training_callback]  # Use training callback only
# )

# print("Model training complete")

# # Save the model
# model.save("ppo_locm_agent")
# print("Model saved")

# # Evaluate the trained agent
# print("Starting evaluation...")
# eval_env = Monitor(create_env(), directory=f"./logs/{int(time.time())}")
# mean_reward, std_reward = evaluate_policy(
#     model,
#     eval_env,
#     n_eval_episodes=EVAL_EPISODES,
#     deterministic=True,
#     render=True  # Render during evaluation
# )
# print(f"Mean Reward: {mean_reward} ± {std_reward}")

##-------------------------------------------------------------------------

import gym
import gym_locm
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from torch import nn
import os
import time

class TrainingCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        # Print done status and reward during training
        if self.num_timesteps % self.check_freq == 0:
            print(f"Time Step: {self.num_timesteps}")
            
            # done = self.locals.get("done", None)
            # if done is not None:
            #     print(f"Done status: {done}")
            
            # reward = self.locals.get("reward", None)
            # if reward is not None:
            #     print(f"Reward: {reward}")
        
        return True

TRAINING_EPISODES = 100
EVAL_EPISODES = 10
SAVE_INTERVAL = 20000
SEED = 42
SWITCH_FREQ = 1000  # Frequency to transfer learning to opponent in self-play

def create_env():
    """Creates and returns the LOCM environment."""
    env = gym.make("LOCM-battle-v0", version="1.5") 
    return env

train_env = make_vec_env(create_env, n_envs=1, seed=SEED)

policy_kwargs = {
    "net_arch": {
        "pi": [256, 256, 256], 
        "vf": [256, 256, 256]  
    },
    "activation_fn": nn.ReLU,  
}

model = MaskablePPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    batch_size=128,
    n_steps=10_000,
    n_epochs=10,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./ppo_locm_tensorboard/",
    device="cuda",  
)

print("Model initialized")
training_callback = TrainingCallback(check_freq=1000, verbose=1)  # Print every 1000 steps

# Create self-play environment setup
# def self_play_opponent(model):
#     """Return a copy of the current model to be used as the opponent in self-play."""
#     opponent = MaskablePPO.load(model.get_parameters())  # Load current model weights
#     return opponent

def self_play_opponent(model):
    """Return a copy of the current model to be used as the opponent in self-play."""
    # Create a new instance of the model with the same environment and policy
    print("Reached copying phase-----------------------")
    opponent = MaskablePPO(
        policy="MlpPolicy",
        env=model.env,  # Use the same environment
        policy_kwargs=model.policy_kwargs,
        device=model.device,
        verbose=0,  # Suppress verbose logging for the opponent
    )
    # Copy parameters from the current model
    opponent.set_parameters(model.get_parameters())
    print("COmpleted copying phase-----------------------")

    return opponent


for episode in range(TRAINING_EPISODES):
    if episode % SWITCH_FREQ == 0:
        opponent_model = self_play_opponent(model)
        print(f"Switching opponent at episode {episode}.")

    model.learn(total_timesteps=10_000, callback=training_callback)  # Adjust timesteps per iteration
    print(f"Episode {episode}: Model training in progress.")

    if episode % 10 == 0:
        obs = train_env.reset()
        # print("Evaluation observation", obs)
        # print("Evaluation observation size", obs.shape)

        done = False
        total_reward = 0
        count=0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # print("Predicted successfully-----------")
            # print("Predicted action", action)
            old_obs = obs
            obs, reward, done, info = train_env.step(action)

            # if (obs==new_obs):
            # if np.allclose(obs, old_obs):
            #     print("Same observation for ", count)
            

            # print("predicted obs size", obs.shape)
            # print("predicted reward", reward)
            total_reward += reward
            count+=1
            if (count>=1000):
                print("Loop broke by reaching max count")
                break
        print(f"Evaluation after episode {episode}: Total Reward = {total_reward}")


# Save the model
model.save("ppo_locm_agent")
print("Model saved")

# Evaluate the trained agent
print("Starting final evaluation...")
eval_env = create_env()
obs = eval_env.reset()
for i in range(10):  # Play 10 evaluation episodes
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        eval_env.render()
    print(f"Episode {i + 1}: Total Reward: {total_reward}")
