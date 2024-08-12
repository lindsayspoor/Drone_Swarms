import jax.numpy as np
from stable_baselines3 import PPO
from utils import plot_log_results
from DronesEnv import DronesEnvironment
import os
from stable_baselines3.common.monitor import Monitor
from CallbackClass import SaveOnBestTrainingRewardCallback



if __name__ == "__main__":

    # Choose settings for the environment

    N = 2 # N is the numer of drones the swarm consists of
    k_a = 4 # k_a equally spaced angles for the actions in range [-theta_max, theta_max]
    k_s = 8 # k_s equally spaced angles for the direction angle in the range [-pi, pi)
    theta_max  = np.pi /2 # theta_max is the maximum turning angle
    boundary_width = 0 # number of grid elements the boundary width consists of (0 means no boundary, currently the only setting that works because of periodic boundary conditions.)
    Rv = 10000 # visibility Radius for each drone, (currently set to 1000 s.t. radius of all drones to infinity (large number))
    L = 50 + (2 * boundary_width) # size of grid of total enviroment (LxL)
    La_x = 10 # x-size of area A
    La_y = 10 # y-size of area A
    Lb_x = 10 # x-size of area B
    Lb_y = 10 # y-size of area B
    origin_Ax = 20 + boundary_width # x origin of area A
    origin_Ay = 20+ boundary_width # y origin of area A
    origin_Bx = 20 # x origin of area B
    origin_By = 30  # y origin of area B
    max_timesteps = 200 # maximum amount of timesteps to play game before truncation
    goal_reward = 1 # initial positional reward value for the exploration area grid tiles
    swarm_factor = 0.5 # swarm factor lambda in the reward function
    collision_factor = 1 # collision factor xi in the reward function
    compactness_const = 1 # compactness constant c
    reward_decay = 0.75 # reward decay parameter eta


    # Choose training settings

    n_episodes = 100 # number of training episodes
    n_steps = 2048 
    batch_size = 64
    n_epochs = 10

    # Choose hyperparameters

    lr = 0.0003
    ent_coef = 0.0
    clip_range = 0.2
    n_layers=2
    n_nodes=128


    settings = {"N": N,
                "k_a": k_a,
                "k_s": k_s,
                "theta_max": theta_max,
                "boundary_width": boundary_width,
                "L": L,
                "Rv": Rv,
                "La_x": La_x,
                "La_y": La_y,
                "Lb_x": Lb_x,
                "Lb_y": Lb_y,
                "origin_Ax": origin_Ax,
                "origin_Ay": origin_Ay,
                "origin_Bx": origin_Bx,
                "origin_By": origin_By,
                "max_timesteps": max_timesteps,
                "goal_reward": goal_reward,
                "swarm_factor": swarm_factor,
                "collision_factor": collision_factor,
                "compactness_const": compactness_const,
                "reward_decay": reward_decay
                }
    

    # Train your model

    # Create log dir
    log_dir = f"log_dir_model/"
    check_freq = 1000
    order_param_check = 10000
    model_path = f"{n_episodes=}_{n_nodes=}_{N=}_{k_a=}_{k_s=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{swarm_factor=}_{collision_factor=}_{compactness_const=}_{reward_decay=}_{goal_reward=}"
    os.makedirs(log_dir, exist_ok=True)

    env = DronesEnvironment(settings=settings, render_mode='rgb_array')
    env = Monitor(env, log_dir)

    auto_save_callback =  SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_path=model_path)

    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range, policy_kwargs={"net_arch":dict(pi=[n_nodes]*n_layers, vf=[n_nodes]*n_layers)})
    model.learn(total_timesteps = max_timesteps*n_episodes, callback=auto_save_callback, progress_bar=True)
    model.save(f"models/pposb3_"+model_path)


    plot_log_results(log_dir, model_path)



