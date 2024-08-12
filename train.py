import jax.numpy as np
from stable_baselines3 import PPO
from utils import plot_log_results
from DronesEnv import DronesEnvironment
import os
from stable_baselines3.common.monitor import Monitor
from CallbackClass import SaveOnBestTrainingRewardCallback



if __name__ == "__main__":

    N = 2
    k_a = 4
    k_s = 8
    theta_max  = np.pi /2
    boundary_width = 0
    Rv = 10000 # set radius of all drones to infinity (large number)
    L = 50 + (2 * boundary_width)
    La_x = 10
    La_y = 10
    Lb_x = 10
    Lb_y = 10
    origin_Ax = 20 + boundary_width
    origin_Ay = 20+ boundary_width
    origin_Bx = 20 
    origin_By = 30 
    max_timesteps = 200
    goal_reward = 1
    swarm_factor = 0.5
    collision_factor = 1
    compactness_const = 1
    reward_decay = 0.75



    n_episodes = 10000
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
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



