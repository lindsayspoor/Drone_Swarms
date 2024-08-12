import numpy as np
from stable_baselines3 import PPO
from utils import plot_order_param, plot_grid_visits, plot_MSD
from DronesEnv import DronesEnvironment
from tqdm import tqdm
import matplotlib.pyplot as plt



def evaluate(env, model, n_eval_episodes, max_timesteps, N, render):
    '''Evaluates a saved model.'''

    episode_rewards = []
    order_params=[]
    grid_visits=[]
    pos_rewards_all=[]
    avg_dispersion_all = []
    
    for i in tqdm(range(n_eval_episodes)):
        order_param = []
        avg_dispersion = []
        position = np.zeros((max_timesteps+1,N,2))
        obs, info = env.reset()
        if render:
            env.render()
        done = False
        trunc = False
        episode_reward = 0

        order_param.append(env.order_param)
        avg_dispersion.append(env.avg_dispersion)
        position[0,:,:]=env.state[:,0:2]
        t=1
        pos_rew = 0

        while not trunc:
            
            action, _ = model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)

            if render:
                env.render()
            
            pos_rewards_N = env.pos_rewards_N
            cumulative_pos_rewards = np.sum(pos_rewards_N)
            pos_rew+=cumulative_pos_rewards
            order_param.append(env.order_param)
            avg_dispersion.append(env.avg_dispersion)
            position[t,:,:]=env.state[:,0:2]
            episode_reward += reward
            t+=1
        

        episode_rewards.append(episode_reward)
        order_params.append(order_param)
        avg_dispersion_all.append(avg_dispersion)
        grid_visits_i = env.grid_visits
        grid_visits.append(grid_visits_i)
        pos_rewards_all.append(pos_rew)
    

    mean_order_params = np.mean(np.mean(np.array(order_params), axis=0))
    mean_reward = np.mean(episode_rewards)
    mean_grid_visits = np.mean(np.array(grid_visits), axis=0)
    mean_pos_rew = np.mean(np.array(pos_rewards_all))
    mean_avg_dispersion = np.mean(np.mean(np.array(avg_dispersion_all), axis=0))

    std_reward = np.std(episode_rewards)
    std_order_params = np.std(mean_order_params)
    std_pos_rew = np.std(np.array(pos_rewards_all))

    print(f"{mean_reward=}")
    print(f"{mean_order_params=}")
    print(f"{mean_pos_rew=}")
    print(f"{mean_avg_dispersion=}")

    print(f"{std_reward=}")
    print(f"{std_order_params=}")
    print(f"{std_pos_rew=}")  

    return np.mean(np.array(order_params), axis=0), mean_reward, mean_grid_visits, mean_pos_rew, np.mean(np.array(avg_dispersion_all), axis=0), std_reward, std_order_params, std_pos_rew  






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


    # Choose training settings the model was trained on

    n_episodes = 100 # number of training episodes
    n_steps = 2048 
    batch_size = 64
    n_epochs = 10

    # Choose hyperparameter settings the model was trained on

    lr = 0.0003
    ent_coef = 0.0
    clip_range = 0.2
    n_layers=2
    n_nodes=128


    # Choose evaluation settings
    
    n_eval_episodes = 10
    render = False


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
    

    
    

    # Get log dir to load model from
    log_dir = f"log_dir_model/"
    model_path = f"best_model_{n_episodes=}_{n_nodes=}_{N=}_{k_a=}_{k_s=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{swarm_factor=}_{collision_factor=}_{compactness_const=}_{reward_decay=}_{goal_reward=}_2"

    # initialize environment
    env = DronesEnvironment(settings=settings, render_mode='rgb_array')

    # initialize model
    model = PPO.load(log_dir+model_path)


    mean_order_params, mean_reward, mean_grid_visits, mean_pos_rew, mean_avg_dispersion, std_reward, std_order_params, std_pos_rew = evaluate(env, model, n_eval_episodes, max_timesteps, N, render)


    plot_order_param(max_timesteps, mean_order_params)
    plot_MSD(max_timesteps, mean_avg_dispersion)
    plot_grid_visits(mean_grid_visits, origin_Bx, origin_By, Lb_x, Lb_y)






