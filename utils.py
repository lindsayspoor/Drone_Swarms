import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, "valid")



############### During training ######################################

def plot_log_results(log_folder,  save_model_path, title="Average training reward"):
    """
    plot the log results during training.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "episodes")

    y = moving_average(y, window=50)

    # Truncate x
    x = x[len(x) - len(y) :]

    np.savetxt(f"{log_folder}{save_model_path}.csv",(x,y) )

    fig = plt.figure(title)
    plt.plot(x, y, color = 'blue', linewidth=0.9)
    plt.yscale("linear")
    plt.xlabel("Number of training episodes")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(f'{log_folder}{save_model_path}.pdf')


############### During evaluation ######################################

def plot_order_param(max_timesteps, order_params):
    '''Plot the order parameter time evolution across an episode.'''

    plt.figure(figsize=(10,8))
    plt.grid()
    plt.plot(np.arange(max_timesteps+1), order_params, color="darkblue")
    plt.axhline(np.mean(order_params), color='red', label=r"mean $\Psi$")
    plt.xlabel("Timestep", fontsize=25)
    plt.ylabel(f"$\Psi$", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=17)
    plt.savefig("plots/order_param.pdf")


def plot_grid_visits(grid_visits, origin_Bx, origin_By, Lb_x, Lb_y):
    '''Plot the number of grid visits across the entire grid.'''

    plt.figure(figsize=(10,10))
    plt.imshow(grid_visits, cmap = "nipy_spectral", origin='lower')
    plt.plot([origin_Bx-(1/2), origin_Bx-(1/2)], [origin_By-(1/2), origin_By+Lb_y-(1/2)], color='yellow', linewidth=4 )
    plt.plot([origin_Bx-(1/2), origin_Bx+Lb_x-(1/2)], [origin_By-(1/2), origin_By-(1/2)], color='yellow', linewidth=4 )
    plt.plot([origin_Bx+Lb_x-(1/2), origin_Bx+Lb_x-(1/2)], [origin_By-(1/2), origin_By+Lb_y-(1/2)], color='yellow', linewidth=4 )
    plt.plot([origin_Bx-(1/2), origin_Bx+Lb_x-(1/2)], [origin_By+Lb_y-(1/2), origin_By+Lb_y-(1/2)], color='yellow', linewidth=4 )
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=19)
    plt.xlabel("x grid positions", fontsize=25)
    plt.ylabel("y grid positions", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig("plots/grid_visits.pdf")


def plot_MSD(max_timesteps, avg_dispersion):
    '''Plot the Mean Squared Displacement (MSD) time evolution across an episode.'''

    plt.figure(figsize=(10,8))
    plt.grid()
    plt.plot(np.arange(max_timesteps+1), avg_dispersion, color="darkblue")
    plt.axhline(np.mean(avg_dispersion), color='red', label=r"mean $MSD$")
    plt.xlabel("Timestep", fontsize=25)
    plt.ylabel(f"$MSD$", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=17)
    plt.savefig("plots/MSD.pdf")


def plot_swarm_factor_order_param(swarm_factors, order_params_4, order_params_2, std_order_params_4, std_order_params_2):
    '''Plot the relation between the order parameter and the swarm factor for both N=2 and N=4 drones.'''

    plt.figure(figsize = (10,8))
    plt.style.use("ggplot")
    plt.plot(swarm_factors, order_params_4, label="N=4")
    plt.fill_between(swarm_factors, order_params_4-std_order_params_4, order_params_4+std_order_params_4, alpha=0.3)
    plt.plot(swarm_factors, order_params_2, label="N=2")
    plt.fill_between(swarm_factors, order_params_2-std_order_params_2, order_params_2+std_order_params_2, alpha=0.3)
    plt.xlabel(r"$\lambda$", fontsize=25)
    plt.ylabel(r"$\Psi$", fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=22)
    plt.savefig("plots/swarm_order_param.pdf")


def plot_swarm_factor_rewards(swarm_factors, rewards_4, rewards_2, std_rewards_4, std_rewards_2):
    '''Plot the relation between the episode rewards and the swarm factor for both N=2 and N=4 drones.'''

    plt.figure(figsize = (10,8))
    plt.style.use("ggplot")
    plt.plot(swarm_factors, rewards_4, label="N=4")
    plt.fill_between(swarm_factors, rewards_4-std_rewards_4, rewards_4+std_rewards_4, alpha=0.3)
    plt.plot(swarm_factors, rewards_2, label="N=2")
    plt.fill_between(swarm_factors, rewards_2-std_rewards_2, rewards_2+std_rewards_2, alpha=0.3)
    plt.xlabel(r"$\lambda$", fontsize=25)
    plt.ylabel(r"Episode reward", fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=22)
    plt.savefig("plots/swarm_reward.pdf")


def plot_order_param_rewards(order_params_4, order_params_2, rewards_4, rewards_2, std_rewards_4, std_rewards_2):
    '''Plot the relation between the order parameter and the episode rewards for both N=2 and N=4 drones.'''

    plt.figure(figsize = (10,8))
    plt.style.use("ggplot")
    plt.plot(order_params_4, rewards_4, label="N=4")
    plt.fill_between(order_params_4, rewards_4-std_rewards_4, rewards_4+std_rewards_4, alpha=0.3)
    plt.plot(order_params_2,rewards_2, label="N=2")
    plt.fill_between(order_params_2, rewards_2-std_rewards_2, rewards_2+std_rewards_2, alpha=0.3)
    plt.xlabel(r"$\Psi$", fontsize=25)
    plt.ylabel(r"Episode reward", fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=22)
    plt.savefig("plots/order_param_reward.pdf")


def plot_swarm_factor_positional_rewards_percentage(swarm_factors, pos_rew_4, pos_rew_2, std_pos_rew_4, std_pos_rew_2):
    '''Plot the positional rewards for both the N=2 and N=4 drones as a percentage of the minimum optimal positional reward as a function of swarm factor.'''

    plt.figure(figsize = (10,8))
    plt.style.use("ggplot")
    plt.plot(swarm_factors, np.array(pos_rew_4)/273.43*100, label="N=4")
    plt.fill_between(swarm_factors, np.array(pos_rew_4)/273.43*100-np.array(std_pos_rew_4)/273.43*100, np.array(pos_rew_4)/273.43*100+np.array(std_pos_rew_4)/273.43*100,alpha=0.3)
    plt.plot(swarm_factors, np.array(pos_rew_2)/175*100, label="N=2")
    plt.fill_between(swarm_factors, np.array(pos_rew_2)/175*100-np.array(std_pos_rew_2)/175*100, np.array(pos_rew_2)/175*100+np.array(std_pos_rew_2)/175*100,alpha=0.3)
    plt.xlabel(r"$\lambda$", fontsize=25)
    plt.ylabel(f"% of optimal positional reward", fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=22)
    plt.savefig("plots/swarm_pos_rew.pdf")


############### During hyperparameter tuning ######################################

def plot_learning_rate(data0003, data0001, data001, data00001, x, saving_path, filename): 
    '''Plot the learning curves for hyperparameter tuning of the learning rate.'''

    average_learning_curve0003 = np.average(data0003, axis=0)
    std_learning_curve0003 = np.std(data0003, axis=0)

    average_learning_curve0003 = moving_average(average_learning_curve0003, window=500)
    std_learning_curve0003 = moving_average(std_learning_curve0003, window=500)

    average_learning_curve0001 = np.average(data0001, axis=0)
    std_learning_curve0001 = np.std(data0001, axis=0)

    average_learning_curve0001 = moving_average(average_learning_curve0001, window=500)
    std_learning_curve0001 = moving_average(std_learning_curve0001, window=500)

    average_learning_curve001 = np.average(data001, axis=0)
    std_learning_curve001 = np.std(data001, axis=0)

    average_learning_curve001 = moving_average(average_learning_curve001, window=500)
    std_learning_curve001 = moving_average(std_learning_curve001, window=500)

    average_learning_curve00001 = np.average(data00001, axis=0)
    std_learning_curve00001 = np.std(data00001, axis=0)

    average_learning_curve00001 = moving_average(average_learning_curve00001, window=500)
    std_learning_curve00001 = moving_average(std_learning_curve00001, window=500)


    # Truncate x
    x = x[len(x) - len(average_learning_curve0003) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(x,average_learning_curve00001, label=r"$\alpha=0.00001$")
    plt.fill_between(x, average_learning_curve00001-std_learning_curve00001, average_learning_curve00001+std_learning_curve00001, alpha=0.5)
    plt.plot(x,average_learning_curve0003, label=r"$\alpha=0.0003$")
    plt.fill_between(x, average_learning_curve0003-std_learning_curve0003, average_learning_curve0003+std_learning_curve0003, alpha=0.5)
    plt.plot(x,average_learning_curve0001, label=r"$\alpha=0.0001$")
    plt.fill_between(x, average_learning_curve0001-std_learning_curve0001, average_learning_curve0001+std_learning_curve0001, alpha=0.5)
    plt.plot(x,average_learning_curve001, label=r"$\alpha=0.001$")
    plt.fill_between(x, average_learning_curve001-std_learning_curve001, average_learning_curve001+std_learning_curve001, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19 )
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")


def plot_ent_coef(data0 ,data001, data005, data01, x, saving_path, filename):
    '''Plot the learning curves for the hyperparameter tuning of the entropy coefficient.'''

    average_learning_curve0 = np.average(data0, axis=0)
    std_learning_curve0 = np.std(data0, axis=0)

    average_learning_curve0 = moving_average(average_learning_curve0, window=500)
    std_learning_curve0 = moving_average(std_learning_curve0, window=500)

    average_learning_curve001 = np.average(data001, axis=0)
    std_learning_curve001 = np.std(data001, axis=0)

    average_learning_curve001 = moving_average(average_learning_curve001, window=500)
    std_learning_curve001 = moving_average(std_learning_curve001, window=500)

    average_learning_curve005 = np.average(data005, axis=0)
    std_learning_curve005 = np.std(data005, axis=0)

    average_learning_curve005 = moving_average(average_learning_curve005, window=500)
    std_learning_curve005 = moving_average(std_learning_curve005, window=500)

    average_learning_curve01 = np.average(data01, axis=0)
    std_learning_curve01 = np.std(data01, axis=0)

    average_learning_curve01 = moving_average(average_learning_curve01, window=500)
    std_learning_curve01 = moving_average(std_learning_curve01, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve0) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(x,average_learning_curve0, label=r"$c_2=0.0$")
    plt.fill_between(x, average_learning_curve0-std_learning_curve0, average_learning_curve0+std_learning_curve0, alpha=0.5)
    plt.plot(x,average_learning_curve001, label=r"$c_2=0.001$")
    plt.fill_between(x, average_learning_curve001-std_learning_curve001, average_learning_curve001+std_learning_curve001, alpha=0.5)
    plt.plot(x,average_learning_curve005, label=r"$c_2=0.005$")
    plt.fill_between(x, average_learning_curve005-std_learning_curve005, average_learning_curve005+std_learning_curve005, alpha=0.5)
    plt.plot(x,average_learning_curve01, label=r"$c_2=0.01$")
    plt.fill_between(x, average_learning_curve01-std_learning_curve01, average_learning_curve01+std_learning_curve01, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19 )
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")


def plot_clip_range(data1, data2, data3, x, saving_path, filename): 
    '''Plot the learning curves for the hyperparameter tuning of the clipping parameter.'''

    average_learning_curve1 = np.average(data1, axis=0)
    std_learning_curve1 = np.std(data1, axis=0)

    average_learning_curve1 = moving_average(average_learning_curve1, window=500)
    std_learning_curve1 = moving_average(std_learning_curve1, window=500)

    average_learning_curve2 = np.average(data2, axis=0)
    std_learning_curve2 = np.std(data2, axis=0)

    average_learning_curve2 = moving_average(average_learning_curve2, window=500)
    std_learning_curve2 = moving_average(std_learning_curve2, window=500)

    average_learning_curve3 = np.average(data3, axis=0)
    std_learning_curve3 = np.std(data3, axis=0)

    average_learning_curve3 = moving_average(average_learning_curve3, window=500)
    std_learning_curve3 = moving_average(std_learning_curve3, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve2) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(x,average_learning_curve1, label=r"$\epsilon=0.1$")
    plt.fill_between(x, average_learning_curve1-std_learning_curve1, average_learning_curve1+std_learning_curve1, alpha=0.5)
    plt.plot(x,average_learning_curve2, label=r"$\epsilon=0.2$")
    plt.fill_between(x, average_learning_curve2-std_learning_curve2, average_learning_curve2+std_learning_curve2, alpha=0.5)
    plt.plot(x,average_learning_curve3, label=r"$\epsilon=0.3$")
    plt.fill_between(x, average_learning_curve3-std_learning_curve3, average_learning_curve3+std_learning_curve3, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")


def plot_layers(data1, data2, data3, x, saving_path, filename): 
    '''Plot the learning curves of the hyperparameter tuning of the number of MLP layers.'''

    average_learning_curve1 = np.average(data1, axis=0)
    std_learning_curve1 = np.std(data1, axis=0)

    average_learning_curve1 = moving_average(average_learning_curve1, window=500)
    std_learning_curve1 = moving_average(std_learning_curve1, window=500)

    average_learning_curve2 = np.average(data2, axis=0)
    std_learning_curve2 = np.std(data2, axis=0)

    average_learning_curve2 = moving_average(average_learning_curve2, window=500)
    std_learning_curve2 = moving_average(std_learning_curve2, window=500)

    average_learning_curve3 = np.average(data3, axis=0)
    std_learning_curve3 = np.std(data3, axis=0)

    average_learning_curve3 = moving_average(average_learning_curve3, window=500)
    std_learning_curve3 = moving_average(std_learning_curve3, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve2) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(x,average_learning_curve1, label=r"$64$")
    plt.fill_between(x, average_learning_curve1-std_learning_curve1, average_learning_curve1+std_learning_curve1, alpha=0.5)
    plt.plot(x,average_learning_curve2, label=r"$64\times64$")
    plt.fill_between(x, average_learning_curve2-std_learning_curve2, average_learning_curve2+std_learning_curve2, alpha=0.5)
    plt.plot(x,average_learning_curve3, label=r"$64\times64\times64$")
    plt.fill_between(x, average_learning_curve3-std_learning_curve3, average_learning_curve3+std_learning_curve3, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")


def plot_nodes(data1, data2, data3, x, saving_path, filename): 
    '''Plot the learning curves for the hyperparameter tuning of the number of nodes per layer.'''

    average_learning_curve1 = np.average(data1, axis=0)
    std_learning_curve1 = np.std(data1, axis=0)

    average_learning_curve1 = moving_average(average_learning_curve1, window=500)
    std_learning_curve1 = moving_average(std_learning_curve1, window=500)

    average_learning_curve2 = np.average(data2, axis=0)
    std_learning_curve2 = np.std(data2, axis=0)

    average_learning_curve2 = moving_average(average_learning_curve2, window=500)
    std_learning_curve2 = moving_average(std_learning_curve2, window=500)

    average_learning_curve3 = np.average(data3, axis=0)
    std_learning_curve3 = np.std(data3, axis=0)

    average_learning_curve3 = moving_average(average_learning_curve3, window=500)
    std_learning_curve3 = moving_average(std_learning_curve3, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve2) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(x,average_learning_curve1, label=r"$32\times32$")
    plt.fill_between(x, average_learning_curve1-std_learning_curve1, average_learning_curve1+std_learning_curve1, alpha=0.5)
    plt.plot(x,average_learning_curve2, label=r"$64\times64$")
    plt.fill_between(x, average_learning_curve2-std_learning_curve2, average_learning_curve2+std_learning_curve2, alpha=0.5)
    plt.plot(x,average_learning_curve3, label=r"$128\times128$")
    plt.fill_between(x, average_learning_curve3-std_learning_curve3, average_learning_curve3+std_learning_curve3, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")