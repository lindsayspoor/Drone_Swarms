# Autonomous Drone Swarm Optimisation

This repository reproduces the results from the Master Thesis: **"Optimising Group Alignment and Cohesion in Autonomous Drone Swarms for Targeted Military Exploration."** The research focuses on training reinforcement learning (RL) agents to manage the alignment and cohesion of drone swarms during military exploration missions.

## Repository Structure

The repository contains the following files and folders:

### Files

- **`DronesEnv.py`**: Defines the custom environment for the single-agent RL agents, where drones move on an LxL grid, aiming to explore specific regions while maintaining group cohesion and alignment.
  
- **`train.py`**: Trains the single-agent RL model using the PPO algorithm from `stable_baselines3`. Models are saved in the `models/` directory, with the best model during training saved in `log_dir_model/`.

- **`evaluate.py`**: Evaluates the trained models, outputting metrics such as the order parameter, Mean Squared Displacement (MSD), and episode rewards. It also generates plots required to reproduce the thesis results.

- **`utils.py`**: Contains utility functions for plotting and other helper functions used across the project.

- **`CallbackClass.py`**: Implements a custom callback to track and save training progress.

### Folders

- **`plots/`**: Stores the plots generated during evaluation.
- **`models/`**: Contains the trained models.
- **`log_dir_model/`**: Stores logs and the best model checkpoint during training.

## Dependencies

To run the scripts in this repository, the following Python packages are required:

- `stable_baselines3`
- `numpy`
- `matplotlib`
- `jax`

You can install these packages using pip:

```bash
pip install stable-baselines3 numpy matplotlib jax
```

## Usage

### Training

To train the RL agents, run the `train.py` script. This will:

- Initialize the drone environment defined in `DronesEnv.py`.
- Train the model using PPO.
- Save the trained model to the `models/` directory.
- Periodically save the best model to `log_dir_model/`.

You can customize parameters such as the number of drones (`N`) and the `swarm_factor` and other settings directly in `train.py`.

### Evaluation

To evaluate the trained models, run the `evaluate.py` script. This will:

- Load the trained models from `models/`.
- Evaluate the performance using metrics like the order parameter, MSD, and episode rewards.
- Generate and save plots in the `plots/` directory, which are essential for reproducing the results in the thesis.

Please make sure to align the settings used to train the model in the `evaluate.py` file in order to make sure to evaluate the correct model.


## Reproducing Thesis Results

Follow these steps to reproduce the results from the thesis:

1. Adjust the environment and model parameters in `train.py` as needed.
2. Train the model by running `train.py`.
3. Evaluate the trained model by running `evaluate.py`.
4. The plots and metrics necessary for the thesis will be available in the `plots/` directory.



## Contact

For any questions or issues, please contact lindsayspoor (repo maintainer) on GitHub.
```
