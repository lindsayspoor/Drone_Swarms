import jax.numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt


class DronesEnvironment(gym.Env):
    """This environment is designed for the task of training an RL agent to learn N drones to swarm and explore a specific exploration
    region. """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, settings, render_mode=None):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.N = settings["N"] # N is the numer of drones the swarm consists of
        self.k_a = settings["k_a"] # k_a equally spaced angles for the actions in range [-theta_max, theta_max]
        self.k_s = settings["k_s"] # k_s equally spaced angles for the direction angle in the range [-pi, pi)
        self.theta_max = settings["theta_max"] # theta_max is the maximum turning angle
        self.boundary_width = settings["boundary_width"] # number of grid elements the boundary width consists of
        self.L = settings["L"] # size of grid of total enviroment (LxL)
        self.Rv = settings["Rv"] # visibility Radius for each drone
        self.La_x = settings["La_x"] # x-size of area A
        self.La_y = settings["La_y"] # y-size of area A
        self.Lb_x = settings["Lb_x"] # x-size of area B
        self.Lb_y = settings["Lb_y"] # y-size of area B
        self.origin_Ax = settings["origin_Ax"] # x origin of area A
        self.origin_Ay = settings["origin_Ay"] # y origin of area A
        self.origin_Bx = settings["origin_Bx"] # x origin of area B
        self.origin_By = settings["origin_By"] # y origin of area B
        self.max_timesteps = settings["max_timesteps"] # maximum amount of timesteps to play game before truncation
        self.goal_reward = settings["goal_reward"] # initial positional reward value for the exploration area grid tiles
        self.swarm_factor = settings["swarm_factor"] # swarm factor lambda in the reward function
        self.collision_factor = settings["collision_factor"] # collision factor xi in the reward function
        self.compactness_const = settings["compactness_const"] # compactness constant c
        self.reward_decay = settings["reward_decay"] # reward decay parameter eta

        self.action_angles = np.linspace(-self.theta_max, self.theta_max, self.k_a)
        self.direction_angles = np.linspace(0, 2*np.pi, self.k_s+1)[0:-1]

        self.counter = 0 # updates each step to count what timestep we are in
        self.done = False
        self.truncated = False
        self.collective_reward = 0
        self.order_param = 0

        self.action_space = spaces.MultiDiscrete([self.k_a]*self.N) # for all N drones, k_a possible action angles to choose from

        # construct observation space
        low  = np.zeros((5*self.N))
        high =  np.zeros((5*self.N))
        high[0:(5*self.N):5] = self.L
        high[1:(5*self.N):5] = self.L
        high[2:(5*self.N):5] = self.direction_angles[-1]
        high[3:(5*self.N):5] = self.direction_angles[-1]
        high[4:(5*self.N):5] = self.goal_reward

        self.observation_space = spaces.Box(low = low, high=high, dtype = np.float64)


    def seed(self, seed=None):
        '''Setting a random seed.'''

        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]


    def reset(self, **kwargs):
        '''Reset function to reset drone states, observations, reward grid back to the initial conditions.'''

        super().reset(**kwargs)

        self.done = False
        self.truncated = False
        self.counter = 0
        self.collective_reward = 0
        self.order_param = 0
        self.state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle
        self.old_state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle

        self.initialize_grid()
        self.initialize_drones()
        self.initialize_rewards()
        self.grid_visits = np.zeros((self.L, self.L))

        self.avg_dispersion = self.dispersion(self.state[:,0:2])


        obs = self.get_obs()


        return obs, {}
    
    
    def initialize_grid(self):
        '''Initialize the positions of the grid tiles, as well as the positions of the grid tiles for initialization
        area A and exploration area B.'''


        self.grid_positions = [[i,j] for i in range(self.L) for j in range(self.L)]
        self.grid_A_positions = [[i,j] for i in range(self.origin_Ax, self.origin_Ax+self.La_x) for j in range(self.origin_Ay, self.origin_Ay+self.La_y)] # define area A on grid
        self.grid_B_positions = [[i,j] for i in range(self.origin_Bx, self.origin_Bx+self.Lb_x) for j in range(self.origin_By, self.origin_By+self.Lb_y)] # define area B on grid


    def initialize_drones(self):
        '''Initialize drone positions within initialization area A.'''

        drone_grid_indices = np.random.choice(np.arange(self.La_x*self.La_y), size=self.N, replace=False) # randomly choose initial grid locations for all N drones in area A
        # by initialising the drones on the grid positions and setting replace = False, all drones will never be initialised onto the same grid cell

        self.drone_directions = np.random.choice(self.direction_angles, size = self.N) # choose random initial directions for all drones

        self.drone_velocities = np.zeros((self.N,2))
        self.avg_velocities = np.zeros((self.N,2))

        for i in range(self.N):
            self.state[i, 0:2] =  [self.grid_A_positions[drone_grid_indices[i]][0],self.grid_A_positions[drone_grid_indices[i]][1]]
            
            self.drone_velocities[i,:] = self.compute_velocities(self.drone_directions[i]) # compute the initial velocity vector for all drones based on the given direction angle


        self.update_order_parameter()


    def initialize_rewards(self):
        '''Initialize the rewards per grid cell.'''

        self.reward_grid = np.zeros((self.L,self.L))


        for x, y in self.grid_B_positions:
            self.reward_grid[x][y] = self.goal_reward


    def compute_ang_diff_angle(self, phi, theta):
        '''Computes the angular difference angle between angles phi and theta in the range (0,2pi] and selects the discretized
        angle from the list of direction angles afterwards.'''

        alpha = phi - theta
        angular_difference_angle = (alpha + 2*np.pi) % (2*np.pi)

        return angular_difference_angle


    def compute_angular_difference(self, i, connected_drones_i):
        '''Computes the average velocity vector Pi and the angular difference angle, i.e. the difference in direction angle of
        the i-th drone and the average direction angle according to Pi from the other drones in the range of the i-th drone.'''


        Pi = np.zeros(2)
        for j in connected_drones_i:
            if j != i:
                Pi += self.drone_velocities[j,:]
        Pi = Pi / Pi.shape[0]

        self.avg_velocities[i,:] = Pi

        phi_i = self.compute_direction(Pi)
        phi_d_i = self.find_nearest(self.direction_angles, phi_i)

        angular_difference_angle_i = self.compute_ang_diff_angle(phi_d_i , self.drone_directions[i])


        return angular_difference_angle_i


    def get_obs(self):
        '''Produces a state observation s_t for all N drones.'''

        obs = np.zeros((5*self.N))

        ang_diff_angles = np.zeros(self.N)
        pos_rewards = np.zeros(self.N)
        new_positions = self.state[:,0:2].copy()

        for i in range(self.N):

            ang_diff_angles[i] = self.compute_angular_difference(i, np.arange(self.N))
            pos_rewards[i] = self.positional_rewards(i, new_positions)

        obs[0:(5*self.N):5] = self.state[:,0]
        obs[1:(5*self.N):5] = self.state[:,1]
        obs[2:(5*self.N):5] = self.state[:,2]
        obs[3:(5*self.N):5] = ang_diff_angles
        obs[4:(5*self.N):5] = pos_rewards

        return obs


    def compute_velocities(self, direction_angle):
        '''Compute velocity vector given the direction angles of 1 drone.'''

        velocity = np.zeros((2))

        velocity[0] = np.cos(direction_angle)
        velocity[1] = np.sin(direction_angle)

        return velocity
    

    def find_nearest(self, array, value):
        '''Finds element of array closest to given value.'''

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return array[idx]


    def compute_direction(self, vector):
        '''Computes the angle of a vector in the range (0,2pi].'''

        theta = np.arctan2(vector[1],vector[0])% (2*np.pi)

        return theta


    def update_drone_directions(self, i):
        '''Update the directions of the drones according to their velocities.'''

        theta = self.compute_direction(self.drone_velocities[i,:])
        new_angle = self.find_nearest(self.direction_angles, theta)

        self.drone_directions[i] = new_angle

    
    def update_periodic_boundaries(self,i):
        '''Updates the position of the i-th drone corrected for periodic boundaries'''

        self.state[i,0:2] = (self.state[i,0:2] - self.L) % self.L


    def update_drone_positions(self, i):
        '''Update the position of the ith drone.'''

        new_pos = self.state[i,0:2] + self.drone_velocities[i,:]

        self.old_state[i,0:2] = self.state[i,0:2]
        self.state[i,0:2] = new_pos

        self.update_periodic_boundaries(i)


    def compute_angles(self, i, actions):
        '''Compute turning angles from given actions.'''

        new_angles = np.zeros((self.N))

        action_index = np.argwhere(actions[i] == 1)[0][0]
        new_angles[i] = self.action_angles[action_index]

        return new_angles
        

    def update_drone_velocities(self, i, angle):
        '''Update the velocities of the i-th drone given the rotation angle.'''

        new_drone_velocities = np.zeros((2))

        new_drone_velocities[0] = self.drone_velocities[i,0]*np.cos(angle) - self.drone_velocities[i,1]*np.sin(angle) 
        new_drone_velocities[1] = self.drone_velocities[i,0]*np.sin(angle) + self.drone_velocities[i,1]*np.cos(angle) 

        self.drone_velocities[i,:] = new_drone_velocities
        # normalize the velocities
        self.drone_velocities[i,:] = self.drone_velocities[i,:] / np.linalg.norm(self.drone_velocities[i,:])


    def find_connected_drones(self, i, positions):
        '''For the i-th drone find the other drones that are within the connectivity range Rv.'''

        drones_in_range=[]

        for j in range(self.N):
            if j != i:
                if (positions[j,0] < positions[i,0] + self.Rv) and (positions[j,0] > positions[i,0] - self.Rv) and (positions[j,1] < positions[i,1] + self.Rv) and (positions[j,1] > positions[i,1] - self.Rv):
                    drones_in_range.append(j)

        return drones_in_range
    

    def collision_rewards(self, i, positions):
        '''Assign the collision rewards to the i-th drone.'''

        collision_reward_i = 0

        for j in np.arange(self.N):
            if j!=i:
                dist_x_ij = (positions[j,0]-positions[i,0] + self.L/2) % self.L - self.L/2
                dist_y_ij = (positions[j,1]-positions[i,1] + self.L/2) % self.L - self.L/2
                if np.sqrt((dist_x_ij)**2+(dist_y_ij)**2) < 1:
                    collision_reward_i -= (self.compactness_const - np.sqrt((dist_x_ij) ** 2+(dist_y_ij) ** 2))

        return collision_reward_i
    

    def update_order_parameter(self):
        '''Computes the order parameter of the entire system.'''

        self.order_param = np.sqrt(np.sum(self.drone_velocities[:,0])**2+np.sum(self.drone_velocities[:,1])**2)/self.N

    
    def order_param_reward(self):
        '''Update the order parameter and return this value.'''

        self.update_order_parameter()

        return self.order_param
    

    def cast_to_grid(self, i, positions):
        '''Casts i-th drone position to grid.'''
        positions[i,0] = positions[i,0] - (positions[i,0] % 1)
        positions[i,1] = positions[i,1] - (positions[i,1] % 1)

        return positions
    
    
    def positional_rewards(self, i, new_positions):
        '''Gives the positional reward value for the i-th drone based on its position.'''

        new_copied_positions = np.array(new_positions).copy()

        reward_positions = self.cast_to_grid(i, new_copied_positions)

        reward_i = self.reward_grid[int(reward_positions[i,0]), int(reward_positions[i,1])]
        if reward_i != 0:
            self.reward_grid[int(reward_positions[i,0]), int(reward_positions[i,1])] *= self.reward_decay

        self.grid_visits[int(reward_positions[i,1]), int(reward_positions[i,0])] +=1


        return reward_i


    def get_rewards(self):
        '''Total reward function, computes this for all drones.'''

        reward_N = []
        self.pos_rewards_N = []

        new_positions = self.state[:,0:2].copy()

        for i in range(self.N):

            order_param_reward_i = -(1-self.order_param_reward())
            position_reward_i = self.positional_rewards(i, new_positions)
            collision_reward_i = self.collision_rewards(i, new_positions)

            reward_i = (self.swarm_factor * order_param_reward_i) + (self.collision_factor * collision_reward_i) + position_reward_i

            reward_N.append(reward_i)
            self.pos_rewards_N.append(position_reward_i)

        return reward_N, self.pos_rewards_N
    

    def center_of_mass(self,positions):
        '''Computes the center of mass of a given position vector for all N drones.'''

        center_of_mass = (1/self.N)*np.sum(positions, axis=0)

        return center_of_mass


    def dispersion(self, positions):
        '''Computes the average dispersion of the system of all N drones at a given timestep.'''

        R_cm = self.center_of_mass(positions)

        # Minimum Image Convention to correct for periodic boundary conditions
        dist_x = (R_cm[0] - positions[:,0] + self.L/2) % self.L - self.L/2
        dist_y = (R_cm[1] - positions[:,1] + self.L/2) % self.L - self.L/2

        dispersion = np.sum(dist_x**2+dist_y**2)/self.N

        return dispersion


    def step(self, actions):
        '''Step function for all N actions for all N agents.'''

        for i in range(self.N):

            action_angle = self.action_angles[int(actions[i])]
            self.update_drone_velocities(i, action_angle)
            self.update_drone_positions(i)
            self.update_drone_directions(i)
            self.old_state[i,2] = self.state[i,2]
            self.state[i,2] = self.drone_directions[i]
        
        obs = self.get_obs()
        
        reward_N, self.pos_rewards_N = self.get_rewards()

        self.update_order_parameter()

        collective_reward  = np.sum(reward_N)/self.N
        self.avg_dispersion = self.dispersion(self.state[:,0:2])

        self.counter+=1
        if self.counter == self.max_timesteps:
            self.truncated = True

        return obs, collective_reward, self.done, self.truncated, {}
        

    def render(self):
        '''Procudes a snapshot of the environment at a single timestep.'''

        fig, ax = plt.subplots(figsize = (10,10))
        a=1/(self.L)

        # Draw initialization area A
        patch_A = plt.Polygon([[a*(self.origin_Ax), a*(self.origin_Ay)], [a*(self.origin_Ax+self.La_x), a*(self.origin_Ay)], [a*(self.origin_Ax+self.La_x), a*(self.origin_Ay+self.La_y)], [a*(self.origin_Ax), a*(self.origin_Ay+self.La_y)] ], fc = 'lightblue', zorder=3)
        ax.add_patch(patch_A)

        # Draw exploration area B
        for i in range(self.boundary_width, self.L-self.boundary_width):

            for j in range(self.boundary_width,self.L-self.boundary_width):
                patch_B = plt.Rectangle((a*i, a*j), width = a, height = a, fc = 'darkgreen', zorder=5, alpha= (self.reward_grid[i,j]/self.goal_reward))
                ax.add_patch(patch_B)
        
        boundary_X0 = plt.Polygon([[a*0, a*0], [a*self.boundary_width, a*0], [a*self.boundary_width, a*self.L], [a*0, a*self.L] ], fc = 'black', zorder=4)
        boundary_Xend = plt.Polygon([[a*(self.L-self.boundary_width), a*0], [a*self.L, a*0], [a*self.L, a*self.L], [a*(self.L-self.boundary_width), a*self.L] ], fc = 'black', zorder=4)
        boundary_Y0 = plt.Polygon([[a*0, a*0], [a*self.L, a*0], [a*self.L, a*self.boundary_width], [a*0, a*self.boundary_width] ], fc = 'black', zorder=4)
        boundary_Yend = plt.Polygon([[a*0, a*(self.L-self.boundary_width)], [a*self.L, a*(self.L-self.boundary_width)], [a*self.L, a*self.L], [a*0, a*self.L] ], fc = 'black', zorder=4)
        ax.add_patch(boundary_X0)
        ax.add_patch(boundary_Xend)
        ax.add_patch(boundary_Y0)
        ax.add_patch(boundary_Yend)

        # Draw grid
        for x in range(self.L):
            for y in range(self.L):
                pos=(a*x, a*y)
                width=a
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black', linewidth=0.2, zorder=13 )
                ax.add_patch(lattice)

        # Draw drones on grid
        for i in range(self.N):
                
                patch_drone = plt.Circle((a*self.state[i,0], a*self.state[i,1]), 0.5*a, fc = 'darkblue', zorder=10)
                ax.add_patch(patch_drone)
                patch_drone_dir = plt.arrow(a*self.state[i,0], a*self.state[i,1], a*self.drone_velocities[i,0], a*self.drone_velocities[i,1], color='red', zorder=11)
                ax.add_patch(patch_drone_dir)

        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.title(f"timestep {self.counter}")
        plt.axis('off')
        plt.show()


    def close(self):
        self.state = None


if __name__ == "__main__":

    # Test scenario

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
    swarm_factor = 1
    collision_factor = 1
    compactness_const = 1
    reward_decay = 0.75

    n_episodes = 1000
    eval_eps = 100

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
    
    env = DronesEnvironment(settings=settings, render_mode='rgb_array')

    obs_0, info = env.reset()
    env.render()

    for i in range(300):

        actions = np.random.randint(0,k_a, size=N)
        obs, reward, done, trunc, info = env.step(actions)
        env.render()
        if trunc:
            obs_0, info = env.reset()

    env.close()



    
