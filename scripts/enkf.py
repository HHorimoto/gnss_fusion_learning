# %%
import sys
sys.path.append('../scripts/')
from ekf import *

# %%
class EnKFParticle:
    def __init__(self, init_pose):
        self.pose = init_pose
        self.observation = init_pose
        
    def motion_update(self, nu, omega, time, motion_noise_stds):
        if abs(omega) < 1e-5: omega = 1e-5
        M = matM(nu, omega, time, motion_noise_stds)
        pnu = nu + np.random.normal(0, M[0, 0])
        pomega = omega + np.random.normal(0, M[1, 1])
        self.pose = IdealRobot.state_transition(pnu, pomega, time, self.pose)
        
    def observation_update(self, observation, distance_dev_rate, direction_dev,
                           x_dev, y_dev, theta_dev):
        gx, gy, _ = observation
        ex, ey, _ = self.pose
        dx = ex - gx
        dy = ey - gy
        dist = math.sqrt(dx**2 + dy**2)
        theta_dev += distance_dev_rate * dist
        Q = matQ(x_dev, y_dev, theta_dev)
        self.observation = np.array([observation[0]+np.random.normal(0, Q[0, 0]), 
                                     observation[1]+np.random.normal(0, Q[1, 1]), observation[2]+np.random.normal(0, Q[2, 2])])

# %%
class EnsembleKalmanFilter:
    def __init__(self, init_pose, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}, \
                 distance_dev_rate=0.05, x_dev=0.25, y_dev=0.25, theta_dev=0.05, direction_dev=0.05):
        self.particles = [EnKFParticle(init_pose) for i in range(num)]
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev
        
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.theta_dev = theta_dev

        self.motion_noise_stds = motion_noise_stds
        self.pose = np.mean([p.pose for p in self.particles], axis=0)
    
    def set_pose(self):
        self.pose = np.mean([p.pose for p in self.particles], axis=0)
    
    def motion_update(self, nu, omega, time): 
        for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_stds)
    
    def observation_update(self, observation):
        if observation is not None:
            for p in self.particles:
                p.observation_update(observation, self.distance_dev_rate, self.direction_dev,
                                    self.x_dev, self.y_dev, self.theta_dev)
                            
            x_poses = np.array([p.pose for p in self.particles])
            x_mean = np.mean(x_poses, axis=0)
            x_dif = (x_poses - x_mean).T
            
            z_observations = np.array([p.observation for p in self.particles])
            z_mean = np.mean(z_observations, axis=0)
            z_dif = (z_observations - z_mean).T
            
            V = 1/(len(self.particles)-1) * z_dif.dot(z_dif.T)
            U = 1/(len(self.particles)-1) * x_dif.dot(z_dif.T)
            K = U.dot(np.linalg.inv(V))
            for p in self.particles:
                p.pose += K.dot(observation-p.observation)

        self.set_pose()

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, \
                               angles='xy', scale_units='xy', scale=1.5, color="blue", alpha=0.3))

# %%
if __name__ == '__main__':
    time_interval = 0.1
    world = World(30, time_interval, debug=False)   

    initial_pose = np.array([0, 0, 0]).T
    estimator = EnsembleKalmanFilter(initial_pose, 20)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, gnss=Gnss(time_interval, hz=1), agent=a, color="red")
    world.append(r)

    world.draw()


