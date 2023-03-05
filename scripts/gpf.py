# %%
import sys
sys.path.append('../scripts/')
from mcl import *
from scipy.linalg import sqrtm

# %%
class GaussianParticleFilter(Mcl):
    
    def __init__(self, init_pose, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}, \
                 distance_dev_rate=0.14, direction_dev=0.05, x_dev=0.05, y_dev=0.05, theta_dev=0.05):
        
        super().__init__(init_pose, num, motion_noise_stds, distance_dev_rate, direction_dev, x_dev, y_dev, theta_dev)
        
    def weight_normalize(self):
        weight_sum = np.sum([e.weight for e in self.particles])
        if weight_sum < 1e-100:
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
        else:
            for p in self.particles:
                p.weight /= weight_sum
    
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.distance_dev_rate, self.direction_dev,
                                 self.x_dev, self.y_dev, self.theta_dev)
        self.set_ml()
        self.weight_normalize()
        x_est = sum([p.pose.dot(p.weight) for p in self.particles])
        cov_est = sum([p.weight * np.outer((p.pose - x_est), (p.pose - x_est)) for p in self.particles])
        cov_sqrt = sqrtm(cov_est)
        diag_cov_sqrt = np.diag(cov_sqrt)
        for p in self.particles:
            p.pose = x_est + diag_cov_sqrt * np.random.randn(x_est.shape[0])
            p.weight = 1.0 / len(self.particles)

# %%
if __name__ == '__main__':
    time_interval = 0.1
    world = World(30, time_interval, debug=False)   

    initial_pose = np.array([0, 0, 0]).T
    estimator = GaussianParticleFilter(initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, gnss=Gnss(time_interval, hz=1), agent=a, color="red")
    world.append(r)

    world.draw()


