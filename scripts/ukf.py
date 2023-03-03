# %%
import sys
sys.path.append('../scripts/')
from ekf import *
from scipy.linalg import cholesky

# %%
class UnscentedKalmanFilter:
    def __init__(self, init_pose, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}, \
                 distance_dev_rate=0.05, x_dev=0.25, y_dev=0.25, theta_dev=0.05, \
                 dim_x=3, dim_u=2, dim_z=3, alpha=10e-3, beta=2., \
                 rejection=True, rejection_threshold=0.001):
        self.belief = multivariate_normal(mean=init_pose, cov=np.diag([1e-10, 1e-10, 1e-10])) 
        self.pose = self.belief.mean
        self.motion_noise_stds = motion_noise_stds
        self.distance_dev_rate = distance_dev_rate
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.theta_dev = theta_dev
        self.rejection = rejection
        self.rejection_threshold = rejection_threshold
        
        # set up ukf
        ## setting dimensions
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.dim_a = dim_x + dim_u + dim_z
        
        ## setting number of sigma points to be generated
        self.n_sigma = (2*self.dim_a) + 1
        
        ## setting scaling parameters
        self.kappa = 3 - self.dim_a
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = self.alpha**2 * (self.dim_a + self.kappa) - self.dim_a

        ## setting scale coefficient for selecting sigma points
        self.sigma_scale = np.sqrt(self.dim_a + self.kappa)
        
        ## calculate unscented weights
        self.W0 = self.kappa / (self.dim_a + self.kappa)
        self.Wi = 0.5 / (self.dim_a + self.kappa)
        
        ## initializing augmented state x_a and augmented covariance P_a
        self.x_a = np.zeros((self.dim_a, ))
        self.P_a = np.zeros((self.dim_a, self.dim_a))
        
        self.idx1, self.idx2 = dim_x, dim_x + dim_u
        self.P_a[:self.idx1, :self.idx1] = self.belief.cov
        self.P_a[self.idx1:self.idx2, self.idx1:self.idx2] = np.diag([1e-10, 1e-10])
        self.P_a[self.idx2:, self.idx2:] = np.diag([1e-10, 1e-10, 1e-10])
    
    def sigma_points(self, x, P):
        nx = np.shape(x)[0]
        x_sigma = np.zeros((nx, self.n_sigma))  
        x_sigma[:, 0] = x
        S = cholesky(P)
        for i in range(nx):
            x_sigma[:, i + 1] = x + (self.sigma_scale * S[:, i])
            x_sigma[:, i + nx + 1] = x - (self.sigma_scale * S[:, i])     
        return x_sigma

    def calculate_mean_and_covariance(self, y_sigmas):
        ydim = np.shape(y_sigmas)[0]
        # mean calculation
        y = self.W0 * y_sigmas[:, 0]
        for i in range(1, self.n_sigma):
            y += self.Wi * y_sigmas[:, i]
        # covariance calculation
        d = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pyy = self.W0 * d.dot(d.T)
        for i in range(1, self.n_sigma):
            d = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pyy += self.Wi * d.dot(d.T)
        return y, Pyy

    def calculate_cross_correlation(self, x, x_sigmas, y, y_sigmas):
        xdim = np.shape(x)[0]
        ydim = np.shape(y)[0]
        n_sigmas = np.shape(x_sigmas)[1]
        dx = (x_sigmas[:, 0] - x).reshape([-1, 1])
        dy = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pxy = self.W0 * dx.dot(dy.T)
        for i in range(1, n_sigmas):
            dx = (x_sigmas[:, i] - x).reshape([-1, 1])
            dy = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pxy += self.Wi * dx.dot(dy.T)
        return Pxy

    def outlier(self, observation):
        delta = self.pose - observation
        dist = (delta).T.dot(np.linalg.inv(self.belief.cov)).dot(delta)
        if dist <= chi2.ppf(1.0-self.rejection_threshold, 3):
            return False
        else:
            return True
    
    def observation_update(self, observation):
        if observation is None: return
        if self.rejection: 
            if self.outlier(observation): return
        gx, gy, _ = observation
        ex, ey, _ = self.pose
        dx = ex - gx
        dy = ey - gy
        dist = math.sqrt(dx**2 + dy**2)
        self.theta_dev += self.distance_dev_rate * dist
        self.x_a[:self.dim_x] = self.belief.mean
        self.P_a[:self.dim_x, :self.dim_x] = self.belief.cov
        self.P_a[self.idx2:, self.idx2:]  = matQ(self.x_dev, self.y_dev, self.theta_dev)
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)
        xx_sigmas = xa_sigmas[:self.dim_x, :]
        xz_sigmas = xa_sigmas[self.idx2:, :]
        y_sigmas = np.zeros((self.dim_z, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = np.array([xx_sigmas[:, i][0]+xz_sigmas[:, i][0], 
                                       xx_sigmas[:, i][1]+xz_sigmas[:, i][1], xx_sigmas[:, i][2]+xz_sigmas[:, i][2]])
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
        Pxy = self.calculate_cross_correlation(self.belief.mean, xx_sigmas, y, y_sigmas)
        K = Pxy.dot(np.linalg.pinv(Pyy))
        self.belief.mean += K.dot(observation - y)
        self.belief.cov -= K.dot(Pyy).dot(K.T)
        self.pose = self.belief.mean
    
    def motion_update(self, nu, omega, time):
        if abs(omega) < 1e-5: omega = 1e-5
        self.x_a[:self.dim_x] = self.belief.mean
        self.P_a[:self.dim_x, :self.dim_x] = self.belief.cov
        self.P_a[self.idx1:self.idx2, self.idx1:self.idx2] = matM(nu, omega, time, self.motion_noise_stds)
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)
        xx_sigmas = xa_sigmas[:self.dim_x, :]
        xu_sigmas = xa_sigmas[self.idx1:self.idx2, :]
        y_sigmas = np.zeros((self.dim_x, self.n_sigma))
        for i in range(self.n_sigma):
            pnu = nu + xu_sigmas[:, i][0]
            pomega = omega + xu_sigmas[:, i][1]
            y_sigmas[:, i] = IdealRobot.state_transition(pnu, pomega, time, xx_sigmas[:, i])
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
        self.x_a[:self.dim_x] = y
        self.P_a[:self.dim_x, :self.dim_x] = Pyy
        self.belief.mean, self.belief.cov = y, Pyy
        self.pose = self.belief.mean

    def draw(self, ax, elems):
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))
        x, y, c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2, 2])*3
        xs = [x + math.cos(c-sigma3), x, x + math.cos(c+sigma3)]
        ys = [y + math.sin(c-sigma3), y, y + math.sin(c+sigma3)]
        elems += ax.plot(xs, ys, color="blue", alpha=0.5)

# %%
if __name__ == '__main__': 
    time_interval = 0.1
    world = World(30, time_interval, debug=False)        

    ### ロボットを作る ###
    initial_pose = np.array([0, 0, 0]).T
    ukf = UnscentedKalmanFilter(initial_pose)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, ukf)
    r = Robot(initial_pose, gnss=Gnss(time_interval, hz=1), agent=circling, color="red")
    world.append(r)

    world.draw()


