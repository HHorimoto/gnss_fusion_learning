# %%
from ideal_robot import *
from scipy.stats import expon, norm, uniform

# %%
class Robot(IdealRobot):
        
    def __init__(self, pose, agent=None, gnss=None, color="black", \
                           noise_per_meter=5, noise_std=math.pi/60, bias_rate_stds=(0.1,0.1), \
                           expected_stuck_time=1e100, expected_escape_time = 1e-100,\
                           expected_kidnap_time=1e100, kidnap_range_x = (-5.0,5.0), kidnap_range_y = (-5.0,5.0)): #追加
        super().__init__(pose, agent, gnss, color)
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1]) 
        
        self.stuck_pdf = expon(scale=expected_stuck_time) 
        self.escape_pdf = expon(scale=expected_escape_time)
        self.is_stuck = False
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        
        self.kidnap_pdf = expon(scale=expected_kidnap_time) 
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi ))
        
    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()
            
        return pose
        
    def bias(self, nu, omega): 
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega
    
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:            
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck)
    
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
            
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.gnss.data(self.pose) if self.gnss else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval) 

# %%
class Gnss(IdealGnss):
    def __init__(self, time_interval, hz=1,
                 x_noise_stddev=0.2, y_noise_stddev=0.2, theta_noise_stddev=0.2, distance_noise_rate=0.1,
                 oversight_prob=0.1, phantom_prob=0.05, 
                 phantom_range_x=(-5.0,5.0), phantom_range_y=(-5.0,5.0), phantom_range_theta=(-180.0+1e-100,180.0)):
        super().__init__(time_interval, hz)
        
        self.x_noise_stddev = x_noise_stddev
        self.y_noise_stddev = y_noise_stddev
        self.theta_noise_stddev = theta_noise_stddev
        self.distance_noise_rate = distance_noise_rate
        
        self.oversight_prob = oversight_prob

        self.phantom_x = uniform(phantom_range_x[0], phantom_range_x[1])
        self.phantom_y = uniform(phantom_range_y[0], phantom_range_y[1])
        self.phantom_theta = uniform(phantom_range_theta[0], phantom_range_theta[1])
        self.phantom_prob = phantom_prob
        
        self.lastdata = None
    
    def phantom(self, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array([self.phantom_x.rvs(), self.phantom_y.rvs(), self.phantom_theta.rvs()]).T
            return pos
        else:
            return relpos
    
    def noise(self, relpos):
        x = norm.rvs(loc=relpos[0], scale=self.x_noise_stddev)
        y = norm.rvs(loc=relpos[1], scale=self.y_noise_stddev)
        theta = norm.rvs(loc=relpos[2], 
                         scale=self.theta_noise_stddev+(math.sqrt(math.pow(relpos[0]-x, 2) + math.pow(relpos[1]-y, 2))*self.distance_noise_rate))
        return np.array([x, y, theta]).T
        
    def oversight(self, relpose):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpose
        
    def data(self, pose):
        z = self.oversight(pose)
        z = self.phantom(pose)
        if z is None:
            self.lastdata = None
            return None
        else:
            if self.visible():
                z = self.noise(z)
                self.lastdata = z
                return z
            else:
                self.lastdata = None
                return None
    
    def draw(self, ax, elems, pose):
        if self.lastdata is not None:
            x, y, theta = self.lastdata
            p = ax.quiver(x, y, math.cos(theta), math.sin(theta), angles='xy', scale_units='xy', scale=1.0, color="green", alpha=1.0)
            elems.append(p)        
        

# %%
if __name__ == '__main__': 
    world = World(30, 0.1, debug=False)     

    ### ロボットを作る ###
    straight = Agent(0.2, 0.0)    
    circling = Agent(0.2, 10.0/180*math.pi)  
    r = Robot(np.array([2, 2, math.pi/6]).T, gnss=Gnss(0.1), agent=circling) 
    world.append(r)

    ### アニメーション実行 ###
    world.draw()


