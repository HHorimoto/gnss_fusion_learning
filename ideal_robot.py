# %%
import matplotlib
from IPython.display import HTML
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

# %%
class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []  
        self.debug = debug
        self.time_span = time_span  
        self.time_interval = time_interval 
        
    def append(self,obj):  
        self.objects.append(obj)
    
    def draw(self): 
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')             
        ax.set_xlim(-5,5)                  
        ax.set_ylim(-5,5) 
        ax.set_xlabel("X",fontsize=10)                 
        ax.set_ylabel("Y",fontsize=10)                 
        
        elems = []
        
        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            self.ani.save("result.gif", writer='imagemagick')
            plt.close()
            return HTML(self.ani.to_jshtml())
        
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)    

# %%
class IdealRobot:   
    def __init__(self, pose, agent=None, gnss=None, color="black"):    # 引数を追加
        self.pose = pose
        self.r = 0.2  
        self.color = color 
        self.agent = agent
        self.poses = [pose]
        self.gnss = gnss
    
    def draw(self, ax, elems):         ### call_agent_draw
        x, y, theta = self.pose  
        xn = x + self.r * math.cos(theta)  
        yn = y + self.r * math.sin(theta)  
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.gnss and len(self.poses) > 1:
            self.gnss.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"): 
            self.agent.draw(ax, elems)
         
    @classmethod           
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega]) * time
        else:
            return pose + np.array([nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time])

    def one_step(self, time_interval):
        if not self.agent: return
        elif self.gnss:
            obs = self.gnss.data(self.pose)
        else:
            obs = None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

# %%
class Agent: 
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        
    def decision(self, observation=None):
        return self.nu, self.omega

# %%
class IdealGnss:
    def __init__(self, time_interval, hz=1):
        self.time_interval = time_interval
        self.hz = hz
        self.count = 0
        self.t = int((self.time_interval/self.hz)*100)
        self.lastdata = None
    
    def visible(self):
        if self.count == self.t:
            self.count = 0
            return True
        else:
            self.count += 1
            return False
    
    def data(self, pose):
        if self.visible():
            self.lastdata = pose
            return pose
        else:
            self.lastdata = None
            return None

    def draw(self, ax, elems, pose):
        if self.lastdata is not None:
            x, y, theta = self.lastdata
            p = ax.quiver(x, y, math.cos(theta), math.sin(theta), angles='xy', scale_units='xy', scale=1.5, color="green", alpha=1.0)
            elems.append(p)

# %%
if __name__ == '__main__':
    world = World(30, 0.1)         

    ### ロボットを作る ###
    straight = Agent(0.2, 0.0)    
    circling = Agent(0.2, 10.0/180*math.pi)  
    robot1 = IdealRobot(np.array([2, 3, math.pi/6]).T, gnss=IdealGnss(0.1), agent=straight)
    robot2 = IdealRobot(np.array([-2, -1, math.pi/5*6]).T, gnss=IdealGnss(0.1), agent=circling, color="red")
    world.append(robot1)
    world.append(robot2)

    ### アニメーション実行 ###
    world.draw()


