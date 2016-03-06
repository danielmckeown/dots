#My program is largely object oriented and creates as many particles 
#as desired of mass M inside a bounding box of a given size. All of 
#these variables are fully and infinitely adjustable. The density 
#calculation is accomplished by creating a radial distance calculation, 
#and then summing all the particles contained within each radii, then 
#multiplying them by the mass M, and then finally, dividing them by the 
#volume of the space found accordingly. 




import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


# Create particle space in which to place them
class ParticleBox:
    

   # bounds of box: [xmin, xmax, ymin, ymax]
    
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-5, 5, -5, 5],
                 size = 0.04,
                 M = 0.05,
                 G = 0):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
     
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]




#------------------------------------------------------------
# Add particles
M = 0.05
little_r = 0
big_r = 1
count = 0

while( count < 5 ):
    np.random.seed(0)
    mass_coordinates = -0.5 + np.random.random((10000, 3))
    p = len(mass_coordinates)
    mass_coordinates[:, :3] *= 10
    #print mass_coordinates
    box = ParticleBox(mass_coordinates, size=0.0090)
    dt = 1. / 30 # 30fps
    dx = mass_coordinates[:,0] 
    dy = mass_coordinates[:,1] 
    dz = mass_coordinates[:,2]
    rr = np.sqrt(dx**2 + dy**2 + dz**2)
    #print rr
    #print len(rr)
    w = np.where ((rr >= little_r) & (rr < big_r) )
    #print w
    mass = len(w[0]) * (M)
    #print mass
    area = 4/3*3.14*(big_r**3) - 4/3*3.14*(little_r**3)
    
    mass_density = (mass/area)
    print "mass density from" 
    print big_r 
    print "to" 
    print little_r 
    print "is" 
    print mass_density
    little_r = little_r + 1
    big_r = big_r + 1
    count = count + 1




#------------------------------------------------------------
# set up figure and animation

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-5,5), ylim=(-5, 5))

# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=6)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize boxes and particles"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect

def animate(i):
    """this step if I want them to move"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=False, init_func=init)
    
plt.show()
