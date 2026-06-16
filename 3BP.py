import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

k_e = 1.0       # Coulomb stala 
epsilon = 0.1   # epsilon


# masa
m = np.array([1.0, 1.0, 1.0])


# ladunki
q = np.array([1.0, -2.0, 1.0])

# pierwsze pozycje
r0 = np.array([
    [ 1.0,  0.0,  0.0],  
    [-1.0,  0.0,  0.0], 
    [ 0.0,  1.0,  0.5]  
])


# pierwsze predkości
v0 = np.array([
    [ 0.0,  0.5, -0.1],
    [ 0.0, -0.5,  0.1],  
    [-0.5,  0.0,  0.0]
])


state0 = np.concatenate((r0.flatten(), v0.flatten()))

def coulomb_derivatives(t, state, m, q):

    r = state[:9].reshape((3, 3))
    v = state[9:]
    
    a = np.zeros((3, 3)) 
    
    
    for i in range(3):
        for j in range(3):
            if i != j:
                dr = r[i] - r[j] 
                dist_sq = np.sum(dr**2) + epsilon**2
                dist = np.sqrt(dist_sq)
                
                
                a[i] -= (k_e * q[i] * -q[j] / m[i]) * (dr / dist**3)
                
    return np.concatenate((v, a.flatten()))

# czas trwania symulacji
t_span = (0, 20)  
t_eval = np.linspace(t_span[0], t_span[1], 1000) 

solution = solve_ivp(
    coulomb_derivatives, 
    t_span, 
    state0, 
    args=(m, q), 
    t_eval=t_eval, 
    method='RK45'
)


r_sol = solution.y[:9].reshape((3, 3, -1))


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# granice symulacji
axis_limit = 10
ax.set_xlim([-axis_limit, axis_limit])
ax.set_ylim([-axis_limit, axis_limit])
ax.set_zlim([-axis_limit, axis_limit])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D 3-Body Coulomb Simulation")

colors = []
for charge in q:
    if charge > 0: colors.append('red')
    elif charge < 0: colors.append('blue')
    else: colors.append('gray')


points = [ax.plot([], [], [], 'o', color=c, markersize=8)[0] for c in colors]
trails = [ax.plot([], [], [], '-', color=c, alpha=0.4)[0] for c in colors]

def init():
    for p, t in zip(points, trails):
        p.set_data([], [])
        p.set_3d_properties([])
        t.set_data([], [])
        t.set_3d_properties([])
    return points + trails

def update(frame):

    for i in range(3):
  
        points[i].set_data([r_sol[i, 0, frame]], [r_sol[i, 1, frame]])
        points[i].set_3d_properties([r_sol[i, 2, frame]])
        

        start_idx = max(0, frame - 100)
        trails[i].set_data(r_sol[i, 0, start_idx:frame], r_sol[i, 1, start_idx:frame])
        trails[i].set_3d_properties(r_sol[i, 2, start_idx:frame])
        

    ax.view_init(elev=20., azim=frame/5.0)

    return points + trails


ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=False, interval=20)
plt.show()