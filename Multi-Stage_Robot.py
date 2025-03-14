# %% Import libraries
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import control
import do_mpc
from copy import deepcopy
from matplotlib.patches import Ellipse
from sample_utils import EllipsoidSampler as EllSamp
import time as time
# %%
def confidence_ellipse(center,cov, ax,dim=[0,1], facecolor='none', **kwargs):
    """
    Create a plot of the ellipsoid (x-c)^T E^-1 (x-c) <= 1
    
    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics
    
    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    cov : Ellipsoid matrix, symmetric, of the for x
    dim : dimensions to be used for the plot
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    dim0 = dim[0]
    dim1 = dim[1]
    cov = cov[dim,:][:,dim]
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) 
    mean_x = center[dim0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) 
    mean_y = center[dim1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# %% system model robot and human
dt = 0.5
nx = 7
nu = 2
nw = 2




v_nom_h = -0.6
v_ref=1.5
W_cov = 0.4**2*np.eye(nw)
# W_cov[5,5]=0.4**2
# W_cov[6,6]=0.4**2
# continuous time system




rhs = ca.SX.zeros(nx,1)
x = ca.SX.sym('x',nx)
u = ca.SX.sym('u',nu)
w = ca.SX.sym('w',nw)
px = x[0]
py = x[1]
theta = x[2]
v = x[3]
omega = x[4]
px_h = x[5]
py_h = x[6]
rhs[0] = v*ca.cos(theta)# + w[0]
rhs[1] = v*ca.sin(theta)# + w[1]
rhs[2] = omega# + w[2]
rhs[3] = u[0]# + w[3]
rhs[4] = u[1]# + w[4]

rhs[5] = v_nom_h + w[0]
rhs[6] = 0 + w[1]

system_continuous = ca.Function('sys', [x, u, w], [rhs])
# Discretize via RK 4
k1 = system_continuous(x, u, w)
k2 = system_continuous(x + dt/2 * k1, u, w)
k3 = system_continuous(x + dt/2 * k2, u, w)
k4 = system_continuous(x + dt * k3, u, w)
x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
system = ca.Function('system', [x, u, w], [x_next])



# Define the system
model = do_mpc.model.Model('discrete')

# Define the states
px = model.set_variable(var_type='_x', var_name='px')
py = model.set_variable(var_type='_x', var_name='py')
theta = model.set_variable(var_type='_x', var_name='theta')
v = model.set_variable(var_type='_x', var_name='v')
omega = model.set_variable(var_type='_x', var_name='omega')
px_h = model.set_variable(var_type='_x', var_name='px_h')
py_h = model.set_variable(var_type='_x', var_name='py_h')
x= ca.vertcat(px, py, theta, v, omega, px_h, py_h)

# Define the control inputs
a = model.set_variable(var_type='_u', var_name='a')
omega_c = model.set_variable(var_type='_u', var_name='omega_c')
u = ca.vertcat(a, omega_c)

# Define the uncertainties
w1 = model.set_variable(var_type='_p', var_name='w1')
w2 = model.set_variable(var_type='_p', var_name='w2')
w = ca.vertcat(w1, w2)

# Define the reference trajectory
x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(nx,1))

# Define expression for safety constr
expr = ca.vertcat(px-px_h, py-py_h)
model.set_expression(expr=ca.sqrt(expr.T@expr), expr_name='Distance')
# Define expression for CLC
Q = np.diag([50, 50, 0, 2, 0,0 ,0])
R = np.diag([2, 2])
model.set_expression(expr = (x-x_ref).T@Q@(x-x_ref) + u.T@R@u, expr_name='CLC')

# Define the model equations
for i,key in enumerate(model.x.keys()):
    model.set_rhs(key, system(model.x, model.u, model.p)[i])

model.setup()
# %%
# Build multi-stage MPC
N=10
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': N,
    't_step': dt,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# Define the objective function
_x = ca.vertcat(model.x)
_u= ca.vertcat(model.u)
del_x = ca.vertcat(_x - x_ref)

lterm = del_x.T @ Q @ del_x + _u.T @ R @ _u
mterm = del_x.T @ Q @ del_x 
mpc.set_objective(mterm=mterm, lterm=lterm)

# Constraints
# Box constraints
mpc.bounds['lower','_x', 'py'] = -1.3
mpc.bounds['upper','_x', 'py'] = 1.3
mpc.bounds['lower','_x', 'v'] = 0
mpc.bounds['upper','_x', 'v'] = 2
mpc.bounds['lower','_x', 'omega'] = -1
mpc.bounds['upper','_x', 'omega'] = 1

# Terminal box constraints
mpc.terminal_bounds['lower', 'v'] = 0
mpc.terminal_bounds['upper', 'v'] = 1e-2

# Safety distance
Del_safe = 0.3
expr = ca.vertcat(px-px_h, py-py_h)
mpc.set_nl_cons('Safety_dist',expr=-ca.sqrt(expr.T@expr) + Del_safe, ub=0)

# Uncertain values
w1_var = np.array([-0.4**2,0, 0.4**2])
w2_var = np.array([-0.4**2,0, 0.4**2])
mpc.set_uncertainty_values(w1=w1_var, w2=w2_var)

# Setup the tvp template
x_init = ca.DM([-2, 0, 0, v_ref, 0, 3.5, 0])
tvp_template = mpc.get_tvp_template()

def tvp_fun(t):
    tvp_template['_tvp', 0, 'x_ref'] = x_init+ca.DM([t*v_ref,0,0,0,0,0,0])

    for i in range(1,N+1):
        tvp_template['_tvp', i, 'x_ref'] = tvp_template['_tvp', i-1, 'x_ref']+ca.DM([dt*v_ref,0,0,0,0,0,0])
    return tvp_template
mpc.set_tvp_fun(tvp_fun)

mpc.setup()

# %% 
# Test the MPC
x_init = np.array([-2, 0.0, 0, 1.6, 0, 3.5, 0])
mpc.x0 = x_init
mpc.set_initial_guess()
u0=mpc.make_step(x_init)

# %%
# PLot prediction
graphics = do_mpc.graphics.Graphics(mpc.data)
fig, ax = plt.subplots(3,2, figsize=(10,10))
graphics.add_line(var_type='_x', var_name='px', axis=ax[0,0],marker='o')
graphics.add_line(var_type='_x', var_name='py', axis=ax[0,1],marker='o')
graphics.add_line(var_type='_x', var_name='v', axis=ax[1,0])
graphics.add_line(var_type='_x', var_name='omega', axis=ax[1,1])
graphics.add_line(var_type='_x', var_name='px_h', axis=ax[0,0],marker='x')
graphics.add_line(var_type='_x', var_name='py_h', axis=ax[0,1],marker='x')
graphics.add_line(var_type='_aux', var_name='Distance', axis=ax[2,0])
graphics.add_line(var_type='_aux', var_name='CLC', axis=ax[2,1])
graphics.plot_predictions(t_ind=0)
for i in range(2):
    for j in range(3):
        ax[j,i].set_xlim([0, N*dt])
plt.show()
# %%
# Run closed loop
# Setup simulator
sim = do_mpc.simulator.Simulator(model)
sim.set_param(t_step=dt)
p_template = sim.get_p_template()
d, V= np.linalg.eig(W_cov)
D = np.diag(d)
Ell = EllSamp(np.zeros((nw,)), d, V)
np.random.seed(123)
def p_fun(t_now):
    #np.random.seed(0+int(t_now*20))
    w = Ell.sample()
    #print(w)
    p_template['w1'] = w[0]
    p_template['w2'] = w[1]
    return p_template
sim.set_p_fun(p_fun)
tvp_template_sim = sim.get_tvp_template()
def sim_tvp_fun(t_now):
    return tvp_template_sim
sim.set_tvp_fun(sim_tvp_fun)
sim.setup()


#
mpc.reset_history()
mpc.x0 = x_init
mpc.set_initial_guess()

sim.reset_history()
sim.x0 = x_init
x0=deepcopy(x_init)
for k in range(15):
    u0 = mpc.make_step(x0)
    x0 = sim.make_step(u0)

# %%
# Plot closed loop
fig, ax = plt.subplots(3,2, figsize=(10,10))
graphics = do_mpc.graphics.Graphics(mpc.data)
graphics.add_line(var_type='_x', var_name='px', axis=ax[0,0])
graphics.add_line(var_type='_x', var_name='py', axis=ax[0,1],)
graphics.add_line(var_type='_x', var_name='v', axis=ax[1,0])
graphics.add_line(var_type='_x', var_name='omega', axis=ax[1,1])
graphics.add_line(var_type='_x', var_name='px_h', axis=ax[0,0],marker='x')
graphics.add_line(var_type='_x', var_name='py_h', axis=ax[0,1],marker='x')
graphics.add_line(var_type='_aux', var_name='Distance', axis=ax[2,0])
graphics.add_line(var_type='_aux', var_name='CLC', axis=ax[2,1])
graphics.plot_results()
graphics.plot_predictions()
for i in range(2):
    for j in range(3):
        ax[j,i].set_xlim([0, 2*N*dt])

plt.show()

# %% Show specific time step
idx=12
graphics = do_mpc.graphics.Graphics(mpc.data)
fig, ax = plt.subplots(3,2, figsize=(10,10))
graphics.add_line(var_type='_x', var_name='px', axis=ax[0,0])
graphics.add_line(var_type='_x', var_name='py', axis=ax[0,1])
graphics.add_line(var_type='_x', var_name='v', axis=ax[1,0])
graphics.add_line(var_type='_x', var_name='omega', axis=ax[1,1])
graphics.add_line(var_type='_x', var_name='px_h', axis=ax[0,0])
graphics.add_line(var_type='_x', var_name='py_h', axis=ax[0,1])
graphics.add_line(var_type='_aux', var_name='Distance', axis=ax[2,0])
#graphics.add_line(var_type='_aux', var_name='CLC', axis=ax[2,1])
ax[2,0].axhline(Del_safe, color='r', linestyle='--')

graphics.plot_results(t_ind=idx)
graphics.plot_predictions(t_ind=idx)
# plot state space
#Robot
x_res= graphics.result_lines['_x','px'][0].get_ydata()
y_res= graphics.result_lines['_x','py'][0].get_ydata()
for nom_idx in range(len(graphics.pred_lines['_x','px',0])):
    x_pred= graphics.pred_lines['_x','px',0,nom_idx][0].get_ydata() #Nominal prediction
    y_pred= graphics.pred_lines['_x','py',0,nom_idx][0].get_ydata() # Nominal prediction
    ax[2,1].plot(x_pred,y_pred,'--',color='tab:blue',marker='o',label='Robot Prediction')
ax[2,1].plot(x_res[0:idx+1],y_res[0:idx+1],color='tab:blue',marker='o',label='Robot')

#Human
x_res= graphics.result_lines['_x','px_h'][0].get_ydata()
y_res= graphics.result_lines['_x','py_h'][0].get_ydata()
#nom_idx =1
for nom_idx in range(len(graphics.pred_lines['_x','px_h',0])):
    x_pred= graphics.pred_lines['_x','px_h',0,nom_idx][0].get_ydata() #Nominal prediction
    y_pred= graphics.pred_lines['_x','py_h',0,nom_idx][0].get_ydata() # Nominal prediction
    ax[2,1].plot(x_pred,y_pred,'--',color='tab:orange',marker='x',label='Human Prediction')
ax[2,1].plot(x_res[0:idx+1],y_res[0:idx+1],color='tab:orange',marker='x',label='Human')

for i in range(2):
    for j in range(3):
        if j==2 and i==1:
            continue
        ax[j,i].set_xlim([0, (N+idx)*dt])

plt.show()
# %%
# Run 50 closed loop simulations
np.random.seed(123)
n_run = 50
N_sim = 10
data = []
for j in range(n_run):
    
   
    sim.reset_history()
    sim.x0 = x_init
    x0=deepcopy(x_init)
    mpc.reset_history()
    mpc.x0 = x_init
    mpc.set_initial_guess()
    time0 =time.time()
    for k in range(N_sim):
        u0 = mpc.make_step(x0)
        x0 = sim.make_step(u0)
        
        
    dat=dict()
    dat['mpc']=deepcopy(mpc.data)
    dat['sim']=deepcopy(sim.data)
    dat['clc']=mpc.data['_aux','CLC',0]
    dat['sol_info']=mpc.data['success']
    dat['time'] = time.time()-time0
    data.append(dat)



# %% Plot the results
fig, ax = plt.subplots(3,2, figsize=(10,10))
fig2, ax2 = plt.subplots(3,1, figsize=(4,6))
for j in range(n_run):
    graphics = do_mpc.graphics.Graphics(data[j]['mpc'])
    graphics.add_line(var_type='_x', var_name='px', axis=ax[0,0],linewidth=0.5,color='b')
    graphics.add_line(var_type='_x', var_name='py', axis=ax[0,1],linewidth=0.5,color='b')
    graphics.add_line(var_type='_x', var_name='v', axis=ax[1,0],linewidth=0.5,color='b')
    graphics.add_line(var_type='_x', var_name='omega', axis=ax[1,1],linewidth=0.5,color='b')
    graphics.add_line(var_type='_x', var_name='px_h', axis=ax[0,0],linewidth=0.5,color='r')
    graphics.add_line(var_type='_x', var_name='py_h', axis=ax[0,1],linewidth=0.5,color='r')
    graphics.add_line(var_type='_aux', var_name='Distance', axis=ax[2,0],linewidth=0.5,color='k')
    graphics.plot_results()
    graphics.plot_predictions()

    # plot state space
    # Robot
    x_res= graphics.result_lines['_x','px'][0].get_ydata()
    y_res= graphics.result_lines['_x','py'][0].get_ydata()
    if j ==0:
        ax2[0].plot(x_res[0:idx+1],y_res[0:idx+1],color='b',marker='x',label='Robot',lw=0.5)
    else:
        ax2[0].plot(x_res[0:idx+1],y_res[0:idx+1],color='b',marker='x',lw=0.5)
    
    ax[2,1].plot(x_res[0:idx+1],y_res[0:idx+1],color='b',marker='x',label='Robot')
    # Human
    x_res= graphics.result_lines['_x','px_h'][0].get_ydata()
    y_res= graphics.result_lines['_x','py_h'][0].get_ydata()
    #nom_idx =1
    if j ==0:
        ax2[0].plot(x_res[0:idx+1],y_res[0:idx+1],color='r',marker='.',label='Human', lw=0.5)
    else:
        ax2[0].plot(x_res[0:idx+1],y_res[0:idx+1],color='r',marker='.', lw=0.5)
    ax[2,1].plot(x_res[0:idx+1],y_res[0:idx+1],color='r',marker='.',label='Human')

    # Velocity plot
    ax2[1].plot(data[j]['mpc']['_time'],data[j]['mpc']['_x','v'],color='b',label='Robot',lw=0.5)


    # Safety distance
    ax2[2].plot(data[j]['mpc']['_time'],data[j]['mpc']['_aux','Distance',0],color='b',label='Distance',lw=0.5)

    

ax2[1].axhline(0, color='k', linestyle='--')
ax2[1].axhline(2, color='k', linestyle='--')
ax2[2].axhline(Del_safe, color='k', linestyle='--')
ax2[0].axhline(-1.3, color='k', linestyle='--')
ax2[0].axhline(1.3, color='k', linestyle='--')
ax2[0].set_xlabel('$x_{r,x} \\text{ and } x_{h,x}$')
ax2[0].set_ylabel('$x_{r,y} \\text{ and } x_{h,y}$')
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel('Velocity')
ax2[2].set_xlabel('Time')
ax2[2].set_ylabel('Safety distance')
ax2[0].legend()
fig2.align_labels()
fig2.tight_layout()
fig2.savefig('sim_study_MS.pdf')



# %% 
# Print mean time and mean clc
clc = []
time_list = []
for j in range(n_run):
    clc.append(np.sum(data[j]['clc']))
    time_list.append(data[j]['time'])
print('Mean CLC:',np.mean(clc),' STD:',np.std(clc))
print('Mean Time:',np.mean(time_list)/N_sim)
# %% animate
from matplotlib.animation import FuncAnimation, ImageMagickWriter
def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    graphics.plot_results(t_ind=t_ind)
    graphics.plot_predictions(t_ind=t_ind)
    graphics.reset_axes()
    lines = graphics.result_lines.full
    return lines

n_steps = mpc.data['_time'].shape[0]


anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

gif_writer = ImageMagickWriter(fps=5)
anim.save('anim.gif', writer=gif_writer)

# %%
