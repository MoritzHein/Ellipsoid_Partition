# %% Import libraries
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import control
from copy import deepcopy
from matplotlib.patches import Ellipse
from sample_utils import EllipsoidSampler as EllSamp
import time as time
import scienceplots
import os
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
sigma_w=0.4
W_cov = sigma_w**2*np.eye(nw)
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
# Feedback controller - LQR
# Linearized system
A = ca.jacobian(x_next, x)
B = ca.jacobian(x_next, u)
W = ca.jacobian(x_next, w)
A_func = ca.Function('A', [x, u,w], [A])
B_func = ca.Function('B', [x, u,w], [B])
W_func = ca.Function('W', [x, u,w], [W])
x_nom = np.array([-2, 0, 0, 1.6, 0, 2, 0])+np.random.randn(nx)*0.1
u_nom = np.array([0, 0])
w_nom = np.zeros((nw,1))
A_nom = np.array(A_func(x_nom, u_nom, w_nom))
#print(A_nom)
B_nom = np.array(B_func(x_nom, u_nom, w_nom))
Q = np.zeros((nx,nx))
Q[0,0] = 50
Q[1,1] = 50
Q[3,3] = 2
#Q = np.eye(nx)
R= 2*np.eye(nu)

K,_,_ = control.dlqr(A_nom[0:5,0:5], B_nom[0:5,:], Q[0:5,0:5], R)
K=-K
print(K)
K_new = np.zeros((nu,nx))
#K_new[:,0:5] = K
K=K_new
# K[1,1]=-0.1
# K[1,6]=0.1
A_K = A_nom + B_nom@K
#print(np.linalg.eig(A_nom))
print(np.linalg.eig(A_K))





# %%

# Constraint: safety distance
h= []
Del_safe = 0.3
py_con=1.3
h.append(-(px-px_h)**2 - (py-py_h)**2 + (Del_safe)**2)
h.append(-py_con+py)
h.append(-py_con-py)
h.append(-2+v)
h.append(-v)
h.append(-1.0+omega)
h.append(-1.0-omega)
h = ca.vertcat(*h)
nh_k = h.shape[0]
h_N = []
# Terminal velocity constraint
h_N.append(-v)
h_N.append(v-1e-2)
h_N.append(-1+omega)
h_N.append(-1-omega)
h_N.append(-py_con+py)
h_N.append(-py_con-py)
#h_N.append(v+1e-3)
h_N = ca.vertcat(*h_N)
nh_N = h_N.shape[0]
con = ca.Function('con', [x,u], [h])
ter_con = ca.Function('ter_con', [x], [h_N])
P_create = ca.SX.sym('P', nx, nx)
K_create = ca.SX.sym('K', nu, nx)
H= []
for i in range(nh_k):
    H.append( (ca.gradient(h[i], ca.vertcat(x,u))).T@ca.vertcat(np.eye(nx), K_create)@P_create@(ca.vertcat(np.eye(nx), K_create).T)@ca.gradient(h[i], ca.vertcat(x,u)))
H = ca.vertcat(*H)
H_fun = ca.Function('H', [x, u, P_create,K_create], [H])
HN = []
for i in range(nh_N):
    HN.append( (ca.gradient(h_N[i], x)).T@P_create@(ca.gradient(h_N[i], x)))
HN = ca.vertcat(*HN)
HN_fun = ca.Function('HN', [x, P_create], [HN])
# cost function
x_ref = ca.SX.sym('x_ref', nx)
stage_cost = ((x-x_ref).T@Q@(x-x_ref)) + u.T@R@u+ca.trace(0.5*Q@P_create)+ca.trace(0.5*R@K_create@P_create@K_create.T)
stage_cost_fcn = ca.Function('stage_cost', [x, u,P_create,K_create,x_ref], [stage_cost])
terminal_cost = (x-x_ref).T@Q@(x-x_ref) + ca.trace(0.5*Q@P_create)
terminal_cost_fcn = ca.Function('terminal_cost', [x,P_create,x_ref], [terminal_cost])
v_ref = 1.5
#%% Set up ellispoidal optimization problem
N=10
opt_x = ca_tools.struct_symSX([ca_tools.entry('x',repeat = [N+1], shape=(nx)),
                                 ca_tools.entry('P', repeat = [N+1], shape=(nx,nx)),
                                 ca_tools.entry('beta', repeat = [N], shape=(nh_k)),
                                 ca_tools.entry('beta_N', shape=(nh_N)),
                                 ca_tools.entry('u', repeat = [N], shape=(nu))])
#%%
eps = 1e-5
J = 0
g = []
lb_g = []
ub_g = []

x_init = ca.SX.sym('x_init', nx)
g.append((opt_x['P',0] - np.zeros((nx,nx))).reshape((-1,1)))
lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
g.append(opt_x['x',0] - x_init)
lb_g.append(np.zeros((nx,1)))
ub_g.append(np.zeros((nx,1)))
x_ref_traj = []
x_traj = 0*x_init
x_traj[0] += x_init[0]
x_traj[3]=v_ref
x_ref_traj.append(x_traj)
for i in range(N):
    # Cost
    J += stage_cost_fcn(opt_x['x',i], opt_x['u',i], opt_x['P',i], K,x_traj)
    x_traj[0] = x_traj[0] + v_ref*dt
    x_ref_traj.append(x_traj)

    # Nominal dynamics
    g.append(opt_x['x',i+1] - system(opt_x['x',i], opt_x['u',i],np.zeros((nw,1))))
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
    # Uncertainty dynamics
    A_k = A_func(opt_x['x',i], opt_x['u',i],np.zeros((nw,1)))
    B_k = B_func(opt_x['x',i], opt_x['u',i],np.zeros((nw,1)))
    W_jac = W_func(opt_x['x',i], opt_x['u',i],np.zeros((nw,1)))
    if i >0:
        A_K = A_k + B_k@K
    else:
        A_K =A_k
    g.append((opt_x['P',i+1] - A_K@opt_x['P',i]@A_K.T - W_jac@W_cov@W_jac.T).reshape((-1,1)))
    lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    # Constraint
    if i == 0:
        g.append(-opt_x['beta',i] + H_fun(opt_x['x',i], opt_x['u',i], opt_x['P',i], np.zeros((nu,nx))))
        lb_g.append(np.zeros((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))
    else:
        g.append(-opt_x['beta',i] + H_fun(opt_x['x',i], opt_x['u',i], opt_x['P',i], K))
        #lb_g.append(-ca.inf*np.ones((nh_k,1)))
        lb_g.append(np.zeros((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))
    # g.append(-opt_x['beta',i]+eps)
    # lb_g.append(-ca.inf*np.ones((nh_k,1)))
    # ub_g.append(np.zeros((nh_k,1)))

    g.append(con(opt_x['x',i], opt_x['u',i]))
    lb_g.append(-ca.inf*np.ones((nh_k,1)))
    ub_g.append(np.zeros((nh_k,1)))
    g.append(opt_x['beta',i]-con(opt_x['x',i], opt_x['u',i])**2)
    lb_g.append(-ca.inf*np.ones((nh_k,1)))
    ub_g.append(np.zeros((nh_k,1)))

# Terminal cost
J += terminal_cost_fcn(opt_x['x',N], opt_x['P',N],x_traj)
# Terminal constraint
g.append(opt_x['beta_N'] - HN_fun(opt_x['x',N], opt_x['P',N]))
lb_g.append(np.zeros((nh_N,1)))
ub_g.append(np.zeros((nh_N,1)))

g.append(ter_con(opt_x['x',N]))
lb_g.append(-ca.inf*np.ones((nh_N,1)))
ub_g.append(np.zeros((nh_N,1)))
g.append(opt_x['beta_N']+eps-ter_con(opt_x['x',i])**2)
lb_g.append(-ca.inf*np.ones((nh_N,1)))
ub_g.append(np.zeros((nh_N,1)))

g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)

prob = {'f': J, 'x': opt_x.cat, 'g': g,'p':x_init}
solver_opt = {'ipopt.linear_solver': 'MA27','ipopt.print_level':5, 'print_time':1, 'ipopt.tol':1e-8}
solver_eps = ca.nlpsol('solver', 'ipopt', prob, solver_opt)
x_ref_func=ca.Function('x_ref', [x_init], x_ref_traj)
# %% Test solver
x_init = np.array([-2, 0, 0, 1.6, 0, 3.5, 0])
opt_x_init = opt_x(0)
opt_x_init['x'] = x_ref_func(x_init)
opt_x_init['x',:,1]=1
opt_x_init['P'] = np.zeros((nx,nx))+np.diag([0,0,0,0,0,0.4**2,0.4**2])
opt_x_init['beta'] = eps
opt_x_init['beta_N'] = eps
res_nom = solver_eps(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
#opt_x_num = opt_x(res_nom['x'])
opt_x_num_nom = opt_x(res_nom['x'])
print(res_nom['f'])
print(solver_eps.stats()['return_status'])

# %%
# Run 50 closed loops
# 
np.random.seed(123)
data_list_ol = []
N_sim = 10
d,V = np.linalg.eig(sigma_w**2*np.eye(nw*N_sim))
D = np.diag(d)
Ell = EllSamp(np.zeros((nw*N_sim,)), d, V)
for j in range(50):
    x_init = np.array([-2, 0.0, 0, 1.6, 0, 3.5, 0])
    
    #opt_x_init = opt_x_num
    opt_x_init = opt_x(0)
    opt_x_init['x'] = x_ref_func(x_init)
    opt_x_init['P'] = np.zeros((nx,nx))+np.diag([0,0,0,0,0,0.4**2,0.4**2])
    opt_x_init['beta'] = eps
    opt_x_init['beta_N'] = eps
    X=[x_init]
    U=[]
    P=Ell.sample()
    sol =[]
    clc = []
    sol_info = []

    time_start = time.time()
    for i in range(N_sim):
        
        res = solver_eps(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
        opt_x_num = opt_x(res['x'])
        sol_info.append(solver_eps.stats()['return_status'])
        if sol_info[-1] == 'Solve_Succeeded' or sol_info[-1] == 'Solved_To_Acceptable_Level':
            opti_k = i
            opt_x_feas = deepcopy(opt_x(res['x']))
            u0= np.array(opt_x_num['u',0]).reshape((nu,1))
        elif opti_k is not None:
            print(j,i,opti_k)
            ido = i - opti_k
            # Recover last feasible input from the previous iteration
            
            u0= np.array(opt_x_feas['u',ido]).reshape((nu,1))

        else:
            u0= np.array(opt_x_num['u',0]).reshape((nu,1))

        
        opt_x_init = deepcopy(opt_x(res['x']))
        sol.append(opt_x_num)
        
        x_ref = deepcopy(x_init)
        x_ref[1:]=0
        x_ref[3] = v_ref
        clc.append(stage_cost_fcn(x_init, u0, np.zeros((nx,nx)), K, x_ref))
        U.append(u0)
        p0= P[i*nw:(i+1)*nw]
        
        x_init = system(x_init, u0,p0)
        X.append(np.array(x_init).reshape((nx,)))

        
    time_end = time.time()-time_start
    print('Closed loop cost:', np.sum(clc))
    data=dict()
    data['X']=X
    data['U']=U
    data['P']=P
    data['sol']=sol
    data['clc']=clc
    data['time']=time_end
    data['sol_info']=sol_info 
    data_list_ol.append(data)

np.save('data_list_ol.npy',data_list_ol,allow_pickle=True)
# 
# %% Print mean CLC
data_list_ol=np.load('data_list_ol.npy',allow_pickle=True)
clc_ol = []
time_ol = []
for data in data_list_ol:
    clc_ol.append(np.sum(data['clc']))
    time_ol.append(data['time'])
print('Mean CLC: ', np.mean([x for x in clc_ol if x<100]))
print('{} of 50'.format(len([x for x in clc_ol if x<100])))
print('Mean time: ', np.mean(time_ol)/N_sim)

# %% Plot similar figure
l=0
fig, ax = plt.subplots(3,1,figsize=(4,6))
for dat in data_list_ol:
    X_traj = np.array(dat['X']).T
    U_traj = np.array(dat['U']).reshape((-1,nu)).T
    # Trajectory
    if l==0:
        ax[0].plot(X_traj[0,:],X_traj[1,:],'b.-',label='Robot')
        ax[0].plot(X_traj[5,:],X_traj[6,:],'r.-',label='Human')
    else:
        ax[0].plot(X_traj[0,:],X_traj[1,:],'b.-',lw=0.5)
        ax[0].plot(X_traj[5,:],X_traj[6,:],'r.-',lw=0.5)
    # Safety distance
    safe_dist=np.sum((X_traj[0:2,:]-X_traj[5:7,:])**2,axis=0)**0.5
    ax[2].plot(np.arange(0,N_sim*dt+dt,dt),safe_dist,'b',lw=0.5)
    if np.any(safe_dist < Del_safe):
        print('Violation index:', l)
    

    # Velocity
    ax[1].plot(np.arange(0,N_sim*dt+dt,dt),X_traj[3,:],'b',lw=0.5)


    
    l+=1
ax[2].axhline(y=Del_safe, color='k', linestyle='--')
ax[1].axhline(y=2, color='k', linestyle='--')
ax[1].axhline(y=0, color='k', linestyle='--')
ax[2].axhline(y=Del_safe, color='k', linestyle='--')
ax[0].axhline(y=py_con, color='k', linestyle='--')
ax[0].axhline(y=-py_con, color='k', linestyle='--')
ax[0].set_xlabel('$x_{r,x} \\text{ and } x_{h,x}$')
ax[0].set_ylabel('$x_{r,y} \\text{ and } x_{h,y}$')
ax[1].set_xlabel('time')
ax[1].set_ylabel('Velocity')
ax[2].set_xlabel('time')
ax[2].set_ylabel('Safety distance')
ax[0].legend()

fig.align_labels()
fig.tight_layout()
fig.savefig('sim_study_OL.pdf')
# %% Only plot the phase plot
with plt.style.context(['science','ieee']):
    fig, ax = plt.subplots(1,1,figsize=(2,2))
    l=0
    for dat in data_list_ol:
        X_traj = np.array(dat['X']).T
        U_traj = np.array(dat['U']).reshape((-1,nu)).T
        # Trajectory
        if l ==0:
            ax.plot(X_traj[0,:],X_traj[1,:],'b-',alpha=0.5,markersize=0.5,lw=0.2,label='Robot')
            ax.plot(X_traj[5,:],X_traj[6,:],'r-',alpha=0.5,markersize=0.5,lw=0.2,label='Human')
        else:
            ax.plot(X_traj[0,:],X_traj[1,:],'b-',alpha=0.5,markersize=0.5,lw=0.2)
            ax.plot(X_traj[5,:],X_traj[6,:],'r-',alpha=0.5,markersize=0.5,lw=0.2)
        l+=1
    ax.axhline(y=py_con, color='k', linestyle='--')
    ax.axhline(y=-py_con, color='k', linestyle='--')
    ax.set_xlabel('$x_{r,x} $ and $ x_{h,x}$')
    ax.set_ylabel('$x_{r,y}$ and $ x_{h,y}$')
    ax.legend()

    fig.align_labels()
    fig.tight_layout()
    fig.savefig('sim_study_OL_phase.pdf')
# %% Plot the ellipsoids
idx_run = 0
idx_time = 0
dat = data_list_ol[idx_run]
opt_x_num_ell = dat['sol'][idx_time]
fig, ax = plt.subplots(3,2, figsize=(10,10))
#ax[0,0].set_xlim(-2, 2)
#ax[0,0].set_ylim(-2, 2)
for i in range(0,N+1):
    confidence_ellipse(np.array(opt_x_num_ell['x',i]), np.array(opt_x_num_ell['P',i]), ax[0,0],dim=[5,6], edgecolor='r')
    #ax[0,0].plot(opt_x_num_nom['x',i][0], opt_x_num_nom['x',i][1], 'r.')
# Plot the trajectory
x_traj = np.array(ca.vertcat(*opt_x_num_ell['x',:]).reshape((nx,-1)))
u_traj = np.array(ca.vertcat(*opt_x_num_ell['u',:]).reshape((nu,-1)))
ax[0,0].plot(x_traj[0,:], x_traj[1,:], 'b.-')
ax[0,0].plot(x_traj[5,:], x_traj[6,:], 'r.-')
ax[1,0].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj[0,:],'b--')
ax[1,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj[1,:],'b--')
ax[0,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj[4,:],'b--')
ax[0,1].axhline(y=1, color='k', linestyle='--')
ax[0,1].axhline(y=-1, color='k', linestyle='--')

ax[1,1].axhline(y=- py_con, color='k', linestyle='--')
ax[1,1].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=-py_con, color='k', linestyle='--')

#PLot velocity
ax[2,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj[3,:],'b--')
ax[2,1].axhline(y=2, color='k', linestyle='--')
#ax[1,1].axhline(y=-0.5 , color='k', linestyle='--')
#ax[0,0].axhline(y=-0.5 , color='k', linestyle='--')
ax[0,0].set_title('Ellipsoids')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,1].set_title('omega')
ax[0,1].set_xlabel('time')
ax[1,0].set_title('x1')
ax[1,0].set_xlabel('time')
ax[1,1].set_title('x2')
ax[1,1].set_xlabel('time')
ax[2,0].set_title('Safety distance')
ax[2,0].set_xlabel('time')
ax[2,1].set_title('Velocity')
ax[2,1].set_xlabel('time')

X_traj = np.array(dat['X']).T
U_traj = np.array(dat['U']).reshape((-1,nu)).T
# Trajectory
ax[0,0].plot(X_traj[0,:idx_time+1],X_traj[1,:idx_time+1],'kx-')
ax[0,0].plot(X_traj[5,:idx_time+1],X_traj[6,:idx_time+1],'k.-')

# Safety distance
safe_dist=np.sum((X_traj[0:2,:idx_time+1]-X_traj[5:7,:idx_time+1])**2,axis=0)**0.5
ax[2,0].plot(dt*np.arange(0,idx_time+1),safe_dist,'k')

# Velocity
ax[2,1].plot(dt*np.arange(0,idx_time+1),X_traj[3,:idx_time+1],'k')

# States
ax[1,0].plot(dt*np.arange(0,idx_time+1),X_traj[0,:idx_time+1],'k')
ax[1,1].plot(dt*np.arange(0,idx_time+1),X_traj[1,:idx_time+1],'k')

# Omega
ax[0,1].plot(dt*np.arange(0,idx_time+1),X_traj[4,:idx_time+1],'k')


ax[0,1].axhline(y=1, color='k', linestyle='--')
ax[0,1].axhline(y=-1, color='k', linestyle='--')

ax[1,1].axhline(y=- py_con, color='k', linestyle='--')
ax[1,1].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=-py_con, color='k', linestyle='--')
# Plot the safety distance
safe_dist=np.sum((x_traj[0:2,:]-x_traj[5:7,:])**2,axis=0)**0.5
# Get also the backoff from the constraint
beta = np.array(ca.vertcat(*opt_x_num_ell['beta',:]).reshape((nh_k,-1)))
backoff = (np.sum((x_traj[0:2,0:N]-x_traj[5:7,0:N])**2,axis=0)-beta[0,:]**0.5)**0.5
# Plot velocity
ax[2,1].axhline(y=2, color='k', linestyle='--')
ax[2,0].plot(dt*np.arange(idx_time,(idx_time+N+1)),safe_dist,'b--')
ax[2,0].fill_between(dt*np.arange(idx_time,(idx_time+N)), backoff, safe_dist[0:N], alpha=0.5)
ax[2,0].axhline(y=Del_safe, color='k', linestyle='--')
ax[0,0].set_title('Ellipsoids')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,1].set_title('omega')
ax[0,1].set_xlabel('time')
ax[1,0].set_title('x1')
ax[1,0].set_xlabel('time')
ax[1,1].set_title('x2')
ax[1,1].set_xlabel('time')
fig.align_labels()
fig.tight_layout()

# %% Only plot the first subplot
# Also with labels
idx_run = 0
idx_time = 0
dat = data_list_ol[idx_run]
opt_x_num_ell = dat['sol'][idx_time]

with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(figsize=(3,2))

    for i in range(N+1):
        if i == 0:
            confidence_ellipse(np.array(opt_x_num_ell['x',i]), np.array(opt_x_num_ell['P',i]), ax, dim=[5,6],edgecolor='r', label='Uncertainty')
        else:
            confidence_ellipse(np.array(opt_x_num_ell['x',i]), np.array(opt_x_num_ell['P',i]), ax, dim=[5,6],edgecolor='r')

    ax.set_ylim(-1.4, 1.4)
    # Plot the trajectory

    x_traj0 = np.array(ca.vertcat(*opt_x_num_ell['x',:])).reshape((-1,nx)).T
    u_traj0 = np.array(ca.vertcat(*opt_x_num_ell['u',:]).reshape((-1,nu))).T

    ax.plot(x_traj0[0,:], x_traj0[1,:], 'rx--', label = 'Robot')
    ax.plot(x_traj0[5,:], x_traj0[6,:], 'r.--', label = 'Human')


    ax.set_xlabel('$x_{r,x}$ and $x_{h,x}$')
    ax.set_ylabel('$x_{r,y}$ and $ x_{h,y}$')
    X_traj = np.array(dat['X']).T
    U_traj = np.array(dat['U']).reshape((-1,nu)).T
    # Trajectory
    ax.plot(X_traj[0,:idx_time+1],X_traj[1,:idx_time+1],'kx-')
    ax.plot(X_traj[5,:idx_time+1],X_traj[6,:idx_time+1],'k.-')
    ax.axhline(y=py_con, color='k', linestyle='--')
    ax.axhline(y=-py_con, color='k', linestyle='--')
    ax.legend()
    fig.align_labels()
    fig.tight_layout()
    fig.savefig('Prediction_loop_OL.pdf')

########################################################################################
# %% Set up ellipsoidal cutting
ns = 2 # Number of ellipsoids
N=10
opt_x = ca_tools.struct_symSX([ca_tools.entry('x',repeat = [N+1,ns], shape=(nx)),
                                 ca_tools.entry('P', repeat = [N+1,ns], shape=(nx,nx)),
                                 ca_tools.entry('cutting_a', shape = (nx,1)),
                                 ca_tools.entry('cutting_b', shape = (1)),
                                 #ca_tools.entry('alpha', shape = (1)),
                                 ca_tools.entry('gamma', shape = (1)),
                                 ca_tools.entry('beta', repeat = [N,ns], shape=(nh_k)),
                                 ca_tools.entry('beta_N', repeat = [ns],shape=(nh_N)),
                                 ca_tools.entry('u', repeat = [N,ns], shape=(nu))])
# %% Define casadi function for volume
al= ca.SX.sym('alpha')
vol = ((nx**2*(1-al**2))/(nx**2-1))**((nx-1)/2)*(nx*(1-al)/(nx+1))
vol_fun = ca.Function('vol', [al], [vol])
# %%
eps = 1e-5
mode = 'smooth' # Modes are 'ifelse', 'mid' or 'border', 'smooth'
if mode == 'smooth':
    from test_softplus import find_param
    func, _,_ = find_param(-1/nx,100,1e-3)
J = 0
g = []
lb_g = []
ub_g = []

x_init = ca.SX.sym('x_init', nx)
for s in  range(ns):
    g.append((opt_x['P',0,s] ).reshape((-1,1)))
    lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    g.append(opt_x['x',0,s] - x_init)
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
    g.append(opt_x['u',0,s]-opt_x['u',0,0])
    lb_g.append(np.zeros((nu,1)))
    ub_g.append(np.zeros((nu,1)))
x_traj = 0*x_init    
x_traj[0] += x_init[0]
x_traj[5] += x_init[5]
x_traj[3]=v_ref
for i in range(N):
    for s in range(ns):
        # Cost
        # Partitioning the ellipsoid
        if i == 0 and s == 0:
            # Initial ellipsoid from uncertainty propagation
            A_k = A_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            B_k = B_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            W_jac = W_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            # At the first prediction step K=0
            A_K = A_k
            E = A_K@opt_x['P',i,s]@A_K.T + W_jac@W_cov@W_jac.T
            c = system(opt_x['x',0,0], opt_x['u',0,0],np.zeros((nw,1)))

            # Cutting plane
            a = opt_x['cutting_a']
            b = opt_x['cutting_b']

            # Auxillary variable for better numerical stability
            g.append(opt_x['gamma']**2 - a.T@E@a)
            lb_g.append(np.zeros((1,1)))
            ub_g.append(np.zeros((1,1)))
            
            alpha = (a.T @ c - b)/ opt_x['gamma']
            # Constraint to ensure the feasibility of the cutting plane
            g.append(alpha)
            if mode == 'ifelse' or mode == 'smooth':
                lb_g.append(0)
                ub_g.append(1)
            elif mode == 'mid':
                lb_g.append(0)
                ub_g.append(1/nx)
            elif mode == 'border':
                lb_g.append(1/nx)
                ub_g.append(1)
                
            tau = (1+nx*alpha)/(nx+1)
            sigma = (2*(1+nx*alpha))/((nx+1)*(1+alpha))
            delta = (nx**2/ (nx**2-1))*(1-alpha**2)
            c_new = c - tau* (E @ a)/opt_x['gamma']
            g.append(opt_x['x',i+1,0] - c_new)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))

            E_new = delta*(E- sigma*(E @ a @ (E@a).T)/opt_x['gamma']**2)

            g.append((opt_x['P',i+1,0]- E_new).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            # Now the other side of the cut
            a1=-a
            b1=-b
            # This alpha should be the negative one as the previous one
            if mode == 'ifelse':
                alpha1 = ca.if_else(alpha<1/nx, -alpha, -1/nx)
            if mode == 'smooth':
                alpha1 = func(-alpha)
            elif mode == 'mid':
                alpha1 = -alpha
            elif mode == 'border':
                alpha1 = -1/nx
            tau1 = (1+nx*alpha1)/(nx+1)
            # if else formulation
            sigma1 = (2*(1+nx*alpha1))/((nx+1))
            delta1 = (nx**2/ (nx**2-1))*(1-alpha1)

            if mode == 'border':
                c_new1 = c
            else:
                c_new1 = c - tau1* (E @ a1)/opt_x['gamma']
            g.append(opt_x['x',i+1,1]- c_new1)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            if mode == 'border':
                E_new1 = E
            else:
                E_new1 = delta1*(E*(1+alpha1)- sigma1*(E @ a1 @ (E@a1).T)/opt_x['gamma']**2)
            g.append((opt_x['P',i+1,1] - E_new1).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            


            # Uniqueness of a and b
            g.append(opt_x['gamma']-1)
            lb_g.append(np.zeros((1,1)))
            ub_g.append(np.zeros((1,1)))

        elif i >0:
            # Nominal dynamics
            g.append(opt_x['x',i+1,s] - system(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1))))
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            # Uncertainty dynamics
            A_k = A_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            B_k = B_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            W_jac = W_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            A_K = A_k + B_k@K
            g.append((opt_x['P',i+1,s] - A_K@opt_x['P',i,s]@A_K.T - W_jac@W_cov@W_jac.T).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))

        # Constraint
        g.append(opt_x['beta',i,s] - H_fun(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K))
        lb_g.append(np.zeros((nh_k,1)))
        ub_g.append(ca.inf*np.ones((nh_k,1)))
        g.append(con(opt_x['x',i,s], opt_x['u',i,s]))
        lb_g.append(-ca.inf*np.ones((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))
        g.append(opt_x['beta',i,s]+eps-con(opt_x['x',i,s], opt_x['u',i,s])**2)
        lb_g.append(-ca.inf*np.ones((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))

        g.append(opt_x['beta',i,s])
        lb_g.append(eps*np.ones((nh_k,1)))
        ub_g.append(ca.inf*np.ones((nh_k,1)))

        # Cost
        # Possible choice for omega_k^s
        # vol1=vol_fun(alpha)
        # vol2=vol_fun(alpha1)
        # if s == 0:
        #     J += (vol1/(vol1+vol2))*stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K,x_traj)
        # elif s == 1:
        #     J += (vol2/(vol1+vol2))*stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K,x_traj)
        J += stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K,x_traj)
        
        
    # Update the nominal reference trajectory
    x_traj[0] = x_traj[0] + v_ref*dt


for s in range(ns):
    # Terminal cost
    # if s == 0:
    #     J += (vol1/(vol1+vol2))*terminal_cost_fcn(opt_x['x',N,s], opt_x['P',N,s],x_traj)
    # elif s == 1:
    #     J += (vol2/(vol1+vol2))*terminal_cost_fcn(opt_x['x',N,s], opt_x['P',N,s],x_traj)
    J += terminal_cost_fcn(opt_x['x',N,s], opt_x['P',N,s],x_traj)
    # Terminal constraint
    g.append(opt_x['beta_N',s] - HN_fun(opt_x['x',N,s], opt_x['P',N,s]))
    lb_g.append(np.zeros((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))
    g.append(ter_con(opt_x['x',N,s]))
    lb_g.append(-ca.inf*np.ones((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))
    g.append(opt_x['beta_N',s]+eps-ter_con(opt_x['x',i,s])**2)
    lb_g.append(-ca.inf*np.ones((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))

#Slight penalization on a
J+= opt_x['cutting_a'].T@opt_x['cutting_a']


g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)

prob = {'f': J, 'x': opt_x.cat, 'g': g,'p': x_init}
solver_opt = {'ipopt.max_iter':3000,'ipopt.linear_solver': 'MA27','ipopt.print_level':0, 'print_time':0,'ipopt.alpha_min_frac':0.00001}
solver_cut = ca.nlpsol('solver', 'ipopt', prob, solver_opt)

# %% Test solver
x_init = np.array([-2, 0.0, 0, 1.6, 0, 3.5, 0])
opt_x_init = opt_x(0)

# if opt_init_Scen_nofb.npy does not exist,
# initialize the solution with the following and constrain alpha to be zero, then run the solver and save the solution as commented below
    # opt_x_init['x',:,0] = opt_x_num_nom['x']
    # opt_x_init['x',:,1] = opt_x_num_nom['x']
    # opt_x_init['x',:,1,1] = [-opt_x_init['x',i,0,1] for i in range(N+1)] 

    # opt_x_init['x',:,0,1] = 1.0
    # opt_x_init['x',:,1,1] = -1.0

    # opt_x_init['P'] = np.zeros((nx,nx))+np.diag([0,0,0,0,0,0.4**2,0.4**2])

    # opt_x_init['cutting_a'] = np.array([0, 0, 0, 0, 0, 0, 5])
    # opt_x_init['cutting_b'] = 0.01
    # opt_x_init['gamma'] = 1
    # opt_x_init['beta'] = np.ones((nh_k,1))

    # opt_x_init['u',:,0] = opt_x_num_nom['u']
    # opt_x_init['u',:,1] = opt_x_num_nom['u']
    # opt_x_init['u',:,1,1] = [-opt_x_init['u',i,0,1] for i in range(N)]

opt_x_init = np.load('opt_init_Scen_nofb.npy',allow_pickle=True).item()
opt_x_initial00 = deepcopy(opt_x_init)
res = solver_cut(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
opt_x_num = opt_x(res['x'])
print(res['f']-opt_x_num['cutting_a'].T@opt_x_num['cutting_a'])
print(solver_cut.stats()['return_status'])
# Initialize the solution with cutting constrained at zero, save it and then recompute
# This avoids local minima for the choice of the hyperplane
#np.save('opt_init_Scen_nofb.npy',opt_x_num)


# %% Run a simulation study on 50 runs
np.random.seed(123)
data_list=[]
d, V= np.linalg.eig(sigma_w**2*np.eye((nw*N_sim)))
D = np.diag(d)
opti_k= None

Ell = EllSamp(np.zeros((nw*N_sim,)), d, V)
P=Ell.sample()
for j in range(50):
    x_init = np.array([-2, 0.0, 0, 1.6, 0, 3.5, 0])
    
    #opt_x_init = opt_x_num
    opt_x_init = np.load('opt_init_Scen_nofb.npy',allow_pickle=True).item()
    opt_x_initial00 = deepcopy(opt_x_init)
    X=[x_init]
    U=[]
    P=Ell.sample()
    sol =[]
    clc = []
    sol_info = []

    time_start = time.time()
    for i in range(N_sim):
        
        res = solver_cut(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
        opt_x_num = opt_x(res['x'])
        sol_info.append(solver_cut.stats()['return_status'])
        if sol_info[-1] == 'Solve_Succeeded' or sol_info[-1] == 'Solved_To_Acceptable_Level':
            opti_k = i
            opt_x_feas = deepcopy(opt_x(res['x']))
            u0= np.array(opt_x_num['u',0,0]).reshape((nu,1))
        elif opti_k is not None:
            print(j,i,opti_k)
            ido = i - opti_k
            # Recover last feasible input from the previous iteration
            P1 = np.array(opt_x_feas['P',ido,0])
            P2 = np.array(opt_x_feas['P',ido,1])
            x1 = np.array(opt_x_feas['x',ido,0])
            x2 = np.array(opt_x_feas['x',ido,1])
            # Check, in which ellipsoid the current state is
            if ido == 1:
                select0=(x1-x_init).T@P1@(x1-x_init) < (x2-x_init).T@P2@(x2-x_init)
            if select0:
                u0= np.array(opt_x_feas['u',ido,0]).reshape((nu,1))
                
            else:
                u0= np.array(opt_x_feas['u',ido,1]).reshape((nu,1))
        else:
            u0= np.array(opt_x_num['u',0,0]).reshape((nu,1))

        
        opt_x_init = deepcopy(opt_x(res['x']))
        sol.append(opt_x_num)
        
        x_ref = deepcopy(x_init)
        x_ref[1:]=0
        x_ref[3] = v_ref
        clc.append(stage_cost_fcn(x_init, u0, np.zeros((nx,nx)), K, x_ref))
        U.append(u0)
        p0= P[nw*i:nw*(i+1)]
        x_init = system(x_init, u0,p0)
        X.append(np.array(x_init).reshape((nx,)))
        if i == 0:
            indi = p0[-1] > 0
        if indi:
            opt_x_init['x',:,0,1] = -1
            opt_x_init['x',:,1,1] = -1
        else:
            opt_x_init['x',:,0,1] = 1
            opt_x_init['x',:,1,1] = 1

    time_end = time.time()-time_start
    print('Closed loop cost:', np.sum(clc))
    data=dict()
    data['X']=X
    data['U']=U
    data['P']=P
    data['sol']=sol
    data['clc']=clc
    data['time']=time_end
    data['sol_info']=sol_info 
    data_list.append(data)

np.save('data_list.npy',data_list,allow_pickle=True)

# %% Print mean clc
clc_list = [np.sum(dat['clc']) for dat in data_list]
time_list = [dat['time'] for dat in data_list]
print('Mean closed loop cost:', np.mean(clc_list))
print('Mean time:', np.mean(time_list)/N_sim)
# %%

# %% Plot similar figure
with plt.style.context(['science','ieee']):
    l=0
    fig, ax = plt.subplots(3,1,figsize=(4,6))
    for dat in data_list:
        X_traj = np.array(dat['X']).T
        U_traj = np.array(dat['U']).reshape((-1,nu)).T
        # Trajectory
        if l==0:
            ax[0].plot(X_traj[0,:],X_traj[1,:],'b.-',label='Robot')
            ax[0].plot(X_traj[5,:],X_traj[6,:],'r.-',label='Human')
        else:
            ax[0].plot(X_traj[0,:],X_traj[1,:],'b.-',lw=0.5)
            ax[0].plot(X_traj[5,:],X_traj[6,:],'r.-',lw=0.5)

        # Safety distance
        safe_dist=np.sum((X_traj[0:2,:]-X_traj[5:7,:])**2,axis=0)**0.5
        ax[2].plot(np.arange(0,N_sim*dt+dt,dt),safe_dist,'b',lw=0.5)
        if np.any(safe_dist < Del_safe):
            print('Violation index:', l)

        # Velocity
        ax[1].plot(np.arange(0,N_sim*dt+dt,dt),X_traj[3,:],'b',lw=0.5)


        
        l+=1
    ax[0].set_xlabel('$x_{r,x} $ and $ x_{h,x}$')
    ax[0].set_ylabel('$x_{r,y} $ and $ x_{h,y}$')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('Velocity')
    ax[2].axhline(y=Del_safe, color='k', linestyle='--')
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('Safety distance')
    ax[0].axhline(y=-py_con, color='k', linestyle='--')
    ax[0].axhline(y=py_con, color='k', linestyle='--')
    ax[1].axhline(y=2, color='k', linestyle='--')
    ax[1].axhline(y=0, color='k', linestyle='--')
    ax[0].legend()
    fig.align_labels()
    fig.tight_layout()
    fig.savefig('sim_study_Part.pdf')
# %%
# Only show the first plot
with plt.style.context(['science','ieee']):
    l=0
    fig, ax = plt.subplots(1,1,figsize=(2,2)) 
    for dat in data_list:
        X_traj = np.array(dat['X']).T
        U_traj = np.array(dat['U']).reshape((-1,nu)).T
        # Trajectory
        if l==0:
            ax.plot(X_traj[0,:],X_traj[1,:],'b-',lw=0.2,alpha=0.5,markersize=0.5,label='Robot')
            ax.plot(X_traj[5,:],X_traj[6,:],'r-',lw=0.2,alpha=0.5,markersize=0.5,label='Human')
        else:
            ax.plot(X_traj[0,:],X_traj[1,:],'b-',lw=0.2,alpha=0.5,markersize=0.5)
            ax.plot(X_traj[5,:],X_traj[6,:],'r-',lw=0.2,alpha=0.5,markersize=0.5)

        # Safety distance
        # safe_dist=np.sum((X_traj[0:2,:]-X_traj[5:7,:])**2,axis=0)**0.5
        # ax.plot(np.arange(0,N_sim*dt+dt,dt),safe_dist,'b',lw=0.5)
        # if np.any(safe_dist < Del_safe):
        #     print('Violation index:', l)

        # # Velocity
        # ax.plot(np.arange(0,N_sim*dt+dt,dt),X_traj[3,:],'b',lw=0.5)


        
        l+=1
    ax.set_xlabel('$x_{r,x} $ and $ x_{h,x}$')
    ax.set_ylabel('$x_{r,y} $ and $ x_{h,y}$')
    ax.axhline(y=-py_con, color='k', linestyle='--')
    ax.axhline(y=py_con, color='k', linestyle='--')
    # ax.axhline(y=Del_safe, color='k', linestyle='--')
    # ax.axhline(y=0, color='k', linestyle='--')
    # ax.axhline(y=2, color='k', linestyle='--')
    ax.legend()
    fig.align_labels()
    fig.tight_layout()
    fig.savefig('sim_study_Part_Phase.pdf')
# %% 
# Plot a prediction loop run

idx_run= 0
idx_time = 0
da = data_list[idx_run]
opt_x_num = da['sol'][idx_time]

fig, ax = plt.subplots(3,2,figsize=(10,10))
for i in range(N+1):
    if i == 0:
        confidence_ellipse(np.array(opt_x_num['x',i,0]), np.array(opt_x_num['P',i,0]), ax[0,0], dim=[5,6],edgecolor='r', label='Uncertainty Scen. 1')
        confidence_ellipse(np.array(opt_x_num['x',i,1]), np.array(opt_x_num['P',i,1]), ax[0,0],dim=[5,6], edgecolor='b', label= 'Uncertainty Scen. 2')
    else:
        confidence_ellipse(np.array(opt_x_num['x',i,0]), np.array(opt_x_num['P',i,0]), ax[0,0], dim=[5,6],edgecolor='r')
        confidence_ellipse(np.array(opt_x_num['x',i,1]), np.array(opt_x_num['P',i,1]), ax[0,0],dim=[5,6], edgecolor='b')
    # Plot nominal confidence ellispoid
    #confidence_ellipse(np.array(opt_x_num_nom['x',i]), np.array(opt_x_num_nom['P',i]), ax[0,0],dim=[5,6], edgecolor='k')
    # ax[0,0].plot(opt_x_num['x',i,0][5], opt_x_num['x',i,0][6], 'r.')
    # ax[0,0].plot(opt_x_num['x',i,1][5], opt_x_num['x',i,1][6], 'b.')
    # ax[0,0].plot(opt_x_num['x',i,0][0], opt_x_num['x',i,0][1], 'rx')
    # ax[0,0].plot(opt_x_num['x',i,1][0], opt_x_num['x',i,1][1], 'bx')
# Plot half space
x = np.linspace(0, 4, 1000)
b = opt_x_num['cutting_b']
a = opt_x_num['cutting_a']
y = (b - a[-2]*x)/a[-1]
ax[0,0].plot(x, y,linewidth=1, label='Partitioning plane')
#ax[0,0].set_xlim(-0.6, 2)
ax[0,0].set_ylim(-1.4, 1.4)
# Plot ellipsoid
A_k = A_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
B_k = B_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
W_jac = W_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
A_K = A_k + B_k@K
E = A_K@opt_x_num['P',0,0]@A_K.T + W_jac@W_cov@W_jac.T
c = system(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
confidence_ellipse(np.array(c), np.array(E), ax[0,0],dim=[5,6], edgecolor='g', label = 'Original Ellipsoid')
# Plot the trajectory

x_traj0 = np.array(ca.vertcat(*opt_x_num['x',:,0])).reshape((-1,nx)).T
x_traj1 = np.array(ca.vertcat(*opt_x_num['x',:,1])).reshape((-1,nx)).T
u_traj0 = np.array(ca.vertcat(*opt_x_num['u',:,0]).reshape((-1,nu))).T
u_traj1 = np.array(ca.vertcat(*opt_x_num['u',:,1]).reshape((-1,nu))).T
ax[0,0].plot(x_traj0[0,:], x_traj0[1,:], 'rx--', label = 'Robot Scen. 1')
ax[0,0].plot(x_traj1[0,:], x_traj1[1,:], 'bx--', label = 'Robot Scen. 2')
ax[0,0].plot(x_traj0[5,:], x_traj0[6,:], 'r.--', label = 'Human Scen. 1')
ax[0,0].plot(x_traj1[5,:], x_traj1[6,:], 'b.--', label = 'Human Scen. 2')
ax[1,0].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj0[0,:],'r--')
ax[1,0].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj1[0,:],'b--')
ax[1,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj0[1,:],'r--')
ax[1,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj1[1,:],'b--')
ax[0,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj0[4,:],'r--')
ax[0,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj1[4,:],'b--')
ax[0,1].axhline(y=1, color='k', linestyle='--')
ax[0,1].axhline(y=-1, color='k', linestyle='--')

ax[1,1].axhline(y=- py_con, color='k', linestyle='--')
ax[1,1].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=-py_con, color='k', linestyle='--')
# Plot the safety distance
safe_dist0=np.sum((x_traj0[0:2,:]-x_traj0[5:7,:])**2,axis=0)**0.5
safe_dist1=np.sum((x_traj1[0:2,:]-x_traj1[5:7,:])**2,axis=0)**0.5
ax[2,0].plot(dt*np.arange(idx_time,(idx_time+N+1)),safe_dist0,'r--')
ax[2,0].plot(dt*np.arange(idx_time,(idx_time+N+1)),safe_dist1,'b--')
# Plot also the backoff
beta0 = np.array(ca.vertcat(*opt_x_num['beta',:,0]).reshape((nh_k,-1)))
beta1 = np.array(ca.vertcat(*opt_x_num['beta',:,1]).reshape((nh_k,-1)))
backoff0 = (np.sum((x_traj0[0:2,0:N]-x_traj0[5:7,0:N])**2,axis=0)-beta0[0,:]**0.5)**0.5
backoff1 = (np.sum((x_traj1[0:2,0:N]-x_traj1[5:7,0:N])**2,axis=0)-beta1[0,:]**0.5)**0.5
ax[2,0].fill_between(dt*np.arange(idx_time,(idx_time+N)), backoff0, safe_dist0[0:N], alpha=0.5,color='r')
ax[2,0].fill_between(dt*np.arange(idx_time,(idx_time+N)), backoff1, safe_dist1[0:N], alpha=0.5, color='b')
ax[2,0].axhline(y=Del_safe, color='k', linestyle='--')
#PLot velocity
ax[2,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj0[3,:],'r--')
ax[2,1].plot(dt*np.arange(idx_time,(idx_time+N+1)),x_traj1[3,:],'b--')
ax[2,1].axhline(y=2, color='k', linestyle='--')
#ax[1,1].axhline(y=-0.5 , color='k', linestyle='--')
#ax[0,0].axhline(y=-0.5 , color='k', linestyle='--')
ax[0,0].set_title('Ellipsoids')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,1].set_title('omega')
ax[0,1].set_xlabel('time')
ax[1,0].set_title('x1')
ax[1,0].set_xlabel('time')
ax[1,1].set_title('x2')
ax[1,1].set_xlabel('time')
ax[2,0].set_title('Safety distance')
ax[2,0].set_xlabel('time')
ax[2,1].set_title('Velocity')
ax[2,1].set_xlabel('time')

X_traj = np.array(da['X']).T
U_traj = np.array(da['U']).reshape((-1,nu)).T
# Trajectory
ax[0,0].plot(X_traj[0,:idx_time+1],X_traj[1,:idx_time+1],'kx-')
ax[0,0].plot(X_traj[5,:idx_time+1],X_traj[6,:idx_time+1],'k.-')

# Safety distance
safe_dist=np.sum((X_traj[0:2,:idx_time+1]-X_traj[5:7,:idx_time+1])**2,axis=0)**0.5
ax[2,0].plot(dt*np.arange(0,idx_time+1),safe_dist,'k')

# Velocity
ax[2,1].plot(dt*np.arange(0,idx_time+1),X_traj[3,:idx_time+1],'k')

# States
ax[1,0].plot(dt*np.arange(0,idx_time+1),X_traj[0,:idx_time+1],'k')
ax[1,1].plot(dt*np.arange(0,idx_time+1),X_traj[1,:idx_time+1],'k')

# Omega
ax[0,1].plot(dt*np.arange(0,idx_time+1),X_traj[4,:idx_time+1],'k')

ax[0,0].legend()
fig.align_labels()
fig.tight_layout()

print('Closed loop cost:', np.sum(da['clc']))
print('Optimal solution:', da['sol_info'][idx_time])

# %% Only plot the first subplot (0,0)
with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots()

    for i in range(N+1):
        if i == 0:
            confidence_ellipse(np.array(opt_x_num['x',i,0]), np.array(opt_x_num['P',i,0]), ax, dim=[5,6],edgecolor='r', label='Uncertainty Scen. 1')
            confidence_ellipse(np.array(opt_x_num['x',i,1]), np.array(opt_x_num['P',i,1]), ax,dim=[5,6], edgecolor='b', label= 'Uncertainty Scen. 2')
        else:
            confidence_ellipse(np.array(opt_x_num['x',i,0]), np.array(opt_x_num['P',i,0]), ax, dim=[5,6],edgecolor='r')
            confidence_ellipse(np.array(opt_x_num['x',i,1]), np.array(opt_x_num['P',i,1]), ax,dim=[5,6], edgecolor='b')
        # Plot nominal confidence ellispoid
        #confidence_ellipse(np.array(opt_x_num_nom['x',i]), np.array(opt_x_num_nom['P',i]), ax[0,0],dim=[5,6], edgecolor='k')
        # ax[0,0].plot(opt_x_num['x',i,0][5], opt_x_num['x',i,0][6], 'r.')
        # ax[0,0].plot(opt_x_num['x',i,1][5], opt_x_num['x',i,1][6], 'b.')
        # ax[0,0].plot(opt_x_num['x',i,0][0], opt_x_num['x',i,0][1], 'rx')
        # ax[0,0].plot(opt_x_num['x',i,1][0], opt_x_num['x',i,1][1], 'bx')
    # Plot half space
    x = np.linspace(0, 4, 1000)
    b = opt_x_num['cutting_b']
    a = opt_x_num['cutting_a']
    y = (b - a[-2]*x)/a[-1]
    ax.plot(x, y,linewidth=1, label='Halfspace')
    #ax[0,0].set_xlim(-0.6, 2)
    ax.set_ylim(-1.4, 1.4)
    # Plot ellipsoid
    A_k = A_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
    B_k = B_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
    W_jac = W_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
    A_K = A_k + B_k@K
    E = A_K@opt_x_num['P',0,0]@A_K.T + W_jac@W_cov@W_jac.T
    c = system(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
    confidence_ellipse(np.array(c), np.array(E), ax,dim=[5,6], edgecolor='g', label = 'Original Ellipsoid')
    # Plot the trajectory

    x_traj0 = np.array(ca.vertcat(*opt_x_num['x',:,0])).reshape((-1,nx)).T
    x_traj1 = np.array(ca.vertcat(*opt_x_num['x',:,1])).reshape((-1,nx)).T
    u_traj0 = np.array(ca.vertcat(*opt_x_num['u',:,0]).reshape((-1,nu))).T
    u_traj1 = np.array(ca.vertcat(*opt_x_num['u',:,1]).reshape((-1,nu))).T
    ax.plot(x_traj0[0,:], x_traj0[1,:], 'rx--', label = 'Robot Scen. 1')
    ax.plot(x_traj1[0,:], x_traj1[1,:], 'bx--', label = 'Robot Scen. 2')
    ax.plot(x_traj0[5,:], x_traj0[6,:], 'r.--', label = 'Human Scen. 1')
    ax.plot(x_traj1[5,:], x_traj1[6,:], 'b.--', label = 'Human Scen. 2')

    ax.set_xlabel('$x_{r,x}$  and $ x_{h,x}$')
    ax.set_ylabel('$x_{r,y} $ and $ x_{h,y}$')

    # Trajectory
    ax.plot(X_traj[0,:idx_time+1],X_traj[1,:idx_time+1],'kx-')
    ax.plot(X_traj[5,:idx_time+1],X_traj[6,:idx_time+1],'k.-')
    ax.axhline(y=py_con, color='k', linestyle='--')
    ax.axhline(y=-py_con, color='k', linestyle='--')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:4],labels[0:4],ncol=2,loc=[0.0,0.8])
    fig.legend(handles[4:],labels[4:],ncol=2,loc=[0.17,0.18])
    fig.align_labels()
    fig.tight_layout()
    fig.savefig('Prediction_loop.pdf')

###########################################################################################################
