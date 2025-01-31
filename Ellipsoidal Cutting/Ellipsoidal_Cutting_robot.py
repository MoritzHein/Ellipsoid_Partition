# %% Import libraries
import casadi as ca
import casadi.tools as ca_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import control
from matplotlib.patches import Ellipse
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
# %% Play around with ellipsoidal cutting
# Initialize ellispoid with (x-c)^T E^-1 (x-c) <= 1
E = np.array([[1, 0], [0, 1]]) # Propotional to the confidence region
E_inv = np.linalg.inv(E)
c = np.array([[0], [0]])
# Plot ellipsoid
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
#eigenvalues, eigenvectors = np.linalg.eig(E_inv)
#theta = np.linspace(0, 2*np.pi, 1000)
#ellipsis = (1/np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
#ax.plot(ellipsis[0,:]+c[0], ellipsis[1,:]+c[1])
confidence_ellipse(c, E, ax, edgecolor='r')
# Check the equation by meshing
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)
# for i in range(x.shape[0]):
#     for j in range(y.shape[0]):
#         point = np.array([[X[i,j]], [Y[i,j]]])
#         if (point-c).T @ E_inv @ (point-c) <= 1:
#             ax.plot(X[i,j], Y[i,j], 'r.')
# Define cutting plane
a = np.array([[1], [1]])
b = 1
# Plot cutting plane a^T x <= b
x = np.linspace(-5, 5, 1000)
y = (b - a[0]*x)/a[1]
ax.plot(x, y)
# Get the new ellipsoid

def get_next_ellipse(c, E, a, b):
    n= E.shape[0]
    alpha = (a.T @ c - b)/ca.sqrt(a.T @ E @ a)
    print(alpha)
    tau = (1+n*alpha)/(n+1)
    sigma = (2*(1+n*alpha))/((n+1)*(1+alpha))
    delta = (n**2/ (n**2-1))*(1-alpha**2)

    c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
    E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
    return c_new, E_new


c_new, E_new = get_next_ellipse(c, E, a, b)
confidence_ellipse(c_new, E_new, ax, edgecolor='b')

# Get other ellipsoid
c_new, E_new = get_next_ellipse(c, E, -a, -b)
confidence_ellipse(c_new, E_new, ax, edgecolor='g')


# %% system model robot and human
dt = 0.5
nx = 7
nu = 2
nw = 7

v_nom_h = -0.6
W_cov = 0.00*np.eye(nw)
W_cov[5,5]=0.4**2
W_cov[6,6]=0.4**2
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
rhs[0] = v*ca.cos(theta) + w[0]
rhs[1] = v*ca.sin(theta) + w[1]
rhs[2] = omega + w[2]
rhs[3] = u[0] + w[3]
rhs[4] = u[1] + w[4]

rhs[5] = v_nom_h +w[5]#+ w[0]
rhs[6] = 0 + w[6] #+ w[1]

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
R= 2*np.eye(nu)

#K,_,_ = control.dlqr(A_nom, B_nom, Q, R)
#K=-K
#print(K)
K = np.zeros((nu,nx))
A_K = A_nom + B_nom@K
print(np.linalg.eig(A_nom))

# %% Simulate a trajectory
x0 = np.array([-2, 0, 0, 1.6, 0, 2, 0])
N = 12
x_traj = np.zeros((nx, N+1))
x_traj[:,0] = x0.flatten()
for i in range(N):
    x_traj[:,i+1] = (system(x_traj[:,i], K@x_traj[:,i], W_cov@np.random.randn(nw))  ).full().flatten()
# Plot
fig, ax = plt.subplots()
ax.plot(x_traj[0,:], x_traj[1,:], 'r.-')
ax.plot(x_traj[5,:], x_traj[6,:], 'b.-')




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
stage_cost = (x-x_ref).T@Q@(x-x_ref) + u.T@R@u+ca.trace(0.5*Q@P_create)+ca.trace(0.5*R@K_create@P_create@K_create.T)
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
x_traj = 0+ x_init
x_traj[3]=v_ref
x_ref_traj.append(x_traj)
for i in range(N):
    # Cost
    J += stage_cost_fcn(opt_x['x',i], opt_x['u',i], opt_x['P',i], K,x_traj)
    x_traj[0] = x_traj[0] + v_ref*dt
    x_ref_traj.append(x_traj)
    # if i >0:
    #     J+= (opt_x['u',i]-opt_x['u',i-1]).T@R@(opt_x['u',i]-opt_x['u',i-1])
    # else:
    #     J+= opt_x['u',i].T@R@opt_x['u',i]
    # Nominal dynamics
    g.append(opt_x['x',i+1] - system(opt_x['x',i], opt_x['u',i],np.zeros((nw,1))))
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
    # Uncertainty dynamics
    A_k = A_func(opt_x['x',i], opt_x['u',i],np.zeros((nw,1)))
    B_k = B_func(opt_x['x',i], opt_x['u',i],np.zeros((nw,1)))
    W_jac = W_func(opt_x['x',i], opt_x['u',i],np.zeros((nw,1)))
    A_K = A_k + B_k@K
    g.append((opt_x['P',i+1] - A_K.T@opt_x['P',i]@A_K - W_jac@W_cov@W_jac.T).reshape((-1,1)))
    lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    # Constraint
    g.append(-opt_x['beta',i] + H_fun(opt_x['x',i], opt_x['u',i], opt_x['P',i], K))
    #lb_g.append(-ca.inf*np.ones((nh_k,1)))
    lb_g.append(np.zeros((nh_k,1)))
    ub_g.append(np.zeros((nh_k,1)))
    # g.append(-opt_x['beta',i]+eps)
    # lb_g.append(-ca.inf*np.ones((nh_k,1)))
    # ub_g.append(np.zeros((nh_k,1)))
    # g.append(con(opt_x['x',i], opt_x['u',i])+ca.sqrt(opt_x['beta',i]+eps))
    # lb_g.append(-np.ones((nh_k,1))*ca.inf)
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
# g.append(ter_con(opt_x['x',N])+ca.sqrt(opt_x['beta_N']+eps))
# lb_g.append(-np.ones((nh_N,1))*ca.inf)
# ub_g.append(np.zeros((nh_N,1)))
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
x_init = np.array([-2, 0, 0, 1.6, 0, 2, 0])
opt_x_init = opt_x(0)
opt_x_init['x'] = x_ref_func(x_init)
opt_x_init['x',:,1]=1
opt_x_init['P'] = np.zeros((nx,nx))+W_cov
opt_x_init['beta'] = eps
opt_x_init['beta_N'] = eps
#opt_x_init['u'] = np.zeros((1,1))
# for key in opt_x_init.keys():
#     opt_x_init[key] = opt_x_num[key,:,1]
res_nom = solver_eps(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
#opt_x_num = opt_x(res_nom['x'])
opt_x_num_nom = opt_x(res_nom['x'])
print(res_nom['f'])
print(solver_eps.stats()['return_status'])
# %% Plot the ellipsoids
fig, ax = plt.subplots(3,2, figsize=(10,10))
#ax[0,0].set_xlim(-2, 2)
#ax[0,0].set_ylim(-2, 2)
for i in range(0,N+1):
    confidence_ellipse(np.array(opt_x_num_nom['x',i]), np.array(opt_x_num_nom['P',i]), ax[0,0],dim=[5,6], edgecolor='r')
    #ax[0,0].plot(opt_x_num_nom['x',i][0], opt_x_num_nom['x',i][1], 'r.')
# Plot the trajectory
x_traj = np.array(ca.vertcat(*opt_x_num_nom['x',:]).reshape((nx,-1)))
u_traj = np.array(ca.vertcat(*opt_x_num_nom['u',:]).reshape((nu,-1)))
ax[0,0].plot(x_traj[0,:], x_traj[1,:], 'b.-')
#ax[0,0].plot(x_traj[5,:], x_traj[6,:], 'b.-')
ax[1,0].plot(x_traj[0,:])
ax[1,1].plot(x_traj[1,:])
ax[0,1].plot(x_traj[4,:])
ax[0,1].axhline(y=1, color='k', linestyle='--')
ax[0,1].axhline(y=-1, color='k', linestyle='--')

ax[1,1].axhline(y=- py_con, color='k', linestyle='--')
ax[1,1].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=-py_con, color='k', linestyle='--')
# Plot the safety distance
safe_dist=np.sum((x_traj[0:2,:]-x_traj[5:7,:])**2,axis=0)**0.5
# Get also the backoff from the constraint
beta = np.array(ca.vertcat(*opt_x_num_nom['beta',:]).reshape((nh_k,-1)))
backoff = (np.sum((x_traj[0:2,0:N]-x_traj[5:7,0:N])**2,axis=0)-beta[0,:]**0.5)**0.5
# Plot velocity
ax[2,1].plot(x_traj[3,:])
ax[2,1].axhline(y=2, color='k', linestyle='--')
ax[2,0].plot(safe_dist)
ax[2,0].fill_between(np.arange(N), backoff, safe_dist[0:N], alpha=0.5)
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
mode = 'mid' # Modes are 'ifelse', 'mid' or 'border'
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
x_traj = 0+ x_init
x_traj[3]=v_ref
for i in range(N):
    for s in range(ns):
        # Cost
        #J += stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K,x_traj)
        # Partitioning the ellipsoid
        if i == 0 and s == 0:
            # Initial ellipsoid from uncertainty propagation
            A_k = A_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            B_k = B_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            W_jac = W_func(opt_x['x',i,s], opt_x['u',i,s],np.zeros((nw,1)))
            A_K = A_k + B_k@K
            E = A_K.T@opt_x['P',i,s]@A_K + W_jac@W_cov@W_jac.T
            c = system(opt_x['x',0,0], opt_x['u',0,0],np.zeros((nw,1)))
            # Cutting plane
            a = opt_x['cutting_a']
            b = opt_x['cutting_b']
            # a = np.zeros((nx,1))
            # a[-1] = 1
            # b = 0
            # Help by restricting a
            g.append(opt_x['cutting_a',:-1])
            lb_g.append(np.zeros((nx-1,1)))
            ub_g.append(np.zeros((nx-1,1)))
            # # Rewrite for alpha = 0
            # g.append(a.T @ c - b)
            # lb_g.append(np.zeros((1,1)))
            # ub_g.append(np.zeros((1,1)))
            # tau = (1)/(nx+1)
            # sigma = (2)/(nx+1)
            # delta = (nx**2/ (nx**2-1))
            # c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
            # g.append(opt_x['x',i+1,0] - c_new)
            # lb_g.append(np.zeros((nx,1)))
            # ub_g.append(np.zeros((nx,1)))
            # E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
            # g.append((opt_x['P',i+1,0] - E_new).reshape((-1,1)))
            # lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            # ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            # # Now the other side of the cut
            # a1=-a
            # b1=-b
            # c_new1 = c - tau* (E @ a1)/(ca.sqrt(a1.T @ E @ a1))
            # g.append(opt_x['x',i+1,1] - c_new1)
            # lb_g.append(np.zeros((nx,1)))
            # ub_g.append(np.zeros((nx,1)))
            # E_new1 = delta*(E - sigma*(E @ a1 @ (E@a1).T)/(a1.T @ E @ a1))
            # g.append((opt_x['P',i+1,1] - E_new1).reshape((-1,1)))
            # lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            # ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))


            # Constrain alpha to be greater than 0, the other side will also be considered anyway
            # g.append(a.T @ c - b)
            # lb_g.append(np.zeros((1,1)))
            # ub_g.append(ca.inf*np.ones((1,1)))
            
            # g.append(-opt_x['alpha'] +0)
            # lb_g.append(-ca.inf*np.ones((1,1)))
            # ub_g.append(np.zeros((1,1)))

            # g.append(opt_x['alpha'] - 1)
            # lb_g.append(-ca.inf*np.ones((1,1)))
            # ub_g.append(np.zeros((1,1)))

            # g.append(opt_x['alpha'] * opt_x['gamma'] - (a.T@c-b))
            # lb_g.append(np.zeros((1,1)))
            # ub_g.append(np.zeros((1,1)))

            # g.append(opt_x['gamma']**2 - a.T@E@a)
            # lb_g.append(np.zeros((1,1)))
            # ub_g.append(np.zeros((1,1)))

            # g.append(-opt_x['gamma'] +1e-3)
            # lb_g.append(-ca.inf*np.ones((1,1)))
            # ub_g.append(np.zeros((1,1)))

            g.append(opt_x['gamma']**2 - a.T@E@a)
            lb_g.append(np.zeros((1,1)))
            ub_g.append(np.zeros((1,1)))
            
            #alpha = (a.T @ c - b) / ca.sqrt(a.T @ E @ a)
            alpha = (a.T @ c - b)/ opt_x['gamma']
            #alpha = opt_x['alpha']
            # Constraint to ensure the feasibility of the cutting plane
            g.append(alpha)
            if mode == 'ifelse':
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
            #c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
            c_new = c - tau* (E @ a)/opt_x['gamma']
            #c_new = c - tau* (E @ a)
            g.append(opt_x['x',i+1,0] - c_new)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            #E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
            E_new = delta*(E- sigma*(E @ a @ (E@a).T)/opt_x['gamma']**2)
            #E_new = delta*(E- sigma*(E @ a @ (E@a).T))
            g.append((opt_x['P',i+1,0]- E_new).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            # Now the other side of the cut
            a1=-a
            b1=-b
            # This alpha should be the negative one as the previous one
            #alpha1 = (a1.T @ c - b1) / ca.sqrt(a1.T @ E @ a1)
            #alpha1 = (a1.T @ c - b1)/ opt_x['gamma']
            if mode == 'ifelse':
                alpha1 = ca.if_else(alpha<1/nx, -alpha, -1/nx)
            elif mode == 'mid':
                alpha1 = -alpha
            elif mode == 'border':
                alpha1 = -1/nx
            tau1 = (1+nx*alpha1)/(nx+1)
            # if else formulation
            #tau1 = ca.if_else(alpha1>-1/nx, (1+nx*alpha1)/(nx+1), 0)
            sigma1 = (2*(1+nx*alpha1))/((nx+1))
            #sigma1 = ca.if_else(alpha1>-1/nx, (2*(1+nx*alpha1))/((nx+1)*(1+alpha1)), 0)
            delta1 = (nx**2/ (nx**2-1))*(1-alpha1)
            #delta1 = ca.if_else(alpha1>-1/nx, (nx**2/ (nx**2-1))*(1-alpha1**2), 1)
            #c_new1 = c - tau1* (E @ a1)/(ca.sqrt(a1.T @ E @ a1))
            if mode == 'border':
                c_new1 = c
            else:
                c_new1 = c - tau1* (E @ a1)/opt_x['gamma']
            #c_new1 = c - tau1* (E @ a1)
            g.append(opt_x['x',i+1,1]- c_new1)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            #E_new1 = delta1*(E - sigma1*(E @ a1 @ (E@a1).T)/(a1.T @ E @ a1))
            if mode == 'border':
                E_new1 = E
            else:
                E_new1 = delta1*(E*(1+alpha1)- sigma1*(E @ a1 @ (E@a1).T)/opt_x['gamma']**2)
            #E_new1 = delta1*(E- sigma1*(E @ a1 @ (E@a1).T))
            g.append((opt_x['P',i+1,1] - E_new1).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            


            # Uniqueness of a and b
            #g.append(opt_x['cutting_a'].T@opt_x['cutting_a']- 1)
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
            g.append((opt_x['P',i+1,s] - A_K.T@opt_x['P',i,s]@A_K - W_jac@W_cov@W_jac.T).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
        # Cost
        vol1=vol_fun(alpha)
        vol2=vol_fun(alpha1)
        if s == 0:
            J += (vol1/(vol1+vol2))*stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K,x_traj)
        elif s == 1:
            J += (vol2/(vol1+vol2))*stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K,x_traj)
        # Constraint
        g.append(opt_x['beta',i,s] - H_fun(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K))
        lb_g.append(np.zeros((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))
        # g.append(con(opt_x['x',i,s], opt_x['u',i,s])+ca.sqrt(opt_x['beta',i,s]+eps))
        # lb_g.append(-np.ones((nh_k,1))*ca.inf)
        # ub_g.append(np.zeros((nh_k,1)))
        # Reformulating the constriant to get rid of the sqrt
        g.append(con(opt_x['x',i,s], opt_x['u',i,s]))
        lb_g.append(-ca.inf*np.ones((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))
        g.append(opt_x['beta',i,s]+eps-con(opt_x['x',i,s], opt_x['u',i,s])**2)
        lb_g.append(-ca.inf*np.ones((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))

        # g.append(opt_x['beta',i,s])
        # lb_g.append(eps*np.ones((nh_k,1)))
        # ub_g.append(ca.inf*np.ones((nh_k,1)))
    # Update the nominal trajectory
    x_traj[0] = x_traj[0] + v_ref*dt


for s in range(ns):
    # Terminal cost
    if s == 0:
        J += (vol1/(vol1+vol2))*terminal_cost_fcn(opt_x['x',N,s], opt_x['P',N,s],x_traj)
    elif s == 1:
        J += (vol2/(vol1+vol2))*terminal_cost_fcn(opt_x['x',N,s], opt_x['P',N,s],x_traj)
    #J += terminal_cost_fcn(opt_x['x',N,s], opt_x['P',N,s],x_traj)
    # Terminal constraint
    g.append(opt_x['beta_N',s] - HN_fun(opt_x['x',N,s], opt_x['P',N,s]))
    lb_g.append(np.zeros((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))
    # g.append(ter_con(opt_x['x',N,s])+ca.sqrt(opt_x['beta_N',s]+eps))
    # lb_g.append(-np.ones((nh_N,1))*ca.inf)
    # ub_g.append(np.zeros((nh_N,1)))
    g.append(ter_con(opt_x['x',N,s]))
    lb_g.append(-ca.inf*np.ones((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))
    g.append(opt_x['beta_N',s]+eps-ter_con(opt_x['x',i,s])**2)
    lb_g.append(-ca.inf*np.ones((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))

J+= opt_x['cutting_a'].T@opt_x['cutting_a']

g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)

prob = {'f': J, 'x': opt_x.cat, 'g': g,'p': x_init}
solver_opt = {'ipopt.max_iter':3000,'ipopt.linear_solver': 'MA27','ipopt.print_level':5, 'print_time':1}
solver_cut = ca.nlpsol('solver', 'ipopt', prob, solver_opt)

# %% Test solver
x_init = np.array([-2, 0.0, 0, 1.6, 0, 2, 0])
opt_x_init = opt_x(0)
# opt_x_init = opt_x_num
# opt_x_init['x',:,0] = opt_x_num_nom['x']
# opt_x_init['x',:,1] = opt_x_num_nom['x']
# opt_x_init['x',:,1,1] = [-opt_x_init['x',i,0,1] for i in range(N+1)] 
opt_x_init['x',:,0] = x_ref_func(x_init)
opt_x_init['x',:,1] = x_ref_func(x_init)
opt_x_init['x',:,0,1] = 1
opt_x_init['x',:,1,1] = -1.3

opt_x_init['P'] = np.zeros((nx,nx))+W_cov
#opt_x_init['P',:,0] = [opt_x_init['P',i,1] for i in range(N+1)]
opt_x_init['cutting_a'] = np.array([0, 0, 0, 0, 0, 0, 5])
opt_x_init['cutting_b'] = 0.0
opt_x_init['gamma'] = 1
opt_x_init['beta'] = np.ones((nh_k,1))
# opt_x_init['beta',:,0] = [opt_x_num_nom['beta',i] for i in range(N)]
# opt_x_init['beta',:,1] = [opt_x_num_nom['beta',i] for i in range(N)]
# opt_x_init['beta_N'] = np.ones((nh_N,1))
# opt_x_init['u',:,0] = opt_x_num_nom['u']
# opt_x_init['u',:,1] = opt_x_num_nom['u']
# opt_x_init['u',:,1,1] = [-opt_x_init['u',i,0,1] for i in range(N)]
opt_x_init['u'] = np.zeros((nu,1))
#opt_x_init = opt_x_num
#opt_x_init['x',:,0,0] = [opt_x_init['x',i,0,1] for i in range(N+1)]
#opt_x_init['x',:,1,1] = [-opt_x_init['x',i,1,1] for i in range(N+1)]  
#opt_x_init['cutting_a'] = np.array([0, 0, 0, 0, 0, 0, 1])
#opt_x_init['cutting_b'] = 0
res = solver_cut(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
opt_x_num = opt_x(res['x'])
print(res['f']-opt_x_num['cutting_a'].T@opt_x_num['cutting_a'])
print(solver_cut.stats()['return_status'])
# %% Evaluate alpha
A_k = A_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
B_k = B_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
W_jac = W_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
A_K = A_k + B_k@K
E = A_K.T@opt_x_num['P',0,0]@A_K + W_jac@W_cov@W_jac.T
c = system(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
# Cutting plane
a = opt_x_num['cutting_a']
b = opt_x_num['cutting_b']
alpha = (a.T @ c - b)/ca.sqrt(a.T @ E @ a)
print (alpha)
# %% Plot the ellipsoids
fig, ax = plt.subplots(3,2,figsize=(10,10))
for i in range(N+1):
    confidence_ellipse(np.array(opt_x_num['x',i,0]), np.array(opt_x_num['P',i,0]), ax[0,0], dim=[5,6],edgecolor='r')
    confidence_ellipse(np.array(opt_x_num['x',i,1]), np.array(opt_x_num['P',i,1]), ax[0,0],dim=[5,6], edgecolor='b')
    # Plot nominal confidence ellispoid
    confidence_ellipse(np.array(opt_x_num_nom['x',i]), np.array(opt_x_num_nom['P',i]), ax[0,0],dim=[5,6], edgecolor='k')
    ax[0,0].plot(opt_x_num['x',i,0][5], opt_x_num['x',i,0][6], 'r.')
    ax[0,0].plot(opt_x_num['x',i,1][5], opt_x_num['x',i,1][6], 'b.')
    ax[0,0].plot(opt_x_num['x',i,0][0], opt_x_num['x',i,0][1], 'rx')
    ax[0,0].plot(opt_x_num['x',i,1][0], opt_x_num['x',i,1][1], 'bx')
# Plot half space
x = np.linspace(-2, 2, 1000)
b = opt_x_num['cutting_b']
a = opt_x_num['cutting_a']
y = (b - a[-2]*x)/a[-1]
ax[0,0].plot(x, y,linewidth=1)
# ax[0,0].set_xlim(-0.6, 2)
# ax[0,0].set_ylim(-0.6, 2)
# Plot ellipsoid
A_k = A_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
B_k = B_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
W_jac = W_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
A_K = A_k + B_k@K
E = A_K.T@opt_x_num['P',0,0]@A_K + W_jac@W_cov@W_jac.T
c = system(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
confidence_ellipse(np.array(c), np.array(E), ax[0,0],dim=[5,6], edgecolor='g')
# Plot the trajectory

x_traj0 = np.array(ca.vertcat(*opt_x_num['x',:,0])).reshape((-1,nx)).T
x_traj1 = np.array(ca.vertcat(*opt_x_num['x',:,1])).reshape((-1,nx)).T
u_traj0 = np.array(ca.vertcat(*opt_x_num['u',:,0]).reshape((-1,nu))).T
u_traj1 = np.array(ca.vertcat(*opt_x_num['u',:,1]).reshape((-1,nu))).T
ax[0,0].plot(x_traj0[0,:], x_traj0[1,:], 'r.-')
ax[0,0].plot(x_traj1[0,:], x_traj1[1,:], 'b.-')
#ax[0,0].plot(x_traj0[5,:], x_traj0[6,:], 'r.-')
#ax[0,0].plot(x_traj1[5,:], x_traj1[6,:], 'b.-')
ax[1,0].plot(x_traj0[0,:],'r')
ax[1,0].plot(x_traj1[0,:],'b')
ax[1,1].plot(x_traj0[1,:],'r')
ax[1,1].plot(x_traj1[1,:],'b')
ax[0,1].plot(x_traj0[4,:],'r')
ax[0,1].plot(x_traj1[4,:],'b')
ax[0,1].axhline(y=1, color='k', linestyle='--')
ax[0,1].axhline(y=-1, color='k', linestyle='--')

ax[1,1].axhline(y=- py_con, color='k', linestyle='--')
ax[1,1].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=py_con, color='k', linestyle='--')
ax[0,0].axhline(y=-py_con, color='k', linestyle='--')
# Plot the safety distance
safe_dist0=np.sum((x_traj0[0:2,:]-x_traj0[5:7,:])**2,axis=0)**0.5
safe_dist1=np.sum((x_traj1[0:2,:]-x_traj1[5:7,:])**2,axis=0)**0.5
ax[2,0].plot(safe_dist0)
ax[2,0].plot(safe_dist1)
ax[2,0].axhline(y=Del_safe, color='k', linestyle='--')
#PLot velocity
ax[2,1].plot(x_traj0[3,:],'r')
ax[2,1].plot(x_traj1[3,:],'b')
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
fig.align_labels()
fig.tight_layout()
# Plot the trajectory

# %% Plot the divided ellipoid
fig, ax = plt.subplots()
A_k = A_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
B_k = B_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
W_jac = W_func(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1)))
A_K = A_k + B_k@K
E = np.array(A_K.T@opt_x_num['P',0,0]@A_K + W_jac@W_cov@W_jac.T)
c = np.array(system(opt_x_num['x',0,0], opt_x_num['u',0,0],np.zeros((nw,1))))
confidence_ellipse(np.array(c), np.array(E),ax,dim=[5,6], edgecolor='g')
# Plot half space
x = np.linspace(-2, 2, 1000)
b = np.array(opt_x_num['cutting_b'])
a = np.array(opt_x_num['cutting_a'])
y = (b - a[5]*x)/a[6]
ax.plot(x, y.T,linewidth=1)
# The other way around
# y = np.linspace(-2, 2, 1000)
# b = np.array(opt_x_num['cutting_b'])
# a = np.array(opt_x_num['cutting_a'])
# x = (b - a[1]*y)/a[0]
# ax.plot(x.T, y,linewidth=1)
ax.set_xlim(1.4, 2)
ax.set_ylim(-1/4, 0.25)
c_new,E_new = get_next_ellipse(c, E, a, b)
confidence_ellipse(c_new, E_new, ax,dim=[5,6], edgecolor='r',linestyle='--')
c_new1,E_new1 = get_next_ellipse(c, E, -a, -b)
confidence_ellipse(c_new1, E_new1, ax,dim=[5,6], edgecolor='b',linestyle='--')

confidence_ellipse(np.array(opt_x_num_nom['x',1]), np.array(opt_x_num_nom['P',1]), ax,dim=[5,6], edgecolor='k',linestyle = '--')

###########################################################################################################

# %% Try more partitions
# %% Set up ellipsoidal cutting
nc = 2 # Number of ellipsoids per time step
n_rob = 3 # Robust horizon
ns = nc**n_rob # Number of scenarios
N=10
opt_x = ca_tools.struct_symSX([ca_tools.entry('x',repeat = [N+1,ns], shape=(nx)),
                                 ca_tools.entry('P', repeat = [N+1,ns], shape=(nx,nx)),
                                 ca_tools.entry('cutting_a', repeat = [n_rob,ns],shape = (nx,1)),
                                 ca_tools.entry('cutting_b', repeat = [n_rob,ns], shape = (1)),
                                 ca_tools.entry('beta', repeat = [N,ns], shape=(nh_k)),
                                 ca_tools.entry('beta_N', repeat = [ns],shape=(nh_N)),
                                 ca_tools.entry('u', repeat = [N,ns], shape=(1))])
# %%
eps = 1e-5
J = 0
g = []
lb_g = []
ub_g = []

x_init = ca.SX.sym('x_init', nx)

g.append((opt_x['P',0,0] ).reshape((-1,1)))
lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
g.append(opt_x['x',0,0] - x_init)
lb_g.append(np.zeros((nx,1)))
ub_g.append(np.zeros((nx,1)))
    # g.append(opt_x['u',0,s]-opt_x['u',0,0])
    # lb_g.append(np.zeros((nu,1)))
    # ub_g.append(np.zeros((nu,1)))

for i in range(N):
    for s in range(ns):
        # Cost
        J += stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s])
    if i <n_rob:
        # Non anticipativity constraints
        for s in range(ns):
            s0 = s//(nc**(n_rob-i))*(nc**(n_rob-i)) #index of the first branch of the parent node
            g.append(opt_x['x',i,s] - opt_x['x',i,s0])
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            g.append((opt_x['P',i,s] - opt_x['P',i,s0]).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            g.append(opt_x['u',i,s] - opt_x['u',i,s0])
            lb_g.append(np.zeros((nu,1)))
            ub_g.append(np.zeros((nu,1)))
            g.append(opt_x['cutting_a',i,s] - opt_x['cutting_a',i,s0])
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            g.append(opt_x['cutting_b',i,s] - opt_x['cutting_b',i,s0])
            lb_g.append(np.zeros((1,1)))
            ub_g.append(np.zeros((1,1)))
            if s == s0:
                # Initial ellipsoid from uncertainty propagation
                E = A_K.T@opt_x['P',i,s]@A_K + sig_w**2*W_jac.T@W_jac
                c = system(opt_x['x',i,s], opt_x['u',i,s])
                # Cutting plane
                a = opt_x['cutting_a',i,s]
                b = opt_x['cutting_b',i,s]
                # Rewrite for alpha = 0
                g.append(a.T @ c - b)
                lb_g.append(np.zeros((1,1)))
                ub_g.append(np.zeros((1,1)))
                tau = (1)/(nx+1)
                sigma = (2)/(nx+1)
                delta = (nx**2/ (nx**2-1))
                c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
                g.append(opt_x['x',i+1,s] - c_new)
                lb_g.append(np.zeros((nx,1)))
                ub_g.append(np.zeros((nx,1)))
                E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
                g.append((opt_x['P',i+1,s] - E_new).reshape((-1,1)))
                lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                # Now the other side of the cut
                s_next = s0 + nc**(n_rob-i-1)
                a1=-a
                b1=-b
                c_new1 = c - tau* (E @ a1)/(ca.sqrt(a1.T @ E @ a1))
                g.append(opt_x['x',i+1,s_next] - c_new1)
                lb_g.append(np.zeros((nx,1)))
                ub_g.append(np.zeros((nx,1)))
                E_new1 = delta*(E - sigma*(E @ a1 @ (E@a1).T)/(a1.T @ E @ a1))
                g.append((opt_x['P',i+1,s_next] - E_new1).reshape((-1,1)))
                lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                # Add if multiple cuts are considered




                # alpha = (a.T @ c - b)/ca.sqrt(a.T @ E @ a)
                # # Constraint to ensure the feasibility of the cutting plane
                # g.append(alpha)
                # lb_g.append(-1/nx)
                # ub_g.append(1/nx)
                # tau = (1+nx*alpha)/(nx+1)
                # sigma = (2*(1+nx*alpha))/((nx+1)*(1+alpha))
                # delta = (nx**2/ (nx**2-1))*(1-alpha**2)
                # c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
                # g.append(opt_x['x',i+1,s0] - c_new)
                # lb_g.append(np.zeros((nx,1)))
                # ub_g.append(np.zeros((nx,1)))
                # E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
                # g.append((opt_x['P',i+1,s0] - E_new).reshape((-1,1)))
                # lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                # ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                # # Now the other side of the cut
                # a1=-a
                # b1=-b
                # # This alpha should be the negative one as the previous one
                # alpha1 = (a1.T @ c - b1)/ca.sqrt(a1.T @ E @ a1)
                # tau1 = (1+nx*alpha1)/(nx+1)
                # sigma1 = (2*(1+nx*alpha1))/((nx+1)*(1+alpha1))
                # delta1 = (nx**2/ (nx**2-1))*(1-alpha1**2)
                # c_new1 = c - tau1* (E @ a1)/(ca.sqrt(a1.T @ E @ a1))
                # g.append(opt_x['x',i+1,(s//(nc**i)+1)*(nc**i)] - c_new1)
                # lb_g.append(np.zeros((nx,1)))
                # ub_g.append(np.zeros((nx,1)))
                # E_new1 = delta1*(E - sigma1*(E @ a1 @ (E@a1).T)/(a1.T @ E @ a1))
                # g.append((opt_x['P',i+1,(s//(nc**i)+1)*(nc**i)] - E_new1).reshape((-1,1)))
                # lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                # ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
                


                # Uniqueness of a and b
                g.append(ca.sumsqr(opt_x['cutting_a',i,s0]) - 1)
                lb_g.append(np.zeros((1,1)))
                ub_g.append(np.zeros((1,1)))

    elif i >n_rob-1:
        for s in range(ns):
            # Nominal dynamics
            g.append(opt_x['x',i+1,s] - system(opt_x['x',i,s], opt_x['u',i,s]))
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            # Uncertainty dynamics
            g.append((opt_x['P',i+1,s] - A_K.T@opt_x['P',i,s]@A_K - sig_w**2*W_jac.T@W_jac).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    for s in range(ns):
        # Constraint
        g.append(opt_x['beta',i,s] - H_fun(opt_x['x',i,s], opt_x['u',i,s], opt_x['P',i,s], K))
        lb_g.append(np.zeros((nh_k,1)))
        ub_g.append(np.zeros((nh_k,1)))
        g.append(con(opt_x['x',i,s], opt_x['u',i,s])+ca.sqrt(opt_x['beta',i,s]+eps))
        lb_g.append(-np.ones((nh_k,1))*ca.inf)
        ub_g.append(np.zeros((nh_k,1)))
            # g.append(opt_x['beta',i,s])
            # lb_g.append(eps*np.ones((nh_k,1)))
            # ub_g.append(ca.inf*np.ones((nh_k,1)))

for s in range(ns):
    # Terminal cost
    J += terminal_cost_fcn(opt_x['x',N,s])
    # Terminal constraint
    g.append(opt_x['beta_N',s] - HN_fun(opt_x['x',N,s], opt_x['P',N,s]))
    lb_g.append(np.zeros((nh_N,1)))
    ub_g.append(np.zeros((nh_N,1)))
    g.append(ter_con(opt_x['x',N,s])+ca.sqrt(opt_x['beta_N',s]+eps))
    lb_g.append(-np.ones((nh_N,1))*ca.inf)
    ub_g.append(np.zeros((nh_N,1)))

g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)

prob = {'f': J, 'x': opt_x.cat, 'g': g,'p': x_init}
solver_opt = {'ipopt.linear_solver': 'mumps','ipopt.print_level':5, 'print_time':1}
solver_cut = ca.nlpsol('solver', 'ipopt', prob, solver_opt)

# %% Test solver
x_init = np.array([[1], [1]])
opt_x_init = opt_x(0)
opt_x_init['x'] = x_init
opt_x_init['P'] = np.zeros((nx,nx))+eps*np.eye(nx)
opt_x_init['cutting_a'] = ca.DM([1, 1])
opt_x_init['cutting_b'] = 1
opt_x_init['beta'] = np.ones((nh_k,1))*1
opt_x_init['beta_N'] = eps
opt_x_init['u'] = np.zeros((1,1))
res = solver_cut(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
opt_x_num = opt_x(res['x'])
# %% Evaluate alpha
E = A_K.T@opt_x_num['P',0,0]@A_K + sig_w**2*W_jac.T@W_jac
c = system(opt_x_num['x',0,0], opt_x_num['u',0,0])
# Cutting plane
a = -opt_x_num['cutting_a',0,0]
b = -opt_x_num['cutting_b',0,0]
alpha = (a.T @ c - b)/ca.sqrt(a.T @ E @ a)
print (alpha)
# %% Plot the ellipsoids
fig, ax = plt.subplots(2,2)
for i in range(N+1):
    confidence_ellipse(np.array(opt_x_num['x',i,0]), np.array(opt_x_num['P',i,0]), ax[0,0], edgecolor='r')
    confidence_ellipse(np.array(opt_x_num['x',i,1]), np.array(opt_x_num['P',i,1]), ax[0,0], edgecolor='b')
    ax[0,0].plot(opt_x_num['x',i,0][0], opt_x_num['x',i,0][1], 'r.')
    ax[0,0].plot(opt_x_num['x',i,1][0], opt_x_num['x',i,1][1], 'b.')
# Plot half space
x = np.linspace(-2, 2, 1000)
b = opt_x_num['cutting_b',0,0]
a = opt_x_num['cutting_a',0,0]
y = (b - a[0]*x)/a[1]
ax[0,0].plot(x, y,linewidth=1)
ax[0,0].set_xlim(-0.6, 2)
ax[0,0].set_ylim(-0.6, 2)
# Plot ellipsoid
E = A_K.T@opt_x_num['P',0,0]@A_K + sig_w**2*W_jac.T@W_jac
c = system(opt_x_num['x',0,0], opt_x_num['u',0,0])
confidence_ellipse(np.array(c), np.array(E), ax[0,0], edgecolor='g')
# Plot the trajectory
for i in range(N+1):
    ax[1,0].plot(i,opt_x_num['x',i,0][0], 'r.')
    ax[1,1].plot(i,opt_x_num['x',i,0][1], 'r.')
    ax[1,0].plot(i,opt_x_num['x',i,1][0], 'b.')
    ax[1,1].plot(i,opt_x_num['x',i,1][1], 'b.')
    if i < N:
        ax[0,1].plot(i,opt_x_num['u',i,0], 'r.')
        ax[0,1].plot(i,opt_x_num['u',i,1], 'b.')
ax[1,1].axhline(y=-0.5 , color='k', linestyle='--')
ax[0,0].axhline(y=-0.5 , color='k', linestyle='--')
ax[0,0].set_title('Ellipsoids')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,1].set_title('input')
ax[0,1].set_xlabel('time')
ax[1,0].set_title('x1')
ax[1,0].set_xlabel('time')
ax[1,1].set_title('x2')
ax[1,1].set_xlabel('time')
fig.align_labels()
fig.tight_layout()
# Plot the trajectory

# %% Plot the divided ellipoid
fig, ax = plt.subplots()
E = np.array(A_K.T@opt_x_num['P',0,0]@A_K + sig_w**2*W_jac.T@W_jac)
c = np.array(system(opt_x_num['x',0,0], opt_x_num['u',0,0]))
confidence_ellipse(np.array(c), np.array(E), ax, edgecolor='g')
# Plot half space
# x = np.linspace(-2, 2, 1000)
# b = np.array(opt_x_num['cutting_b'])
# a = np.array(opt_x_num['cutting_a'])
# y = (b - a[0]*x)/a[1]
# ax.plot(x, y.T,linewidth=1)
# The other way around
y = np.linspace(-2, 2, 1000)
b = np.array(opt_x_num['cutting_b',0,0])
a = np.array(opt_x_num['cutting_a',0,0])
x = (b - a[1]*y)/a[0]
ax.plot(x.T, y,linewidth=1)
# ax.set_xlim(1, 2)
# ax.set_ylim(0, 1)
c_new,E_new = get_next_ellipse(c, E, a, b)
confidence_ellipse(c_new, E_new, ax, edgecolor='r')
c_new1,E_new1 = get_next_ellipse(c, E, -a, -b)
confidence_ellipse(c_new1, E_new1, ax, edgecolor='b',linestyle='--')

# %%
