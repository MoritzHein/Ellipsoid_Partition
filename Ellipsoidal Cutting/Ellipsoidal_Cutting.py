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
def confidence_ellipse(center,cov, ax, facecolor='none', **kwargs):
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
    mean_x = center[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) 
    mean_y = center[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
# %% Play around with ellipsoidal cutting
# Initialize ellispoid with (x-c)^T E^-1 (x-c) <= 1
E = np.array([[10, 1], [1, 2]]) # Propotional to the confidence region
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
    tau = (1+n*alpha)/(n+1)
    sigma = (2*(1+n*alpha))/((n+1)*(1+alpha))
    delta = (n**2/ (n**2-1))*(1-alpha**2)

    c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
    E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
    return c_new, E_new

# get hyperplane
a = np.array([[1], [1]])
b = 1
c_new, E_new = get_next_ellipse(c, E, a, b)
confidence_ellipse(c_new, E_new, ax, edgecolor='b')

# Get other ellipsoid
c_new, E_new = get_next_ellipse(c, E, -a, -b)
confidence_ellipse(c_new, E_new, ax, edgecolor='g')


# %% system model 2D linear double integrator
T=0.1
nx = 2
nu = 1
nw = 2
A=np.array([[1,1],[0,1]])
B=np.array([[0.5],[1]])
C = np.array([[1, 0]])
D = np.array([[0]])
# casadi function
x = ca.SX.sym('x', 2)
u = ca.SX.sym('u')
x_next = ca.mtimes(A, x) + B*u
system = ca.Function('sys', [x, u], [x_next])
# Feedback controller - LQR
K,_,_ = control.dlqr(A, B, np.eye(2), 1)
K=-K.reshape(1,2)
print(K)
A_K = A + B@K
print(np.linalg.eig(A_K))
# Disturbances: Additive ellipsoidal noise
sig_w = 0.1
W_jac = np.eye(nx)
# %% Simulate a trajectory
x0 = np.array([[1], [1]])
N = 100
x_traj = np.zeros((nx, N+1))
x_traj[:,0] = x0.flatten()
for i in range(N):
    x_traj[:,i+1] = (system(x_traj[:,i], K@x_traj[:,i]) + sig_w*np.random.randn(nx,1)).full().flatten()  
# Plot
fig, ax = plt.subplots()
ax.plot(x_traj[0,:], x_traj[1,:], 'r.-')




# %%

# Constraint: x1 < 2, x2 < 2, x1 > -0.5, x2 > -0.5, u < 5, u > -5
h= []
h.append(x[0] - 2)
h.append(x[1] - 2)
h.append(-x[0] -0.5)
h.append(-x[1] -0.5)
h_N = ca.vertcat(*h)
nh_N = h_N.shape[0]
h.append(u-1)
h.append(-u-1),
h = ca.vertcat(*h)
nh_k = h.shape[0]
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
Q = np.eye(2)
R = 1
stage_cost = x.T@Q@x + u*R@u
stage_cost_fcn = ca.Function('stage_cost', [x, u], [stage_cost])
terminal_cost = 5*x.T@Q@x
terminal_cost_fcn = ca.Function('terminal_cost', [x], [terminal_cost])
#%% Set up ellispoidal optimization problem
N=10
opt_x = ca_tools.struct_symSX([ca_tools.entry('x',repeat = [N+1], shape=(nx)),
                                 ca_tools.entry('P', repeat = [N+1], shape=(nx,nx)),
                                 ca_tools.entry('beta', repeat = [N], shape=(nh_k)),
                                 ca_tools.entry('beta_N', shape=(nh_N)),
                                 ca_tools.entry('u', repeat = [N], shape=(1))])
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

for i in range(N):
    # Cost
    J += stage_cost_fcn(opt_x['x',i], opt_x['u',i])
    # if i >0:
    #     J+= (opt_x['u',i]-opt_x['u',i-1]).T@R@(opt_x['u',i]-opt_x['u',i-1])
    # else:
    #     J+= opt_x['u',i].T@R@opt_x['u',i]
    # Nominal dynamics
    g.append(opt_x['x',i+1] - system(opt_x['x',i], opt_x['u',i]))
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
    # Uncertainty dynamics
    g.append((opt_x['P',i+1] - A_K.T@opt_x['P',i]@A_K - sig_w**2*W_jac.T@W_jac).reshape((-1,1)))
    lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
    # Constraint
    g.append(opt_x['beta',i] - H_fun(opt_x['x',i], opt_x['u',i], opt_x['P',i], K))
    lb_g.append(np.zeros((nh_k,1)))
    ub_g.append(np.zeros((nh_k,1)))
    g.append(con(opt_x['x',i], opt_x['u',i])+ca.sqrt(opt_x['beta',i]+eps))
    lb_g.append(-np.ones((nh_k,1))*ca.inf)
    ub_g.append(np.zeros((nh_k,1)))

# Terminal cost
J += terminal_cost_fcn(opt_x['x',N])
# Terminal constraint
g.append(opt_x['beta_N'] - HN_fun(opt_x['x',N], opt_x['P',N]))
lb_g.append(np.zeros((nh_N,1)))
ub_g.append(np.zeros((nh_N,1)))
g.append(ter_con(opt_x['x',N])+ca.sqrt(opt_x['beta_N']+eps))
lb_g.append(-np.ones((nh_N,1))*ca.inf)
ub_g.append(np.zeros((nh_N,1)))

g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)

prob = {'f': J, 'x': opt_x.cat, 'g': g,'p':x_init}
solver_opt = {'ipopt.linear_solver': 'MA57','ipopt.print_level':5, 'print_time':1, 'ipopt.tol':1e-8}
solver_eps = ca.nlpsol('solver', 'ipopt', prob, solver_opt)
# %% Test solver
x_init = np.array([[1], [1]])
opt_x_init = opt_x(0)
opt_x_init['x'] = x_init
opt_x_init['P'] = np.zeros((nx,nx))+1*np.eye(nx)
opt_x_init['beta'] = np.ones((nh_k,1))*eps
opt_x_init['beta_N'] = eps
opt_x_init['u'] = np.zeros((1,1))
res = solver_eps(x0=opt_x_init,p=x_init,lbg=lb_g, ubg=ub_g)
opt_x_num = opt_x(res['x'])
# %% Plot the ellipsoids
fig, ax = plt.subplots(2,2)
ax[0,0].set_xlim(-2, 2)
ax[0,0].set_ylim(-2, 2)
for i in range(0,N+1):
    confidence_ellipse(np.array(opt_x_num['x',i]), np.array(opt_x_num['P',i]), ax[0,0], edgecolor='r')
    ax[0,0].plot(opt_x_num['x',i][0], opt_x_num['x',i][1], 'r.')
# Plot the trajectory
for i in range(N+1):
    ax[1,0].plot(i,opt_x_num['x',i][0], 'r.')
    ax[1,1].plot(i,opt_x_num['x',i][1], 'r.')
    if i < N:
        ax[0,1].plot(i,opt_x_num['u',i], 'r.')
ax[1,1].axhline(y=-0.5 , color='k', linestyle='--')
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


# %% Set up ellipsoidal cutting
ns = 2 # Number of ellipsoids
N=10
opt_x = ca_tools.struct_symSX([ca_tools.entry('x',repeat = [N+1,ns], shape=(nx)),
                                 ca_tools.entry('P', repeat = [N+1,ns], shape=(nx,nx)),
                                 ca_tools.entry('cutting_a', shape = (nx,1)),
                                 ca_tools.entry('cutting_b', shape = (1)),
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

for i in range(N):
    for s in range(ns):
        # Cost
        J += stage_cost_fcn(opt_x['x',i,s], opt_x['u',i,s])
        if i == 0 and s == 0:
            # Initial ellipsoid from uncertainty propagation
            E = A_K.T@opt_x['P',0,0]@A_K + sig_w**2*W_jac.T@W_jac
            c = system(opt_x['x',0,0], opt_x['u',0,0])
            # Cutting plane
            a = opt_x['cutting_a']
            b = opt_x['cutting_b']
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





            alpha = (a.T @ c - b)/ca.sqrt(a.T @ E @ a)
            # Constraint to ensure the feasibility of the cutting plane
            g.append(alpha)
            lb_g.append(-1)
            ub_g.append(1)
            tau = (1+nx*alpha)/(nx+1)
            sigma = (2*(1+nx*alpha))/((nx+1)*(1+alpha))
            delta = (nx**2/ (nx**2-1))*(1-alpha**2)
            c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
            g.append(opt_x['x',i+1,0] - c_new)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
            g.append((opt_x['P',i+1,0] - E_new).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            # Now the other side of the cut
            a1=-a
            b1=-b
            # This alpha should be the negative one as the previous one
            alpha1 = (a1.T @ c - b1)/ca.sqrt(a1.T @ E @ a1)
            tau1 = (1+nx*alpha1)/(nx+1)
            sigma1 = (2*(1+nx*alpha1))/((nx+1)*(1+alpha1))
            delta1 = (nx**2/ (nx**2-1))*(1-alpha1**2)
            c_new1 = c - tau1* (E @ a1)/(ca.sqrt(a1.T @ E @ a1))
            g.append(opt_x['x',i+1,1] - c_new1)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            E_new1 = delta1*(E - sigma1*(E @ a1 @ (E@a1).T)/(a1.T @ E @ a1))
            g.append((opt_x['P',i+1,1] - E_new1).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            


            # Uniqueness of a and b
            g.append(ca.sumsqr(opt_x['cutting_a']) - 1)
            lb_g.append(np.zeros((1,1)))
            ub_g.append(np.zeros((1,1)))

        elif i >0:
            # Nominal dynamics
            g.append(opt_x['x',i+1,s] - system(opt_x['x',i,s], opt_x['u',i,s]))
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))
            # Uncertainty dynamics
            g.append((opt_x['P',i+1,s] - A_K.T@opt_x['P',i,s]@A_K - sig_w**2*W_jac.T@W_jac).reshape((-1,1)))
            lb_g.append(np.zeros((nx,nx)).reshape((-1,1)))
            ub_g.append(np.zeros((nx,nx)).reshape((-1,1)))
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
opt_x_init['cutting_a'] = ca.DM([0, 1])
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
a = -opt_x_num['cutting_a']
b = -opt_x_num['cutting_b']
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
b = opt_x_num['cutting_b']
a = opt_x_num['cutting_a']
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
b = np.array(opt_x_num['cutting_b'])
a = np.array(opt_x_num['cutting_a'])
x = (b - a[1]*y)/a[0]
ax.plot(x.T, y,linewidth=1)
# ax.set_xlim(1, 2)
# ax.set_ylim(0, 1)
c_new,E_new = get_next_ellipse(c, E, a, b)
confidence_ellipse(c_new, E_new, ax, edgecolor='r')
c_new1,E_new1 = get_next_ellipse(c, E, -a, -b)
confidence_ellipse(c_new1, E_new1, ax, edgecolor='b',linestyle='--')

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
