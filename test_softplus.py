# %%
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def softplus(x,beta=1):
    return np.log(1+np.exp(beta*x))/beta

def mod_softplus(x,c,beta=1,gamma=1,delta=1):
    y = x-c
    return np.log(1+gamma*np.exp(delta*beta*y))/beta +c

x = np.linspace(-1,1,100)
y = softplus(x,10)
# fig, ax = plt.subplots()
# ax.plot(x,y)
# ax.plot(x,x)
# %% reformulate to show behavior 
# if x<c then constant
# if x>c then linear
ce= -0.5
bet = 100
x = np.linspace(-1,1,100)
y = mod_softplus(x,ce,bet)
fig, ax = plt.subplots()
ax.plot(x,y)
ax.plot(x,x)
# %% Set up optimization problem to find gamma and beta
def find_param(ce,bet,eps):
    eps=1e-4
    # Define the decision variables
    beta = ca.SX.sym('beta')
    gamma = ca.SX.sym('gamma')
    delta = ca.SX.sym('delta')
    c = ca.SX.sym('c')
    x = ca.SX.sym('x')
    # Define the structure of the function
    f = ca.log(1+gamma*ca.exp(delta*beta*(x-c)))/beta + c
    Fun = ca.Function('f',[x,beta,gamma,c,delta],[f])
    J = 0
    g = []
    lb_g = []
    ub_g = []
    # First constraint is, that at zero, the function should be zero
    g.append(Fun(0,beta,gamma,c,delta))
    lb_g.append(0)
    ub_g.append(0)
    # Second constraint is, that at c, the function should be c-eps
    g.append(Fun(c,beta,gamma,c,delta)-c-eps)
    lb_g.append(0)
    ub_g.append(0)

    g = ca.vertcat(*g)
    lb_g = ca.vertcat(*lb_g)
    ub_g = ca.vertcat(*ub_g)
    # Setup problem
    nlp = {'x':ca.vertcat(gamma,delta),
        'f':0,
        'g':g,
        'p':ca.vertcat(beta,c)}

    solver = ca.nlpsol('solver','ipopt',nlp)

    res = solver(x0=[0.1,1.1],lbg=lb_g,ubg=ub_g,p=[bet,ce])
    print(res['x'])
    gamma_opt = res['x'][0]
    delta_opt = res['x'][1]
    res_fun = Fun(x,bet,gamma_opt,ce,delta_opt) 
    fun_ret = ca.Function('f',[x],[res_fun])
    return fun_ret,gamma_opt,delta_opt
# Plot the modified softplus function
funy,gamma_opt,delta_opt = find_param(ce,bet,1e-4)
x = np.linspace(-1,0,100)
y = mod_softplus(x,ce,bet,gamma_opt,delta_opt)
z = funy(x)
fig, ax = plt.subplots()
ax.plot(x,y)
ax.plot(x,z)
ax.plot(x,x)
ax.plot(x,[x[i] if x[i]>ce else ce for i in range(len(x))])

# %%
