# %% Import stuff
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import matplotlib.transforms as transforms
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
    print(cov)
    
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

# %%
def get_next_ellipse(c, E, a, b):
    n= E.shape[0]
    print(n)
    alpha = (a.T @ c - b)/ca.sqrt(a.T @ E @ a)
    print(alpha)
    tau = (1+n*alpha)/(n+1)
    print(tau)
    sigma = (2*(1+n*alpha))/((n+1)*(1+alpha))
    delta = (n**2/ (n**2-1))*(1-alpha**2)

    c_new = c - tau* (E @ a)/(ca.sqrt(a.T @ E @ a))
    E_new = delta*(E - sigma*(E @ a @ (E@a).T)/(a.T @ E @ a))
    return c_new, E_new


# %%
# PLot tau over alpha
n = 2
alpha = np.linspace(-1,1,100)
tau = (1+n*alpha)/(n+1)
fig, ax = plt.subplots()
ax.plot(alpha, tau)
ax.axvline(-1/n, color='k', linestyle='--')
# %%
# Plot sigma over alpha
n = 2
alpha = np.linspace(-1,1,100)
sigma = (2*(1+n*alpha))/((n+1)*(1+alpha))
fig, ax = plt.subplots()
ax.plot(alpha, sigma)
ax.axvline(-1/n, color='k', linestyle='--')
ax.axhline(0)
# %%
# Plot delta over alpha
n = 4
alpha = np.linspace(-1,1,100)
delta = (n**2/ (n**2-1))*(1-alpha)
fig, ax = plt.subplots()
ax.plot(alpha, delta)
ax.axvline(-1/n, color='k', linestyle='--')
ax.axhline(1, color='k', linestyle='--')
# %% Plot a circular ellisoid
B = np.array([[1, 0], [0, 1]]) # Propotional to the confidence region
#E_inv = np.linalg.inv(E)
c = np.array([[0], [0]])
a = np.array([[1], [1]])
b = 0
c1, B1 = get_next_ellipse(c, B, a, b)
c1 = np.array(c1).reshape(-1,1)
B1 = np.array(B1)
fig, ax = plt.subplots()
confidence_ellipse(c, B, ax, edgecolor='r')
# Plot line
x = np.linspace(-2,2,100)
y = a[1]/a[0]*x
confidence_ellipse(c1, B1, ax, edgecolor='b')
ax.plot(x,y)

# %%
