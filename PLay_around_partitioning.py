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
E = np.array([[2, 1], [1, 2]]) # Propotional to the confidence region
E_inv = np.linalg.inv(E)
c = np.array([[0], [0]])
# Plot ellipsoid
fig, ax = plt.subplots()
#eigenvalues, eigenvectors = np.linalg.eig(E_inv)
#theta = np.linspace(0, 2*np.pi, 1000)
#ellipsis = (1/np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
#ax.plot(ellipsis[0,:]+c[0], ellipsis[1,:]+c[1])
confidence_ellipse(c, E, ax, edgecolor='k',label='Original Ellipoid')
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
#b = -1.22474487 #alpha = 0.5
b = 0
# Plot cutting plane a^T x <= b
x = np.linspace(-2, 2, 100)
y = (b - a[0]*x)/a[1]
ax.plot(x, y,'k')
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
confidence_ellipse(c_new, E_new, ax, edgecolor='g',linestyle='--',lw=5,label='Löwner-John ellipsoid')
fig.legend()
fig.tight_layout()
# Get other ellipsoid
# c_new, E_new = get_next_ellipse(c, E, -a, -b)
# confidence_ellipse(c_new, E_new, ax, edgecolor='k',linestyle=':')
fig.savefig('Löwner-John_alpha=0.pdf')