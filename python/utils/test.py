# import numpy as np
# # from scipy.interpolate import griddata
# # random_state = np.random.RandomState(19680801)
# # # random data to interpolate
# # x = np.array([0, 10, 13, 17, 20, 50, 55, 60.0])
# # y = np.array([10, 20, 40, 80, 90, 95, 100, 120.0])
# # zg = np.random.randn(8, 8)
# #
# # # x1x = random_state.uniform(-2, 2, 200)
# # # y1y = random_state.uniform(-2, 2, 200)
# # # z1z = x1x*np.exp(-x1x**2 - y1y**2)
# # #select one of the following two line, it depends on the order in z
# # #xg, yg = np.broadcast_arrays(x[:, None], y[None, :])
# # xg, yg = np.broadcast_arrays(x[None, :], y[:, None])
# #
# # yg2, xg2 = np.mgrid[y.min()-10:y.max()+10:100j, x.min()-10:x.max()+10:100j]
# #
# # zg2 = griddata((xg.ravel(), yg.ravel()), zg.ravel(), (xg2.ravel(), yg2.ravel()), method="nearest")
# # zg2.shape = yg2.shape
# #
# # import pylab as pl
# #
# # pl.pcolormesh(xg2, yg2, zg2)
# # pl.scatter(xg.ravel(), yg.ravel(), c=zg.ravel())
# # pl.show()


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
# make up some randomly distributed data
seed(1234)
npts = 200
x = uniform(-2,2,npts)
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2)
# define grid.
xi = np.linspace(-2.1,2.1,100)
yi = np.linspace(-2.1,2.1,100)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('griddata test (%d points)' % npts)
plt.show()