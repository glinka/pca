import pca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_pca():
    """Generates a noisy 2d plane embedded in 3d, then performs PCA to find better coordinates for the data"""
    # define a planar equation
    z = lambda x, y: 3*x + y + 4
    npoints = 200
    stdev = 1.0
    noise = stdev*np.random.normal(size=npoints)
    xvals = np.random.uniform(low=2, high=6, size=npoints)
    yvals = np.random.uniform(low=2, high=10, size=npoints)
    zvals = z(xvals, yvals) + noise

    data = np.transpose(np.array([xvals, yvals, zvals]))
    pcomp, pvar = pca.pca(data, 2)

    # plot projections along various components
    ncomp = pvar.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cs = ['b', 'r', 'g']
    data_avg = np.average(data, 0)
    centered_data = data - data_avg
    # do not bother plotting "full" projection, which would give back the original data
    # proj = np.dot(pcomp.T, centered_data.T)
    # print np.dot(pcomp.T, pcomp), pvar
    # ax.scatter(proj[0,:]+data_avg[0], proj[1,:]+data_avg[1], 0, c='g')
    proj = np.dot(pcomp, np.dot(pcomp.T, centered_data.T))
    ax.scatter(data[:,0], data[:,1], data[:,2], alpha=1.0, s=80, c='#96031E')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # hide labels and grid, too squashed/noisy
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.tick_params(axis='both', which='major', labelsize=0)
    # ax.scatter(proj[0,:]+data_avg[0], proj[1,:]+data_avg[1], proj[2,:]+data_avg[2], color='g')
    # # plot lines from orig data to planar projection
    # for i in range(npoints):
    #     pts = np.array((proj[:,i] + data_avg, data[i,:]))
    #     ax.plot(pts[:,0], pts[:,1], pts[:,2], c='y')

    # sort based on z val and wireframe
    # data_avg.shape = (3,1)
    # sorted_indices = np.argsort(np.linalg.norm(proj + data_avg, axis=0))
    # proj = proj[:,sorted_indices]

    # fake pca result, true in the limit of infinite data
    xgrid, ygrid = np.meshgrid(np.linspace(2,6,10), np.linspace(2,10,10))
    ax.plot_wireframe(xgrid, ygrid, z(xgrid, ygrid), color='#12227A', alpha=0.5)
    grid_coord = (2,2)
    for i in range(2):
        ax.plot((xgrid[grid_coord], xgrid[grid_coord] + pcomp[0,i]), (ygrid[grid_coord], ygrid[grid_coord] + pcomp[1,i]), (z(xgrid[grid_coord], ygrid[grid_coord]), z(xgrid[grid_coord], ygrid[grid_coord]) + pcomp[2,i]), c='k')


    # # handmade wireframe
    # nnearest_neighbors = 6
    # for i in range(npoints-nnearest_neighbors):
    #     # need to change shape to (3,1) for broadcasting
    #     current_pt = proj[:,i]
    #     current_pt.shape = (3,1)
    #     # sort indices based on distance from current_pt
    #     sorted_distances = np.argsort(np.linalg.norm(proj - current_pt, axis=0))
    #     # add nnearest_neighbors lines from current_pt to nnearest_neighbors nearest neighbors
    #     for j in range(nnearest_neighbors):
    #         # sorted_distances[0] refers to current_pt, offset by one
    #         line = np.array((proj[:,i] + data_avg, proj[:,sorted_distances[1+j]] + data_avg))
    #         ax.plot(line[:,0], line[:,1], line[:,2], c='k')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # proj = np.dot(pcomp.T, centered_data.T)
    # ax.scatter(proj[0,:], proj[1,:])
    # ax.set_xlabel('First principal component')
    # ax.set_ylabel('Second principal component')
    # # ax.set_title('Projection of dataset along first two principal components')

    plt.show()
    # for i in range(2):
    #     # need to add back the averages
    #     proj = np.dot(np.dot(centered_data, pcomp[:,:i+1]), np.transpose(pcomp[:,:i+1])) + data_avg
    #     ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=cs[i], lw=0)
    #     print str(i+1) + 'd projection norm: ', np.linalg.norm(data - proj)
    # # plot original data
    # ax.scatter(data[:,0], data[:,1], data[:,2], c='g', lw=0)
    plt.show(fig)

if __name__ == '__main__':
    test_pca()
