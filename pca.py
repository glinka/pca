import numpy as np
import matplotlib.pyplot as plt

def pca(data, ncomp=None):
    # expect individual data vectors to
    # be input as rows in data, so
    # data.shape = (npoints, nvars)
    # ncomp represents the number of
    # components to store
    X = np.array(data)
    npoints = X.shape[0]
    nvars = X.shape[1]

    # center the data at the origin
    # (trust the magic of broadcasting)
    X = X - np.average(X, 0)

    # an unset ncomp will default to 
    # returning the full transformation
    if ncomp == None:
        ncomp = nvars
    princ_comps = np.zeros((nvars, ncomp))
    princ_vars = np.zeros(ncomp)

    for i in range(ncomp):
        eigvals, eigvects = np.linalg.eig(np.dot(np.transpose(X), X))
        sorted_indices = np.argsort(eigvals)
        princ_vars[i] = eigvals[sorted_indices[0]]
        princ_comp = eigvects[:,sorted_indices[0]]
        princ_comps[:,i] = princ_comp
        # need to change shape to take transpose
        princ_comp.shape = (nvars, 1)
        X = X - np.dot(np.dot(X, princ_comp), np.transpose(princ_comp))
    return princ_vars, princ_comps
        
def test_pca():
    from mpl_toolkits.mplot3d import Axes3D

    # define a planar equation
    z = lambda x, y: 3*x - y
    npoints = 50
    noise = np.random.normal(size=npoints)
    xvals = np.random.uniform(low=-1, high=1, size=npoints)
    yvals = np.random.uniform(low=-1, high=1, size=npoints)
    zvals = z(xvals, yvals) + noise

    data = np.transpose(np.array([xvals, yvals, zvals]))
    pvar, pcomp = pca(data)

    # plot projections along various components
    ncomp = pvar.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(ncomp):
        proj = np.dot(np.dot(data, pcomp[:,:i+1]), np.transpose(pcomp[:,:i+1]))
        ax.scatter(proj[:,0], proj[:,1], proj[:,2])
        plt.show()
    

if __name__ == '__main__':
    test_pca()
