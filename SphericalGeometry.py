import numpy as np
from scipy.interpolate import interp1d
import cartopy.crs as ccrs

unit_sphere = ccrs.Globe(semimajor_axis=1., semiminor_axis=1., ellipse=None)

def vec2mapcoords(v, to_crs=None, from_crs=None):
    """Converts radial vectors v into longitude, latitude, altitude triples.
    Any cartopy CRSs may be used, but the defaults are
    `to_crs = Geodetic(globe=unit_sphere)`, and
    `from_crs = to_crs.as_geocentric()`
    """
    to_crs = to_crs or ccrs.Geodetic(globe=unit_sphere)
    from_crs = from_crs or to_crs.as_geocentric()
    v = np.asanyarray(v)
    ret = to_crs.transform_points(from_crs, x=v[:,0], y=v[:,1], z=v[:,2])
    return ret

class SphericalPolygon:
    """A polygon on a sphere and some related functions.
    Assumptions:
        1. vertices are listed in order so that there is an edge between v[n] and v[n+1].
        2. no edges cover exactly pi radians. i.e. vertices connected by an edge are not antipodal.
        3. edges are always defined to be the shorter arc between vertices (less than pi radians).
        4. the polygon is not self intersecting.
    Of course, if you would like your polygon to have an edge equal to or longer than pi radians
    just use an extra vertex to make it into two edges on the same great circle.
    """

    def __init__(self, vertices, inside):
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert inside.ndim == 1
        assert inside.shape[0] == 3

        assert np.allclose(np.linalg.norm(vertices, axis=-1), 1.0)

        # self.vertices should be 'cyclic' in the sense that the last vertex is the
        # same as the first vertex. This closes the polygon.
        if np.allclose(vertices[0], vertices[-1]):
            self.vertices = vertices
        else:
            self.vertices = np.empty((vertices.shape[0]+1, 3))
            self.vertices[:-1,:] = vertices
            self.vertices[-1,:] = vertices[0,:]

        self.inside = inside/np.linalg.norm(inside)

    @property
    def edges(self):
        return zip(self.vertices[:-1], self.vertices[1:])

    @property
    def num_edges(self):
        """Normally in polygons the number of edges is equal to the number of vertices, but
        here the first and last vertex are the same, so we subtract 1 to prevent double counting
        """
        return len(self.vertices) - 1

    def contains_point(self, P):
        """Checks if the point P is in the polygon. Each edge splits the sphere into two halfs,
        we check that P is on the same half as X for each edge. Points on the edge are defined
        as being inside.
        """
        X = self.inside
        P = P/np.linalg.norm(P, axis=-1, keepdims=True)
        ret = np.ones(P.shape[0], dtype=bool)
        for A,B in self.edges:
            # Find a normal to the plane defined by AB that is on the same
            # half of the sphere that the inside is on.
            normal = np.cross(A,B)
            if normal.dot(X) < 0:
                normal = -normal

            # If the normal makes an acute angle with P, then P must be on the same half
            # of the sphere.
            ret = np.logical_and(ret, np.inner(normal, P) >= 0, out=ret)

        return ret

    def interpolate_edges(self, N=100):
        """Interpolate all edges with N+1 points on each edge (including corners).
        Returns an array of shape (N*num_edges+1,3). Each vertex is hit exactly.
        >>> np.all(sp.interpolate_edges(N)[::N] == sp.vertices)
        True
        """
        ret = np.empty((N*self.num_edges+1,3))
        for i,(A,B) in enumerate(self.edges):
            ret[i*N:(i+1)*N] = self._interpolate_edge(A,B,N)[:-1]
        ret[-1] = B
        return ret

    def _interpolate_edge(self, A, B, N):
        """interpolates points on the sphere between A and B
        including only the first end point."""
        t = np.linspace(0,1,N+1)
        interpolator = interp1d([0,1], [A,B], axis=0)
        ret = interpolator(t)
        return ret/np.linalg.norm(ret, axis=-1, keepdims=True)

    def plot_boundary(self, ax, N=100, *args, **kwargs):
        interp = self.interpolate_edges(N)
        proj = ax.projection
        map_coords = vec2mapcoords(interp, to_crs=proj)
        return ax.plot(map_coords[:,0], map_coords[:,1], *args, **kwargs)

class SphericalGrid:
    """Someday, refactor methods common to spherical grids into this class.
    I'll know better how to do that once I've implemented one or two
    spherical grids.
    """
    pass

class EqualArea_UVSphere(SphericalGrid):
    """Physics convention, theta is angle from the z axis, phi is right handed from the x axis"""
    def __init__(self, n, r=1.0):
        self.n = n

        self.phi, delta = np.linspace(0, 2*np.pi, n, retstep=True)
        delta_S = delta/n
        self.theta = 1-np.arange(2*n+1) * delta_S / (r**2 * delta)
        self.theta = np.arccos(self.theta)

        t,p = np.meshgrid(self.theta,self.phi)

        self.x = r*np.cos(p)*np.sin(t)
        self.y = r*np.sin(p)*np.sin(t)
        self.z = r*np.cos(t)



def lonlat(v, radians=False):
    v = np.asanyarray(v)
    lat = np.arctan2(v[...,2], np.sqrt(v[...,0]**2 + v[...,1]**2))
    lon = np.arctan2(v[...,1], v[...,0])

    if radians:
        return lon, lat
    else:
        return np.degrees(lon), np.degrees(lat)

def thetaphi(v, radians=False):
    """Physics convention, theta is angle from the z axis, phi is right handed from the x axis"""
    v = np.asanyarray(v)
    theta = np.arctan2(np.sqrt(v[...,0]**2 + v[...,1]**2), v[...,2])
    phi = np.arctan2(v[...,1], v[...,0])

    phi[phi<=0] += 2*np.pi

    if radians:
        return theta, phi
    else:
        return np.degrees(theta), np.degrees(phi)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    S = EqualArea_UVSphere(10)
    print S.theta.min(), S.theta.max()
    print S.phi.min(), S.phi.max()
    # make some random data
    normals = np.random.normal(size=(1000,3))
    sphere = normals/np.linalg.norm(normals, axis=-1, keepdims=True)
    shell = sphere + np.array([[1.,0.,0.]])
    t,p = thetaphi(shell, radians=True)
    print t.min(), t.max()
    print p.min(), p.max()

    H, xedges, yedges = np.histogram2d(t, p, bins=(S.theta, S.phi))
    H = H.T
    X,Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X,Y,H)
    plt.scatter(t,p, alpha=0.1)
    plt.show()
