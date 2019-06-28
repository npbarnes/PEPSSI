import numpy as np
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
import itertools

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
        1. Vertices are listed in order so that there is an edge between v[n] and v[n+1] and from v[-1] to v[0].
        2. No edges cover exactly pi radians. i.e. vertices connected by an edge are not antipodal.
        3. Edges are always defined to be the shorter arc between vertices (less than pi radians).
        4. The polygon is simply connected and not self intersecting.
        5. The given interior point is not on an edge and not antipodal to any vertex.
    Of course, if you would like your polygon to have an edge equal to or longer than pi radians
    just use an extra vertex to make it into two edges on the same great circle.
    """

    def __init__(self, vertices, inside):
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert inside.ndim == 1
        assert inside.shape[0] == 3

        self.vertices = vertices/np.linalg.norm(vertices, axis=-1, keepdims=True)
        self.inside = inside/np.linalg.norm(inside)

    @property
    def edges(self):
        most = zip(self.vertices[:-1], self.vertices[1:])
        return itertools.chain(most, ((self.vertices[-1], self.vertices[0]),))

    @property
    def num_edges(self):
        """Normally in polygons the number of edges is equal to the number of vertices, but
        here the first and last vertex are the same, so we subtract 1 to prevent double counting
        """
        return len(self.vertices)

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
        proj = ax.projection # map coordinates, whatever they may be.
        geodetic_proj = proj.as_geodetic() # latitude, longitude with spherical topology
        geodetic_coords = vec2mapcoords(interp, to_crs=geodetic_proj) # Convert geocentric vectors to geodetic coords (lon, lat, height).
        ax.plot(geodetic_coords[:,0], geodetic_coords[:,1], transform=geodetic_proj) # plot with the transform that converts from geodetic to map coords
