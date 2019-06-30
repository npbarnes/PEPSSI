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
        for i in range(self.num_edges):
            yield (self.vertices[i], self.vertices[(i+1) % self.num_edges])

    @property
    def num_edges(self):
        return len(self.vertices)

    def _intersections(self, A):
        """Get all the intersection points between the arcs A-self.inside
        and the edges of the polygon.

        I don't really understand how this algorithm works, I copied it from
        the spherical_geometry astropy project, and the link to their citation
        was broken :(

        We're not even using the intersection locations as of this writing, only
        counting how many interections there are.
        """
        ret = np.empty((A.shape[0], self.num_edges,3), dtype='d')
        for i,(C,D) in enumerate(self.edges):
            ABX = np.cross(A, self.inside)
            CDX = np.cross(C, D)
            T = np.cross(ABX, CDX)
            T = T/np.linalg.norm(T, axis=-1, keepdims=True)

            s = np.zeros(T.shape[0])
            s += np.sign(np.einsum('ij,ij->i',np.cross(ABX, A), T))
            s += np.sign(np.einsum('ij,ij->i',np.cross(self.inside, ABX), T))
            s += np.sign(np.einsum('j,ij->i',np.cross(CDX, C), T))
            s += np.sign(np.einsum('j,ij->i',np.cross(D, CDX), T))
            s = np.expand_dims(s, 1)

            cross = np.where(s == -4, -T, np.where(s == 4, T, np.nan))

            # If they share a common point, it's not an intersection.  This
            # gets around some rounding-error/numerical problems with the
            # above.
            equals = (np.all(A == C, axis=-1) |
                    np.all(A == D, axis=-1) |
                    np.all(self.inside == C, axis=-1) |
                    np.all(self.inside == D, axis=-1))

            equals = np.expand_dims(equals, 2)

            ret[:,i,:] = np.where(equals, np.nan, cross)
        return ret

    def _count_intersections(self, A):
        I = self._intersections(A)
        return np.count_nonzero(~np.isnan(I[:,:,0]), axis=1)

    def contains_point(self,P):
        c = self._count_intersections(P)
        return np.mod(c, 2) == 0

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

    def plot_boundary(self, ax, N=100, vertices=False, *args, **kwargs):
        interp = self.interpolate_edges(N)
        proj = ax.projection # map coordinates, whatever they may be.
        geodetic_proj = proj.as_geodetic() # latitude, longitude with spherical topology
        geodetic_coords = vec2mapcoords(interp, to_crs=geodetic_proj) # Convert geocentric vectors to geodetic coords (lon, lat, height).
        ax.plot(geodetic_coords[:,0], geodetic_coords[:,1], transform=geodetic_proj) # plot with the transform that converts from geodetic to map coords
        if vertices:
            geodetic_vertices = vec2mapcoords(self.vertices, to_crs=geodetic_proj)
            ax.scatter(geodetic_vertices[:,0], geodetic_vertices[:,1], transform=geodetic_proj)

    def _angle(self,A,B,C):
        """Compute the angle ABC.

        This formula follows from a standard result in spherical trigonometry.
        """
        cb = A.dot(C)
        cc = B.dot(A)
        ca = C.dot(B)

        sc = np.linalg.norm(np.cross(A,B))
        sa = np.linalg.norm(np.cross(B,C))

        return np.arccos((cb - cc*ca)/(sc*sa))

