import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import spice_tools as st
import scipy.spatial as spatial

def location_masks(x, points, r, p=np.inf):
    """Generates a list of masks for particles near each point.
    I.e. for each point, p, in points find the particles (with positions x)
    that are within radius r under the given p-norm. x[locaton_masks(x,points,r)[i]]
    will be the list of particle locations near points[i]
    """
    kdparts  = spatial.cKDTree(x)
    kdpoints = spatial.cKDTree(points)
    local = kdpoints.query_ball_tree(kdparts, r, p=p)
    return local

def energy_window(v, mrat, top, bottom):
    E = 0.5*v**2/mrat
    return (E<top) & (E>bottom)

def spectrum(v, mrat, n, bins=None, N=None):
    """An energy spectrum is a histogram of particle energies weighted by flux.
    n is the macroparticle density i.e. 1/(Vol*beta).
    If bins is None, it defaults to a geometric series.
    """
    if N is not None:
        assert bins is None
    if bins is not None:
        assert N is None
    v = 1000.*v # convert km/s to m/s
    mrat = 1e8*mrat # convert to C/kg
    E = (0.5*v**2/mrat) # fun fact: Joules per Coulomb is the same as electron volts per elementary charge
    nv = n*np.linalg.norm(v, axis=-1)
    bins = bins or np.geomspace(E.min(), E.max(), num=(N or 50))

    # We may want to devide the final histogram by the energy window width and FOV steradians.
    # I think that's what differential intensity is.
    return np.histogram(E, bins, weights=nv)

if __name__ == '__main__':
    et = st.hi_burst

    pepssi_sectors = [st.fov_polygon("NH_PEPSSI_S{}".format(s), et) for s in range(5)]
    swap = st.fov_polygon("NH_SWAP", et)
    # Plot
    unit_sphere = ccrs.Globe(semimajor_axis=1., semiminor_axis=1., ellipse=None)
    map_crs = ccrs.Mollweide(globe=unit_sphere)
    fig, ax = plt.subplots(subplot_kw={'projection':map_crs})

    #ax.plot(*lonlat([1,0,0]), marker='*', color='gold')
    for S in pepssi_sectors:
        S.plot_boundary(ax, color='red')
    swap.plot_boundary(ax, color='blue')

    # Set the limits to the projection
    ax.set_global()
    # Show gridlines
    ax.gridlines()#crs=ccrs.RotatedPole(pole_latitude=0.0, globe=unit_sphere))

    plt.show()
