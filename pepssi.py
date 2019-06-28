import numpy as np
import spice_tools as st
import scipy.spatial as spatial

def location_masks(x, points, r, p=np.inf, ret_indices=False):
    """Generates a list of masks for particles near each point.
    I.e. for each point, p, in points find the particles (with positions x)
    that are within radius r under the given p-norm. x[locaton_masks(x,points,r)[i]]
    will be the list of particle locations near points[i]
    """
    kdparts  = spatial.cKDTree(x)
    kdpoints = spatial.cKDTree(points)
    inds = kdpoints.query_ball_tree(kdparts, r, p=p)
    if ret_indices:
        return inds
    local = np.zeros((len(points),len(x)), dtype=bool)
    for i,ind in enumerate(inds):
        local[i,ind] = True
    return local

def Erat(v,mrat):
    v = 1000.*v # convert km/s to m/s
    mrat = 9.578832e7*mrat # convert elementary charge per proton mass to C/kg
    vsqr = np.einsum('ij,ij->i',v,v)
    E = (0.5*vsqr/mrat) # fun fact: Joules per Coulomb is the same as electron volts per elementary charge
    return E

def energy_window(v, mrat, bottom, top):
    E = Erat(v,mrat)
    return (E<top) & (E>bottom)

def spectrum(v, mrat, n, Emin=None, Emax=None, bins=None, N=None):
    """An energy spectrum is a histogram of particle energies weighted by flux.
    n is the macroparticle density i.e. 1/(Vol*beta).
    If bins is None, it defaults to a geometric series.
    Specify either bins, or N, or neither bins or N. Do not give both bins and N.
    """
    if N is not None:
        assert bins is None
    if bins is not None:
        assert N is None
    v = 1000.*v # convert km/s to m/s
    mrat = 1e8*mrat # convert to C/kg
    vsqr = np.einsum('ij,ij->i',v,v)
    E = (0.5*vsqr/mrat) # fun fact: Joules per Coulomb is the same as electron volts per elementary charge
    nv = n*np.linalg.norm(v, axis=-1)
    if Emin is None:
        Emin = E.min()
    if Emax is None:
        Emax = E.max()
    bins = bins or np.geomspace(Emin, Emax, num=(N or 50))

    # TODO: We may want to devide the final histogram by the energy window width and FOV steradians.
    # I think that's what differential intensity is?
    return np.histogram(E, bins, weights=nv)

# Plot
if __name__ == '__main__':
    from HybridParticleReader import particle_data
    import matplotlib.pyplot as plt
    import SphericalGeometry as sg
    import cartopy.crs as ccrs
    unit_sphere = ccrs.Globe(semimajor_axis=1., semiminor_axis=1., ellipse=None)
    map_crs = ccrs.Mollweide(globe=unit_sphere)
    fig, ax = plt.subplots(subplot_kw={'projection':map_crs})

    et = st.plutopause
    fov = st.fov_polygon("NH_PEPSSI_S0", et, frame="HYBRID_SIMULATION_INTERNAL")
    #swap = st.fov_polygon("NH_SWAP", et)
    para, x, v, mrat, beta, tags = particle_data("/home/nathan/data/chinook/pluto-1/data", n=[125])
    x[:,2] -= np.mean(x[:,2]) # Shift coordinates so particles are near the xy plane.
    #pos = st.coordinate_at_time(et, mccomas=False)
    #pos[2] = 0.0 # Project spacecraft position into the xy plane
    pos = np.array([7.*1187., 0., 0.])
    dx = 1000.
    location_mask, = location_masks(x, [pos], dx)
    E_mask = energy_window(v,mrat, 1000., 100000.)
    print np.count_nonzero(location_mask)
    print np.count_nonzero(E_mask)
    mask = location_mask & E_mask
    print np.count_nonzero(mask)
    fov.plot_boundary2(ax)
    print(np.arccos(np.dot(np.array([1.,0.,0.]), fov.inside)))
    for t,lab in zip([1.0,2.0,3.0,4.0], ['H_sw', 'He_sw', 'H_shell', 'He_shell']):
        mc = sg.vec2mapcoords(-v[mask & (tags==t)], ax.projection)
        ax.scatter(mc[:,0], mc[:,1], marker='.', s=0.1, label=lab)
    mc = sg.vec2mapcoords(-v[mask & ((tags==5.0) | (tags==6.0) | (tags==7.0))], ax.projection)
    ax.scatter(mc[:,0], mc[:,1], marker='.', s=0.1, label='CH4')
    ax.set_global()
    ax.legend()
    plt.show()


    fov_mask = fov.contains_point(-v)
    print np.count_nonzero(fov_mask)
    n = (np.sum(1./beta[mask])/dx**3)/1000**3 # density due to a macroparticle in m^-3
    H, edges = spectrum(v[mask], mrat[mask], n, Emin=20., Emax=9000.)
    centers = (edges[:-1]+edges[1:])/2

    plt.figure()
    plt.plot(centers, H)
    plt.xlim([edges[0], edges[-1]])
    plt.xlabel('Energy')
    plt.ylabel('"Differential Intensity"')
    plt.show()
