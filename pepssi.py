import numpy as np
import spice_tools as st
import instrument as inst
# "/home/nathan/data/chinook/pluto-1/data"

def load_and_plot(name):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.dates as mdates

    spec = np.load(name)
    times = [st.et2pydatetime(t) for t in spec['times']]
    responses = spec['responses']
    bins = spec['bins']

    centers = 0.5*(bins[0,0,1:] + bins[0,0,:-1])
    total_resps = np.sum(responses, axis=1)
    plt.pcolormesh(times, centers, total_resps.T, norm=LogNorm())
    plt.yscale('log')

    hours = mdates.HourLocator()
    hours_fmt = mdates.DateFormatter('%H:%M')
    minutes = mdates.MinuteLocator(byminute=[10,20,30,40,50])
    plt.gca().xaxis.set_major_locator(hours)
    plt.gca().xaxis.set_major_formatter(hours_fmt)
    plt.gca().xaxis.set_minor_locator(minutes)


    plt.figure()
    n = 23
    plt.plot(times, total_resps[:,n])
    plt.gca().xaxis.set_major_formatter(hours_fmt)

    plt.show()

def compute_and_save(name):
    times, responses, bins = inst.spectrogram_by_subpop(
        "/home/nathan/data/chinook/pluto-1/data",
        "NH_SWAP",
        inst.flux_density_per_keV,
        dx=3000.,
        start_et=st.flyby_start, stop_et=st.flyby_end, step=60.,
        E_bins=np.geomspace(20., 20000., 60)
    )

    np.savez(name, times=times, responses=responses, bins=bins)


if __name__ == '__main__':
    load_and_plot('spectrogram.npz')
    #compute_and_save('spectrogram.npz')
