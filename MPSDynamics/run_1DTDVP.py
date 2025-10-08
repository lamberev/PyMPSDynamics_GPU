import time
import cupy as cp

from .measure import measure
from .treeDTDVP import tdvp1_sweep_dynamic
from .treeTDVP import mps_embed


def run_1dtdvp(dt, tmax, A_in, H, Dmax, obs=None, timed=False, **kwargs):
    """
    Runs the dynamic/adaptive 1-site TDVP (DTDVP1) algorithm on a tree-MPS.

    Args:
        dt (complex): The time step.
        tmax (float): The maximum simulation time.
        A_in: The initial Tree MPS state.
        H: The Tree MPO Hamiltonian.
        Dmax (int): The maximum bond dimension (embedding cap).
        obs (list, optional): Observables to measure. Defaults to None.
        timed (bool, optional): Whether to time the sweeps. Defaults to False.
        **kwargs: Additional keyword arguments for the sweep, e.g. localV, Dlim, Dplusmax.

    Returns:
        tuple: (final_state, data_dictionary)
    """
    if obs is None:
        obs = []

    A = A_in
    data = {}

    start_time = 0.0
    num_steps = int(round(abs(tmax - start_time) / abs(dt)))
    times = [start_time + i * dt for i in range(num_steps + 1)]

    print(f"Dmax : {Dmax}")

    if obs:
        exp = measure(A, obs, t=times[0])
        for i, ob in enumerate(obs):
            val = exp[i]
            data[ob.name] = cp.asarray(val).reshape(cp.asarray(val).shape + (1,))

    ttdvp = [] if timed else None

    # Ensure A is embedded to the maximum requested manifold before starting
    mps_embed(A, Dmax)

    F = None
    for tstep in range(num_steps):
        print(f"{tstep + 1}/{num_steps}, t = {times[tstep].real:.3f} + {times[tstep].imag:.3f}im")

        # Optional per-step on-site drives
        localV = None
        if kwargs.get('timedep', False):
            Ndrive = kwargs.get('Ndrive', None)
            Htime = kwargs.get('Htime', None)
            if Ndrive is not None and Htime is not None:
                localV = {}
                if isinstance(Ndrive, int):
                    site_idx0 = int(Ndrive) - 1
                    if 0 <= tstep < len(Htime):
                        localV[site_idx0] = Htime[tstep]
                elif isinstance(Ndrive, (list, tuple)):
                    if isinstance(Htime, dict):
                        for site in Ndrive:
                            site_idx0 = int(site) - 1
                            if site in Htime and 0 <= tstep < len(Htime[site]):
                                localV[site_idx0] = Htime[site][tstep]
                    else:
                        if 0 <= tstep < len(Htime):
                            for site in Ndrive:
                                site_idx0 = int(site) - 1
                                localV[site_idx0] = Htime[tstep]

        if timed:
            t0 = time.perf_counter()
            if localV is not None:
                A, F = tdvp1_sweep_dynamic(dt, A, H, F, tstep + 1, localV=localV, **kwargs)
            else:
                A, F = tdvp1_sweep_dynamic(dt, A, H, F, tstep + 1, **kwargs)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"\tΔT = {elapsed}")
            ttdvp.append(elapsed)
        else:
            if localV is not None:
                A, F = tdvp1_sweep_dynamic(dt, A, H, F, tstep + 1, localV=localV, **kwargs)
            else:
                A, F = tdvp1_sweep_dynamic(dt, A, H, F, tstep + 1, **kwargs)

        if obs:
            exp = measure(A, obs, t=times[tstep + 1])
            for i, ob in enumerate(obs):
                new_val = cp.asarray(exp[i])[..., cp.newaxis]
                data[ob.name] = cp.concatenate((data[ob.name], new_val), axis=-1)

    if timed:
        data["deltat"] = ttdvp

    data["times"] = times
    return A, data
