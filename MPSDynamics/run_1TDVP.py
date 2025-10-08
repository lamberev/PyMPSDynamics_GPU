import time
import cupy as cp

from .measure import measure
from .treeTDVP import tdvp1_sweep
from .treeTDVP import mps_embed

def run_1tdvp(dt, tmax, A_in, H, Dmax, obs=None, timed=False, **kwargs):
    """
    Runs the 1-site Time-Dependent Variational Principle (1TDVP) algorithm.

    Args:
        dt (complex): The time step.
        tmax (float): The maximum simulation time.
        A_in (list): The initial Matrix Product State (MPS).
        H (list): The Matrix Product Operator (MPO) for the Hamiltonian.
        Dmax (int): The maximum bond dimension.
        obs (list, optional): A list of observables to measure. Defaults to None.
        timed (bool, optional): Whether to time the TDVP sweeps. Defaults to False.
        **kwargs: Additional keyword arguments for the tdvp1_sweep function.

    Returns:
        tuple: A tuple containing:
            - list: The final MPS.
            - dict: A dictionary containing the simulation data.
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
            # Add a new dimension for time series
            data[ob.name] = cp.asarray(val).reshape(cp.asarray(val).shape + (1,))

    if timed:
        ttdvp = []

    F = None
    mps_embed(A, Dmax)

    for tstep in range(num_steps):
        print(f"{tstep + 1}/{num_steps}, t = {times[tstep].real:.3f} + {times[tstep].imag:.3f}im")

        # Build per-step local on-site drives if requested
        localV = None
        if kwargs.get('timedep', False):
            Ndrive = kwargs.get('Ndrive', None)
            Htime = kwargs.get('Htime', None)
            if Ndrive is not None and Htime is not None:
                localV = {}
                # Single driven site
                if isinstance(Ndrive, int):
                    site_idx0 = int(Ndrive) - 1  # external sites are 1-based
                    if 0 <= tstep < len(Htime):
                        localV[site_idx0] = Htime[tstep]
                # Multiple driven sites
                elif isinstance(Ndrive, (list, tuple)):
                    # Support Htime as dict: site_id -> list(per-step operators)
                    if isinstance(Htime, dict):
                        for site in Ndrive:
                            site_idx0 = int(site) - 1
                            if site in Htime and 0 <= tstep < len(Htime[site]):
                                localV[site_idx0] = Htime[site][tstep]
                    else:
                        # Fallback: same per-step operator applied to all driven sites
                        if 0 <= tstep < len(Htime):
                            for site in Ndrive:
                                site_idx0 = int(site) - 1
                                localV[site_idx0] = Htime[tstep]

        if timed:
            start_tdvp_time = time.perf_counter()
            if localV is not None:
                A, F = tdvp1_sweep(dt, A, H, F, tstep + 1, localV=localV, **kwargs)
            else:
                A, F = tdvp1_sweep(dt, A, H, F, tstep + 1, **kwargs)
            end_tdvp_time = time.perf_counter()
            elapsed = end_tdvp_time - start_tdvp_time
            print(f"\tΔT = {elapsed}")
            ttdvp.append(elapsed)
        else:
            if localV is not None:
                A, F = tdvp1_sweep(dt, A, H, F, tstep + 1, localV=localV, **kwargs)
            else:
                A, F = tdvp1_sweep(dt, A, H, F, tstep + 1, **kwargs)

        if obs:
            exp = measure(A, obs, t=times[tstep + 1])
            for i, ob in enumerate(obs):
                # Reshape to allow concatenation along the new time-series axis
                new_val = cp.asarray(exp[i])[..., cp.newaxis]
                data[ob.name] = cp.concatenate((data[ob.name], new_val), axis=-1)

    if timed:
        data["deltat"] = ttdvp
        
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccckm,lllllllllllllllll
    return A, data
