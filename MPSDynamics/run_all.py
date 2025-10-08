import cupy as cp
from .run_1TDVP import run_1tdvp
from .run_1DTDVP import run_1dtdvp

def run_all(dt, tmax, A, H,
            method='TDVP1',
            obs=None,
            convobs=None,
            convparams=None,
            **kwargs):
    """
    High-level driver for running a simulation, potentially with multiple
    convergence checks.

    This function serves as a wrapper around a core simulation engine (e.g.,
    run_1tdvp). It can run a simulation multiple times with different sets of
    parameters (e.g., varying bond dimensions) to check for convergence.

    Args:
        dt (float): The time step for the simulation.
        tmax (float): The maximum simulation time.
        A (TreeNetwork): The initial state as a TreeNetwork.
        H (TreeNetwork): The Hamiltonian as a TreeNetwork (MPO).
        method (str, optional): The TDVP method to use. Defaults to 'TDVP1'.
        obs (list, optional): A list of observables to measure during the
                              final run. Defaults to None.
        convobs (list, optional): A list of observables to measure during the
                                  convergence runs. Defaults to None.
        convparams (list, optional): A list of parameter sets for convergence
                                     runs. Each element is a tuple of arguments
                                     for the core simulation function.
                                     Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the core
                  simulation function.

    Returns:
        tuple: A tuple containing:
            - B (TreeNetwork): The final state of the system.
            - data (dict): A dictionary containing the simulation results.
                           If convergence checks were performed, it will have
                           'data' for the final run and 'convdata' for the
                           convergence runs.
    """
    if obs is None:
        obs = []
    if convobs is None:
        convobs = []

    # Ensure all observables are measured in the final run
    obs = list(set(obs) | set(convobs))

    convcheck = isinstance(convparams, list) and len(convparams) > 1

    convdat = None
    if convcheck:
        for i, cps in enumerate(convparams[:-1]):
            # Ensure cps is a list/tuple for unpacking
            if not isinstance(cps, (list, tuple)):
                cps = [cps]
            if method == 'TDVP1':
                _, dat = run_1tdvp(dt, tmax, A, H, *cps, obs=convobs, **kwargs)
            elif method == 'DTDVP1':
                _, dat = run_1dtdvp(dt, tmax, A, H, *cps, obs=convobs, **kwargs)
            else:
                raise ValueError(f"Method {method} not recognised")

            if i == 0:
                convdat = {key: val[..., cp.newaxis] for key, val in dat.items()}
            else:
                for key, val in dat.items():
                    # Stack the new data along a new axis
                    new_val = val[..., cp.newaxis]
                    convdat[key] = cp.concatenate((convdat[key], new_val), axis=-1)

    # Use the last (or only) parameter set for the final run
    cps = convparams[-1] if convcheck else convparams
    if cps is None:
        raise ValueError("convparams must be provided.")

    # Ensure cps is a list/tuple for unpacking
    if not isinstance(cps, (list, tuple)):
        cps = [cps]

    if method == 'TDVP1':
        B, dat = run_1tdvp(dt, tmax, A, H, *cps, obs=obs, **kwargs)
    elif method == 'DTDVP1':
        B, dat = run_1dtdvp(dt, tmax, A, H, *cps, obs=obs, **kwargs)
    else:
        raise ValueError(f"Method {method} not recognised")

    if convcheck:
        for key in convdat.keys():
            new_val = dat[key][..., cp.newaxis]
            convdat[key] = cp.concatenate((convdat[key], new_val), axis=-1)

    if convcheck:
        data = {"data": dat, "convdata": convdat}
    else:
        data = {"data": dat}

    return B, data 