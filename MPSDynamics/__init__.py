# MPSDynamics.py

from .run_1TDVP import *
from .run_1DTDVP import *
from .run_all import *
from .machines import *
from .treeTDVP import *
from .treeDTDVP import *
from .utilities import *
from .treeBasics import *
from .treeIterators import *
from .treeMeasure import *
from .observables import *
from .tensorOps import *
from .mpsBasics import *
from .models import *
from .fundamentals import *
from .measure import *
from .logiter import *
from .flattendict import *
from .reshape import *
from .logging import *
from .compute_lineshape_from_dcf import *

import os
import uuid
import traceback
import functools
from datetime import datetime

class MPSDynamicsError(Exception):
    """Custom exception for the MPSDynamics package."""
    pass

def run_sim(dt, tmax, A, H, *,
            method='TDVP1',
            machine=None,
            params=None,
            obs=None,
            convobs=None,
            convparams=None,
            save=False,
            plot=None,
            savedir=None,
            unid=None,
            name=None,
            **kwargs):
    """
    Propagates the MPS A with the MPO H up to time tmax in time steps of dt.

    Args:
        dt (float): Time step.
        tmax (float): Maximum propagation time.
        A: The initial MPS.
        H: The MPO Hamiltonian.
        method (str): The TDVP method to use (e.g., 'TDVP1').
        machine (Machine): The machine to run on (LocalMachine or RemoteMachine).
        params (list): List of tuples with parameters to log.
        obs (list): List of observables to measure.
        convobs (list): List of observables for convergence checks.
        convparams (list or dict): Parameters for convergence runs. Must be specified.
        save (bool): Whether to save the data.
        plot (bool): Whether to generate and save plots. Defaults to `save`.
        savedir (str): Directory to save results. Defaults to '~/MPSDynamics/'.
        unid (str): Unique ID for the run. Auto-generated if not provided.
        name (str): A name for the calculation for the log file.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple of (final_state, data_dictionary).
    """
    # Set default values for mutable arguments
    if machine is None:
        machine = LocalMachine()
    if params is None:
        params = []
    if obs is None:
        obs = []
    if convobs is None:
        convobs = []
    if plot is None:
        plot = save
    if savedir is None:
        savedir = os.path.join(os.path.expanduser("~"), "MPSDynamics/")
    if unid is None:
        unid = uuid.uuid4().hex[:5]

    if convparams is None:
        raise ValueError("Must specify convergence parameters in 'convparams'")

    remote = isinstance(machine, RemoteMachine)
    if remote:
        update_machines([machine])

    if isinstance(convparams, list) and len(convparams) > 1:
        convcheck = True
    else:
        convcheck = False
        if isinstance(convparams, list):
            convparams = convparams[0]
            
    onthefly_save = 'onthefly' in kwargs and kwargs['onthefly'].get('save_obs') and kwargs['onthefly'].get('savedir') == 'auto'

    if save or plot or onthefly_save:
        if not savedir.endswith('/'):
            savedir = savedir + '/'
        os.makedirs(savedir, exist_ok=True)
        open_log(dt, tmax, convparams, method, machine, savedir, unid, name, params, obs, convobs, convcheck, **kwargs)
        if onthefly_save:
            tmp_dir = os.path.join(savedir, unid, "tmp/")
            os.makedirs(tmp_dir, exist_ok=True)
            kwargs['onthefly']['savedir'] = tmp_dir
    
    param_dict = dict(params)
    param_dict.update({
        "dt": dt,
        "tmax": tmax,
        "method": method,
        "convparams": convparams,
        "unid": unid,
        "name": name
    })
    
    error_file = f"{unid}.e"

    t_start = datetime.now()
    A0, dat = (None, None)
    
    try:
        task_func = functools.partial(run_all, dt, tmax, A, H, method=method, obs=obs, convobs=convobs, convparams=convparams, **kwargs)
        
        out = launch_workers(machine, task_func)
        
        if isinstance(out, RemoteException):
            raise out
        else:
            A0, dat = out
            if save:
                conv_data = dat.get("convdata") if convcheck else None
                save_data(savedir, unid, convcheck, dat["data"], conv_data, param_dict)
            if plot:
                data_to_plot = dat.get("convdata") if convcheck else dat["data"]
                save_plot(savedir, convcheck, unid, dat["data"]["times"], data_to_plot, convparams, convobs)
            dat = flatten_dict(dat)
            return A0, dat
            
    except Exception as e:
        if save:
            error_log(savedir, unid)
        traceback.print_exc()
        if save:
            error_path = os.path.join(savedir, unid, error_file)
            with open(error_path, "w+") as f:
                traceback.print_exc(file=f)
        return None, None
    finally:
        t_elapsed = datetime.now() - t_start
        if save:
            run_dir = os.path.join(savedir, unid)
            if os.path.exists(run_dir):
                output_files = [f for f in os.listdir(run_dir) if f not in [error_file, "info.txt"]]
                output = len(output_files) > 0
                close_log(savedir, unid, output, t_elapsed)
        print(f"total run time : {t_elapsed}")
        
    return A0, dat 
