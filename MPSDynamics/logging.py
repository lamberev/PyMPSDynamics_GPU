import os
import socket
import datetime
import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Tuple

from .machines import Machine

def log_params(**kwargs: Any) -> List[List[Union[str, Any]]]:
    """
    Creates a list of [name, value] pairs from keyword arguments.
    This is a replacement for the @LogParams macro in Julia.
    
    Example:
    a = 1
    b = "test"
    params = log_params(a=a, b=b) 
    # params will be [['a', 1], ['b', 'test']]
    """
    return [[name, value] for name, value in kwargs.items()]

def open_log(
    dt: float, 
    tmax: float, 
    convparams: Any, 
    method: str, 
    machine: Machine, 
    savedir: str, 
    unid: str, 
    name: str, 
    params: List[List[Any]], 
    obs: List[Any], 
    convobs: List[Any], 
    convcheck: bool, 
    **kwargs: Any
):
    """
    Sets up logging for a simulation run. Creates a directory for the run,
    and writes a header to a global log file and a run-specific info file.
    """
    run_dir = os.path.join(savedir, unid)
    os.makedirs(run_dir, exist_ok=True)
    
    logname = os.path.join(savedir, f"log-{socket.gethostname()}.txt")
    info_filename = os.path.join(run_dir, "info.txt")

    message_parts = []
    message_parts.append(f"[{datetime.datetime.now()}] => RUN <{unid}> START")
    if isinstance(name, str):
        message_parts.append(f"\t name : {name}")
    
    message_parts.append(f"\t machine : {machine.name}")
    message_parts.append(f"\t method : {method}")
    message_parts.append(f"\t dt = {dt}")
    message_parts.append(f"\t tmax = {tmax}")
    
    param_str = ", ".join([f"{p[0]} = {p[1]}" for p in params])
    message_parts.append(f"\t parameters : {param_str}")

    obs_str = ", ".join([o.name for o in obs])
    message_parts.append(f"\t observables : {obs_str}")

    if convcheck:
        convobs_str = ", ".join([o.name for o in convobs])
        message_parts.append(f"\t convergence observables : {convobs_str}")
    
    message_parts.append(f"\t convparams : {convparams}")
    
    log_message = "\n".join(message_parts)

    with open(info_filename, "w") as f0:
        f0.write(log_message + "\n")

    with open(logname, "a") as f:
        f.write(log_message + "\n\n")

def error_log(savedir: str, unid: str):
    """Logs an error message for a run."""
    logname = os.path.join(savedir, f"log-{socket.gethostname()}.txt")
    with open(logname, "a") as f:
        f.write(f"[{datetime.datetime.now()}] => RUN <{unid}> ERROR\n")
        f.write(f"\t see {unid}/{unid}.e for details\n\n")

def close_log(savedir: str, unid: str, output: bool, telapsed: Any):
    """Logs the end of a run, including total elapsed time."""
    logname = os.path.join(savedir, f"log-{socket.gethostname()}.txt")
    with open(logname, "a") as f:
        f.write(f"[{datetime.datetime.now()}] => RUN <{unid}> END\n")
        if output:
            f.write("\t output files produced\n")
        else:
            f.write("\t no output files produced\n")
        f.write(f"\t total run time : {telapsed}\n\n")

def _save_dict_to_h5_group(h5_group: h5py.Group, data_dict: Dict[str, Any]):
    """Helper to save a dictionary to an HDF5 group."""
    for key, value in data_dict.items():
        if isinstance(value, cp.ndarray):
            value = value.get()
        
        if value is None:
            value = "None"

        try:
            h5_group[key] = value
        except TypeError:
            print(f"Warning: could not save {key} to HDF5, converting to string.")
            h5_group[key] = str(value)

        if isinstance(value, np.ndarray) and np.iscomplexobj(value):
            h5_group[f"{key}-re"] = np.real(value)
            h5_group[f"{key}-im"] = np.imag(value)

def save_data(
    savedir: str, 
    unid: str, 
    convcheck: bool, 
    datadict: Dict[str, Any], 
    convdatadict: Dict[str, Any], 
    paramdatadict: Dict[str, Any]
):
    """
    Saves data from dictionaries into an HDF5 file.
    Converts cupy arrays to numpy arrays before saving.
    """
    filepath = os.path.join(savedir, unid, f"dat_{unid}.h5")
    with h5py.File(filepath, "w") as file:
        g1 = file.create_group("data")
        _save_dict_to_h5_group(g1, datadict)

        if convcheck:
            g2 = file.create_group("convdata")
            _save_dict_to_h5_group(g2, convdatadict)

        g3 = file.create_group("parameters")
        _save_dict_to_h5_group(g3, paramdatadict)

def save_plot(
    savedir: str, 
    convcheck: bool, 
    unid: str, 
    times: np.ndarray, 
    convdatadict: Dict[str, Any], 
    convparams: List[Any], 
    convobs: List[Any]
):
    """
    Creates and saves plots for convergence data using matplotlib.
    Also generates lineshape plots from dcf data if present.
    """
    plt.rc('figure', figsize=(8, 6))
    
    if isinstance(convparams, (list, tuple, np.ndarray)):
        labels = convparams
    else:
        labels = [convparams]

    # Check if dcf observable is present and generate lineshape plot
    dcf_present = any(ob.name == "dcf" for ob in convobs)
    if dcf_present and "dcf" in convdatadict:
        try:
            # Import the lineshape computation function
            from .compute_lineshape_from_dcf import compute_lineshape_from_dcf_data
            
            # Get dcf data and ensure it's a numpy array
            dcf_data = convdatadict["dcf"]
            if isinstance(dcf_data, cp.ndarray):
                dcf_data = dcf_data.get()
            dcf_data = np.asarray(dcf_data)
            
            # Ensure times is also a numpy array
            times_arr = np.asarray(times)

            # Save DCF as .dat. If complex, save three columns: time, Re, Im; else two: time, value.
            try:
                if times_arr.ndim > 1:
                    times_arr = times_arr.flatten()

                # If dcf is 2D, take the first column
                dcf_series = dcf_data[:, 0] if (hasattr(dcf_data, 'ndim') and dcf_data.ndim == 2) else dcf_data

                # Try to align lengths using optional times from convdatadict
                if dcf_series.shape[0] != times_arr.shape[0]:
                    alt_times = convdatadict.get("times", None)
                    if alt_times is not None:
                        if isinstance(alt_times, cp.ndarray):
                            alt_times = alt_times.get()
                        alt_times = np.asarray(alt_times)
                        if alt_times.ndim == 2:
                            alt_times = alt_times[:, 0]
                        if alt_times.shape[0] == dcf_series.shape[0]:
                            times_arr = alt_times

                # Fallback: synthetic time index if still mismatched
                if dcf_series.shape[0] != times_arr.shape[0]:
                    times_arr = np.arange(dcf_series.shape[0])

                if np.iscomplexobj(dcf_series):
                    dcf_dat = np.column_stack((times_arr, np.real(dcf_series), np.imag(dcf_series)))
                else:
                    dcf_dat = np.column_stack((times_arr, dcf_series))
                dcf_dat_path = os.path.join(savedir, unid, f"dcf_{unid}.dat")
                np.savetxt(dcf_dat_path, dcf_dat)
                print(f"Wrote DCF dat file: {dcf_dat_path}")
            except Exception as e_inner:
                print(f"Warning: Failed to write DCF .dat file for {unid}: {e_inner}")
            
            # Generate lineshape plot
            lineshape_filename = os.path.join(savedir, unid, f"lineshape_{unid}.pdf")
            spectrum = compute_lineshape_from_dcf_data(times_arr, dcf_data, lineshape_filename)
            
            print(f"Generated lineshape plot: {lineshape_filename}")
            
        except Exception as e:
            print(f"Warning: Failed to generate lineshape plot for {unid}: {e}")
            import traceback
            traceback.print_exc()

    for ob in convobs:
        if ob.name not in convdatadict:
            continue
            
        data = convdatadict[ob.name]
        if isinstance(data, cp.ndarray):
            data = data.get()

        fig, ax = plt.subplots()

        times_arr = np.asarray(times)
        if times_arr.ndim > 1:
            times_arr = times_arr.flatten()

        if data.shape[0] != times_arr.shape[0]:
            alt_times = convdatadict.get("times", None)
            if alt_times is not None:
                if isinstance(alt_times, cp.ndarray):
                    alt_times = alt_times.get()
                alt_times = np.asarray(alt_times)
                if alt_times.ndim == 2:
                    alt_times = alt_times[:, 0]
                if alt_times.shape[0] == data.shape[0]:
                    times_arr = alt_times

        if data.shape[0] != times_arr.shape[0]:
            times_arr = np.arange(data.shape[0])

        if np.iscomplexobj(data):
            if data.ndim == 2:
                for i, label in enumerate(labels):
                    ax.plot(np.real(data[:, i]), np.imag(data[:, i]), label=label)
            else:
                ax.plot(np.real(data), np.imag(data), label=labels[0] if labels else None)
            ax.set_xlabel(f"Re({ob.name})")
            ax.set_ylabel(f"Im({ob.name})")
        else:
            if data.ndim == 2:
                for i in range(data.shape[1]):
                    label = labels[i] if i < len(labels) else None
                    ax.plot(times_arr, data[:, i], label=label)
            else:
                ax.plot(times_arr, data, label=labels[0] if len(labels) == 1 else None)
                if data.ndim == 1 and len(labels) == 1:
                    ax.legend(labels)
            ax.set_xlabel("t")
            ax.set_ylabel(ob.name)
        
        ax.set_title(unid)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        
        plot_filename = os.path.join(savedir, unid, f"convplot_{ob.name}_{unid}.pdf")
        fig.savefig(plot_filename)
        plt.close(fig) 