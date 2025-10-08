import time
import sys
import os
import re
from datetime import timedelta
import h5py
import numpy as np
import matplotlib.pyplot as plt

class Observable:
    def __init__(self, name):
        self.name = name

class ProgressBar:
    """
    An iterable returning values from 1 to `numsteps`. Displays a progress bar 
    for the for loop where it has been called. If ETA is true then displays an 
    estimation of the remaining time calculated based on the time spent 
    computing the last `last` values.
    """
    def __init__(self, numsteps, ETA=False, last=10):
        self.numsteps = numsteps
        self.ETA = ETA
        self.times = [0.0] * (last + 1 if ETA else 2)
        self.Dmax = 0
        try:
            self.length = min(os.get_terminal_size()[0] - 54, 50)
        except OSError: # In non-interactive environments
             self.length = 50
        if self.length < 0:
            raise ValueError("Error : Terminal window too narrow")
        self.state = 1

    def __iter__(self):
        self.state = 1
        return self

    def __next__(self):
        Ntimes = len(self.times) - 1
        if self.state > self.numsteps:
            print()
            raise StopIteration
        
        if self.state == 1:
            print("\n\033[1;31mCompiling...\033[0m", end='', flush=True)
            self.times = [time.time()] * (Ntimes + 1)
        else:
            tnow = time.time()
            dtelapsed = tnow - self.times[0]
            
            if self.ETA:
                dtETA = (tnow - self.times[1 + (self.state - 1) % Ntimes]) * (self.numsteps - self.state) / min(self.state - 1, Ntimes)
            else:
                dtETA = 0
                
            dtiter = tnow - self.times[1 + (self.state - 2) % Ntimes]
            self.times[1 + (self.state-1) % Ntimes] = tnow

            elapsed_str = str(timedelta(seconds=int(dtelapsed)))
            eta_str = str(timedelta(seconds=int(dtETA)))

            if dtiter > 60:
                iter_str = time.strftime("%M:%S", time.gmtime(dtiter)) + f".{int((dtiter%1)*1000):03d}"
            else:
                iter_str = f"{dtiter:02.3f}"
            
            sys.stdout.write('\r')
            
            progress = self.state / self.numsteps
            progress_percent = f"{progress:.1%}"
            
            bar_len = int(progress * self.length)
            bar = "┣" + "#" * bar_len + " " * (self.length - bar_len) + "┫"

            dmax_str = f"; Dmax={self.Dmax}" if self.Dmax > 0 else ""
            eta_full_str = f"; ETA:{eta_str}s" if self.ETA else ""

            status = (f"\033[1;32m{progress_percent} \033[0m"
                      f"{bar}"
                      f"\033[1;32m {self.state}/{self.numsteps} [{elapsed_str}s"
                      f"{eta_full_str}; {iter_str}s/it"
                      f"{dmax_str}]\033[0m")
            
            sys.stdout.write(status)
            sys.stdout.flush()

        result = self.state
        self.state += 1
        return result

def onthefly(plot_obs=None, save_obs=None, savedir="auto", step=10, func=lambda x: x, compare=None, clear=None):
    """
    Helper function returning a dictionary containing the necessary arguments for on-the-fly plotting or saving.

    Args:
        plot_obs (Observable, optional): Observable to plot.
        save_obs (list[Observable], optional): List of Observables to save.
        savedir (str, optional): Path to store temporary files. Defaults to "auto".
        step (int, optional): Number of time steps between plots/saves. Defaults to 10.
        func (function, optional): Function to apply to plot_obs data. Defaults to identity.
        compare (tuple, optional): (times, data) of previous results to compare against.
        clear (function, optional): Function to clear output (e.g., in Jupyter).

    Returns:
        dict: A dictionary of on-the-fly parameters.
    """
    if save_obs is None:
        save_obs = []
    if plot_obs is None and not save_obs:
        raise ValueError("Must provide an observable to plot/save")

    fig, ax = (None, None)
    if plot_obs is not None:
        fig, ax = plt.subplots()
        ax.set_title("Intermediate Results")
        ax.set_xlabel("t")
        ax.set_ylabel(plot_obs.name)
        if compare is not None:
            ax.plot(compare[0], compare[1])
        # Create an empty line object to update later
        line, = ax.plot([], [])
        ax.legend()

    print("On the fly mode activated")
    return {
        "plot_obs": plot_obs.name if plot_obs else None,
        "save_obs": [ob.name for ob in save_obs],
        "savedir": savedir,
        "step": step,
        "func": func,
        "clear": clear,
        "compare": compare,
        "fig": fig,
        "ax": ax,
        "line": line if plot_obs else None,
    }

def onthefly_plot(onthefly_dict, tstep, times, data):
    """Plots data according to the arguments of the onthefly dictionary."""
    if onthefly_dict["line"] is None:
        return

    times_to_plot = np.asarray(times[:tstep+1])
    data_to_plot = np.asarray(data[onthefly_dict['plot_obs']][:tstep+1])
    
    onthefly_dict["line"].set_data(times_to_plot, onthefly_dict["func"](data_to_plot))
    
    onthefly_dict["ax"].relim()
    onthefly_dict["ax"].autoscale_view()
    onthefly_dict["fig"].canvas.draw()
    onthefly_dict["fig"].canvas.flush_events()

    if onthefly_dict["clear"] is not None:
        try:
            onthefly_dict["clear"](wait=True)
        except TypeError:
            onthefly_dict["clear"]()
    
    try:
        from IPython.display import display
        display(onthefly_dict["fig"]) # Assumes an IPython environment
    except ImportError:
        plt.pause(0.01) # fallback for non-IPython environments

    time.sleep(0.05)


def onthefly_save(onthefly_dict, tstep, times, data):
    """Saves data according to the arguments of the onthefly dictionary."""
    filename = os.path.join(onthefly_dict["savedir"], f"tmp{tstep // onthefly_dict['step']}.h5")
    with h5py.File(filename, "w") as f:
        start_idx = max(0, tstep - onthefly_dict["step"] + 1)
        end_idx = tstep + 1
        f.create_dataset("times", data=times[start_idx:end_idx])
        for name in onthefly_dict["save_obs"]:
            f.create_dataset(name, data=data[name][start_idx:end_idx])

def merge_tmp(tmpdir, fields=None, overwrite=True):
    """
    Merges the temporary files created by onthefly_save at the directory `tmpdir` 
    and returns a dictionary containing the resulting data.
    By default all fields are present but one can select the fields of interest 
    with a list of names in `fields`.
    """
    if not os.path.isdir(tmpdir):
        raise ValueError("Choose a valid directory")

    files = [f for f in os.listdir(tmpdir) if re.match(r"tmp(\d+)\.h5", f)]
    files.sort(key=lambda x: int(re.match(r"tmp(\d+)\.h5", x).groups()[0]))

    if not files:
        print("No temporary files found to merge.")
        return {}

    if fields is None:
        with h5py.File(os.path.join(tmpdir, files[0]), 'r') as f:
            fields = list(f.keys())

    merged_data = {ob: [] for ob in fields}

    for file in files:
        filepath = os.path.join(tmpdir, file)
        with h5py.File(filepath, 'r') as f:
            for ob in fields:
                merged_data[ob].append(f[ob][:])
    
    for key in merged_data:
        merged_data[key] = np.concatenate(merged_data[key])

    if overwrite:
        last_file_path = os.path.join(tmpdir, files[-1])
        with h5py.File(last_file_path, 'w') as f:
            for key, val in merged_data.items():
                f.create_dataset(key, data=val)
        
        for file in files[:-1]:
            os.remove(os.path.join(tmpdir, file))
        os.rename(last_file_path, os.path.join(tmpdir, "merged_data.h5"))


    return merged_data 