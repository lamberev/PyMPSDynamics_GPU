# machines.py
#
# This file is a skeleton of the original machines.jl.
# Given that the primary computations are done on GPUs, the
# CPU parallelism in the original Julia version is
# unnecessary. Only the classes and functions necessary to
# be compatible with the run_sim are included here.

import functools
from .run_all import run_all

class Machine:
    """Base class for execution environments."""
    pass

class LocalMachine(Machine):
    """Represents the local machine environment."""
    def __init__(self, name="local"):
        self.name = name

class RemoteMachine(Machine):
    """Represents a remote machine environment."""
    def __init__(self, name, exename=None, wdir=None):
        self.name = name
        self.exename = exename
        self.wdir = wdir

class RemoteException(Exception):
    """Custom exception for remote execution errors."""
    pass

def update_machines(machines):
    """Placeholder function to simulate updating machines."""
    print(f"Updating machines: {machines}")

def launch_workers(machine, func):
    """
    Trivial worker launch.
    """
    print(f"Launching task on {machine.name}.")
    # In a real distributed scenario this would
    # serialize 'func' and send it to a worker process on the specified machine.
    # Here, we just call it directly. The 'func' is a functools.partial
    # that has the target function (run_all) and its arguments bundled.
    try:
        return func()
    except Exception as e:
        print(f"An error occurred during simplified execution: {e}")
        raise RemoteException(e) 