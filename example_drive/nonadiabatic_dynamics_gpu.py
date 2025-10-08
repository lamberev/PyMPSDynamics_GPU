import sys
import os
import numpy as np

sys_path = '/home/lamberev/local/PyMPSDynamics_GPU-latest'
sys.path.insert(0, sys_path)

import cupy as cp
import h5py
import copy

from MPSDynamics import (
    run_sim, MPSDynamicsError,
    phys_dims, to_gpu, norm_mps,
    methyl_blue_mpo_2, read_chain_coeffs,
    unitvec, numb,
    OneSiteObservable,
    TreeNetwork, find_head_node,
    log_params
)
from MPSDynamics.treeTDVP import product_state_mps

def main():
    ########################################################################
    ########################## INPUT PARAMETERS ############################
    ########################################################################

    run_name = "example"

    energy_1 = 3.0 / 27.2114 + 2500 / 219463
    energy_2 = 3.2 / 27.2114 + 2500 / 219463
    coupling = 0.0
    mu_01 = 1.0
    mu_02 = 0.0

    savedir = os.getcwd()
    coeffsdir = os.path.join(savedir, "chain_coeffs.hdf5")

    time_step = 1.0
    total_time_absorption = 500.0
    total_time_emission = 1.0
    temp = 0
    chain_length = 45
    num_fock = 8
    bond_dims = [20]

    # --- Gaussian-enveloped cosine pulse parameters on system site (site id = 1) ---
    # E(t) = E_amp * cos(omega_carrier * (t - t_pulse_center)) * exp(-4 ln 2 * ((t - t_pulse_center)/t_pulse_duration)^2)
    E_amp = 0.1
    omega_carrier = 3.05 / 27.2114
    t_pulse_center = 30
    t_pulse_duration = 10

    PROP_IMAG = False
    imag_time_step = -50.0j
    max_imag_time = -3000.0j
    chunk_imag_time = -50.0j

    ########################################################################
    ########################################################################
    ########################################################################

    if PROP_IMAG and temp != 0:
        print("Error: temp not 0 for imaginary time propagation.")
        exit(1)

    def get_temp_str(temp_val):
        if temp_val == 0:
            return "0"
        elif int(temp_val) == temp_val:
            return str(int(round(temp_val)))
        else:
            return str(temp_val)

    temp_str = get_temp_str(temp)
    s1_chain_coeffs = read_chain_coeffs(coeffsdir, f"temp_{temp_str}", "s1")
    s2_chain_coeffs = read_chain_coeffs(coeffsdir, f"temp_{temp_str}", "s2")
    s1_s2_coupling = read_chain_coeffs(coeffsdir, f"temp_{temp_str}", "s1_s2_coupling")
    
    with h5py.File(coeffsdir, 'r') as f:
        long_range_s1_to_s2 = f[f"temp_{temp_str}/long_range_s1_to_s2"][()]
        long_range_s2_to_s1 = f[f"temp_{temp_str}/long_range_s2_to_s1"][()]

    # Hamiltonian
    mpo_cpu = methyl_blue_mpo_2(
        energy_1, energy_2, coupling,
        chain_length, chain_length, chain_length,
        num_fock, num_fock, num_fock,
        s1_chain_coeffs, long_range_s1_to_s2, long_range_s2_to_s1, s2_chain_coeffs,
        s1_s2_coupling
    )

    mpo = to_gpu(mpo_cpu)

    s2 = unitvec(0, 3)
    s1 = unitvec(1, 3)
    s0 = unitvec(2, 3)
    
    # Initial state for absorption: purely s0
    psi = s0

    # Dipole operator
    mu = mu_01 * cp.outer(s0, s1.conj()) + mu_02 * cp.outer(s0, s2.conj())

    # Initial product state
    initial_states = [psi] + [unitvec(0, num_fock) for _ in range(3 * chain_length)]
    mps = product_state_mps(mpo.tree, phys_dims(mpo), state=initial_states)
    mps = to_gpu(mps)

    # System-site Pauli-x-like couplings (between specified levels)
    # sigma_x01 couples S0<->S1; sigma_x02 couples S0<->S2. No S1<->S2 term.
    sigma_x01 = cp.outer(s0, s1.conj()) + cp.outer(s1, s0.conj())
    sigma_x02 = cp.outer(s0, s2.conj()) + cp.outer(s2, s0.conj())

    # Build time lists and per-step on-site drive operators for site 1
    def gaussian_cos_pulse(t):
        dt = t - t_pulse_center
        return E_amp * np.cos(omega_carrier * dt) * np.exp(-4.0 * np.log(2.0) * (dt / t_pulse_duration) ** 2)

    num_steps_abs = int(round(abs(total_time_absorption) / abs(time_step)))
    times_abs = [i * time_step for i in range(num_steps_abs + 1)]
    Htime_abs = [
        gaussian_cos_pulse(t) * (mu_01 * sigma_x01 + mu_02 * sigma_x02)
        for t in times_abs[:-1]
    ]

    num_steps_em = int(round(abs(total_time_emission) / abs(time_step)))
    times_em = [i * time_step for i in range(num_steps_em + 1)]
    Htime_em = [
        gaussian_cos_pulse(t) * (mu_01 * sigma_x01 + mu_02 * sigma_x02)
        for t in times_em[:-1]
    ]

    # Observables
    ob4 = OneSiteObservable("occ1", numb(num_fock), sites=(2, chain_length + 1))
    ob5 = OneSiteObservable("occ2", numb(num_fock), sites=(chain_length + 2, 2 * chain_length + 1))
    ob6 = OneSiteObservable("occ12", numb(num_fock), sites=(2 * chain_length + 2, 3 * chain_length + 1))
    ob7 = OneSiteObservable("dcf", mu, sites=1)
    ob8 = OneSiteObservable("s0", cp.outer(s0, s0.conj()), sites=1)
    ob9 = OneSiteObservable("s1", cp.outer(s1, s1.conj()), sites=1)
    ob10 = OneSiteObservable("s2", cp.outer(s2, s2.conj()), sites=1)
    ob11 = OneSiteObservable("s1s2", cp.outer(s1, s2.conj()), sites=1)
    ob12 = OneSiteObservable("s0s2", cp.outer(s0, s2.conj()), sites=1)
    ob13 = OneSiteObservable("s0s1", cp.outer(s0, s1.conj()), sites=1)
    absorption_emission_convobs = [ob4, ob5, ob6, ob7, ob8, ob9, ob10, ob11, ob12, ob13]

    # Propagate excited state (for absorption)
    mps_relaxed, dat_abs = run_sim(
        time_step, total_time_absorption, mps, mpo,
        name=run_name + ": absorption",
        unid="absorption",
        method='TDVP1',
        savedir=savedir,
        obs=[],
        convobs=absorption_emission_convobs,
        convparams=bond_dims,
        verbose=False,
        save=True,
        plot=True,
        params=log_params(
            psi=psi,
            energy_1=energy_1, energy_2=energy_2, coupling=coupling,
            mu_01=mu_01, mu_02=mu_02,
            temp=temp, chain_length=chain_length, num_fock=num_fock
        ),
        # Time-dependent drive on system site (id=1); site indices are 1-based
        timedep=True,
        Ndrive=1, 
        Htime=Htime_abs,
    )

    if mps_relaxed is None:
        print("Simulation failed to produce a final MPS. Exiting.")
        sys.exit(1)

    if PROP_IMAG and temp == 0:
        mps_ground_current = copy.deepcopy(mps_relaxed)
        cumulative_imag_time = 0.0j
        chunk_num = 0

        while abs(cumulative_imag_time) < abs(max_imag_time):
            chunk_num += 1
            print(f"\n--- Chunk {chunk_num} ---")

            remaining_time = max_imag_time - cumulative_imag_time
            time_this_chunk_val = min(abs(chunk_imag_time), abs(remaining_time))
            time_this_chunk = np.sign(imag_time_step) * time_this_chunk_val

            mps_chunk_end, dat_chunk = run_sim(
                imag_time_step, time_this_chunk, mps_ground_current, mpo,
                name=f"{run_name}: chunk {chunk_num}",
                unid=f"chunk_{chunk_num}",
                method='TDVP1',
                savedir=savedir,
                obs=[],
                convobs=[],
                convparams=bond_dims,
                verbose=False,
                save=True,
                plot=False,
                params=log_params(
                    imag_time_step=imag_time_step, time_this_chunk=time_this_chunk, cumulative_imag_time=cumulative_imag_time,
                    temp=temp, chain_length=chain_length, num_fock=num_fock
                )
            )

            mps_ground_current = mps_chunk_end
            cumulative_imag_time += time_this_chunk

            head_node_index = find_head_node(mps_ground_current.tree)
            norm_val = norm_mps(mps_ground_current, mpsorthog=head_node_index)
            print(f"Chunk {chunk_num} norm: {cp.sqrt(norm_val)}")

            # Normalize the state
            mps_ground_current.sites[head_node_index - 1] /= cp.sqrt(norm_val)

            print(f"Cumulative time: {cumulative_imag_time}")

        print("-" * 40)
        print(f"Reached max imag time: {cumulative_imag_time}")
        print("-" * 40)

        mps_ground_final = mps_ground_current
    else:
        mps_ground_final = copy.deepcopy(mps_relaxed)
        if mps_ground_final is None:
            print("Error: mps_ground_final is None before emission calculation. Exiting.")
            sys.exit(1)

    # Prepare initial state for emission
    site0_tensor = mps_ground_final.sites[0]
    site0_tensor[..., 2] = mu_01 * site0_tensor[..., 1] + mu_02 * site0_tensor[..., 0]

    mps_emitted, dat_emission = run_sim(
        time_step, total_time_emission, mps_ground_final, mpo,
        name=run_name + ": emission",
        unid="emission",
        method='TDVP1',
        savedir=savedir,
        obs=[],
        convobs=absorption_emission_convobs,
        convparams=bond_dims,
        verbose=False,
        save=True,
        plot=True,
        params=log_params(
            energy_1=energy_1, energy_2=energy_2, coupling=coupling,
            mu_01=mu_01, mu_02=mu_02,
            temp=temp, chain_length=chain_length, num_fock=num_fock
        ),
        timedep=False,
        Ndrive=1,
        Htime=Htime_em,
    )

if __name__ == "__main__":
    main()
