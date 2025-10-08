#!/usr/bin/env python3
"""
Compute dipole response spectrum from an HDF5/H5 or dat input files, 
apply damping envelope, then save both data and a PDF plot.

Steps:
1. Read HDF5/H5 or dat file to extract dipole response data (real and imaginary parts) and time points.
2. Build a response function array with time, dipole amplitudes, and phase information.
3. Enforce continuous phase and apply an energy shift.
4. Apply a damping envelope e^(-t/TAU) * cos(pi*t/(2*T_FINAL)) to the response amplitude.
5. Zero-pad the response function, ensuring correct time axis for padded part.
6. Symmetrically extend the padded response function using C(-t) = C*(t) to cover -T_pad < t < T_pad.
7. Compute effective complex response from the extended data and integrate over energy to generate the spectrum.
8. Save the extended dipole response and spectrum to output files, renaming any .jld file found.
9. Plot the spectrum and save as a PDF.

Usage:
    compute_lineshape_from_dcf.py my_file.h5
    compute_lineshape_from_dcf.py my_file.hdf5
    compute_lineshape_from_dcf.py my_file.dat
"""

import sys
import os
import glob
import h5py
import numpy as np
import cmath
import math
from scipy import integrate
import matplotlib.pyplot as plt

TOTAL_ENERGY = 0.000       # Energy shift (Ha)
SPECTRUM_START = -0.045    # Start of energy range for spectrum (Ha)
SPECTRUM_END = 0.045       # End of energy range for spectrum (Ha)
NUM_POINTS = 10000         # Number of points in the computed spectrum
PADDING_LENGTH = 10000     # Number of zero-padding time steps
TAU = 35000.0              # Damping time constant for envelope (a.u.)


def full_spectrum_integrand(response, energy):
    """
    Compute the real part of the time-domain integrand for a given energy.

    response: array of shape (N, 2) with columns [time, amplitude]
    energy: float, energy value at which to evaluate the integrand
    """
    times = response[:, 0]
    amplitudes = response[:, 1]
    return (amplitudes * np.exp(1j * times * energy)).real


def compute_spectrum(response, num_points, start, end):
    """
    Integrate the time-domain response function over energy to compute the spectrum.

    response: array shape (N, 2) with [time, amplitude]
    Returns: spectrum array shape (num_points, 2) [energy, intensity]
    """
    dt = response[1, 0] - response[0, 0]
    energies = np.linspace(start, end, num_points)
    spectrum = np.zeros((num_points, 2))

    for i, E in enumerate(energies):
        integrand = full_spectrum_integrand(response, E)
        intensity = integrate.simpson(integrand, dx=dt)
        spectrum[i, 0] = E
        spectrum[i, 1] = intensity

    # Handle output file naming and .jld renaming
    cwd = os.getcwd()
    dir_parts = cwd.split(os.sep)[-4:-1]
    base_name = '_'.join(dir_parts)
    output_dat = base_name + '.dat'
    jld_list = glob.glob('*.jld')
    if jld_list:
        os.rename(jld_list[0], base_name + '.jld')

    np.savetxt(output_dat, spectrum)
    return spectrum


def compute_spectrum_clean(response, num_points, start, end):
    """
    Integrate the time-domain response function over energy to compute the spectrum.
    Clean version without file handling side effects.

    response: array shape (N, 2) with [time, amplitude]
    Returns: spectrum array shape (num_points, 2) [energy, intensity]
    """
    dt = response[1, 0] - response[0, 0]
    energies = np.linspace(start, end, num_points)
    spectrum = np.zeros((num_points, 2))

    for i, E in enumerate(energies):
        integrand = full_spectrum_integrand(response, E)
        intensity = integrate.simpson(integrand, dx=dt)
        spectrum[i, 0] = E
        spectrum[i, 1] = intensity

    return spectrum


def plot_spectrum(spectrum, filename='spectrum.pdf'):
    """
    Plot the computed spectrum and save to a PDF file.

    spectrum: array shape (M, 2) [energy, intensity]
    filename: output PDF file name
    """
    energies = spectrum[:, 0]
    intensities = spectrum[:, 1]
    plt.figure()
    plt.fill(energies, intensities, color='gray', alpha=0.8)
    plt.xlabel('Energy (Ha)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def load_response_from_hdf5(filename):
    """
    Read the HDF5 file and assemble the time-domain dipole response function.
    Handles both original HDF5 structure (with nested groups) and new H5 structure
    (with data directly in the 'data' group).

    Returns:
      response_func: array shape (N, 5) columns [time, real, imag, magnitude, phase]
    """
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        print(f'Keys in HDF5 file: {keys}')
        
        # Check if this is the new H5 format with 'data' group
        if 'data' in keys:
            print(f'Using new H5 format with data group')
            group = f['data']
            print(f'Data group keys: {list(group.keys())}')
        else:
            # Original format with nested groups
            group = f[keys[2]]
            print(f'Using original format with group: {keys[2]}, type: {type(group)}')
            for name in group:
                print(f'  {name}')

        real_part = group['dcf-re'][()]
        imag_part = group['dcf-im'][()]
        times = group['times'][()]

    N = times.shape[0]
    response = np.zeros((N, 5))
    for i in range(N):
        t = times[i]
        re = real_part[i]
        im = imag_part[i]
        mag, phase = cmath.polar(re + 1j * im)
        response[i] = [t, re, im, mag, phase]
    return response


def enforce_continuous_phase(response):
    """
    Correct phase jumps larger than 0.7*pi to ensure a smooth phase evolution.
    Modifies the phase column in place.
    """
    phase_offset = 0.0
    N = response.shape[0]
    for i in range(N - 1):
        response[i, 4] += phase_offset
        delta = response[i+1, 4] + phase_offset - response[i, 4]
        if abs(delta) > 0.7 * math.pi:
            n = int(round(delta / math.pi))
            phase_offset -= n * math.pi
    response[-1, 4] += phase_offset


def apply_energy_shift(response, shift):
    """
    Shift the phase by energy * time to account for total energy offset.
    """
    response[:, 4] -= shift * response[:, 0]


def apply_damping(response, tau):
    """
    Apply a damping envelope e^(-t/tau) * cos(pi*t/(2*T_final))
    to the magnitude column of the response.
    """
    times = response[:, 0]
    T_final = times[-1]
    envelope = np.exp(-times / tau) * np.cos(np.pi * times / (2 * T_final))
    response[:, 3] *= envelope


def pad_response(response, padding_length):
    """
    Add zero values to the end of response for improved spectral resolution
    and extend the time axis accordingly.
    """
    N_orig, M = response.shape
    
    if padding_length == 0:
        return response.copy() # No padding needed

    N_total_padded = N_orig + padding_length
    padded_array = np.zeros((N_total_padded, M))

    # Copy original data if it exists
    if N_orig > 0:
        padded_array[:N_orig, :] = response
        original_times = response[:, 0]
        last_original_time = original_times[-1]
        
        # Determine dt for extending the time axis
        if N_orig > 1:
            dt = original_times[1] - original_times[0]
        elif N_orig == 1 and padding_length > 0: # Only t=0 point exists, and we need to pad
            print("Warning: Original response has only t=0. `dt` for padding times is assumed as 1.0.")
            dt = 1.0 # Default dt, as it cannot be derived from a single point.
        elif N_orig == 1 and padding_length == 0: # Only t=0, no padding
            dt = 0 # dt is not strictly needed here, but avoid unassigned variable
        else: # N_orig == 0, should not happen if load_response_from_hdf5 works
            dt = 1.0 
            last_original_time = -dt # To start padded times from 0 if N_orig=0

        # Fill time for the padded part only if there's padding to do and dt is sensible
        if padding_length > 0 and dt > 0 : # dt > 0 to avoid issues with arange if dt is 0 or negative
            padded_time_extension = last_original_time + dt * np.arange(1, padding_length + 1)
            padded_array[N_orig:, 0] = padded_time_extension
        elif padding_length > 0 and dt == 0 and N_orig == 1: # Special case: single point at t=0, padding
             # Times for padded region will also be 0, effectively.
            padded_array[N_orig:, 0] = last_original_time


    elif N_orig == 0 and padding_length > 0: # padding an empty response
        # This case implies creating a signal from scratch, which is unusual for this script.
        print("Warning: Padding an empty response. Time axis for padding starts at 0 with dt=1.0.")
        dt = 1.0 # Default dt
        padded_array[:, 0] = dt * np.arange(padding_length)
        # Other data columns (real, imag, mag, phase) are already zero.
        
    return padded_array


def symmetrically_extend_response(padded_response):
    """
    Extends the response C(t) for t>=0 to -T < t < T using C(-t) = C*(t).
    Assumes padded_response[0,:] is for t=0.
    Input `padded_response` has columns [time, real, imag, magnitude, phase].
    """
    if padded_response.shape[0] == 0:
        return padded_response # Nothing to extend

    # Data for t=0 (first row)
    response_at_t0 = padded_response[0:1, :]
    
    # Data for t > 0 (all rows after the first)
    response_for_t_gt_0 = padded_response[1:, :]

    if response_for_t_gt_0.shape[0] == 0: # Only t=0 point was provided
        # Symmetric extension of C(0) is just C(0)
        return response_at_t0

    # Construct C(-t) from C(t) for t > 0 based on C(-t) = C*(t)
    # C*(t) means: real part same, imag part negated, mag same, phase negated.
    response_for_t_lt_0 = np.copy(response_for_t_gt_0)
    response_for_t_lt_0[:, 0] *= -1  # time t -> -t
    # response_for_t_lt_0[:, 1] (real part) remains the same
    response_for_t_lt_0[:, 2] *= -1  # imag part  Im(C(t)) -> -Im(C(t))
    # response_for_t_lt_0[:, 3] (magnitude) remains the same
    response_for_t_lt_0[:, 4] *= -1  # phase phi(t) -> -phi(t)

    # Concatenate:
    # 1. Flipped C(-t) data (times from most negative up to -dt)
    # 2. C(0) data (time t=0)
    # 3. C(t) data (times from dt up to most positive)
    extended_response = np.concatenate((
        np.flipud(response_for_t_lt_0), # Orders time from -T_max to -dt
        response_at_t0,
        response_for_t_gt_0
    ))
    return extended_response


def compute_effective_response(extended_response_data):
    """
    Build complex effective response from magnitude and phase of the extended data.
    Saves the input extended_response_data (5 columns) to 'dipole_response.txt'.
    """
    N = extended_response_data.shape[0]
    eff = np.zeros((N, 2), dtype=complex) # Array for [time, complex_value]
    eff[:, 0] = extended_response_data[:, 0] # Time column
    # Complex value = magnitude * exp(i * phase)
    eff[:, 1] = extended_response_data[:, 3] * np.exp(1j * extended_response_data[:, 4])
    
    # Save the 5-column extended data, which now includes negative times if extended
    np.savetxt('dipole_response.txt', extended_response_data)
    return eff


def compute_effective_response_clean(extended_response_data):
    """
    Build complex effective response from magnitude and phase of the extended data.
    Clean version without file saving side effects.
    """
    N = extended_response_data.shape[0]
    eff = np.zeros((N, 2), dtype=complex) # Array for [time, complex_value]
    eff[:, 0] = extended_response_data[:, 0] # Time column
    # Complex value = magnitude * exp(i * phase)
    eff[:, 1] = extended_response_data[:, 3] * np.exp(1j * extended_response_data[:, 4])
    
    return eff


def compute_lineshape_from_dcf_data(times, dcf_data, output_filename='spectrum.pdf'):
    """
    Compute lineshape directly from dcf data arrays.
    
    Args:
        times: array of time points
        dcf_data: complex array of dcf values
        output_filename: filename for the output PDF plot
    
    Returns:
        spectrum: array shape (num_points, 2) [energy, intensity]
    """
    # Ensure inputs are numpy arrays
    times = np.asarray(times)
    dcf_data = np.asarray(dcf_data)
    
    # Convert to the format expected by the processing functions
    N = times.shape[0]
    response = np.zeros((N, 5))
    for i in range(N):
        t = times[i]
        dcf_val = dcf_data[i]
        re = np.real(dcf_val)
        im = np.imag(dcf_val)
        mag, phase = cmath.polar(dcf_val)
        response[i] = [t, re, im, mag, phase]
    
    # Apply the same processing as in main()
    enforce_continuous_phase(response)
    apply_energy_shift(response, TOTAL_ENERGY)
    apply_damping(response, TAU)
    
    # Zero-pad and compute effective response
    padded = pad_response(response, PADDING_LENGTH)
    extended_resp = symmetrically_extend_response(padded)
    eff_response = compute_effective_response_clean(extended_resp)
    
    # Compute spectrum using clean version
    spectrum = compute_spectrum_clean(eff_response, NUM_POINTS, SPECTRUM_START, SPECTRUM_END)
    
    # Save spectrum data to .dat file (matching original behavior)
    # Extract base name from output_filename for the .dat file
    base_name = os.path.splitext(os.path.basename(output_filename))[0]
    dat_filename = os.path.join(os.path.dirname(output_filename), f"{base_name}.dat")
    np.savetxt(dat_filename, spectrum)
    
    # Plot and save
    plot_spectrum(spectrum, output_filename)
    
    return spectrum


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 script.py input_file.h5')
        print('       python3 script.py input_file.hdf5')
        sys.exit(1)
    filename = sys.argv[1]

    # Load and process response function
    response = load_response_from_hdf5(filename)
    enforce_continuous_phase(response)
    apply_energy_shift(response, TOTAL_ENERGY)

    # Apply damping envelope to magnitude
    apply_damping(response, TAU)

    # Zero-pad and compute effective response
    padded = pad_response(response, PADDING_LENGTH)
    extended_resp = symmetrically_extend_response(padded)
    eff_response = compute_effective_response(extended_resp)

    # Compute spectrum and save data
    spectrum = compute_spectrum(eff_response, NUM_POINTS, SPECTRUM_START, SPECTRUM_END)

    # Plot and save spectrum PDF
    plot_spectrum(spectrum)

if __name__ == '__main__':
    main()
