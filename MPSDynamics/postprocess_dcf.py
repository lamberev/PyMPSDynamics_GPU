"""Postprocess DCF (dipole correlation function) outputs for multi-phase runs.

This module aggregates and reshapes DCF data produced by a 3-phase workflow
(phase1/phase2 absorption and phase3 emission), and computes lineshape
spectra. If the helper from MPSDynamics is available, it is used for PDF
generation; otherwise, a local FFT-based method is used.

Features
- Absorption:
  - Reads phase1 and phase2 DCF files under
    `phase1/absorption_phase1/dcf_absorption_phase1.dat` and
    `phase2/absorption_phase2/dcf_absorption_phase2.dat`.
  - Shifts phase2 times to absolute time using `t_phase1_end`, removes overlap,
    concatenates, time-sorts, and interpolates onto a uniform grid with step
    `dt_small` (cubic via SciPy if available, otherwise linear).
  - Writes `postprocess_dcf_output/combined_absorption_dcf.dat` and, if
    available, a lineshape PDF.

- Emission:
  - Scans `phase3/` for subdirectories named `emission_from_*` and copies each
    run's DCF into `postprocess_dcf_output/emission/<id>/combined_<id>.dat`.
  - Computes a lineshape for each emission dataset and writes
    `lineshape_<id>.dat` and `lineshape_<id>.pdf`.
  - Forms the average of all emission lineshapes and writes
    `lineshape_emission_average.dat` and `lineshape_emission_average.pdf`.

Main entry points
- `postprocess_absorption(run_root, t_phase1_end, t_phase2_end, dt_small, dt_large)` -> Optional[str]
- `postprocess_emission(run_root)` -> int
- `postprocess_for_run(run_root, t_phase1_end, t_phase2_end, dt_small, dt_large)` -> None

Notes
- Input `.dat` files may contain either [t, Re, Im] or [t, Re] columns.
- Lineshapes are computed via FFT after interpolating DCFs to a uniform
  time grid when the external helper is not available. If `matplotlib` is not
  installed, only `.dat` outputs are written.
"""

import os
import sys
from typing import Tuple, Optional, List

import numpy as np

# Optional SciPy interpolation; fallback to numpy if unavailable
try:
    from scipy.interpolate import interp1d as _scipy_interp1d
except Exception:
    _scipy_interp1d = None

# Reuse the MPSDynamics path from the main script if available on PYTHONPATH
try:
    from MPSDynamics.compute_lineshape_from_dcf import compute_lineshape_from_dcf_data
except Exception:
    compute_lineshape_from_dcf_data = None

import matplotlib.pyplot as plt

def _read_dcf_dat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns [t, Re, Im] in DCF file, got shape {arr.shape} for {path}")
    times = arr[:, 0]
    re = arr[:, 1]
    im = arr[:, 2]
    dcf = re + 1j * im
    return times, dcf


def _write_dcf_dat(path: str, times: np.ndarray, dcf: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = np.column_stack((times, np.real(dcf), np.imag(dcf)))
    np.savetxt(path, out)


def _build_uniform_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    n = int(np.floor((t_end - t_start) / dt)) + 1
    t = t_start + dt * np.arange(n)
    # Clamp last grid point <= t_end with numerical safety
    if t[-1] > t_end:
        t[-1] = t_end
    return t


def _interp_complex(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray, kind: str = 'cubic') -> np.ndarray:
    # Prefer SciPy cubic; fallback to linear or numpy.interp
    if _scipy_interp1d is not None:
        k = kind if times_src.size >= 4 else 'linear'
        f_re = _scipy_interp1d(times_src, np.real(values_src), kind=k, fill_value="extrapolate", assume_sorted=False)
        f_im = _scipy_interp1d(times_src, np.imag(values_src), kind=k, fill_value="extrapolate", assume_sorted=False)
        return f_re(times_dst) + 1j * f_im(times_dst)
    # numpy.interp is 1D linear only
    re = np.interp(times_dst, times_src, np.real(values_src))
    im = np.interp(times_dst, times_src, np.imag(values_src))
    return re + 1j * im


def _uniformize_series(times: np.ndarray, values: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate complex series onto a uniform time grid.

    If dt is not provided, uses the median of time diffs. Assumes times are
    1D increasing (will sort if needed).
    """
    if times.size == 0:
        return times, values
    order = np.argsort(times)
    t_sorted = times[order]
    v_sorted = values[order]
    if dt is None:
        diffs = np.diff(t_sorted)
        if diffs.size == 0:
            dt = 1.0
        else:
            dt = float(np.median(diffs))
            if dt <= 0:
                dt = float(np.max(diffs)) if np.max(diffs) > 0 else 1.0
    t_uniform = _build_uniform_grid(float(t_sorted[0]), float(t_sorted[-1]), dt)
    v_uniform = _interp_complex(t_sorted, v_sorted, t_uniform, kind='cubic')
    return t_uniform, v_uniform


def _compute_lineshape_fft(times: np.ndarray, dcf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute complex lineshape spectrum via FFT of the complex DCF.

    Returns (frequencies>=0, complex_spectrum). The spectrum is the positive-
    frequency part of the full FFT of the DCF sampled on a uniform grid.
    """
    if times.size == 0:
        return np.array([]), np.array([])
    # Uniformize first
    t_u, dcf_u = _uniformize_series(times, dcf)
    if t_u.size <= 1:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(t_u)))
    # Full FFT, then keep non-negative frequencies
    spec_c = np.fft.fft(dcf_u)
    freqs = np.fft.fftfreq(t_u.size, d=dt)
    mask = freqs >= 0
    return freqs[mask], spec_c[mask]


def _save_lineshape_dat_pdf(freqs: np.ndarray, spec: np.ndarray, dat_path: str, pdf_path: str, title: str) -> None:
    """Save lineshape (complex spectrum) to .dat and plot magnitude to PDF."""
    os.makedirs(os.path.dirname(dat_path), exist_ok=True)
    if freqs.size == 0:
        np.savetxt(dat_path, np.empty((0, 3)))
        return
    # Save as three columns: freq, Re(spec), Im(spec)
    data = np.column_stack((freqs, np.real(spec), np.imag(spec)))
    np.savetxt(dat_path, data)
    # Plot magnitude
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, np.abs(spec), lw=1.5)
    ax.set_xlabel('Frequency (arb.)')
    ax.set_ylabel('|Lineshape| (arb.)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


def postprocess_absorption(run_root: str, t_phase1_end: float, t_phase2_end: float, dt_small: float, dt_large: float) -> Optional[str]:
    phase1_dat = os.path.join(run_root, 'phase1', 'absorption_phase1', 'dcf_absorption_phase1.dat')
    phase2_dat = os.path.join(run_root, 'phase2', 'absorption_phase2', 'dcf_absorption_phase2.dat')

    if not (os.path.isfile(phase1_dat) and os.path.isfile(phase2_dat)):
        return None

    t1, dcf1 = _read_dcf_dat(phase1_dat)
    t2, dcf2 = _read_dcf_dat(phase2_dat)

    # Make absolute time axis for phase 2 and remove boundary overlap with phase 1
    t2_abs = t2 + t_phase1_end
    tol = 1e-12
    last_t1 = float(np.max(t1)) if t1.size > 0 else -np.inf
    mask2 = t2_abs > (last_t1 + tol)
    t2_abs_nz = t2_abs[mask2]
    dcf2_nz = dcf2[mask2]

    # Concatenate raw series (not uniform) and sort by time
    t_cat = np.concatenate([t1, t2_abs_nz])
    dcf_cat = np.concatenate([dcf1, dcf2_nz])
    if t_cat.size == 0:
        return None
    order = np.argsort(t_cat)
    t_cat = t_cat[order]
    dcf_cat = dcf_cat[order]

    # Build uniform grid at dt_small across [0, t_final]
    t_final = max(t_cat.max(), t_phase2_end)
    t_uniform = _build_uniform_grid(0.0, t_final, dt_small)

    # Interpolate onto uniform grid (cubic where possible)
    dcf_uniform = _interp_complex(t_cat, dcf_cat, t_uniform, kind='cubic')

    outdir = os.path.join(run_root, 'postprocess_dcf_output')
    os.makedirs(outdir, exist_ok=True)

    combined_path = os.path.join(outdir, 'combined_absorption_dcf.dat')
    _write_dcf_dat(combined_path, t_uniform, dcf_uniform)

    # Compute lineshape if function available
    if compute_lineshape_from_dcf_data is not None:
        ls_pdf = os.path.join(outdir, 'lineshape_combined_absorption.pdf')
        compute_lineshape_from_dcf_data(t_uniform, dcf_uniform, ls_pdf)

    return combined_path

def postprocess_emission(run_root: str) -> int:
    phase3_dir = os.path.join(run_root, 'phase3')
    inter_dirs: List[str] = []
    if os.path.isdir(phase3_dir):
        for d in sorted(os.listdir(phase3_dir)):
            full = os.path.join(phase3_dir, d)
            if os.path.isdir(full) and d.startswith('emission_from_'):
                inter_dirs.append(full)

    count = 0
    outdir_base = os.path.join(run_root, 'postprocess_dcf_output', 'emission')
    spectra_freqs: List[np.ndarray] = []
    spectra_vals: List[np.ndarray] = []
    for ed in inter_dirs:
        unid = os.path.basename(ed)
        dcf_dat = os.path.join(ed, unid, f'dcf_{unid}.dat')
        if not os.path.isfile(dcf_dat):
            continue
        t, dcf = _read_dcf_dat(dcf_dat)
        outdir = os.path.join(outdir_base, unid)
        os.makedirs(outdir, exist_ok=True)
        outdat = os.path.join(outdir, f'combined_{unid}.dat')
        _write_dcf_dat(outdat, t, dcf)
        # Always compute lineshape: write .dat and .pdf
        ls_dat = os.path.join(outdir, f'lineshape_{unid}.dat')
        ls_pdf = os.path.join(outdir, f'lineshape_{unid}.pdf')
        try:
            if compute_lineshape_from_dcf_data is not None:
                # Use external helper to create the PDF for consistency
                compute_lineshape_from_dcf_data(t, dcf, ls_pdf)
            # Always compute numeric complex spectrum for .dat and our PDF
            f_i, s_i = _compute_lineshape_fft(t, dcf)
            _save_lineshape_dat_pdf(f_i, s_i, ls_dat, ls_pdf, f'Emission lineshape: {unid}')
        except Exception:
            # Fallback to local method if external helper fails
            f_i, s_i = _compute_lineshape_fft(t, dcf)
            _save_lineshape_dat_pdf(f_i, s_i, ls_dat, ls_pdf, f'Emission lineshape: {unid}')
        if 'f_i' in locals() and f_i.size > 0:
            spectra_freqs.append(f_i)
            spectra_vals.append(s_i)
        count += 1
    # Average emission lineshapes across datasets and save
    try:
        if len(spectra_freqs) > 0:
            f_min = max(float(f[0]) for f in spectra_freqs)
            f_max = min(float(f[-1]) for f in spectra_freqs)
            if f_max > f_min:
                # Determine a reasonable df
                dfs = [float(np.min(np.diff(f))) for f in spectra_freqs if f.size > 1]
                df = min(dfs) if len(dfs) > 0 and np.isfinite(min(dfs)) and min(dfs) > 0 else (f_max - f_min) / 1024.0
                n_pts = int(np.floor((f_max - f_min) / df)) + 1
                f_common = f_min + df * np.arange(n_pts)
                specs_on_common = []
                for f_i, s_i in zip(spectra_freqs, spectra_vals):
                    s_interp = (
                        np.interp(f_common, f_i, np.real(s_i))
                        + 1j * np.interp(f_common, f_i, np.imag(s_i))
                    )
                    specs_on_common.append(s_interp)
                avg_spec = np.mean(np.vstack(specs_on_common), axis=0)
                avg_dat = os.path.join(outdir_base, 'lineshape_emission_average.dat')
                avg_pdf = os.path.join(outdir_base, 'lineshape_emission_average.pdf')
                _save_lineshape_dat_pdf(f_common, avg_spec, avg_dat, avg_pdf, 'Average emission lineshape')
    except Exception as e:
        print(f"[Postprocess] Averaging emission lineshapes failed: {e}")
    return count


def postprocess_for_run(run_root: str, t_phase1_end: float, t_phase2_end: float, dt_small: float, dt_large: float) -> None:
    try:
        combined_abs_path = postprocess_absorption(run_root, t_phase1_end, t_phase2_end, dt_small, dt_large)
        if combined_abs_path is None:
            print(f"[Postprocess] Absorption DCF files missing under {run_root}; skipping absorption postprocess")
        else:
            print(f"[Postprocess] Wrote combined absorption DCF: {combined_abs_path}")
    except Exception as e:
        print(f"[Postprocess] Absorption postprocess failed: {e}")

    try:
        n_em = postprocess_emission(run_root)
        print(f"[Postprocess] Processed {n_em} emission DCF(s)")
    except Exception as e:
        print(f"[Postprocess] Emission postprocess failed: {e}")
