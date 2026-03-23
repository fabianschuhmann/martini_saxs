#!/usr/bin/env python3

import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from datetime import datetime
import re

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
from MDAnalysis.lib.distances import distance_array

from saxs_tools.saxs_martini_beads import get_base_electrons, get_bead_radii
from saxs_tools.saxs_pdb_beadname_converter import ensure_pdb_has_bead_names
from saxs_tools.saxs_plot import add_plot_parser, run_plot_command

def make_default_out_base():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"martini_saxs_{timestamp}"

def gaussian_smooth_1d(q, Iq, sigma_q):
    q = np.asarray(q, dtype=float)
    Iq = np.asarray(Iq, dtype=float)

    if sigma_q <= 0:
        return Iq.copy()

    dq = np.mean(np.diff(q))
    half_width = int(np.ceil(4.0 * sigma_q / dq))

    # Keep kernel from exceeding signal length
    max_half = (len(Iq) - 1) // 2
    half_width = min(half_width, max_half)

    x = np.arange(-half_width, half_width + 1) * dq
    kernel = np.exp(-0.5 * (x / sigma_q) ** 2)
    kernel /= kernel.sum()

    return np.convolve(Iq, kernel, mode="same")


def read_cg_pdb(pdb_path):
    """
    Load a Martini CG PDB with MDAnalysis.

    Returns
    -------
    coords : (N, 3) ndarray
        Coordinates in nm.
    bead_types : list[str]
        Atom names, assumed to be Martini bead types.
    dimensions : ndarray
        MDAnalysis dimensions, with box lengths converted to nm.
    """
    u = mda.Universe(str(pdb_path))
    coords = u.atoms.positions.copy() / 10.0  # Å -> nm
    bead_types = list(u.atoms.names)
    dimensions = u.dimensions.copy()
    dimensions[:3] /= 10.0  # Å -> nm
    return coords, bead_types, dimensions


def get_form_factors(bead_types, envelope_dict):
    """
    Map bead types to solvent-subtracted zero-angle form factors.
    """
    missing = set(bead_types) - set(envelope_dict.keys())
    if missing:
        raise KeyError(f"Missing form factor for bead types: {sorted(missing)}")
    return np.array([envelope_dict[bt] for bt in bead_types], dtype=float)


def sphere_ff(q, f0, R):
    """
    Sphere form factor:
        f(q) = f0 * 3 * (sin(x) - x*cos(x)) / x^3
    with x = qR.
    """
    x = q * R
    if x == 0.0:
        sf = 1.0
    else:
        sf = 3.0 * (np.sin(x) - x * np.cos(x)) / (x ** 3)
    return f0 * sf


def compute_saxs_periodic_fourier(
    q_values,
    coords,
    form_factors,
    bead_types,
    dimensions,
    return_counts=False,
):
    """
    Periodic SAXS from direct particle Fourier amplitudes on the reciprocal lattice.
    """
    q_values = np.asarray(q_values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    form_factors = np.asarray(form_factors, dtype=float)

    dims = np.asarray(dimensions, dtype=float)
    Lx, Ly, Lz = dims[:3]

    bead_radii = get_bead_radii()

    # Wrap coordinates into the primary cell
    r = coords.copy()
    r[:, 0] %= Lx
    r[:, 1] %= Ly
    r[:, 2] %= Lz

    n_q = len(q_values)
    if n_q == 1:
        dq = q_values[0] * 0.1 if q_values[0] > 0 else 2.0 * np.pi / min(Lx, Ly, Lz)
        edges = np.array([q_values[0] - dq / 2, q_values[0] + dq / 2])
    else:
        mids = 0.5 * (q_values[:-1] + q_values[1:])
        dq0 = q_values[1] - q_values[0]
        dql = q_values[-1] - q_values[-2]
        edges = np.concatenate(
            (
                [q_values[0] - 0.5 * dq0],
                mids,
                [q_values[-1] + 0.5 * dql],
            )
        )

    q_max = edges[-1]

    nx_max = int(np.ceil(q_max * Lx / (2.0 * np.pi)))
    ny_max = int(np.ceil(q_max * Ly / (2.0 * np.pi)))
    nz_max = int(np.ceil(q_max * Lz / (2.0 * np.pi)))

    I_sum = np.zeros(n_q, dtype=np.float64)
    counts = np.zeros(n_q, dtype=np.int64)

    for nx in range(-nx_max, nx_max + 1):
        qx = 2.0 * np.pi * nx / Lx
        for ny in range(-ny_max, ny_max + 1):
            qy = 2.0 * np.pi * ny / Ly
            for nz in range(-nz_max, nz_max + 1):
                qz = 2.0 * np.pi * nz / Lz
                qmag = np.sqrt(qx * qx + qy * qy + qz * qz)

                k = np.digitize(qmag, edges) - 1
                if k < 0 or k >= n_q:
                    continue

                fq = np.array(
                    [
                        sphere_ff(qmag, form_factors[i], bead_radii[bt])
                        for i, bt in enumerate(bead_types)
                    ],
                    dtype=float,
                )

                phase = qx * r[:, 0] + qy * r[:, 1] + qz * r[:, 2]
                amp = np.sum(fq * np.exp(1j * phase))

                I_sum[k] += np.abs(amp) ** 2
                counts[k] += 1

    Iq = np.zeros(n_q, dtype=np.float64)
    mask = counts > 0
    Iq[mask] = I_sum[mask] / counts[mask]

    if return_counts:
        return Iq, counts
    return Iq


def _Iq_single(q, coords, bead_types, form_factors, bead_radii, distance_box):
    """
    Single-q Debye intensity.
    """
    N = coords.shape[0]

    fq = np.array(
        [sphere_ff(q, form_factors[i], bead_radii[bt]) for i, bt in enumerate(bead_types)],
        dtype=float,
    )

    Iq_k = np.sum(fq * fq)

    for i in range(N):
        dists = distance_array(coords[i:i+1], coords[i+1:], box=distance_box)[0]
        sinc = np.sinc((q * dists) / np.pi)
        Iq_k += 2.0 * fq[i] * np.sum(fq[i+1:] * sinc)

    return Iq_k


def compute_Iq_parallel(
    coords,
    bead_types,
    form_factors,
    q_values,
    use_pbc_distances=False,
    dimensions=None,
    max_workers=None,
):
    """
    Parallel Debye evaluation over q values.
    """
    bead_radii = get_bead_radii()
    distance_box = dimensions if use_pbc_distances else None

    func = partial(
        _Iq_single,
        coords=coords,
        bead_types=bead_types,
        form_factors=form_factors,
        bead_radii=bead_radii,
        distance_box=distance_box,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        Iq_list = list(tqdm(pool.map(func, q_values), total=len(q_values)))

    return np.array(Iq_list, dtype=float)


def prepare_pdb_for_saxs(pdb_path, itp_dir=".", itp_pattern="*.itp", verbose=True):
    """
    Ensure the input PDB has valid bead names in atom.name.

    Detection is done against saxs_martini_beads.
    If invalid names are found, conversion is done via ITP files.
    """
    prepared_pdb = ensure_pdb_has_bead_names(
        pdb_path=pdb_path,
        itp_dir=itp_dir,
        itp_pattern=itp_pattern,
        converted_suffix="_updated",
        verbose=verbose,
    )
    return prepared_pdb


def estimate_saxs_from_cg(
    pdb_path,
    q_min=0.0,
    q_max=3.5,
    n_q=25,
    debye=False,
    use_pbc_distances=False,
    smooth_sigma_factor=None,
    max_workers=None,
    itp_dir=".",
    itp_pattern="*.itp",
    verbose=True,
):
    """
    High-level SAXS driver.

    Flow
    ----
    1. Ensure bead names are present in the PDB (convert via ITP if needed)
    2. Read coordinates / bead types
    3. Compute Debye or periodic Fourier SAXS
    """
    prepared_pdb = prepare_pdb_for_saxs(
        pdb_path=pdb_path,
        itp_dir=itp_dir,
        itp_pattern=itp_pattern,
        verbose=verbose,
    )

    coords, bead_types, dimensions = read_cg_pdb(prepared_pdb)
    form_factors = get_form_factors(bead_types, get_base_electrons())

    q_values = np.linspace(q_min, q_max, n_q)
    trustbar = 2.0 * np.pi / np.min(dimensions[:3])

    if debye:
        Iq = compute_Iq_parallel(
            coords=coords,
            bead_types=bead_types,
            form_factors=form_factors,
            q_values=q_values,
            use_pbc_distances=use_pbc_distances,
            dimensions=dimensions,
            max_workers=max_workers,
        )
        counts = None
    else:
        Iq, counts = compute_saxs_periodic_fourier(
            q_values=q_values,
            coords=coords,
            form_factors=form_factors,
            bead_types=bead_types,
            dimensions=dimensions,
            return_counts=True,
        )

        if smooth_sigma_factor is not None and smooth_sigma_factor > 0:
            dq = np.mean(np.diff(q_values))
            sigma_q = smooth_sigma_factor * dq
            Iq = gaussian_smooth_1d(q_values, Iq, sigma_q=sigma_q)

    return q_values, Iq, trustbar, counts, prepared_pdb


def save_outputs(out_base, q, Iq, trustbar, counts=None):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{out_base}_q_values.npy", q)
    np.save(f"{out_base}_Iq.npy", Iq)

    if counts is not None:
        np.save(f"{out_base}_counts.npy", counts)

    np.savetxt(
        f"{out_base}_saxs_curve.dat",
        np.column_stack((q, Iq)),
        header=f"q I(q)\nTrust {trustbar}",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Martini SAXS utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compute
    compute_parser = subparsers.add_parser("compute", help="Compute SAXS from a PDB")
    compute_parser.add_argument("--pdb", required=True, help="Path to input PDB file.")
    compute_parser.add_argument("--out-base",default=None,help="Base output path. Default: martini_saxs_<timestamp>")
    compute_parser.add_argument("--debye", action="store_true", help="Use Debye calculation.")
    compute_parser.add_argument(
        "--use-pbc-distances",
        action="store_true",
        help="For Debye mode, use PBC minimum-image distances."
    )
    compute_parser.add_argument("--q-min", type=float, default=0.0, help="Default: 0.0 1/nm")
    compute_parser.add_argument("--q-max", type=float, default=3.5, help="Default: 3.5 1/nm")
    compute_parser.add_argument("--n-q", type=int, default=200, help="Default: 200")
    compute_parser.add_argument("--smooth-sigma-factor", type=float, default=None)
    compute_parser.add_argument("--max-workers", type=int, default=None)
    compute_parser.add_argument("--itp-dir", default=".", help="Default: '.'")
    compute_parser.add_argument("--itp-pattern", default="*.itp", help="Default: '*.itp'")
    compute_parser.add_argument("--quiet", action="store_true")

    add_plot_parser(subparsers)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compute":
        out_base = args.out_base or make_default_out_base()
        q, Iq, trustbar, counts, prepared_pdb = estimate_saxs_from_cg(
            pdb_path=args.pdb,
            q_min=args.q_min,
            q_max=args.q_max,
            n_q=args.n_q,
            debye=args.debye,
            use_pbc_distances=args.use_pbc_distances,
            smooth_sigma_factor=args.smooth_sigma_factor,
            max_workers=args.max_workers,
            itp_dir=args.itp_dir,
            itp_pattern=args.itp_pattern,
            verbose=not args.quiet,
        )

        save_outputs(
            out_base=out_base,
            q=q,
            Iq=Iq,
            trustbar=trustbar,
            counts=counts,
        )

        if not args.quiet:
            print(f"Prepared PDB: {prepared_pdb}")
            print(f"Saved outputs with base: {args.out_base}")
            print(f"Trustbar: {trustbar:.6f}")
            if counts is not None:
                print("Counts per q-bin:")
                print(counts)

    elif args.command == "plot":
        run_plot_command(args)


if __name__ == "__main__":
    main()