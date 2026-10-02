from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scripts.common.timer import timer, print0

from scripts.common.project_config import DFT_RESULTS_DIR, RESULTS_DIR, ensure_directories


MD_RESULTS_DIR = RESULTS_DIR / "md"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Molecular Dynamics for ZnO + CO2 adsorption system.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--site", default="bridge")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--timestep", type=float, default=1.0, help="Time step in fs.")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ecut", type=float, default=350.0)
    parser.add_argument("--kpts", nargs=3, type=int, default=(2, 2, 1))
    return parser.parse_args()


def build_gpaw_calculator(out_dir: Path, ecut: float, kpts: tuple[int, int, int]):
    from gpaw import GPAW, PW, FermiDirac, Mixer

    return GPAW(
        mode=PW(ecut),
        xc="PBE",
        occupations=FermiDirac(0.10),
        kpts=kpts,
        symmetry="off",
        mixer=Mixer(0.05, 5, 50),
        txt=str(out_dir / "md_gpaw.txt"),
    )


def apply_bottom_layer_constraint(atoms, adsorbate_atoms: int = 3, slab_free_fraction: float = 0.60):
    from ase.constraints import FixAtoms

    slab_count = len(atoms) - adsorbate_atoms
    slab_indices = np.arange(slab_count)

    slab_z = atoms.positions[slab_indices, 2]
    z_min = float(np.min(slab_z))
    z_max = float(np.max(slab_z))

    threshold = z_min + (1.0 - slab_free_fraction) * (z_max - z_min)

    fixed_indices = [
        int(i)
        for i in slab_indices
        if atoms.positions[i, 2] <= threshold
    ]

    atoms.set_constraint(FixAtoms(indices=fixed_indices))


def get_geometry_metrics(atoms):
    """
    Assumes the last 3 atoms are O-C-O, same convention used in the DFT code.
    """
    o1_idx = len(atoms) - 3
    c_idx = len(atoms) - 2
    o2_idx = len(atoms) - 1

    slab_indices = np.arange(len(atoms) - 3)

    co1 = float(atoms.get_distance(c_idx, o1_idx))
    co2 = float(atoms.get_distance(c_idx, o2_idx))
    oco_angle = float(atoms.get_angle(o1_idx, c_idx, o2_idx))

    slab_z_max = float(np.max(atoms.positions[slab_indices, 2]))
    carbon_z = float(atoms.positions[c_idx, 2])
    c_surface_distance = carbon_z - slab_z_max

    return co1, co2, oco_angle, c_surface_distance


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def run_md(args):
    from ase.io import read, write
    from ase import units
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        force_temperature,
        )

    ensure_directories()

    material = args.material
    site = args.site

    input_xyz = DFT_RESULTS_DIR / material / site / f"{material}_{site}_optimized.xyz"

    if not input_xyz.exists():
        raise SystemExit(
            f"Missing optimized DFT structure: {input_xyz}\n"
            f"Run DFT first for {material} | {site}."
        )

    out_dir = MD_RESULTS_DIR / material / site
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(input_xyz)

    # Apply slab constraints first
    apply_bottom_layer_constraint(atoms, adsorbate_atoms=3)

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)

    # Remove momenta from fixed atoms
    momenta = atoms.get_momenta()

    for constraint in atoms.constraints:
        if hasattr(constraint, "get_indices"):
            fixed_indices = constraint.get_indices()
            momenta[fixed_indices] = 0.0

    atoms.set_momenta(momenta)

    # Rescale temperature after removing constrained atom velocities
    force_temperature(
        atoms,
        temperature=args.temperature,
        unit="K",
    )

    # Attach GPAW calculator
    calc = build_gpaw_calculator(
        out_dir=out_dir,
        ecut=args.ecut,
        kpts=tuple(args.kpts),
    )
    atoms.calc = calc

    timestep_fs = args.timestep

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.02,
        fixcm=False,
    )

    traj = Trajectory(str(out_dir / "md_trajectory.traj"), "w", atoms)

    rows = []

    def record(step: int):
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temp = float(atoms.get_temperature())

        co1, co2, oco_angle, c_surface_distance = get_geometry_metrics(atoms)

        rows.append(
            {
                "step": step,
                "time_ps": step * timestep_fs / 1000.0,
                "potential_energy_ev": epot,
                "kinetic_energy_ev": ekin,
                "total_energy_ev": epot + ekin,
                "temperature_K": temp,
                "co_bond_1_ang": co1,
                "co_bond_2_ang": co2,
                "co_bond_avg_ang": (co1 + co2) / 2.0,
                "oco_angle_deg": oco_angle,
                "c_surface_distance_ang": c_surface_distance,
            }
        )

        print0(
            f"step={step:5d} "
            f"time={step * timestep_fs / 1000.0:7.4f} ps "
            f"Epot={epot:12.6f} eV "
            f"Ekin={ekin:10.6f} eV "
            f"T={temp:8.2f} K "
            f"C-surf={c_surface_distance:8.3f} Å "
            f"OCO={oco_angle:8.2f} deg"
        )

    record(0)
    traj.write(atoms)

    for step in range(args.log_interval, args.steps + 1, args.log_interval):
        dyn.run(args.log_interval)
        record(step)
        traj.write(atoms)

    traj.close()

    write(out_dir / "md_final.xyz", atoms)
    write(out_dir / "md_final.cif", atoms)

    save_csv(out_dir / "md_summary.csv", rows)

    print0(f"\n[md] Done.")
    print0(f"[md] Results saved in: {out_dir}")
    print0(f"[md] Main table: {out_dir / 'md_summary.csv'}")


def main():
    args = parse_args()

    with timer(f"Molecular Dynamics | Material={args.material} | Site={args.site}"):
        run_md(args)

if __name__ == "__main__":
    main()
