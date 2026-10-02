from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.common.timer import timer, print0
from scripts.common.project_config import DFT_RESULTS_DIR, RESULTS_DIR, ensure_directories


NEB_RESULTS_DIR = RESULTS_DIR / "neb"

DEFAULT_SITES = ["top_metal", "top_oxygen", "bridge"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run NEB calculations between ZnO + CO2 adsorption sites.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--sites", nargs="*", default=DEFAULT_SITES,)
    parser.add_argument("--initial-site", default=None,)
    parser.add_argument("--final-site", default=None,)
    parser.add_argument("--images", type=int, default=3,)
    parser.add_argument("--fmax", type=float, default=0.08,)
    parser.add_argument("--steps", type=int, default=120,)
    parser.add_argument("--ecut", type=float, default=350.0)
    parser.add_argument("--kpts", nargs=3, type=int, default=(2, 2, 1), metavar=("KX", "KY", "KZ"),)
    parser.add_argument("--interpolation", choices=["linear", "idpp"], default="idpp",)
    parser.add_argument("--climb", action="store_true",)
    parser.add_argument("--optimizer", choices=["FIRE", "BFGS"], default="FIRE",)
    parser.add_argument("--climb-start-fraction", type=float, default=0.3,)
    parser.add_argument("--parallel-images", action="store_true",)
    parser.add_argument("--overwrite", action="store_true",)

    return parser.parse_args()

def build_gpaw_calculator(
    out_dir: Path,
    label: str,
    ecut: float,
    kpts: tuple[int, int, int],
    communicator=None,
):
    from gpaw import GPAW, PW, FermiDirac, Mixer

    kwargs = {}
    if communicator is not None:
        kwargs["communicator"] = communicator

    return GPAW(
        mode=PW(ecut),
        xc="PBE",
        occupations=FermiDirac(0.10),
        kpts=kpts,
        symmetry="off",
        mixer=Mixer(0.05, 5, 50),
        txt=str(out_dir / f"{label}.txt"),
        **kwargs,
    )


def apply_bottom_layer_constraint(
    atoms,
    adsorbate_atoms: int = 3,
    slab_free_fraction: float = 0.60,
):
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


def get_pairs(args) -> list[tuple[str, str]]:
    if args.initial_site or args.final_site:
        if not args.initial_site or not args.final_site:
            raise SystemExit("Use both --initial-site and --final-site together.")

        if args.initial_site == args.final_site:
            raise SystemExit("Initial and final sites must be different.")

        return [(args.initial_site, args.final_site)]

    sites = list(args.sites)

    if len(sites) < 2:
        raise SystemExit("At least two sites are required for NEB.")

    return list(itertools.combinations(sites, 2))


def read_dft_structure(material: str, site: str):
    from ase.io import read

    path = DFT_RESULTS_DIR / material / site / f"{material}_{site}_optimized.xyz"

    if not path.exists():
        raise SystemExit(
            f"Missing optimized DFT structure for {material} | {site}: {path}"
        )

    return read(path, parallel=False)


def check_compatible_endpoints(initial, final, initial_site: str, final_site: str):
    if len(initial) != len(final):
        raise SystemExit(
            f"Initial and final structures have different atom counts: "
            f"{initial_site}={len(initial)}, {final_site}={len(final)}"
        )

    symbols_initial = initial.get_chemical_symbols()
    symbols_final = final.get_chemical_symbols()

    if symbols_initial != symbols_final:
        raise SystemExit(
            f"Initial and final structures have different atom ordering. "
            f"NEB requires the same atoms in the same order."
        )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def append_csv(path: Path, row: dict):
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def run_single_neb(
    *,
    material: str,
    initial_site: str,
    final_site: str,
    args,
):
    from ase.io import write
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE
    from gpaw.mpi import world

    kpts = tuple(args.kpts)

    pair_name = f"{initial_site}_to_{final_site}"
    out_dir = NEB_RESULTS_DIR / material / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "neb_run_summary.json"

    if summary_path.exists() and not args.overwrite:
        print0(f"[neb] Skipping existing path: {material} | {pair_name}")
        return

    print0(
        f"\n[neb] Start | material={material} | "
        f"path={initial_site} -> {final_site}"
    )

    initial = read_dft_structure(material, initial_site)
    final = read_dft_structure(material, final_site)

    check_compatible_endpoints(initial, final, initial_site, final_site)

    apply_bottom_layer_constraint(initial, adsorbate_atoms=3)
    apply_bottom_layer_constraint(final, adsorbate_atoms=3)

    if world.rank == 0:
        write(out_dir / "initial.xyz", initial, parallel=False)
        write(out_dir / "final.xyz", final, parallel=False)

    images = [initial]

    for _ in range(args.images):
        image = initial.copy()
        apply_bottom_layer_constraint(image, adsorbate_atoms=3)
        images.append(image)

    images.append(final)

    use_parallel = args.parallel_images and world.size > 1

    if use_parallel:
        n_internal = args.images

        if world.size % n_internal != 0:
            raise SystemExit(
                f"Parallel NEB over images requires MPI ranks divisible by "
                f"number of internal images.\n"
                f"world.size={world.size}, internal_images={n_internal}\n"
                f"Example: for --images 3, use -np 3, 6, 9, 12, ..."
            )

        ranks_per_image = world.size // n_internal

        for i, image in enumerate(images[1:-1]):
            ranks = range(i * ranks_per_image, (i + 1) * ranks_per_image)

            if world.rank in ranks:
                communicator = world.new_communicator(ranks)
                image.calc = build_gpaw_calculator(
                    out_dir=out_dir,
                    label=f"neb_image_{i + 1}",
                    ecut=args.ecut,
                    kpts=kpts,
                    communicator=communicator,
                )

        neb = NEB(
            images,
            parallel=True,
            climb=False,
            method="improvedtangent",
            allow_shared_calculator=False,
        )

    else:
        for i, image in enumerate(images[1:-1]):
            image.calc = build_gpaw_calculator(
                out_dir=out_dir,
                label=f"neb_image_{i + 1}",
                ecut=args.ecut,
                kpts=kpts,
            )

        neb = NEB(
            images,
            parallel=False,
            climb=False,
            method="improvedtangent",
            allow_shared_calculator=False,
        )

    if args.interpolation == "idpp":
        neb.interpolate(method="idpp", apply_constraint=False)
    else:
        neb.interpolate(method="linear", apply_constraint=False)

    optimizer_cls = FIRE if args.optimizer == "FIRE" else BFGS

    optimizer = optimizer_cls(
        neb,
        logfile=str(out_dir / "neb_opt.log"),
        trajectory=str(out_dir / "neb.traj"),
    )

    with timer(
        f"NEB | {material} | {initial_site} -> {final_site} | "
        f"images={args.images}"
    ):
        if args.climb:
            climb_start_step = int(args.climb_start_fraction * args.steps)

            print0(f"[neb] Phase 1: without climbing image, up to step {climb_start_step}")
            converged = optimizer.run(fmax=args.fmax, steps=climb_start_step)

            if not converged:
                print0("[neb] Phase 2: activating climbing image (climb=True)")
                neb.climb = True
                converged = optimizer.run(fmax=args.fmax, steps=args.steps)
            else:
                print0("[neb] Converged before the climb phase — climb was not activated.")
        else:
            converged = optimizer.run(fmax=args.fmax, steps=args.steps)

    if world.rank == 0:
        for i, image in enumerate(images):
            write(out_dir / f"image_{i:02d}.xyz", image, parallel=False)

        write(out_dir / "neb_final_images.traj", images, parallel=False)

        metadata = {
            "material": material,
            "initial_site": initial_site,
            "final_site": final_site,
            "path": pair_name,
            "internal_images": args.images,
            "total_images": args.images + 2,
            "fmax": args.fmax,
            "max_steps": args.steps,
            "ecut": args.ecut,
            "kpts": list(kpts),
            "interpolation": args.interpolation,
            "climb": bool(args.climb),
            "climb_start_fraction": args.climb_start_fraction,
            "optimizer": args.optimizer,
            "parallel_images": bool(use_parallel),
            "mpi_size": int(world.size),
            "status": "converged" if converged else "max_steps_reached",
        }

        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        append_csv(
            NEB_RESULTS_DIR / "neb_paths_summary.csv",
            {
                "material": material,
                "initial_site": initial_site,
                "final_site": final_site,
                "path": pair_name,
                "internal_images": args.images,
                "total_images": args.images + 2,
                "fmax": args.fmax,
                "max_steps": args.steps,
                "ecut": args.ecut,
                "kpts": " ".join(str(v) for v in kpts),
                "interpolation": args.interpolation,
                "climb": bool(args.climb),
                "optimizer": args.optimizer,
                "parallel_images": bool(use_parallel),
                "mpi_size": int(world.size),
                "status": "converged" if converged else "max_steps_reached",
            },
        )

        print0(
            f"[neb] Done | status={'converged' if converged else 'max_steps_reached'} "
            f"| results saved in: {out_dir}"
        )


def run_neb(args):
    from gpaw.mpi import world

    ensure_directories()
    NEB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_csv = NEB_RESULTS_DIR / "neb_paths_summary.csv"

    if args.overwrite and world.rank == 0 and summary_csv.exists():
        summary_csv.unlink()

    pairs = get_pairs(args)

    if world.rank == 0:
        print0("\n=== Nudged Elastic Band calculations ===")
        print0(f"[neb] material: {args.material}")
        print0(f"[neb] pairs: {pairs}")
        print0(f"[neb] internal images: {args.images}")
        print0(f"[neb] interpolation: {args.interpolation}")
        print0(f"[neb] climb: {args.climb}")
        print0(f"[neb] optimizer: {args.optimizer}")
        print0(f"[neb] parallel images: {args.parallel_images}")

    for initial_site, final_site in pairs:
        run_single_neb(
            material=args.material,
            initial_site=initial_site,
            final_site=final_site,
            args=args,
        )
        #world.barrier()

    if world.rank == 0:
        print0("\n[neb] All NEB paths finished.")


def main():
    args = parse_args()
    run_neb(args)


if __name__ == "__main__":
    main()
