from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

BASE = Path("results/neb/ZnO")

CASES = [
    ("climb_0.2", "top_oxygen_to_top_metal_02"),
    ("climb_0.4", "top_oxygen_to_top_metal_04"),
    ("climb_0.5", "top_oxygen_to_top_metal_05"),
]

XYZ_ENERGY_RE = re.compile(r"energy=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
LOG_ENERGY_RE = re.compile(r"Extrapolated:\s*([-+]?\d+\.\d+)")


def read_xyz_energy(path: Path) -> float:
    try:
        with open(path) as f:
            f.readline()
            comment = f.readline()
        m = XYZ_ENERGY_RE.search(comment)
        return float(m.group(1)) if m else float("nan")
    except Exception:
        return float("nan")


def read_log_energy(path: Path) -> float:
    if not path.exists():
        return float("nan")
    last = None
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                m = LOG_ENERGY_RE.search(line)
                if m:
                    last = float(m.group(1))
    except Exception:
        return float("nan")
    return last if last is not None else float("nan")


def get_n_internal_images(folder: Path) -> int:
    summary_path = folder / "neb_run_summary.json"
    if summary_path.exists():
        try:
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
            return int(meta["internal_images"])
        except Exception:
            pass
    return len(list(folder.glob("neb_image_*.txt")))


def get_energies(folder: Path) -> list[float]:
    n_internal = get_n_internal_images(folder)
    if n_internal == 0:
        return []

    energies = [read_xyz_energy(folder / "image_00.xyz")]
    for i in range(1, n_internal + 1):
        energies.append(read_log_energy(folder / f"neb_image_{i}.txt"))
    final_idx = n_internal + 1
    energies.append(read_xyz_energy(folder / f"image_{final_idx:02d}.xyz"))
    return energies


def get_run_metadata(folder: Path) -> dict:
    summary_path = folder / "neb_run_summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


rows = []

print(f"{'case':<12}{'image':>7}{'abs_energy_eV':>16}{'rel_energy_eV':>16}")
print("-" * 51)

for label, folder_name in CASES:
    folder = BASE / folder_name
    energies = get_energies(folder)

    if not energies or not math.isfinite(energies[0]):
        print(f"{label:<12}  SEM DADOS VALIDOS em {folder}")
        continue

    e0 = energies[0]
    rel = [e - e0 for e in energies]
    finite_rel = [r for r in rel if math.isfinite(r)]
    barrier = max(finite_rel) if finite_rel else float("nan")
    barrier_idx = rel.index(barrier) if math.isfinite(barrier) else -1
    rxn_e = rel[-1]
    meta = get_run_metadata(folder)

    for i, (e_abs, e_rel) in enumerate(zip(energies, rel)):
        print(f"{label:<12}{i:>7}{e_abs:>16.6f}{e_rel:>16.6f}")
        rows.append({
            "case": label,
            "image_index": i,
            "abs_energy_eV": e_abs,
            "rel_energy_eV": e_rel,
        })

    print(f"  -> barrier = {barrier:.4f} eV (image {barrier_idx})  |  reaction dE = {rxn_e:.4f} eV")
    print(f"  -> status = {meta.get('status', '?')}  |  climb_start_fraction = {meta.get('climb_start_fraction', '?')}  |  fmax = {meta.get('fmax', '?')}")
    print()

out_csv = BASE / "comparison_plots" / "neb_energy_table.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Tabela completa salva em: {out_csv}")