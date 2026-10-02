from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

base = Path("results/my_analysis/md/ZnO")

files = {
    "Top metal": base / "top_metal" / "md_summary_topMetal.csv",
    "Top oxygen": base / "top_oxygen" / "md_summary_topOxygen.csv",
    "Bridge": base / "bridge" / "md_summary_bridge.csv",
}

out_dir = Path("results/my_analysis/md")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))

for label, path in files.items():
    if path.exists():
        df = pd.read_csv(path)
        plt.plot(
            df["time_ps"],
            df["c_surface_distance_ang"],
            linewidth=1.8,
            label=label,
        )
    else:
        print(f"Warning: file not found: {path}")

plt.xlabel("Time (ps)")
plt.ylabel(r"C--surface distance ($\AA$)")
plt.title(r"AIMD evolution of the CO$_2$--surface distance")
plt.legend()
plt.tight_layout()

output = out_dir / "md_c_surface_distance_comparison.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()
