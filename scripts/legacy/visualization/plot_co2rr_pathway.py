from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

csv_path = Path("results/my_analysis/co2rr_pathway_summary.csv")
out_dir = Path("results/my_analysis/co2rr")
out_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    raise FileNotFoundError(f"Data file not found: {csv_path}")

df = pd.read_csv(csv_path)

site_order = ["top_metal", "top_oxygen", "bridge"]
site_labels = {
    "top_metal": "Top metal",
    "top_oxygen": "Top oxygen",
    "bridge": "Bridge",
}

states = [
    "CO2*",
    "COOH*",
    "CO*",
    "after CO desorption",
    "products",
]

profiles = []

for site in site_order:
    row = df[df["site"] == site]
    if row.empty:
        continue

    row = row.iloc[0]

    d1 = float(row["deltaG_CO2_to_COOH_ev"])
    d2 = float(row["deltaG_COOH_to_CO_ev"])
    d3 = float(row["deltaG_CO_desorption_ev"])
    d4 = float(row["deltaG_O_removal_ev"])

    profile = [
        0.0,
        d1,
        d1 + d2,
        d1 + d2 + d3,
        d1 + d2 + d3 + d4,
    ]

    # The source CSV retains legacy headers, but these are electronic-energy estimates in eV.
    profiles.append({
        "site": site,
        "profile": profile,
        "limiting_step": row["limiting_step"],
        "reaction_descriptor_ev": float(row["limiting_potential_v"]),
    })

plot_rows = []
for item in profiles:
    for state, energy in zip(states, item["profile"]):
        plot_rows.append({
            "site": item["site"],
            "state": state,
            "relative_free_energy_ev": energy,
        })

plot_df = pd.DataFrame(plot_rows)
table_path = out_dir / "co2rr_free_energy_profile_table.csv"
plot_df.to_csv(table_path, index=False)

plt.figure(figsize=(8, 5))

for item in profiles:
    plt.plot(
        states,
        item["profile"],
        marker="o",
        linewidth=2,
        label=(
            f"{site_labels[item['site']]} "
            f"($D_{{\\mathrm{{reaction}}}}$ = {item['reaction_descriptor_ev']:.3f} eV)"
        ),
    )

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Reaction coordinate")
plt.ylabel("Relative electronic energy (eV)")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()

fig_path = out_dir / "co2rr_free_energy_pathway.png"
plt.savefig(fig_path, dpi=300)
plt.close()

for item in profiles:
    print(f"Site: {item['site']}")
    print(f"  Limiting step: {item['limiting_step']}")
    print(f"  Reaction-energy descriptor: {item['reaction_descriptor_ev']:.6f} eV")
    print(f"  Profile: {item['profile']}\n")
