import pandas as pd
from pathlib import Path

FEATURE_SETS_QSVM = {
    "md3": [
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_c_surface_distance_std_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
    "dynamic4": [
        "feature_tddft_transition_energy_ev",
        "feature_tddft_transition_oscillator_strength",
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_c_surface_distance_std_a",
    ],
    "dynamic6": [
        "feature_tddft_transition_energy_ev",
        "feature_tddft_transition_oscillator_strength",
        "feature_tddft_transition_delta_from_peak_ev",
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_c_surface_distance_std_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
    "all12": [
        "feature_tddft_transition_energy_ev",
        "feature_tddft_transition_oscillator_strength",
        "feature_tddft_transition_delta_from_peak_ev",
        "feature_dft_adsorption_energy_ev",
        "feature_dft_c_surface_distance_a",
        "feature_dft_gap_change_ev",
        "feature_co2rr_deltaG_CO2_to_COOH_ev",
        "feature_co2rr_deltaG_COOH_to_CO_ev",
        "feature_co2rr_limiting_potential_v",
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_c_surface_distance_std_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
}

def build_table(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, features in feature_sets.items():
        n_features = len(features)
        rows.append({
            "feature_set": name,
            "n_features": n_features,
            "n_qubits": n_features,  # ZZFeatureMap maps 1 feature to 1 qubit
        })

    return pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)

def main():
    df = build_table(FEATURE_SETS_QSVM)

    output_csv = Path("results/feature_qubit_table.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("Table: Number of features and qubits per feature set (QSVM)\n")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_csv}")

if __name__ == "__main__":
    main()