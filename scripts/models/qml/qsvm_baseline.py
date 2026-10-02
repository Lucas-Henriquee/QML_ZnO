from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC


FEATURE_SETS = {
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
    "static6": [
        "feature_dft_adsorption_energy_ev",
        "feature_dft_c_surface_distance_a",
        "feature_dft_gap_change_ev",
        "feature_co2rr_deltaG_CO2_to_COOH_ev",
        "feature_co2rr_deltaG_COOH_to_CO_ev",
        "feature_co2rr_limiting_potential_v",
    ],
    "dynamic6": [
        "feature_tddft_transition_energy_ev",
        "feature_tddft_transition_oscillator_strength",
        "feature_tddft_transition_delta_from_peak_ev",
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
    "tddft3": [
        "feature_tddft_transition_energy_ev",
        "feature_tddft_transition_oscillator_strength",
        "feature_tddft_transition_delta_from_peak_ev",
    ],
    "md3": [
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_c_surface_distance_std_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
}

SCALERS = ["none", "standard", "minmax", "minmax_pi", "power", "power_minmax"]
ENTANGLEMENTS = ["linear", "circular", "full"]
SIMULATORS = ["reference", "aer_mps"]


def parse_args():
    parser = argparse.ArgumentParser(description="General ideal/simulated quantum-kernel SVM benchmark.")
    parser.add_argument("--csv", type=Path, default=Path("results/datasets/multitech_expanded_ml_dataset_clean.csv"))
    parser.add_argument("--label", default="label_site_id")
    parser.add_argument("--feature-set", default="md3", choices=sorted(FEATURE_SETS))
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--scaler", default="power_minmax", choices=SCALERS)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--entanglement", default="linear", choices=ENTANGLEMENTS)
    parser.add_argument("--n-qubits", type=int, default=0, help="0 = one qubit per feature; >features enables repeated multi-qubit encoding.")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses every available row.")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional CSV with row_id and split=train/test.")
    parser.add_argument("--cv", type=int, default=0, help="Optional stratified CV")
    parser.add_argument(
        "--simulator",
        default="reference",
        choices=SIMULATORS,
        help=(
            "reference = Qiskit FidelityStatevectorKernel; "
            "aer_mps = Qiskit Aer matrix-product-state simulation."
        ),
    )
    parser.add_argument(
        "--mps-max-bond-dimension",
        type=int,
        default=0,
        help=(
            "MPS only. 0 leaves the bond dimension unlimited. "
        ),
    )
    parser.add_argument(
        "--mps-truncation-threshold",
        type=float,
        default=0.0,
        help=(
            "MPS only. 0 disables coefficient truncation as far as Aer permits. "
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qsvm"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_weight_value(name: str):
    return None if name == "none" else name


def diagnose_fit(train_score: float, test_score: float, cv_mean: float | None = None,
                 gap_threshold: float = 0.10, low_threshold: float = 0.60) -> dict:
    gap = train_score - test_score
    if train_score < low_threshold and test_score < low_threshold:
        status = "possible_underfitting"
    elif gap > gap_threshold:
        status = "possible_overfitting"
    else:
        status = "no_obvious_train_test_gap"
    return {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "cv_score": None if cv_mean is None else float(cv_mean),
        "train_test_gap": float(gap),
        "diagnosis": status,
    }


def get_features(args) -> list[str]:
    return list(args.features) if args.features is not None else FEATURE_SETS[args.feature_set].copy()


def balanced_subsample(df: pd.DataFrame, label: str, max_samples: int, random_state: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    classes = list(df[label].drop_duplicates())
    n_classes = len(classes)
    if max_samples < 2 * n_classes:
        raise ValueError(f"--max-samples must be at least {2 * n_classes} for {n_classes} classes.")
    base, remainder = divmod(max_samples, n_classes)
    rng = np.random.default_rng(random_state)
    class_order = list(classes)
    rng.shuffle(class_order)
    chunks, selected = [], set()
    for i, cls in enumerate(class_order):
        target = base + (1 if i < remainder else 0)
        group = df[df[label] == cls]
        part = group.sample(n=min(target, len(group)), random_state=random_state + i)
        chunks.append(part)
        selected.update(part.index.tolist())
    sampled = pd.concat(chunks, axis=0)
    remaining = max_samples - len(sampled)
    if remaining > 0:
        rest = df.drop(index=list(selected), errors="ignore")
        if len(rest):
            sampled = pd.concat([sampled, rest.sample(n=min(remaining, len(rest)), random_state=random_state + 10_000)])
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_base_data(args, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    df["_row_id"] = np.arange(len(df), dtype=int)
    required = features + [args.label]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.dropna(subset=required).copy()


def split_data(args, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.split_file is not None:
        split = pd.read_csv(args.split_file)
        if not {"row_id", "split"}.issubset(split.columns):
            raise ValueError("split-file must contain columns: row_id, split")
        if split["row_id"].duplicated().any():
            raise ValueError("split-file contains duplicated row_id values")
        invalid_splits = sorted(set(split["split"]) - {"train", "test"})
        if invalid_splits:
            raise ValueError(f"split-file contains invalid split values: {invalid_splits}")
        train_ids = split.loc[split["split"] == "train", "row_id"].astype(int).tolist()
        test_ids = split.loc[split["split"] == "test", "row_id"].astype(int).tolist()
        if not train_ids or not test_ids:
            raise ValueError("split-file must contain at least one train and one test row")
        indexed = df.set_index("_row_id", drop=False)
        missing_ids = sorted((set(train_ids) | set(test_ids)) - set(indexed.index))
        if missing_ids:
            raise ValueError(f"Split contains row_ids unavailable after dropna: {missing_ids[:10]}")
        if "true_label" in split.columns:
            expected = split.set_index("row_id")["true_label"]
            actual = indexed.loc[expected.index, args.label]
            mismatched = expected.index[expected.astype(str) != actual.astype(str)]
            if len(mismatched):
                raise ValueError(
                    "split-file true_label does not match the dataset for row_ids: "
                    f"{mismatched[:10].tolist()}"
                )
        return indexed.loc[train_ids].copy(), indexed.loc[test_ids].copy()

    sampled = balanced_subsample(df, args.label, args.max_samples, args.random_state)
    if sampled[args.label].nunique() < 2 or sampled[args.label].value_counts().min() < 2:
        raise ValueError("Need at least two classes with at least two samples each.")
    n_classes = sampled[args.label].nunique()
    n_test = math.ceil(len(sampled) * args.test_size)
    if n_test < n_classes or len(sampled) - n_test < n_classes:
        raise ValueError("Split too small for stratification. Increase --max-samples or --test-size.")
    train_df, test_df = train_test_split(sampled, test_size=args.test_size, random_state=args.random_state, stratify=sampled[args.label])
    return train_df.copy(), test_df.copy()


def fit_transform_scaler(X_train, X_test, scaler_name: str):
    X_train_np = np.asarray(X_train, dtype=float)
    X_test_np = np.asarray(X_test, dtype=float)
    fitted = []
    if scaler_name == "none":
        train_start = time.perf_counter()
        Xtr = X_train_np.copy()
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = X_test_np.copy()
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, fitted, train_wall, test_wall
    if scaler_name == "standard":
        scaler = StandardScaler()
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "minmax_pi":
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "power":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "power_minmax":
        power = PowerTransformer(method="yeo-johnson", standardize=True)
        train_start = time.perf_counter()
        train_p = power.fit_transform(X_train_np)
        mm = MinMaxScaler(feature_range=(0.0, np.pi))
        Xtr = mm.fit_transform(train_p)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        test_p = power.transform(X_test_np)
        Xte = mm.transform(test_p)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [power, mm], train_wall, test_wall
    raise ValueError(f"Unknown scaler: {scaler_name}")


def entanglement_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if entanglement == "linear":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    if entanglement == "circular":
        pairs = [(q, q + 1) for q in range(n_qubits - 1)]
        if n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        return pairs
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    raise ValueError(f"Unsupported entanglement: {entanglement}")


def build_multiqubit_feature_map(n_features: int, n_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    if n_qubits < n_features:
        raise ValueError("n_qubits must be >= n_features")
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_qubits, name=f"multi_zz_{n_features}f_{n_qubits}q")
    pairs = entanglement_pairs(n_qubits, entanglement)
    for rep in range(reps):
        for q in range(n_qubits):
            k = q % n_features
            qc.h(q)
            qc.p(2.0 * x[k], q)
        for q1, q2 in pairs:
            f1, f2 = q1 % n_features, q2 % n_features
            angle = 2.0 * (np.pi - x[f1]) * (np.pi - x[f2])
            qc.cx(q1, q2)
            qc.p(angle, q2)
            qc.cx(q1, q2)
        if rep != reps - 1:
            qc.barrier()
    if qc.num_parameters != n_features:
        raise RuntimeError("Unexpected number of feature-map parameters.")
    return qc


def build_feature_map(n_features: int, requested_qubits: int, reps: int, entanglement: str):
    n_qubits = n_features if requested_qubits == 0 else requested_qubits
    if n_qubits < n_features:
        raise ValueError(f"n_qubits={n_qubits} cannot be smaller than n_features={n_features}")
    if n_qubits == n_features:
        return zz_feature_map(feature_dimension=n_features, reps=reps, entanglement=entanglement), "standard_zz_feature_map"
    return build_multiqubit_feature_map(n_features, n_qubits, reps, entanglement), "custom_multiqubit_zz_reupload"


def feature_map_fingerprint(circuit: QuantumCircuit) -> str:

    instructions = []
    for item in circuit.data:
        operation = item.operation
        instructions.append({
            "name": operation.name,
            "qubits": [circuit.find_bit(bit).index for bit in item.qubits],
            "clbits": [circuit.find_bit(bit).index for bit in item.clbits],
            "params": [str(value) for value in operation.params],
        })
    payload = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "global_phase": str(circuit.global_phase),
        "instructions": instructions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metric_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _project_to_psd(kernel_matrix: np.ndarray) -> np.ndarray:

    sym = 0.5 * (kernel_matrix + kernel_matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).real


def _bind_feature_map(feature_map: QuantumCircuit, values: np.ndarray) -> QuantumCircuit:

    params = list(feature_map.parameters)
    values = np.asarray(values, dtype=float)

    if len(params) != len(values):
        raise ValueError(
            f"Feature map expects {len(params)} values, received {len(values)}."
        )

    bound = feature_map.assign_parameters(
        dict(zip(params, values)),
        inplace=False,
    )

    if bound.num_parameters != 0:
        raise RuntimeError(
            "Feature-map circuit still has unbound parameters after binding."
        )
    return bound


def _make_mps_fidelity_circuit(
    feature_map: QuantumCircuit,
    x: np.ndarray,
    y: np.ndarray,
) -> QuantumCircuit:
    
    ux = _bind_feature_map(feature_map, x)
    uy = _bind_feature_map(feature_map, y)

    circuit = ux.compose(uy.inverse())
    circuit.global_phase = 0

    try:
        import qiskit_aer 
    except ImportError as exc:
        raise RuntimeError(
            "Simulator 'aer_mps' requires qiskit-aer. "
            "Install it in this environment with: pip install qiskit-aer"
        ) from exc

    circuit.save_amplitudes_squared(
        [0],
        label="fidelity_zero",
    )
    return circuit


def _build_mps_simulator(
    max_bond_dimension: int,
    truncation_threshold: float,
):
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError(
            "Simulator 'aer_mps' requires qiskit-aer. "
            "Install it in this environment with: pip install qiskit-aer"
        ) from exc

    if max_bond_dimension < 0:
        raise ValueError("--mps-max-bond-dimension must be >= 0.")
    if truncation_threshold < 0:
        raise ValueError("--mps-truncation-threshold must be >= 0.")

    options = {
        "method": "matrix_product_state",
        "max_parallel_experiments": 1,
        "matrix_product_state_truncation_threshold": float(truncation_threshold),
    }
    if max_bond_dimension > 0:
        options["matrix_product_state_max_bond_dimension"] = int(max_bond_dimension)

    return AerSimulator(**options)


def _run_mps_pair_batch(
    simulator,
    feature_map: QuantumCircuit,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    if not pairs:
        return []

    circuits = [
        _make_mps_fidelity_circuit(feature_map, x, y)
        for x, y in pairs
    ]

    result = simulator.run(circuits).result()

    values: list[float] = []
    for i in range(len(circuits)):
        saved = result.data(i)["fidelity_zero"]
        arr = np.asarray(saved, dtype=float).reshape(-1)
        if arr.size != 1:
            raise RuntimeError(
                f"Unexpected MPS fidelity result shape for circuit {i}: {arr.shape}"
            )
        values.append(float(arr[0]))
    return values


def _evaluate_mps_kernel_train(
    X_train: np.ndarray,
    feature_map: QuantumCircuit,
    simulator,
) -> np.ndarray:
    n = len(X_train)
    kernel = np.eye(n, dtype=float)

    indices: list[tuple[int, int]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n):
        for j in range(i + 1, n):
            indices.append((i, j))
            pairs.append((X_train[i], X_train[j]))

    values = _run_mps_pair_batch(simulator, feature_map, pairs)
    for (i, j), value in zip(indices, values):
        kernel[i, j] = value
        kernel[j, i] = value

    return _project_to_psd(kernel)


def _evaluate_mps_kernel_cross(
    X_left: np.ndarray,
    X_right: np.ndarray,
    feature_map: QuantumCircuit,
    simulator,
) -> np.ndarray:
    kernel = np.empty((len(X_left), len(X_right)), dtype=float)

    indices: list[tuple[int, int]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for i, x in enumerate(X_left):
        for j, y in enumerate(X_right):
            indices.append((i, j))
            pairs.append((x, y))

    values = _run_mps_pair_batch(simulator, feature_map, pairs)
    for (i, j), value in zip(indices, values):
        kernel[i, j] = value

    return kernel


def evaluate_one_split(
    X_train,
    y_train,
    X_test,
    y_test,
    scaler_name: str,
    feature_map: QuantumCircuit,
    C: float,
    class_weight: str,
    simulator_name: str,
    mps_max_bond_dimension: int = 0,
    mps_truncation_threshold: float = 0.0,
):
    Xtr, Xte, _, preprocess_train_time, preprocess_test_time = fit_transform_scaler(
        X_train,
        X_test,
        scaler_name,
    )

    if simulator_name == "reference":
        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
            shots=None,
            enforce_psd=True,
        )

        ktrain_start = time.perf_counter()
        K_train = kernel.evaluate(x_vec=Xtr)
        ktrain_time = time.perf_counter() - ktrain_start

        ktest_start = time.perf_counter()
        K_test = kernel.evaluate(x_vec=Xte, y_vec=Xtr)
        ktest_time = time.perf_counter() - ktest_start

        kernel_implementation = "FidelityStatevectorKernel"

    elif simulator_name == "aer_mps":
        simulator = _build_mps_simulator(
            max_bond_dimension=mps_max_bond_dimension,
            truncation_threshold=mps_truncation_threshold,
        )

        print(
            "[MPS] AerSimulator | "
            f"max_bond_dimension={'unlimited' if mps_max_bond_dimension == 0 else mps_max_bond_dimension} | "
            f"truncation_threshold={mps_truncation_threshold}"
        )

        ktrain_start = time.perf_counter()
        K_train = _evaluate_mps_kernel_train(Xtr, feature_map, simulator)
        ktrain_time = time.perf_counter() - ktrain_start

        ktest_start = time.perf_counter()
        K_test = _evaluate_mps_kernel_cross(Xte, Xtr, feature_map, simulator)
        ktest_time = time.perf_counter() - ktest_start

        kernel_implementation = "manual_compute_uncompute_aer_mps"

    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    model = SVC(
        kernel="precomputed",
        C=C,
        class_weight=class_weight_value(class_weight),
    )

    fit_start = time.perf_counter()
    model.fit(K_train, y_train)
    model_fit_time = time.perf_counter() - fit_start

    predict_train_start = time.perf_counter()
    pred_train = model.predict(K_train)
    predict_train_time = time.perf_counter() - predict_train_start

    predict_test_start = time.perf_counter()
    pred_test = model.predict(K_test)
    predict_test_time = time.perf_counter() - predict_test_start

    return {
        "K_train": K_train,
        "K_test": K_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "kernel_train_time_s": ktrain_time,
        "kernel_test_time_s": ktest_time,
        "preprocess_train_time_s": preprocess_train_time,
        "preprocess_test_time_s": preprocess_test_time,
        "model_fit_time_s": model_fit_time,
        "predict_train_time_s": predict_train_time,
        "predict_test_time_s": predict_test_time,
        "svc_total_time_s": model_fit_time + predict_train_time + predict_test_time,
        "kernel_implementation": kernel_implementation,
    }


def run_cv(args, selected_df: pd.DataFrame, features: list[str], n_qubits: int) -> tuple[float | None, float | None, float | None, int]:
    requested = int(args.cv)
    if requested <= 0:
        return None, None, None, 0
    y = selected_df[args.label]
    folds = min(requested, int(y.value_counts().min()))
    if folds < 2:
        return None, None, None, 0

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.random_state)
    scores = []
    start = time.perf_counter()
    for train_idx, val_idx in splitter.split(selected_df[features], y):
        tr, va = selected_df.iloc[train_idx], selected_df.iloc[val_idx]
        fmap, _ = build_feature_map(len(features), n_qubits, args.reps, args.entanglement)
        result = evaluate_one_split(
            tr[features],
            tr[args.label],
            va[features],
            va[args.label],
            args.scaler,
            fmap,
            args.C,
            args.class_weight,
            args.simulator,
            args.mps_max_bond_dimension,
            args.mps_truncation_threshold,
        )
        scores.append(balanced_accuracy_score(va[args.label], result["pred_test"]))
    elapsed = time.perf_counter() - start
    return float(np.mean(scores)), float(np.std(scores)), elapsed, folds


def make_run_dir(args, feature_set_name: str, n_qubits: int) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        prefix = f"{feature_set_name}_{n_qubits}q_{args.entanglement}_{args.scaler}_seed{args.random_state}"
        index = 1
        while (args.output_dir / f"{prefix}_{index:02d}").exists():
            index += 1
        name = f"{prefix}_{index:02d}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    args = parse_args()
    if args.C <= 0 or args.reps < 1:
        raise ValueError("--C and --reps must be positive")

    features = get_features(args)
    feature_set_name = "custom" if args.features is not None else args.feature_set
    df = load_base_data(args, features)
    train_df, test_df = split_data(args, df)
    selected_df = pd.concat([train_df, test_df], ignore_index=True)

    feature_map, encoding_name = build_feature_map(len(features), args.n_qubits, args.reps, args.entanglement)
    n_qubits = feature_map.num_qubits
    result = evaluate_one_split(
        train_df[features],
        train_df[args.label],
        test_df[features],
        test_df[args.label],
        args.scaler,
        feature_map,
        args.C,
        args.class_weight,
        args.simulator,
        args.mps_max_bond_dimension,
        args.mps_truncation_threshold,
    )

    train_metrics = metric_bundle(train_df[args.label], result["pred_train"])
    test_metrics = metric_bundle(test_df[args.label], result["pred_test"])
    cv_mean, cv_std, cv_time, cv_folds = run_cv(args, train_df, features, n_qubits)
    diagnosis = diagnose_fit(train_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"], cv_mean)

    run_dir = make_run_dir(args, feature_set_name, n_qubits)
    labels = sorted(selected_df[args.label].unique().tolist())

    pd.DataFrame(result["K_train"]).to_csv(run_dir / "K_train.csv", index=False)
    pd.DataFrame(result["K_test"]).to_csv(run_dir / "K_test.csv", index=False)
    pd.DataFrame({
        "row_id": test_df["_row_id"].to_numpy(),
        "true_label": test_df[args.label].to_numpy(),
        "pred_label": result["pred_test"],
    }).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(test_df[args.label], result["pred_test"], labels=labels), index=labels, columns=labels).to_csv(run_dir / "confusion_matrix.csv")
    report = classification_report(test_df[args.label], result["pred_test"], zero_division=0)
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (run_dir / "used_features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    (run_dir / "feature_map.txt").write_text(str(feature_map.draw(output="text", fold=-1)) + "\n", encoding="utf-8")
    pd.concat([
        pd.DataFrame({"row_id": train_df["_row_id"].astype(int), "split": "train", "true_label": train_df[args.label].to_numpy()}),
        pd.DataFrame({"row_id": test_df["_row_id"].astype(int), "split": "test", "true_label": test_df[args.label].to_numpy()}),
    ], ignore_index=True).to_csv(run_dir / "split.csv", index=False)

    metrics = {
        "model": "QSVM_IDEAL",
        "implementation": result["kernel_implementation"] + " + SVC(precomputed)",
        "input_csv": str(args.csv),
        "feature_set": feature_set_name,
        "features": features,
        "n_features": len(features),
        "n_qubits": n_qubits,
        "encoding": encoding_name,
        "feature_map_fingerprint": feature_map_fingerprint(feature_map),
        "feature_map": "zz_feature_map" if n_qubits == len(features) else "custom_multiqubit_zz_reupload",
        "reps": args.reps,
        "entanglement": args.entanglement,
        "feature_map_depth": feature_map.depth(),
        "feature_map_size": feature_map.size(),
        "samples": len(selected_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "scaler": args.scaler,
        "simulator": args.simulator,
        "mps_max_bond_dimension": int(args.mps_max_bond_dimension),
        "mps_truncation_threshold": float(args.mps_truncation_threshold),
        "mps_approximate": bool(
            args.simulator == "aer_mps"
            and (args.mps_max_bond_dimension > 0 or args.mps_truncation_threshold > 0)
        ),
        "C": args.C,
        "class_weight": args.class_weight,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_samples": args.max_samples,
        "split_file": None if args.split_file is None else str(args.split_file),
        "split_sha256": file_sha256(args.split_file),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_folds": cv_folds,
        "cv_balanced_accuracy_mean": cv_mean,
        "cv_balanced_accuracy_std": cv_std,
        "fit_diagnosis": diagnosis,
        "timing": {
            "preprocess_train_wall_s": result["preprocess_train_time_s"],
            "preprocess_test_wall_s": result["preprocess_test_time_s"],
            "kernel_train_wall_s": result["kernel_train_time_s"],
            "kernel_test_wall_s": result["kernel_test_time_s"],
            "kernel_total_wall_s": result["kernel_train_time_s"] + result["kernel_test_time_s"],
            "model_fit_wall_s": result["model_fit_time_s"],
            "predict_train_wall_s": result["predict_train_time_s"],
            "predict_test_wall_s": result["predict_test_time_s"],
            "quantum_execution_s": None,
            "transpile_wall_s": None,
            "job_wait_wall_s": None,
            "end_to_end_test_wall_s": (
                result["preprocess_train_time_s"]
                + result["preprocess_test_time_s"]
                + result["kernel_train_time_s"]
                + result["kernel_test_time_s"]
                + result["model_fit_time_s"]
                + result["predict_test_time_s"]
            ),
            "cv_wall_s": cv_time,
            "kernel_train_time_s": result["kernel_train_time_s"],
            "kernel_test_time_s": result["kernel_test_time_s"],
            "kernel_total_time_s": result["kernel_train_time_s"] + result["kernel_test_time_s"],
            "precomputed_svc_time_s": result["svc_total_time_s"],
            "total_test_pipeline_s": (
                result["kernel_train_time_s"]
                + result["kernel_test_time_s"]
                + result["svc_total_time_s"]
            ),
            "cv_time_s": cv_time,
        },
        "environment": {
            "python": platform.python_version(),
            "qiskit": package_version("qiskit"),
            "qiskit_machine_learning": package_version("qiskit-machine-learning"),
            "qiskit_aer": package_version("qiskit-aer"),
            "scikit_learn": package_version("scikit-learn"),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== QSVM =====")
    print(f"Feature set: {feature_set_name} ({len(features)} features)")
    print(f"Qubits: {n_qubits} | encoding={encoding_name} | reps={args.reps} | entanglement={args.entanglement}")
    print(f"Scaler: {args.scaler} | C={args.C} | class_weight={args.class_weight}")
    print(f"Simulator: {args.simulator}")
    print(f"Test accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test macro F1:          {test_metrics['f1_macro']:.4f}")
    print(f"Train balanced accuracy:{train_metrics['balanced_accuracy']:.4f}")
    print(f"Train-test BA gap:      {diagnosis['train_test_gap']:.4f}")
    print(f"Diagnosis:              {diagnosis['diagnosis']}")
    if cv_mean is not None:
        print(f"CV balanced accuracy:   {cv_mean:.4f} +/- {cv_std:.4f} ({cv_folds} folds)")
    print(f"K_train time:           {result['kernel_train_time_s']:.6f} s")
    print(f"K_test time:            {result['kernel_test_time_s']:.6f} s")
    print(f"SVC fit time:           {result['model_fit_time_s']:.6f} s")
    print(f"Predict test time:      {result['predict_test_time_s']:.6f} s")
    print(
        "Preprocess train/test:  "
        f"{result['preprocess_train_time_s']:.6f} / "
        f"{result['preprocess_test_time_s']:.6f} s"
    )
    print(report)
    print(f"RESULT_DIR={run_dir}")


if __name__ == "__main__":
    main()
