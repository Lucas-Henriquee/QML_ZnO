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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
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
KERNELS = ["linear", "rbf", "poly", "sigmoid"]


def parse_args():
    parser = argparse.ArgumentParser(description="General classical SVM benchmark.")
    parser.add_argument("--csv", type=Path, default=Path("results/datasets/multitech_expanded_ml_dataset_clean.csv"))
    parser.add_argument("--label", default="label_site_id")
    parser.add_argument("--feature-set", default="all12", choices=sorted(FEATURE_SETS))
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--kernel", default="rbf", choices=KERNELS)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale", help="'scale', 'auto', or a positive float.")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--scaler", default="power", choices=SCALERS)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv", type=int, default=5, help="0 disables cross-validation.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses every available row.")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional CSV with row_id and split=train/test.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/svm"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shuffle-label", action="store_true")
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


def parse_gamma(value: str):
    if value in {"scale", "auto"}:
        return value
    gamma = float(value)
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return gamma


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


def make_scaler(name: str):
    if name == "none":
        return "passthrough"
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    if name == "minmax_pi":
        return MinMaxScaler(feature_range=(0.0, np.pi))
    if name == "power":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if name == "power_minmax":
        return Pipeline([
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("minmax_pi", MinMaxScaler(feature_range=(0.0, np.pi))),
        ])
    raise ValueError(f"Unknown scaler: {name}")


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

    chunks = []
    selected = set()
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
            sampled = pd.concat([
                sampled,
                rest.sample(n=min(remaining, len(rest)), random_state=random_state + 10_000),
            ])
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

    train_df, test_df = train_test_split(
        sampled,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=sampled[args.label],
    )
    return train_df.copy(), test_df.copy()


def build_svc(args) -> SVC:
    return SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=parse_gamma(args.gamma),
        degree=args.degree,
        class_weight=class_weight_value(args.class_weight),
        max_iter=100_000,
        random_state=args.random_state,
    )


def build_model(args):
    return Pipeline([("scaler", make_scaler(args.scaler)), ("svm", build_svc(args)),])

def preprocess_with_timing(X_train, X_test, scaler_name: str):

    scaler = make_scaler(scaler_name)

    train_start = time.perf_counter()
    if scaler == "passthrough":
        X_train_scaled = np.asarray(X_train, dtype=float)
    else:
        X_train_scaled = scaler.fit_transform(X_train)
    train_wall = time.perf_counter() - train_start

    test_start = time.perf_counter()
    if scaler == "passthrough":
        X_test_scaled = np.asarray(X_test, dtype=float)
    else:
        X_test_scaled = scaler.transform(X_test)
    test_wall = time.perf_counter() - test_start

    return X_train_scaled, X_test_scaled, train_wall, test_wall


def metric_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def get_cv(y: pd.Series, requested: int) -> int:
    if requested <= 0:
        return 0
    return max(0, min(int(requested), int(y.value_counts().min())))


def make_run_dir(args, feature_set_name: str) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{stamp}_{feature_set_name}_{args.kernel}_{args.scaler}_seed{args.random_state}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    args = parse_args()
    if args.C <= 0:
        raise ValueError("--C must be > 0")

    features = get_features(args)
    feature_set_name = "custom" if args.features is not None else args.feature_set
    df = load_base_data(args, features)
    train_df, test_df = split_data(args, df)

    if args.shuffle_label:
        combined = pd.concat([train_df, test_df], ignore_index=True)
        shuffled = combined[args.label].sample(frac=1.0, random_state=args.random_state).to_numpy()
        combined[args.label] = shuffled
        train_df = combined.iloc[:len(train_df)].copy()
        test_df = combined.iloc[len(train_df):].copy()

    X_train, y_train = train_df[features], train_df[args.label]
    X_test, y_test = test_df[features], test_df[args.label]

    X_train_scaled, X_test_scaled, preprocess_train_time, preprocess_test_time = (preprocess_with_timing(X_train, X_test, args.scaler))

    model = build_svc(args)
    fit_start = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    fit_time = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    predict_time = time.perf_counter() - predict_start

    train_predict_start = time.perf_counter()
    y_train_pred = model.predict(X_train_scaled)
    train_predict_time = time.perf_counter() - train_predict_start

    test_metrics = metric_bundle(y_test, y_pred)
    train_metrics = metric_bundle(y_train, y_train_pred)

    all_selected = pd.concat([train_df, test_df], ignore_index=True)
    cv = get_cv(y_train, args.cv)
    cv_mean = cv_std = cv_time = None
    if cv >= 2:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=args.random_state)
        cv_start = time.perf_counter()
        scores = cross_val_score(
            build_model(args),
            X_train,
            y_train,
            cv=splitter,
            scoring="balanced_accuracy",
        )
        cv_time = time.perf_counter() - cv_start
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    diagnosis = diagnose_fit(train_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"], cv_mean)
    run_dir = make_run_dir(args, feature_set_name)

    labels = sorted(all_selected[args.label].unique().tolist())
    pd.DataFrame({
        "row_id": test_df["_row_id"].to_numpy(),
        "true_label": y_test.to_numpy(),
        "pred_label": y_pred,
    }).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_text = classification_report(y_test, y_pred, zero_division=0)
    (run_dir / "classification_report.txt").write_text(classification_report_text, encoding="utf-8")
    (run_dir / "used_features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")

    split_out = pd.concat([
        pd.DataFrame({"row_id": train_df["_row_id"].astype(int), "split": "train", "true_label": y_train.to_numpy()}),
        pd.DataFrame({"row_id": test_df["_row_id"].astype(int), "split": "test", "true_label": y_test.to_numpy()}),
    ], ignore_index=True)
    split_out.to_csv(run_dir / "split.csv", index=False)

    metrics = {
        "model": "SVM",
        "input_csv": str(args.csv),
        "feature_set": feature_set_name,
        "features": features,
        "n_features": len(features),
        "samples": len(all_selected),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "classes": labels,
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "degree": args.degree,
        "class_weight": args.class_weight,
        "scaler": args.scaler,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_samples": args.max_samples,
        "split_file": None if args.split_file is None else str(args.split_file),
        "split_sha256": file_sha256(args.split_file),
        "shuffle_label": bool(args.shuffle_label),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_folds": cv,
        "cv_balanced_accuracy_mean": cv_mean,
        "cv_balanced_accuracy_std": cv_std,
        "fit_diagnosis": diagnosis,
        "timing": {
            "preprocess_train_wall_s": preprocess_train_time,
            "preprocess_test_wall_s": preprocess_test_time,
            "kernel_train_wall_s": None,
            "kernel_test_wall_s": None,
            "kernel_total_wall_s": None,
            "model_fit_wall_s": fit_time,
            "predict_train_wall_s": train_predict_time,
            "predict_test_wall_s": predict_time,
            "quantum_execution_s": None,
            "transpile_wall_s": None,
            "job_wait_wall_s": None,
            "end_to_end_test_wall_s": (
                preprocess_train_time
                + preprocess_test_time
                + fit_time
                + predict_time
            ),
            "cv_wall_s": cv_time,
            "fit_time_s": fit_time,
            "predict_test_time_s": predict_time,
            "predict_train_time_s": train_predict_time,
            "fit_plus_test_predict_s": fit_time + predict_time,
            "cv_time_s": cv_time,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": package_version("scikit-learn"),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCLASSICAL SVM ")
    print(f"Feature set: {feature_set_name} ({len(features)} features)")
    print(f"Samples: {len(all_selected)} | train={len(train_df)} | test={len(test_df)}")
    print(f"Kernel: {args.kernel} | C={args.C} | gamma={args.gamma} | scaler={args.scaler}")
    print(f"Test accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test macro F1:          {test_metrics['f1_macro']:.4f}")
    print(f"Train balanced accuracy:{train_metrics['balanced_accuracy']:.4f}")
    print(f"Train-test BA gap:      {diagnosis['train_test_gap']:.4f}")
    print(f"Diagnosis:              {diagnosis['diagnosis']}")
    if cv_mean is not None:
        print(f"CV balanced accuracy:   {cv_mean:.4f} +/- {cv_std:.4f} ({cv} folds)")
    print(f"Fit time:               {fit_time:.6f} s")
    print(f"Predict test time:      {predict_time:.6f} s")
    print(f"Preprocess train/test:  {preprocess_train_time:.6f} / {preprocess_test_time:.6f} s")
    print(classification_report_text)
    print(f"RESULT_DIR={run_dir}")


if __name__ == "__main__":
    main()