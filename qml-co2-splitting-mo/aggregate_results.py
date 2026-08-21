from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics(base_dir: Path, model_type: str) -> list[dict]:
    rows = []
    for metrics_path in base_dir.rglob("metrics.json"):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"AVISO: nao consegui ler {metrics_path}, pulando")
            continue

        data["model_type"] = model_type
        data["run_dir"] = str(metrics_path.parent)
        rows.append(data)

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm-dir", type=Path, default=Path("results/svm"))
    parser.add_argument("--qsvm-dir", type=Path, default=Path("results/qsvm"))
    parser.add_argument("--output", type=Path, default=Path("results/comparison_table.csv"))
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    if args.svm_dir.exists():
        rows.extend(load_metrics(args.svm_dir, "SVM"))
    if args.qsvm_dir.exists():
        rows.extend(load_metrics(args.qsvm_dir, "QSVM"))

    if not rows:
        print("Nenhum metrics.json encontrado. Rode o sweep primeiro.")
        return

    df = pd.DataFrame(rows)

    # colunas comuns aos dois modelos, na ordem que importa pra comparacao
    common_cols = [
        "model_type",
        "feature_set",
        "n_features",
        "scaler",
        "accuracy",
        "balanced_accuracy",
        "train_balanced_accuracy",
        "train_test_gap",
        "fit_diagnosis",
        "cv_balanced_accuracy_mean",
        "fit_time_s",
        "predict_time_s",
        "total_time_s",
        "test_size",
        "samples",
        "max_samples",
        "kernel",
        "reps",
        "entanglement",
        "run_dir",
    ]
    ordered_cols = [c for c in common_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    df = df.sort_values(
        by=["model_type", "balanced_accuracy"],
        ascending=[True, False],
        na_position="last",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Tabela comparativa salva em: {args.output}")
    print(f"Total de runs: {len(df)} ({(df['model_type'] == 'SVM').sum()} SVM, {(df['model_type'] == 'QSVM').sum()} QSVM)")
    print()
    print("Top 5 por balanced_accuracy (cada modelo):")
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type].head(5)
        print(f"\n-- {model_type} --")
        cols_to_show = [c for c in ["feature_set", "scaler", "balanced_accuracy", "train_test_gap", "fit_diagnosis"] if c in subset.columns]
        print(subset[cols_to_show].to_string(index=False))

    print("\nMaior gap treino-teste (possível overfitting) por modelo:")
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type].nlargest(3, "train_test_gap")
        print(f"\n-- {model_type} --")
        print(subset[["feature_set", "scaler", "balanced_accuracy", "train_test_gap"]].to_string(index=False))    


if __name__ == "__main__":
    main()