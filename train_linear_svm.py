import argparse
import os
import pickle
import random
import time
from typing import List,NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from cld.models.asr_model import ASRModel


class RunResults(NamedTuple):
    train_acc: float
    eval_acc: float
    eval_split: str
    n_train: int
    n_eval: int
    n_classes: int
    model_path: str
    csv_path: str
    log_path: str
    training_time_s: float


def _maybe_init_wandb(enable: bool, run_name: str, config: dict):
    if not enable:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        print("wandb not available; continuing without wandb logging.")
        return None
    wandb.init(project="CLD", name=run_name, config=config)
    return wandb


def _load_split_embeddings(
    asr_model: ASRModel,
    data_dir: str,
    dataset_split: str,
    data_seed: int,
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Returns:
        A: (N, D) float array of pooled encoder embeddings
        y: (N,) int labels for language index
        n_classes: optional int (returned by MMS backend)
    """
    out = asr_model.load_data(
        data_dir,
        data_seed=data_seed,
        dataset_split=dataset_split,
        shuffle=shuffle,
    )
    if isinstance(out, tuple) and len(out) == 2:
        A, y = out
        return np.asarray(A), np.asarray(y), None
    if isinstance(out, tuple) and len(out) == 3:
        A, y, n_classes = out
        return np.asarray(A), np.asarray(y), int(n_classes)
    raise ValueError(f"Unexpected return from ASRModel.load_data: type={type(out)} value={out}")


def run(
    model_name: str,
    data_dir: str,
    output_dir: str,
    data_seed: int,
    eval_split: str,
    c: float,
    max_iter: int,
    class_weight: Optional[str],
    dual: str,
    wandb_enable: bool,
    languages: List[str],
) -> RunResults:
    # Lazy import sklearn so importing this module doesn't require it.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    asr_model = ASRModel.from_pretrained(model_name, config={"languages": languages})

    print(f"Loading embeddings from {data_dir} (train split)...")
    X_train, y_train, n_classes_train = _load_split_embeddings(
        asr_model=asr_model,
        data_dir=data_dir,
        dataset_split="train",
        data_seed=data_seed,
        shuffle=True,
    )

    print(f"Loading embeddings from {data_dir} (eval split: {eval_split})...")
    X_eval, y_eval, n_classes_eval = _load_split_embeddings(
        asr_model=asr_model,
        data_dir=data_dir,
        dataset_split=eval_split,
        data_seed=data_seed,
        shuffle=False,
    )

    n_classes = int(
        n_classes_train
        or n_classes_eval
        or len(np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_eval)], axis=0)))
    )

    if dual not in {"auto", "true", "false"}:
        raise ValueError("--dual must be one of: auto, true, false")
    dual_value = "auto" if dual == "auto" else (dual == "true")

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=float(c),
                    max_iter=int(max_iter),
                    class_weight=class_weight,
                    dual=dual_value,
                    random_state=int(data_seed),
                ),
            ),
        ]
    )

    config = {
        "model_name": model_name,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "data_seed": data_seed,
        "eval_split": eval_split,
        "C": c,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "dual": dual,
        "n_train": int(X_train.shape[0]),
        "n_eval": int(X_eval.shape[0]),
        "n_classes": n_classes,
        "embedding_dim": int(X_train.shape[1]) if X_train.ndim == 2 else None,
    }

    wandb = _maybe_init_wandb(wandb_enable, run_name=f"linear_svm_{model_name}", config=config)

    print("Training Linear SVM (sklearn LinearSVC) ...")
    start = time.time()
    clf.fit(X_train, y_train)
    training_time_s = time.time() - start

    yhat_train = clf.predict(X_train)
    yhat_eval = clf.predict(X_eval)

    train_acc = float(accuracy_score(y_train, yhat_train))
    eval_acc = float(accuracy_score(y_eval, yhat_eval))

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"{eval_split} accuracy: {eval_acc:.4f}")

    # Prepare output directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{model_name}_linear_svm.pkl").replace("/", "_")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save metrics CSV (tab-separated to match train_cvxnn.py)
    csv_path = os.path.join(model_dir, "linear_svm_metrics.csv")
    metrics_df = pd.DataFrame(
        {
            "train_acc": [train_acc],
            "eval_acc": [eval_acc],
            "eval_split": [eval_split],
            "n_train": [int(X_train.shape[0])],
            "n_eval": [int(X_eval.shape[0])],
            "n_classes": [n_classes],
            "embedding_dim": [int(X_train.shape[1]) if X_train.ndim == 2 else None],
            "training_time_s": [training_time_s],
            "model_path": [model_path],
        }
    )
    metrics_df.to_csv(csv_path, sep="\t", encoding="utf-8", index=False, header=True)

    # Save summary text file
    log_path = os.path.join(model_dir, "linear_svm_results.txt")
    with open(log_path, "w") as f:
        for k, v in metrics_df.iloc[0].to_dict().items():
            f.write(f"{k}: {v}\n")

    if wandb is not None:
        try:
            wandb.log(
                {
                    "train_acc": train_acc,
                    "eval_acc": eval_acc,
                    "training_time_s": training_time_s,
                    "n_train": int(X_train.shape[0]),
                    "n_eval": int(X_eval.shape[0]),
                    "n_classes": n_classes,
                    "model_path": model_path,
                }
            )
            wandb.finish()
        except Exception as e:
            print(f"wandb logging failed (continuing): {e}")

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {csv_path}")

    return RunResults(
        train_acc=train_acc,
        eval_acc=eval_acc,
        eval_split=eval_split,
        n_train=int(X_train.shape[0]),
        n_eval=int(X_eval.shape[0]),
        n_classes=n_classes,
        model_path=model_path,
        csv_path=csv_path,
        log_path=log_path,
        training_time_s=training_time_s,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Linear SVM on ASR encoder embeddings for language classification")

    # Required Arguments
    parser.add_argument("--model_name", type=str, required=True, help="ASR model name (e.g. facebook/mms-1b, openai/whisper-small)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HF dataset directory (load_from_disk), with splits train/valid/test")
    parser.add_argument("--languages", type=str, required=True, help="Comma-separated list of languages to train on")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory (models + metrics will be saved here)")

    # Optional Arguments
    parser.add_argument("--data_seed", type=int, default=None, help="Data seed (default: random 1-10)")
    parser.add_argument("--eval_split", type=str, default="valid", help="Which split to evaluate on: valid or test (default: valid)")

    # Linear SVM Hyperparameters
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)")
    parser.add_argument("--max_iter", type=int, default=5000, help="Max iterations for LinearSVC (default: 5000)")
    parser.add_argument(
        "--class_weight",
        type=str,
        default=None,
        help='Class weighting; use "balanced" for inverse-frequency weights (default: None)',
    )
    parser.add_argument(
        "--dual",
        type=str,
        default="auto",
        help='LinearSVC dual formulation: "auto", "true", or "false" (default: auto)',
    )

    # Logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")

    args = parser.parse_args()

    if args.data_seed is None:
        args.data_seed = random.randint(1, 10)

    # Normalize class_weight
    class_weight = args.class_weight
    if isinstance(class_weight, str) and class_weight.lower() == "none":
        class_weight = None

    results = run(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        data_seed=int(args.data_seed),
        eval_split=args.eval_split,
        c=float(args.C),
        max_iter=int(args.max_iter),
        class_weight=class_weight,
        dual=args.dual,
        wandb_enable=not args.no_wandb,
        languages=args.languages.split(","),
    )

    # Print a small final summary (human-readable)
    print("\nFinal Results:")
    print(
        pd.DataFrame(
            {
                "train_acc": [results.train_acc],
                f"{results.eval_split}_acc": [results.eval_acc],
                "n_train": [results.n_train],
                "n_eval": [results.n_eval],
                "n_classes": [results.n_classes],
                "training_time_s": [results.training_time_s],
                "model_path": [results.model_path],
            }
        )
    )
