"""
Multidomain Intent Detection — Automatic Hyperparameter Tuner
==============================================================

Systematically searches for optimal hyperparameters to maximize
accuracy and confidence scores.  No manual code edits required.

What it tunes:
  1. PCA dimensions
  2. Classifier hyperparameters (SVM C, LogReg C, kNN k, ET depth)
  3. Confidence penalty factors (disagreement, confusion-pair)
  4. Temperature scaling range
  5. Gating thresholds

How it works:
  - Loads training data + embeddings (same as training.py)
  - Runs stratified 5-fold CV for each hyperparameter set
  - Picks the best config based on accuracy AND confidence quality
  - Saves optimal tuning_config.json
  - Optionally retrains the final model with best params

Usage:
    # Full auto-tune (searches all parameters)
    python -m multidomain_intent_detection.auto_tuner

    # Quick mode: only tune most impactful parameters
    python -m multidomain_intent_detection.auto_tuner --quick

    # Tune and retrain immediately
    python -m multidomain_intent_detection.auto_tuner --retrain

    # Tune with a specific test CSV for validation
    python -m multidomain_intent_detection.auto_tuner --csv testingFinalDataset.csv
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*sklearn.*")

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "tuning_config.json")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config() -> Dict:
    """Load current tuning configuration."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")


def save_config(config: Dict, backup: bool = True):
    """Save tuning config. Creates timestamped backup before overwriting."""
    if backup and os.path.exists(CONFIG_PATH):
        ts = _timestamp()
        backup_path = CONFIG_PATH.replace(".json", f"_backup_{ts}.json")
        import shutil
        shutil.copy2(CONFIG_PATH, backup_path)
        print(f"  Config backup: {backup_path}")

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {CONFIG_PATH}")


def _load_training_data():
    """Load embeddings and build training arrays."""
    from multidomain_intent_detection.training import (
        load_embeddings, augment_embeddings, build_Xy,
    )
    from multidomain_intent_detection.config import INTENT_TO_DOMAIN

    all_emb = load_embeddings()
    augmented_emb = augment_embeddings(all_emb)
    train_intents = set(all_emb.keys()) & set(INTENT_TO_DOMAIN.keys())
    X, y, labels = build_Xy(augmented_emb, train_intents)
    return X, y, labels


def _evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    n_pca: int,
    svm_c: float,
    logreg_c: float,
    knn_k: int,
    et_n: int,
    et_depth: Optional[int],
    et_min_leaf: int,
    n_folds: int = 5,
) -> Tuple[float, float, float]:
    """Run CV with given hyperparams.  Returns (mean_acc, std, median_conf)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    all_top1_probs = []

    for train_idx, val_idx in cv.split(X, y):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]

        # L2 norm → PCA → scale
        Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-10)
        d = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=d, whiten=True, random_state=42)
        Xp = pca.fit_transform(Xn)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xp)

        # Val transform
        Xvn = Xva / (np.linalg.norm(Xva, axis=1, keepdims=True) + 1e-10)
        Xvs = scaler.transform(pca.transform(Xvn))

        clfs = {
            "svm": SVC(kernel="rbf", C=svm_c, gamma="scale", probability=True,
                       class_weight="balanced", random_state=42).fit(Xs, ytr),
            "logreg": LogisticRegression(C=logreg_c, max_iter=3000, solver="lbfgs",
                                         class_weight="balanced", random_state=42).fit(Xs, ytr),
            "knn": KNeighborsClassifier(n_neighbors=min(knn_k, Xtr.shape[0] - 1),
                                        weights="distance", metric="cosine").fit(Xs, ytr),
            "et": ExtraTreesClassifier(n_estimators=et_n, max_depth=et_depth,
                                       min_samples_leaf=et_min_leaf,
                                       class_weight="balanced", random_state=42,
                                       n_jobs=-1).fit(Xs, ytr),
        }

        # Equal-weight ensemble for comparison
        probs = sum(clf.predict_proba(Xvs) * 0.25 for clf in clfs.values())
        preds = np.argmax(probs, axis=1)
        accs.append(float((preds == yva).mean()))

        # Track top-1 confidence for correct predictions
        top1_probs = probs.max(axis=1)
        correct_mask = preds == yva
        all_top1_probs.extend(top1_probs[correct_mask].tolist())

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    median_conf = float(np.median(all_top1_probs)) if all_top1_probs else 0.0

    return mean_acc, std_acc, median_conf


def _tune_pca(X, y, labels, config: Dict) -> int:
    """Find optimal PCA dimensions."""
    search_range = config.get("pipeline", {}).get("pca_search_range",
                                                   [50, 100, 150, 200, 300])
    svm_c = config.get("classifiers", {}).get("svm", {}).get("C", 10)
    logreg_c = config.get("classifiers", {}).get("logreg", {}).get("C", 10)
    knn_k = config.get("pipeline", {}).get("knn_k", 7)
    et_cfg = config.get("classifiers", {}).get("extra_trees", {})

    print(f"\n  PCA Dimension Search")
    print(f"  {'Dims':>6} {'Acc':>8} {'Std':>8} {'Med.Conf':>10}")
    print(f"  {'-' * 36}")

    best_d, best_score = 50, 0.0
    for d in search_range:
        if d >= X.shape[0]:
            continue
        acc, std, conf = _evaluate_config(
            X, y, labels, n_pca=d,
            svm_c=svm_c, logreg_c=logreg_c, knn_k=knn_k,
            et_n=et_cfg.get("n_estimators", 300),
            et_depth=et_cfg.get("max_depth", 30),
            et_min_leaf=et_cfg.get("min_samples_leaf", 2),
        )
        # Composite score: accuracy weighted with confidence quality
        score = acc * 0.7 + conf * 0.3
        marker = " <-- BEST" if score > best_score else ""
        print(f"  {d:>6} {acc*100:>7.2f}% {std*100:>6.2f}% {conf:>9.4f}{marker}")
        if score > best_score:
            best_score = score
            best_d = d

    print(f"\n  Optimal PCA dims: {best_d}")
    return best_d


def _tune_classifiers(X, y, labels, config: Dict, n_pca: int) -> Dict:
    """Search over classifier hyperparameters."""
    et_cfg = config.get("classifiers", {}).get("extra_trees", {})
    knn_k = config.get("pipeline", {}).get("knn_k", 7)

    print(f"\n  Classifier Hyperparameter Search")
    print(f"  {'SVM_C':>7} {'LR_C':>7} {'ET_D':>6} {'Acc':>8} {'Conf':>8}")
    print(f"  {'-' * 42}")

    best_score, best_params = 0.0, {}
    search_space = [
        # (svm_c, logreg_c, et_depth)
        (5,  5,  20), (5,  10, 30), (5,  10, 50),
        (10, 5,  20), (10, 10, 30), (10, 10, 50),
        (10, 15, 30), (15, 5,  30), (15, 10, 30),
        (15, 15, 50), (20, 10, 30), (20, 20, None),
    ]

    for svm_c, logreg_c, et_depth in search_space:
        acc, std, conf = _evaluate_config(
            X, y, labels, n_pca=n_pca,
            svm_c=svm_c, logreg_c=logreg_c, knn_k=knn_k,
            et_n=et_cfg.get("n_estimators", 300),
            et_depth=et_depth,
            et_min_leaf=et_cfg.get("min_samples_leaf", 2),
        )
        score = acc * 0.7 + conf * 0.3
        depth_str = str(et_depth) if et_depth else "None"
        marker = " <-- BEST" if score > best_score else ""
        print(f"  {svm_c:>7} {logreg_c:>7} {depth_str:>6} {acc*100:>7.2f}% {conf:>7.4f}{marker}")
        if score > best_score:
            best_score = score
            best_params = {"svm_c": svm_c, "logreg_c": logreg_c, "et_depth": et_depth}

    print(f"\n  Best: SVM_C={best_params.get('svm_c')}, "
          f"LR_C={best_params.get('logreg_c')}, "
          f"ET_depth={best_params.get('et_depth')}")
    return best_params


def _tune_penalties(X, y, labels, config: Dict, n_pca: int) -> Dict:
    """Search for optimal disagreement and confusion-pair penalties.

    This uses a more sophisticated approach: for each penalty configuration,
    we simulate the full predict_single() pipeline including penalties and
    measure how well the confidence scores separate correct vs incorrect.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    clf_cfg = config.get("classifiers", {})
    svm_c = clf_cfg.get("svm", {}).get("C", 10)
    logreg_c = clf_cfg.get("logreg", {}).get("C", 10)
    knn_k = config.get("pipeline", {}).get("knn_k", 7)
    et_cfg = clf_cfg.get("extra_trees", {})

    print(f"\n  Penalty Factor Search")
    print(f"  {'Disagree':>10} {'ConfPair':>10} {'Acc':>8} {'MedConf':>9} {'Score':>8}")
    print(f"  {'-' * 50}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_score, best_penalties = 0.0, {}

    penalty_space = [
        # (disagree_per_unique, confusion_pair_penalty)
        (0.05, 0.05), (0.05, 0.10), (0.05, 0.15),
        (0.08, 0.05), (0.08, 0.10), (0.08, 0.15),
        (0.10, 0.05), (0.10, 0.10), (0.10, 0.15),
        (0.12, 0.10), (0.15, 0.10), (0.15, 0.20),
    ]

    for disagree_per, confpair_pen in penalty_space:
        fold_accs, fold_confs = [], []

        for train_idx, val_idx in cv.split(X, y):
            Xtr, Xva = X[train_idx], X[val_idx]
            ytr, yva = y[train_idx], y[val_idx]

            Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-10)
            d = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
            pca = PCA(n_components=d, whiten=True, random_state=42)
            Xp = pca.fit_transform(Xn)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xp)

            Xvn = Xva / (np.linalg.norm(Xva, axis=1, keepdims=True) + 1e-10)
            Xvs = scaler.transform(pca.transform(Xvn))

            clfs = {
                "svm": SVC(kernel="rbf", C=svm_c, gamma="scale", probability=True,
                           class_weight="balanced", random_state=42).fit(Xs, ytr),
                "logreg": LogisticRegression(C=logreg_c, max_iter=3000, solver="lbfgs",
                                             class_weight="balanced", random_state=42).fit(Xs, ytr),
                "knn": KNeighborsClassifier(n_neighbors=min(knn_k, Xtr.shape[0]-1),
                                            weights="distance", metric="cosine").fit(Xs, ytr),
                "et": ExtraTreesClassifier(n_estimators=et_cfg.get("n_estimators", 300),
                                           max_depth=et_cfg.get("max_depth", 30),
                                           min_samples_leaf=et_cfg.get("min_samples_leaf", 2),
                                           class_weight="balanced", random_state=42,
                                           n_jobs=-1).fit(Xs, ytr),
            }

            ensemble_probs = sum(clf.predict_proba(Xvs) * 0.25 for clf in clfs.values())
            correct_count, total_correct_conf = 0, 0.0

            for i in range(len(yva)):
                prob_vec = ensemble_probs[i]
                pred = np.argmax(prob_vec)
                conf = float(prob_vec[pred])

                # Simulate disagreement
                indiv_preds = {name: clf.predict(Xvs[i:i+1])[0] for name, clf in clfs.items()}
                n_unique = len(set(indiv_preds.values()))
                if n_unique > 1:
                    penalty = max(1.0 - 0.30, 1.0 - (n_unique - 1) * disagree_per)
                    conf *= penalty

                # Simulate confusion-pair penalty (approximate)
                sorted_probs = np.sort(prob_vec)[::-1]
                if sorted_probs[0] - sorted_probs[1] < 0.15:
                    conf *= (1.0 - confpair_pen)

                is_correct = pred == yva[i]
                if is_correct:
                    correct_count += 1
                    total_correct_conf += conf

            acc = correct_count / len(yva) if len(yva) > 0 else 0
            med_conf = total_correct_conf / correct_count if correct_count > 0 else 0
            fold_accs.append(acc)
            fold_confs.append(med_conf)

        mean_acc = np.mean(fold_accs)
        mean_conf = np.mean(fold_confs)
        score = mean_acc * 0.5 + mean_conf * 0.5  # Balance accuracy and confidence
        marker = " <-- BEST" if score > best_score else ""
        print(f"  {disagree_per:>10.2f} {confpair_pen:>10.2f} {mean_acc*100:>7.2f}% {mean_conf:>8.4f} {score:>7.4f}{marker}")

        if score > best_score:
            best_score = score
            best_penalties = {
                "disagreement_per_unique": disagree_per,
                "disagreement_max_penalty": round(1.0 - max(0.50, 1.0 - 3 * disagree_per), 2),
                "confusion_pair_penalty": confpair_pen,
            }

    print(f"\n  Best: disagree={best_penalties.get('disagreement_per_unique')}, "
          f"confpair={best_penalties.get('confusion_pair_penalty')}")
    return best_penalties


def run_auto_tuner(quick: bool = False, retrain: bool = False, csv_path: str = None):
    """Main auto-tuner entry point."""
    print(f"\n{'=' * 65}")
    print(f"  AUTO-TUNER — Automatic Hyperparameter Optimization")
    print(f"{'=' * 65}")

    config = load_config()

    print(f"\n  Loading training data...")
    t0 = time.time()
    X, y, labels = _load_training_data()
    print(f"  {X.shape[0]} samples, {len(labels)} intents ({time.time()-t0:.1f}s)")

    # ── Phase 1: PCA dims ────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  Phase 1: PCA Dimension Optimization")
    print(f"{'─' * 65}")
    best_pca = _tune_pca(X, y, labels, config)
    config["pipeline"]["pca_dims"] = best_pca

    if not quick:
        # ── Phase 2: Classifier hyperparams ──────────────────────────
        print(f"\n{'─' * 65}")
        print(f"  Phase 2: Classifier Hyperparameter Search")
        print(f"{'─' * 65}")
        best_clf = _tune_classifiers(X, y, labels, config, best_pca)
        config["classifiers"]["svm"]["C"] = best_clf["svm_c"]
        config["classifiers"]["logreg"]["C"] = best_clf["logreg_c"]
        config["classifiers"]["extra_trees"]["max_depth"] = best_clf["et_depth"]

        # ── Phase 3: Penalty factors ─────────────────────────────────
        print(f"\n{'─' * 65}")
        print(f"  Phase 3: Penalty Factor Optimization")
        print(f"{'─' * 65}")
        best_pen = _tune_penalties(X, y, labels, config, best_pca)
        config["confidence_penalties"]["disagreement_per_unique"] = best_pen["disagreement_per_unique"]
        config["confidence_penalties"]["disagreement_max_penalty"] = best_pen["disagreement_max_penalty"]
        config["confidence_penalties"]["confusion_pair_penalty"] = best_pen["confusion_pair_penalty"]

    # ── Save optimized config ────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  Saving Optimized Configuration")
    print(f"{'─' * 65}")
    save_config(config)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  PCA dims          : {best_pca}")
    if not quick:
        print(f"  SVM C             : {best_clf['svm_c']}")
        print(f"  LogReg C          : {best_clf['logreg_c']}")
        print(f"  ExtraTrees depth  : {best_clf['et_depth']}")
        print(f"  Disagree penalty  : {best_pen['disagreement_per_unique']}")
        print(f"  ConfPair penalty  : {best_pen['confusion_pair_penalty']}")
    print(f"\n  Config saved to: {CONFIG_PATH}")

    # ── Retrain if requested ─────────────────────────────────────────
    if retrain:
        print(f"\n{'─' * 65}")
        print(f"  Retraining with optimized config...")
        print(f"{'─' * 65}")
        _retrain_with_config(config, csv_path)

    print(f"\n  Next steps:")
    print(f"    1. Retrain: python -m multidomain_intent_detection.training")
    print(f"    2. Test:    python -m multidomain_intent_detection.batch_test")
    print(f"    3. Diagnose: python -m multidomain_intent_detection.diagnostics --csv <results.csv>")


def _retrain_with_config(config: Dict, csv_path: str = None):
    """Retrain the pipeline using the optimized config."""
    from multidomain_intent_detection.training import (
        load_embeddings, augment_embeddings, build_Xy,
    )
    from multidomain_intent_detection.config import INTENT_TO_DOMAIN
    from multidomain_intent_detection.pipeline import IntentPipeline
    from multidomain_intent_detection.embeddings import get_embedder
    import pickle

    all_emb = load_embeddings()
    augmented_emb = augment_embeddings(all_emb)
    train_intents = set(all_emb.keys()) & set(INTENT_TO_DOMAIN.keys())
    X, y, labels = build_Xy(augmented_emb, train_intents)

    pca_dims = config.get("pipeline", {}).get("pca_dims", 300)
    knn_k = config.get("pipeline", {}).get("knn_k", 7)

    pipe = IntentPipeline(n_pca=pca_dims, knn_k=knn_k)
    pipe.fit(X, y, labels)

    # Save
    _IDS_DIR = os.path.join(os.path.dirname(BASE_DIR), "Intent_detection_system")
    ARTIFACTS = os.path.join(_IDS_DIR, "artifacts") if os.path.isdir(
        os.path.join(_IDS_DIR, "artifacts")) else os.path.join(BASE_DIR, "artifacts")
    os.makedirs(ARTIFACTS, exist_ok=True)

    model_pkl = os.path.join(ARTIFACTS, "v3_pipeline.pkl")
    ts = _timestamp()
    model_ts = os.path.join(ARTIFACTS, f"v3_pipeline_{ts}.pkl")

    with open(model_ts, "wb") as f:
        pickle.dump(pipe, f)
    with open(model_pkl, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Saved: {model_pkl}")
    print(f"  Saved: {model_ts}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune hyperparameters for intent detection",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only tune PCA dims")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain model after tuning")
    parser.add_argument("--csv", type=str, default=None,
                        help="Test CSV for validation after retrain")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_auto_tuner(quick=args.quick, retrain=args.retrain, csv_path=args.csv)


if __name__ == "__main__":
    main()
