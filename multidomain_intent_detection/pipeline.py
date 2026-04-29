"""
Multidomain Intent Detection — PCA + Ensemble Pipeline
========================================================

Core ML pipeline:
  1. L2-normalize raw 768-d embeddings
  2. PCA whitening → reduced dims (default 50)
  3. StandardScaler
  4. Calibrated Ensemble:
       - SVM-RBF   (weight 0.40)
       - LogReg    (weight 0.35)
       - kNN       (weight 0.25)
  5. Temperature-scaled soft voting

This module is purely the sklearn pipeline — no I/O, no LLM calls.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Known confusion pairs (intents with high embedding overlap) ─────────────
# When the top-2 predictions are both from the same confusion pair,
# the confidence is penalized even if the ensemble average is high.
CONFUSION_PAIRS: Dict[str, set] = {
    "approval_info":       {"prior_auth_info", "claim_status"},
    "prior_auth_info":     {"approval_info", "pa_summary"},
    "rejection_reasons":   {"help", "claim_status"},
    "help":                {"rejection_reasons"},
    "pricing_info":        {"compound_info", "medicare_part_d", "cob_info"},
    "compound_info":       {"pricing_info"},
    "claim_status":        {"approval_info", "rejection_reasons"},
    "generic_availability": {"Generic", "daw_info"},
    "fill_date_info":      {"rx_details", "audit_info"},
    "member_demographics": {"member_contact_info"},
    "member_contact_info": {"member_demographics"},
}

CONFUSION_PRONE_INTENTS = set(CONFUSION_PAIRS.keys())


class IntentPipeline:
    """PCA → Ensemble (SVM-RBF + LogReg + kNN) with learned temperature scaling."""

    def __init__(self, n_pca: int = 50, knn_k: int = 5, temperature: float = 1.5):
        self.n_pca = n_pca
        self.knn_k = knn_k
        self.temperature = temperature  # learned during fit(); >1 = softer (reduces overconfidence)
        self.pca = None
        self.scaler = None
        self.clfs: Dict = {}
        self.label_names: List[str] = []
        self.weights = {"svm": 0.40, "logreg": 0.35, "knn": 0.25}

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, X_raw: np.ndarray, y: np.ndarray, label_names: List[str]):
        """Train the ensemble on (X_raw, y) with given label names."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier

        self.label_names = label_names

        # L2-normalize → PCA → scale
        X_n = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-10)
        d = min(self.n_pca, X_raw.shape[0] - 1, X_raw.shape[1])
        self.pca = PCA(n_components=d, whiten=True, random_state=42)
        X_p = self.pca.fit_transform(X_n)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X_p)

        var_kept = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA: 768 → {d} dims ({var_kept * 100:.1f}% variance)")

        self.clfs["svm"] = SVC(
            kernel="rbf", C=10, gamma="scale", probability=True,
            class_weight="balanced", random_state=42,
        ).fit(X_s, y)

        self.clfs["logreg"] = LogisticRegression(
            C=10, max_iter=3000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ).fit(X_s, y)

        self.clfs["knn"] = KNeighborsClassifier(
            n_neighbors=min(self.knn_k, X_raw.shape[0] - 1),
            weights="distance", metric="cosine",
        ).fit(X_s, y)

        logger.info("Ensemble ready: SVM-RBF + LogReg + kNN")
        # Learn optimal temperature on held-out data
        self.temperature = self._learn_temperature(X_s, y)
        logger.info(f"Learned temperature: {self.temperature:.3f}")

    # ── Temperature Calibration ─────────────────────────────────────────────

    def _learn_temperature(self, X_scaled: np.ndarray, y: np.ndarray) -> float:
        """Find temperature T that minimizes NLL on held-out data.

        Temperature scaling (Guo et al. 2017) is the simplest post-hoc
        calibration method.  T > 1 softens overconfident predictions so
        that the confidence gate can correctly identify uncertain queries.

        Uses 3-fold CV with LogReg (fastest sub-classifier) to collect
        held-out probabilities, then optimizes T via bounded scalar search.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from scipy.optimize import minimize_scalar

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        all_probs, all_labels = [], []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            lr = LogisticRegression(
                C=10, max_iter=3000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            ).fit(X_tr, y_tr)
            all_probs.append(lr.predict_proba(X_val))
            all_labels.append(y_val)

        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)

        def nll(T):
            """Negative log-likelihood with temperature scaling."""
            scaled = np.log(all_probs + 1e-12) / T
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            softmax = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            correct_probs = softmax[np.arange(len(all_labels)), all_labels]
            return -np.log(correct_probs + 1e-12).mean()

        result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
        # T ≥ 1.0: never sharpen, only soften (sharpening worsens overconfidence)
        return max(round(result.x, 3), 1.0)

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to a probability vector."""
        log_p = np.log(probs + 1e-12) / self.temperature
        log_p -= log_p.max()
        exp_p = np.exp(log_p)
        return exp_p / exp_p.sum()
    # ── Transform ────────────────────────────────────────────────────────

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        return self.scaler.transform(self.pca.transform(X_n))

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability with log-space temperature scaling.

        Uses the same _apply_temperature logic as predict_single() to ensure
        consistency between batch CV evaluation and inference.
        """
        X_f = self._transform(X)
        raw_p = sum(
            clf.predict_proba(X_f) * self.weights[name]
            for name, clf in self.clfs.items()
        )
        # Apply log-space temperature per row (same math as _apply_temperature)
        if self.temperature != 1.0:
            log_p = np.log(raw_p + 1e-12) / self.temperature
            log_p -= log_p.max(axis=1, keepdims=True)
            exp_p = np.exp(log_p)
            return exp_p / (exp_p.sum(axis=1, keepdims=True) + 1e-10)
        return raw_p / (raw_p.sum(axis=1, keepdims=True) + 1e-10)

    def predict_single(self, vec: np.ndarray) -> Dict:
        """Predict intent for a single 768-d vector.

        Applies three layers of calibration to prevent confidently-wrong
        predictions from bypassing the LLM fallback gate:

          1. Learned temperature scaling  — softens overconfident probabilities
             using a data-driven T learned during fit().
          2. Disagreement penalty         — when sub-classifiers disagree,
             confidence is reduced proportionally.
          3. Confusion-pair penalty       — when the top-2 predictions are a
             known confusion pair, an additional penalty is applied.

        Returns:
            {
                "intent":         str,
                "confidence":     float,  # after all calibration layers
                "raw_confidence": float,  # before any calibration
                "margin":         float,
                "top_5":          [(intent, prob), ...],
                "individual":     {"svm": ..., "logreg": ..., "knn": ...},
                "agreement":      bool,
                "is_confusion_pair": bool,
            }
        """
        if vec.ndim == 1:
            if vec.shape[0] != self.pca.n_features_in_:
                raise ValueError(
                    f"Expected {self.pca.n_features_in_}-d vector, got {vec.shape[0]}-d"
                )
        # ── Raw ensemble probabilities (before temperature) ───────────────
        X_f = self._transform(vec.reshape(1, -1))
        raw_p = sum(
            clf.predict_proba(X_f) * self.weights[name]
            for name, clf in self.clfs.items()
        )[0]

        # ── Layer 1: Temperature scaling ───────────────────────────────
        calibrated_p = self._apply_temperature(raw_p)

        idx = np.argsort(calibrated_p)[::-1]
        top5 = [(self.label_names[i], float(calibrated_p[i])) for i in idx[:5]]

        # ── Sub-classifier agreement ──────────────────────────────────
        indiv = {
            name: self.label_names[clf.predict(X_f)[0]]
            for name, clf in self.clfs.items()
        }
        agreement = len(set(indiv.values())) == 1

        raw_confidence = float(raw_p[idx[0]])
        confidence = float(calibrated_p[idx[0]])
        margin = float(calibrated_p[idx[0]] - calibrated_p[idx[1]]) if len(idx) > 1 else 1.0

        # ── Layer 2: Disagreement penalty ──────────────────────────────
        if not agreement:
            n_unique = len(set(indiv.values()))
            penalty = 1.0 - (n_unique - 1) * 0.15  # 2-way=0.85×, 3-way=0.70×
            confidence *= penalty
            margin *= penalty

        # ── Layer 3: Confusion-pair penalty ────────────────────────────
        top1_intent = self.label_names[idx[0]]
        top2_intent = self.label_names[idx[1]] if len(idx) > 1 else ""
        is_confusion_pair = (
            top1_intent in CONFUSION_PAIRS
            and top2_intent in CONFUSION_PAIRS.get(top1_intent, set())
        )
        if is_confusion_pair:
            confidence *= 0.80  # 20% additional penalty
            margin *= 0.80

        return {
            "intent": top1_intent,
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "margin": margin,
            "top_5": top5,
            "individual": indiv,
            "agreement": agreement,
            "is_confusion_pair": is_confusion_pair,
        }

    # ── Cross-validation ────────────────────────────────────────────────

    def cross_validate(self, X_raw: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
        """5-fold stratified CV.  Returns (mean_acc, std_acc, fold_accs)."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs: List[float] = []

        for train_idx, val_idx in cv.split(X_raw, y):
            Xtr, Xva = X_raw[train_idx], X_raw[val_idx]
            ytr, yva = y[train_idx], y[val_idx]

            fold_pipe = IntentPipeline(self.n_pca, self.knn_k, self.temperature)
            fold_pipe.label_names = self.label_names

            Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-10)
            d = min(self.n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
            fold_pipe.pca = PCA(n_components=d, whiten=True, random_state=42)
            Xp = fold_pipe.pca.fit_transform(Xn)
            fold_pipe.scaler = StandardScaler()
            Xs = fold_pipe.scaler.fit_transform(Xp)

            fold_pipe.clfs["svm"] = SVC(
                kernel="rbf", C=10, gamma="scale", probability=True,
                class_weight="balanced", random_state=42,
            ).fit(Xs, ytr)
            fold_pipe.clfs["logreg"] = LogisticRegression(
                C=10, max_iter=3000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            ).fit(Xs, ytr)
            fold_pipe.clfs["knn"] = KNeighborsClassifier(
                n_neighbors=min(self.knn_k, Xtr.shape[0] - 1),
                weights="distance", metric="cosine",
            ).fit(Xs, ytr)

            preds = np.argmax(fold_pipe.predict_proba(Xva), axis=1)
            accs.append(float((preds == yva).mean()))

        return float(np.mean(accs)), float(np.std(accs)), accs
