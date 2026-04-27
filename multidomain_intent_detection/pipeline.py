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


class IntentPipeline:
    """PCA → Ensemble (SVM-RBF + LogReg + kNN) with temperature scaling."""

    def __init__(self, n_pca: int = 50, knn_k: int = 5, temperature: float = 0.3):
        self.n_pca = n_pca
        self.knn_k = knn_k
        self.temperature = temperature  # <1 = sharper, >1 = softer
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

    # ── Transform ────────────────────────────────────────────────────────

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        return self.scaler.transform(self.pca.transform(X_n))

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability with temperature scaling."""
        X_f = self._transform(X)
        p = sum(
            clf.predict_proba(X_f) * self.weights[name]
            for name, clf in self.clfs.items()
        )
        if self.temperature != 1.0:
            p = np.power(p + 1e-10, 1.0 / self.temperature)
        return p / (p.sum(axis=1, keepdims=True) + 1e-10)

    def predict_single(self, vec: np.ndarray) -> Dict:
        """Predict intent for a single 768-d vector.

        Returns:
            {
                "intent":      str,
                "confidence":  float,
                "margin":      float,
                "top_5":       [(intent, prob), ...],
                "individual":  {"svm": ..., "logreg": ..., "knn": ...},
                "agreement":   bool,
            }
        """
        p = self.predict_proba(vec.reshape(1, -1))[0]
        idx = np.argsort(p)[::-1]
        top5 = [(self.label_names[i], float(p[i])) for i in idx[:5]]

        X_f = self._transform(vec.reshape(1, -1))
        indiv = {
            name: self.label_names[clf.predict(X_f)[0]]
            for name, clf in self.clfs.items()
        }

        return {
            "intent": self.label_names[idx[0]],
            "confidence": float(p[idx[0]]),
            "margin": float(p[idx[0]] - p[idx[1]]) if len(idx) > 1 else 1.0,
            "top_5": top5,
            "individual": indiv,
            "agreement": len(set(indiv.values())) == 1,
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
