"""
Multidomain Intent Detection — PCA + Ensemble Pipeline
========================================================

Core ML pipeline:
  1. L2-normalize raw 768-d embeddings
  2. PCA whitening → reduced dims (tuned via CV)
  3. StandardScaler
  4. Calibrated Ensemble of 4 classifiers with LEARNED weights:
       - SVM-RBF        — non-linear decision boundaries
       - LogReg         — well-calibrated probabilities
       - kNN (cosine)   — preserves local decision boundaries
       - ExtraTrees     — captures feature interactions, adds diversity
  5. Learned temperature scaling (Guo et al. 2017)
  6. Disagreement + confusion-pair confidence penalties

Key improvements over hardcoded ensemble:
  - Weights are LEARNED via held-out accuracy optimization (not hardcoded)
  - 4th classifier (ExtraTrees) adds diversity — reduces correlated errors
  - Temperature learned from data, not assumed

This module is purely the sklearn pipeline — no I/O, no LLM calls.
"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Load tuning config (all tunable params in one file) ─────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_config.json")

def _load_tuning_config() -> Dict:
    """Load tuning_config.json.  Returns empty dict if missing."""
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

_TUNING_CONFIG = _load_tuning_config()


# ── Known confusion pairs (intents with high embedding overlap) ─────────────
# Updated based on model accuracy report confusion analysis.
CONFUSION_PAIRS: Dict[str, set] = {
    "approval_info":       {"prior_auth_info", "claim_status"},
    "prior_auth_info":     {"approval_info", "pa_summary", "PriorAuth"},
    "rejection_reasons":   {"help", "claim_status", "pa_suggest_override"},
    "help":                {"rejection_reasons"},
    "pricing_info":        {"compound_info", "medicare_part_d", "cob_info", "Pricing"},
    "compound_info":       {"pricing_info"},
    "claim_status":        {"approval_info", "rejection_reasons", "audit_info", "ClaimNum"},
    "generic_availability": {"Generic", "daw_info", "Brand", "drug_info"},
    "fill_date_info":      {"rx_details", "audit_info", "DrugLast", "date_range_claims"},
    "member_demographics": {"member_contact_info"},
    "member_contact_info": {"member_demographics"},
    "ClaimNum":            {"claim_status", "rx_details", "date_range_claims"},
    "settlement_info":     {"Settlement"},
    "Settlement":          {"settlement_info"},
    "beneficiary_info":    {"approval_info", "member_coverage"},
    "greeting":            {"out_of_scope"},
    "out_of_scope":        {"greeting"},
    # ── From accuracy report confusion analysis ─────────────────────
    "mail_order_info":     {"PharmType"},
    "PharmType":           {"mail_order_info", "Pharmacy", "compound_info"},
    "Prescriber":          {"prescriber_info"},
    "prescriber_info":     {"Prescriber"},
    "plan_summary":        {"member_coverage", "plan_history"},
    "member_coverage":     {"plan_summary", "beneficiary_info"},
    "pa_follow_me_logic":  {"member_transition_status"},
    "member_transition_status": {"pa_follow_me_logic"},
    "pa_field_help":       {"pa_ignore_status", "pa_summary", "pa_reason_code_fields"},
    "pa_agent_code":       {"pa_modification_history"},
    "medicare_part_d":     {"medicare_coverage", "pricing_info", "pa_part_b_override"},
    "DrugLast":            {"fill_date_info", "drug_info"},
    "related_cagm":        {"cvs_id_lookup", "family_members"},
    "Pharmacy":            {"pharmacy_info", "PharmType"},
    "pharmacy_info":       {"Pharmacy"},
    "family_type":         {"beneficiary_info", "family_members"},
    "rx_details":          {"Refills", "fill_date_info", "DaysSupply"},
    "Refills":             {"rx_details", "Status"},
    # ── NEW: Added from 02-06-2026 misclassification analysis ───────
    "PriorAuth":           {"prior_auth_info", "date_range_claims"},
    "multi_claim_summary": {"date_range_claims", "Pharmacy", "Month", "pricing_info", "Plan", "Manufacturer"},
    "date_range_claims":   {"multi_claim_summary", "fill_date_info", "medicare_part_d", "Status"},
    "daw_info":            {"generic_availability", "Brand", "Generic"},
    "Generic":             {"Brand", "daw_info", "date_range_claims"},
    "Brand":               {"Generic", "daw_info", "Settlement", "Prescriber"},
    "NDC":                 {"Settlement", "Status", "medicare_part_d", "date_range_claims"},
    "RejectCode":          {"Status", "rejection_reasons"},
    "DaysSupply":          {"rx_details"},
    "Status":              {"Settlement", "RejectCode", "Refills"},
    # ── NEW: Override domain operational how-to confusion pairs ──────
    "pa_contingent_therapy_override": {"pa_smart_pa_override", "pa_field_help", "pa_override_reject"},
    "pa_smart_pa_override":           {"pa_contingent_therapy_override", "pa_field_help", "prior_auth_info"},
    "pa_part_b_override":             {"pa_reason_code", "pa_esrd_override", "medicare_part_d"},
    "pa_esrd_override":               {"pa_part_b_override", "pa_reason_code", "pa_override_reject"},
    "pa_skip_deductible":             {"pa_copay_pricing", "pa_copay_setup", "pa_field_help"},
    "pa_send_expiration":             {"pa_effective_dates", "pa_field_help"},
    "pa_tf_letter_setup":             {"pa_copay_setup", "pa_reason_code", "approval_info"},
    "pa_copay_setup":                 {"pa_copay_pricing", "pa_skip_deductible", "pa_tf_letter_setup"},
    "pa_suggest_override":            {"pa_override_reject", "pa_reason_code", "rejection_reasons"},
    "pa_reason_code_fields":          {"pa_reason_code", "pa_field_help", "pa_suggest_override"},
}

CONFUSION_PRONE_INTENTS = set(CONFUSION_PAIRS.keys())


class IntentPipeline:
    """PCA → 4-classifier ensemble with learned weights and temperature.

    The ensemble uses 4 diverse classifiers whose weights are optimized
    on held-out data rather than hardcoded. Adding ExtraTrees as a 4th
    voter provides diversity — it uses randomized splits that are
    uncorrelated with SVM/LogReg, reducing correlated errors.
    """

    def __init__(self, n_pca: int = None, knn_k: int = None, temperature: float = 1.0):
        cfg = _TUNING_CONFIG
        pipe_cfg = cfg.get("pipeline", {})
        self.n_pca = n_pca or pipe_cfg.get("pca_dims") or 50
        self.knn_k = knn_k or pipe_cfg.get("knn_k") or 7
        self.temperature = temperature  # learned during fit()
        self.pca = None
        self.scaler = None
        self.clfs: Dict = {}
        self.label_names: List[str] = []
        self.weights: Dict[str, float] = {}  # learned during fit()

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, X_raw: np.ndarray, y: np.ndarray, label_names: List[str],
            **kwargs):
        """Train 4 classifiers, learn optimal weights and temperature."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import ExtraTreesClassifier

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

        # ── 4 diverse classifiers (params from tuning_config.json) ────
        clf_cfg = _TUNING_CONFIG.get("classifiers", {})
        svm_cfg = clf_cfg.get("svm", {})
        lr_cfg = clf_cfg.get("logreg", {})
        knn_cfg = clf_cfg.get("knn", {})
        et_cfg = clf_cfg.get("extra_trees", {})

        self.clfs["svm"] = SVC(
            kernel=svm_cfg.get("kernel", "rbf"),
            C=svm_cfg.get("C", 10),
            gamma=svm_cfg.get("gamma", "scale"),
            probability=True,
            class_weight="balanced", random_state=42,
        ).fit(X_s, y)

        self.clfs["logreg"] = LogisticRegression(
            C=lr_cfg.get("C", 10),
            max_iter=lr_cfg.get("max_iter", 3000),
            solver=lr_cfg.get("solver", "lbfgs"),
            class_weight="balanced", random_state=42,
        ).fit(X_s, y)

        self.clfs["knn"] = KNeighborsClassifier(
            n_neighbors=min(self.knn_k, X_raw.shape[0] - 1),
            weights=knn_cfg.get("weights", "distance"),
            metric=knn_cfg.get("metric", "cosine"),
        ).fit(X_s, y)

        self.clfs["et"] = ExtraTreesClassifier(
            n_estimators=et_cfg.get("n_estimators", 300),
            max_depth=et_cfg.get("max_depth", 30),
            min_samples_leaf=et_cfg.get("min_samples_leaf", 2),
            class_weight="balanced", random_state=42, n_jobs=-1,
        ).fit(X_s, y)

        logger.info("Classifiers: SVM-RBF(C=%s), LogReg(C=%s), kNN(k=%d), ExtraTrees(%s)"
                    % (svm_cfg.get('C', 10), lr_cfg.get('C', 10), self.knn_k,
                       et_cfg.get('n_estimators', 300)))

        # ── Learn optimal ensemble weights ───────────────────────────
        self.weights = self._learn_weights(X_s, y)
        logger.info(f"Learned weights: {self.weights}")

        # ── Learn optimal temperature ────────────────────────────────
        self.temperature = self._learn_temperature(X_s, y)
        logger.info(f"Learned temperature: {self.temperature:.3f}")

    # ── Weight Learning ────────────────────────────────────────────────

    def _learn_weights(self, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Learn optimal ensemble weights by maximizing held-out accuracy.

        Uses 3-fold CV: for each fold, each classifier type predicts held-out
        probabilities. Then a grid search over weight combinations finds the
        mix that maximizes accuracy on the held-out predictions.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import ExtraTreesClassifier

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf_names = list(self.clfs.keys())

        # Load classifier config (same params used in fit())
        clf_cfg = _TUNING_CONFIG.get("classifiers", {})
        svm_cfg = clf_cfg.get("svm", {})
        lr_cfg = clf_cfg.get("logreg", {})
        knn_cfg = clf_cfg.get("knn", {})
        et_cfg = clf_cfg.get("extra_trees", {})

        all_probs = {name: [] for name in clf_names}
        all_labels = []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            all_labels.append(y_val)

            fold_clfs = {
                "svm": SVC(kernel=svm_cfg.get("kernel", "rbf"),
                           C=svm_cfg.get("C", 10), gamma=svm_cfg.get("gamma", "scale"),
                           probability=True,
                           class_weight="balanced", random_state=42).fit(X_tr, y_tr),
                "logreg": LogisticRegression(C=lr_cfg.get("C", 10),
                                             max_iter=lr_cfg.get("max_iter", 3000),
                                             solver=lr_cfg.get("solver", "lbfgs"),
                                             class_weight="balanced", random_state=42).fit(X_tr, y_tr),
                "knn": KNeighborsClassifier(n_neighbors=min(self.knn_k, X_tr.shape[0]-1),
                                            weights=knn_cfg.get("weights", "distance"),
                                            metric=knn_cfg.get("metric", "cosine")).fit(X_tr, y_tr),
                "et": ExtraTreesClassifier(n_estimators=et_cfg.get("n_estimators", 300),
                                           max_depth=et_cfg.get("max_depth", 30),
                                           min_samples_leaf=et_cfg.get("min_samples_leaf", 2),
                                           class_weight="balanced", random_state=42,
                                           n_jobs=-1).fit(X_tr, y_tr),
            }
            for name in clf_names:
                all_probs[name].append(fold_clfs[name].predict_proba(X_val))

        for name in clf_names:
            all_probs[name] = np.vstack(all_probs[name])
        all_labels_arr = np.concatenate(all_labels)

        # Grid search over weight combinations (step=0.05 on 4-simplex)
        best_acc, best_w = 0.0, None
        step = 0.05
        for w1 in np.arange(0.10, 0.55, step):
            for w2 in np.arange(0.10, 0.55, step):
                for w3 in np.arange(0.05, 0.40, step):
                    w4 = round(1.0 - w1 - w2 - w3, 2)
                    if w4 < 0.05 or w4 > 0.40:
                        continue
                    w = {"svm": w1, "logreg": w2, "knn": w3, "et": w4}
                    avg_p = sum(all_probs[n] * w[n] for n in clf_names)
                    acc = (np.argmax(avg_p, axis=1) == all_labels_arr).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_w = {k: round(v, 2) for k, v in w.items()}

        logger.info(f"Weight search: best CV accuracy = {best_acc * 100:.2f}%")
        return best_w or {"svm": 0.30, "logreg": 0.25, "knn": 0.20, "et": 0.25}

    # ── Temperature Calibration ─────────────────────────────────────────

    def _learn_temperature(self, X_scaled: np.ndarray, y: np.ndarray) -> float:
        """Find temperature T that minimizes NLL on held-out data.

        Temperature scaling (Guo et al. 2017) is a simple post-hoc
        calibration method.  T > 1 softens, T < 1 sharpens.

        IMPORTANT: We learn T on the FULL ENSEMBLE predictions (not just
        one classifier) so the temperature matches the actual prediction
        pipeline output.  With 89 intents, softening (T > 1) often
        crushes confidence too aggressively, so we allow T < 1 for
        sharpening when the ensemble is already well-calibrated.

        Uses 3-fold CV with all 4 classifiers to collect held-out
        ensemble probabilities, then optimizes T via bounded search.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from scipy.optimize import minimize_scalar

        clf_cfg = _TUNING_CONFIG.get("classifiers", {})
        svm_cfg = clf_cfg.get("svm", {})
        lr_cfg = clf_cfg.get("logreg", {})
        knn_cfg = clf_cfg.get("knn", {})
        et_cfg = clf_cfg.get("extra_trees", {})
        pipe_cfg = _TUNING_CONFIG.get("pipeline", {})

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        all_probs, all_labels = [], []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Fit all 4 classifiers on this fold (using config params)
            fold_clfs = {
                "svm": SVC(kernel=svm_cfg.get("kernel", "rbf"),
                           C=svm_cfg.get("C", 10), gamma=svm_cfg.get("gamma", "scale"),
                           probability=True,
                           class_weight="balanced", random_state=42).fit(X_tr, y_tr),
                "logreg": LogisticRegression(C=lr_cfg.get("C", 10),
                                             max_iter=lr_cfg.get("max_iter", 3000),
                                             solver=lr_cfg.get("solver", "lbfgs"),
                                             class_weight="balanced", random_state=42).fit(X_tr, y_tr),
                "knn": KNeighborsClassifier(n_neighbors=min(self.knn_k, X_tr.shape[0]-1),
                                            weights=knn_cfg.get("weights", "distance"),
                                            metric=knn_cfg.get("metric", "cosine")).fit(X_tr, y_tr),
                "et": ExtraTreesClassifier(n_estimators=et_cfg.get("n_estimators", 300),
                                           max_depth=et_cfg.get("max_depth", 30),
                                           min_samples_leaf=et_cfg.get("min_samples_leaf", 2),
                                           class_weight="balanced", random_state=42,
                                           n_jobs=-1).fit(X_tr, y_tr),
            }

            # Weighted ensemble probability (same as predict_proba)
            fold_probs = sum(
                fold_clfs[name].predict_proba(X_val) * self.weights[name]
                for name in self.weights
            )
            all_probs.append(fold_probs)
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

        result = minimize_scalar(nll, bounds=(
            pipe_cfg.get("temperature_min", 0.3),
            pipe_cfg.get("temperature_max", 4.0),
        ), method="bounded")
        return max(round(result.x, 3), pipe_cfg.get("temperature_floor", 0.3))

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to a probability vector."""
        log_p = np.log(probs + 1e-12) / self.temperature
        log_p -= log_p.max()
        exp_p = np.exp(log_p)
        return exp_p / exp_p.sum()

    # ── Transform & Predict ──────────────────────────────────────────────

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        return self.scaler.transform(self.pca.transform(X_n))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability with log-space temperature scaling."""
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

    def predict_single(self, vec: np.ndarray, **kwargs) -> Dict:
        """Predict intent for a single 768-d vector.

        Applies three layers of calibration:
          1. Learned temperature scaling
          2. Disagreement penalty (scales with number of unique predictions)
          3. Confusion-pair penalty (0.80×)
        """
        if vec.ndim == 1 and hasattr(self.pca, 'n_features_in_'):
            if vec.shape[0] != self.pca.n_features_in_:
                raise ValueError(
                    f"Expected {self.pca.n_features_in_}-d vector, got {vec.shape[0]}-d"
                )

        # ── Raw ensemble probabilities ────────────────────────────────
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

        # ── Layer 2: Disagreement penalty (configurable) ──────────────
        pen_cfg = _TUNING_CONFIG.get("confidence_penalties", {})
        disagree_per = pen_cfg.get("disagreement_per_unique", 0.08)
        disagree_max = pen_cfg.get("disagreement_max_penalty", 0.30)
        if not agreement:
            n_unique = len(set(indiv.values()))
            penalty = max(1.0 - disagree_max, 1.0 - (n_unique - 1) * disagree_per)
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
            confpair_pen = pen_cfg.get("confusion_pair_penalty", 0.10)
            confidence *= (1.0 - confpair_pen)
            margin *= (1.0 - confpair_pen)

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
        from sklearn.ensemble import ExtraTreesClassifier

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

            # Use config-driven classifier params
            cv_clf_cfg = _TUNING_CONFIG.get("classifiers", {})
            cv_svm = cv_clf_cfg.get("svm", {})
            cv_lr = cv_clf_cfg.get("logreg", {})
            cv_knn = cv_clf_cfg.get("knn", {})
            cv_et = cv_clf_cfg.get("extra_trees", {})

            fold_pipe.clfs["svm"] = SVC(
                kernel=cv_svm.get("kernel", "rbf"),
                C=cv_svm.get("C", 10),
                gamma=cv_svm.get("gamma", "scale"),
                probability=True,
                class_weight="balanced", random_state=42,
            ).fit(Xs, ytr)
            fold_pipe.clfs["logreg"] = LogisticRegression(
                C=cv_lr.get("C", 10),
                max_iter=cv_lr.get("max_iter", 3000),
                solver=cv_lr.get("solver", "lbfgs"),
                class_weight="balanced", random_state=42,
            ).fit(Xs, ytr)
            fold_pipe.clfs["knn"] = KNeighborsClassifier(
                n_neighbors=min(self.knn_k, Xtr.shape[0] - 1),
                weights=cv_knn.get("weights", "distance"),
                metric=cv_knn.get("metric", "cosine"),
            ).fit(Xs, ytr)
            fold_pipe.clfs["et"] = ExtraTreesClassifier(
                n_estimators=cv_et.get("n_estimators", 300),
                max_depth=cv_et.get("max_depth", 30),
                min_samples_leaf=cv_et.get("min_samples_leaf", 2),
                class_weight="balanced", random_state=42, n_jobs=-1,
            ).fit(Xs, ytr)

            # Equal weights for CV estimation
            fold_pipe.weights = {"svm": 0.25, "logreg": 0.25, "knn": 0.25, "et": 0.25}
            preds = np.argmax(fold_pipe.predict_proba(Xva), axis=1)
            accs.append(float((preds == yva).mean()))

        return float(np.mean(accs)), float(np.std(accs)), accs
