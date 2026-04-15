"""
NLIDomainRouter (End-to-End, Production Ready)

High-confidence intent/domain routing using MNLI-style Natural Language Inference.
Uses raw entailment logits from a DeBERTa MNLI-trained model.

Key properties:
- No zero-shot pipeline
- Raw entailment logits (no softmax compression)
- Multi-hypothesis per domain
- Verb-bias mitigation (retrieve/show vs explain/why)
- Margin-based confidence
- Explicit audit_info separation
"""

import os
import re
import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# NLI Domain Router
# ---------------------------------------------------------------------

class NLIDomainRouter:
    """
    Domain router powered by MNLI Natural Language Inference.
    """

    # DeBERTa MNLI label mapping
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2

    def __init__(
        self,
        domain_hypotheses: Dict[str, List[str]],
        model_dir: str,
        device: Optional[int] = None,
        max_length: int = 256,
    ):
        self.domain_hypotheses = domain_hypotheses
        self.model_dir = model_dir
        self.max_length = max_length

        # Device selection
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self._load_model()

    # -----------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------

    def _load_model(self):
        logger.info(f"Loading NLI model from: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        if self.device >= 0:
            self.model.to(self.device)

        self.model.eval()
        logger.info("NLI model loaded successfully")

    # -----------------------------------------------------------------
    # Core NLI Scoring
    # -----------------------------------------------------------------

    def _entailment_logit(self, premise: str, hypothesis: str) -> float:
        """
        Compute raw entailment logit (MNLI ENTAILMENT index).
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        if self.device >= 0:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits[0, self.ENTAILMENT].item()

    # -----------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------

    def route(self, query: str) -> Dict:
        """
        Route a single query.

        Returns dict with:
        - domain
        - score
        - all_scores
        - margin
        - confidence
        - per-hypothesis scores
        """
        domain_scores = {}
        details = {}

        for domain, hypotheses in self.domain_hypotheses.items():
            scored = []
            for h in hypotheses:
                logit = self._entailment_logit(query, h)
                scored.append(logit)

            best = max(scored)
            domain_scores[domain] = best
            details[domain] = [
                {"hypothesis": h, "score": round(s, 3)}
                for h, s in zip(hypotheses, scored)
            ]

        ranked = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        top_domain, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")

        margin = top_score - second_score
        confidence = self._confidence_level(top_score, margin)

        return {
            "domain": top_domain,
            "score": round(top_score, 3),
            "all_scores": {k: round(v, 3) for k, v in domain_scores.items()},
            "margin": round(margin, 3),
            "confidence": confidence,
            "details": details,
        }

    def batch_route(self, queries: List[str]) -> List[Dict]:
        return [self.route(q) for q in queries]

    # -----------------------------------------------------------------
    # Confidence Calibration
    # -----------------------------------------------------------------

    @staticmethod
    def _confidence_level(score: float, margin: float) -> str:
        """
        Confidence thresholds tuned empirically for MNLI logits.
        """
        if score >= 4.5 and margin >= 1.5:
            return "high"
        elif score >= 2.0:
            return "medium"
        return "low"


# ---------------------------------------------------------------------
# Query Preprocessing
# ---------------------------------------------------------------------

def preprocess_query(query: str) -> str:
    """
    Remove long numeric IDs and normalize whitespace.
    """
    query = re.sub(r"\b\d{6,}\b", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


# ---------------------------------------------------------------------
# Example / CLI Runner
# ---------------------------------------------------------------------

if __name__ == "__main__":

    DOMAIN_HYPOTHESES = {

        # -------------------------------------------------------------
        # CAP API – explanation, reasoning, decision logic
        # -------------------------------------------------------------
        "cap_api": [
            "The user is requesting an explanation or reasoning for how or why a specific claim was adjudicated, approved, denied, or processed, not simply retrieving claim data.",
            "The user is asking to understand transition fill logic, such as why TF was applied, its type, eligibility, or configuration for a specific claim.",
            "The user is asking which plan overrides, rules, or edits influenced the approval or denial of a specific claim.",
            "The user is asking which configuration, benefit rules, or processing setup influenced how a specific claim was adjudicated.",
            "The user is asking to understand the adjudication pathway or decision logic used to process a claim, not to search claim records.",
            "The user is analyzing a single claim decision rather than searching or listing claims."
        ],

        # -------------------------------------------------------------
        # Claims Search – listing / finding multiple claims
        # -------------------------------------------------------------
        "claims_search": [
            "The user is searching for multiple claims or lists of claim records.",
            "The user is requesting claim records across date ranges, history, or multiple claim results.",
            "The user wants to find or browse claims rather than understand why a claim was processed a certain way."
        ],

        # -------------------------------------------------------------
        # Audit Information – logs / change history
        # -------------------------------------------------------------
        "audit_info": [
            "The user is requesting audit logs, audit trails, change history, or modification records for a claim.",
            "The user wants to see historical audit information rather than an explanation of adjudication logic."
        ],
    }

    MODEL_DIR = r"C:\ProjectData\POC-Flow-1\local_model"

    router = NLIDomainRouter(
        domain_hypotheses=DOMAIN_HYPOTHESES,
        model_dir=MODEL_DIR,
    )

    print("Type a user prompt (empty to exit):")

    while True:
        try:
            user_input = input("User prompt: ").strip()
            if not user_input:
                break

            cleaned = preprocess_query(user_input)
            result = router.route(cleaned)

            print("\nCleaned Query:", cleaned)
            print("Domain:", result["domain"])
            print("Score:", result["score"])
            print("Margin:", result["margin"])
            print("Confidence:", result["confidence"])
            print("All Scores:", result["all_scores"])
            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break