"""
Intent Classifier v3 — Production Module
=========================================

Loads the pre-trained PCA + Ensemble pipeline from intent_detection_v3.py
and provides a simple API for classifying user prompts into intents and domains.

Usage (sync):
    from Intent_detection_system.intent_classifier_v3 import get_classifier
    
    classifier = get_classifier()
    result = classifier.classify("What is the copay on claim 132435151040074 sequence 001?")
    
    print(result["intent"])      # "pricing_info"
    print(result["domain"])      # "cap_api"
    print(result["confidence"])  # 0.97
    print(result["source"])      # "ensemble" or "llm"

Usage (async):
    result = await classifier.classify_async("Show all rejected claims for this member")

Architecture:
    1. Normalize query (strip claim numbers for embedding focus)
    2. Embed with Vertex AI text-embedding-005
    3. Predict with PCA + Ensemble (SVM-RBF / LogReg / kNN)
    4. If confidence < threshold → LLM fallback (Gemini Flash)
    5. Return intent, domain, confidence, entities, metadata
"""

import os
import re
import json
import sys
import time
import pickle
import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")
MODEL_PKL = os.path.join(ARTIFACTS, "v3_pipeline.pkl")


# ── Query Normalizer (same as intent_detection_v3.py) ───────────────────────
_CLAIM_NUM_PATTERN = re.compile(r'\b\d{12,18}\b')
_SEQ_PATTERN = re.compile(r'\bsequence\s+\d{1,3}\b', re.IGNORECASE)
_SEQ_NUM = re.compile(r'\bseq\s+\d{1,3}\b', re.IGNORECASE)
_PA_NUM_PATTERN = re.compile(r'\bPA\s+[A-Z0-9]{5,15}\b', re.IGNORECASE)
_WHITESPACE = re.compile(r'\s+')

# ── Known confusion pairs ────────────────────────────────────────────────
_CONFUSION_PRONE_INTENTS = {
    "approval_info", "prior_auth_info", "rejection_reasons", "help",
    "pricing_info", "compound_info", "claim_status", "generic_availability",
    "fill_date_info", "member_demographics", "member_contact_info",
}

# Entity extraction patterns
_ENTITY_CLAIM_NUM = re.compile(r'\b(\d{15})\b')
_ENTITY_SEQ_NUM = re.compile(r'\bsequence\s+(\d{1,3})\b', re.IGNORECASE)
_ENTITY_NPI = re.compile(r'\bNPI\s+(\d{10})\b', re.IGNORECASE)
_ENTITY_NDC = re.compile(r'\bNDC\s+([\d-]{10,13})\b', re.IGNORECASE)
_ENTITY_MEMBER_ID = re.compile(r'\bmember\s+(?:ID\s+)?(\d{6,12})\b', re.IGNORECASE)


def _normalize_query(text: str) -> str:
    """Strip claim/sequence/PA numbers so embedding focuses on intent semantics."""
    t = text.lower().strip()
    t = _SEQ_PATTERN.sub('', t)
    t = _SEQ_NUM.sub('', t)
    t = _CLAIM_NUM_PATTERN.sub('', t)
    t = _PA_NUM_PATTERN.sub('pa', t)
    t = t.replace('.', ' ').replace('?', ' ').replace('!', ' ')
    t = _WHITESPACE.sub(' ', t).strip()
    return t


def _extract_entities(text: str) -> Dict[str, Optional[str]]:
    """Extract structured entities from the raw query text."""
    entities = {
        "claim_number": None,
        "sequence_number": None,
        "npi": None,
        "ndc": None,
        "member_id": None,
    }
    m = _ENTITY_CLAIM_NUM.search(text)
    if m:
        entities["claim_number"] = m.group(1)
    m = _ENTITY_SEQ_NUM.search(text)
    if m:
        entities["sequence_number"] = m.group(1).zfill(3)
    m = _ENTITY_NPI.search(text)
    if m:
        entities["npi"] = m.group(1)
    m = _ENTITY_NDC.search(text)
    if m:
        entities["ndc"] = m.group(1)
    m = _ENTITY_MEMBER_ID.search(text)
    if m:
        entities["member_id"] = m.group(1)
    # Strip None values for cleaner output
    return {k: v for k, v in entities.items() if v is not None}


# ── Intent → Domain mapping ─────────────────────────────────────────────────
INTENT_TO_DOMAIN = {
    # cap_api (single-claim operations)
    "claim_status": "cap_api", "multi_claim_summary": "cap_api",
    "pharmacy_info": "cap_api", "prescriber_info": "cap_api",
    "pricing_info": "cap_api", "reimbursement_info": "cap_api",
    "rejection_reasons": "cap_api", "settlement_info": "cap_api",
    "rx_details": "cap_api", "reversal_info": "cap_api",
    "cob_info": "cap_api", "generic_availability": "cap_api",
    "daw_info": "cap_api", "government_claim_type": "cap_api",
    "mail_order_info": "cap_api", "medicare_part_d": "cap_api",
    "network_info": "cap_api", "prior_auth_info": "cap_api",
    # benefits_api
    "approval_info": "benefits_api", "audit_info": "benefits_api",
    "beneficiary_info": "benefits_api", "plan_summary": "benefits_api",
    "plan_history": "benefits_api", "plan_finder": "benefits_api",
    # claim_history_search
    "compound_info": "claim_history_search", "date_range_claims": "claim_history_search",
    "drug_info": "claim_history_search", "drug_interaction_info": "claim_history_search",
    "fill_date_info": "claim_history_search",
    "Refills": "claim_history_search", "DaysSupply": "claim_history_search",
    "PriorAuth": "claim_history_search", "Diagnosis": "claim_history_search",
    "Settlement": "claim_history_search", "PharmType": "claim_history_search",
    "Plan": "claim_history_search", "Pharmacy": "claim_history_search",
    "Prescriber": "claim_history_search", "Pricing": "claim_history_search",
    "Status": "claim_history_search", "RejectCode": "claim_history_search",
    "DrugLast": "claim_history_search", "Month": "claim_history_search",
    "ClaimNum": "claim_history_search", "NDC": "claim_history_search",
    "Manufacturer": "claim_history_search", "Generic": "claim_history_search",
    "Brand": "claim_history_search",
    # general
    "greeting": "general", "help": "general", "out_of_scope": "general",
    # member_domain
    "member_coverage": "member_domain", "member_hierarchy": "member_domain",
    "benefit_reset_date": "member_domain", "family_type": "member_domain",
    "family_members": "member_domain", "alternate_insurance": "member_domain",
    "medicare_coverage": "member_domain", "lics_status": "member_domain",
    "stcob_linkage": "member_domain", "cvs_id_lookup": "member_domain",
    "related_cagm": "member_domain", "alternate_ids": "member_domain",
    "member_demographics": "member_domain", "member_contact_info": "member_domain",
    "member_eligibility_copay": "member_domain", "member_transition_status": "member_domain",
    "member_dur_config": "member_domain", "member_mbi_number": "member_domain",
    "member_caretaker_info": "member_domain", "member_language_pref": "member_domain",
    "member_discount_program": "member_domain", "member_override_plan": "member_domain",
    # override_domain
    "pa_summary": "override_domain", "pa_override_reject": "override_domain",
    "pa_field_help": "override_domain", "pa_copay_pricing": "override_domain",
    "pa_drug_coverage": "override_domain", "pa_claim_usage": "override_domain",
    "pa_reason_code": "override_domain", "pa_effective_dates": "override_domain",
    "pa_agent_code": "override_domain", "pa_ignore_status": "override_domain",
    "pa_specialty_rx_override": "override_domain", "pa_clinical_admin_code": "override_domain",
    "pa_transform_care": "override_domain", "pa_follow_me_logic": "override_domain",
    "pa_drug_type_indicator": "override_domain", "pa_modification_history": "override_domain",
}

# Domain → API endpoint mapping
DOMAIN_ENDPOINTS = {
    "cap_api": "/myclaims/claims/v1/claim/byclaimnumber",
    "benefits_api": "/myclaims/benefits/v1/member",
    "claim_history_search": "/myclaims/claims/v1/claim/history",
    "member_domain": "/myclaims/members/v1/member",
    "override_domain": "/myclaims/overrides/v1/pa",
    "general": None,
}

# Domain → friendly name
DOMAIN_NAMES = {
    "cap_api": "Cap-API",
    "benefits_api": "Benefits API",
    "claim_history_search": "Claim History Search",
    "member_domain": "Member Domain",
    "override_domain": "Override Domain",
    "general": "General",
}

# Intent descriptions (for LLM fallback context)
INTENT_DESC = {
    "claim_status": "General claim status, adjudication outcome, paid/rejected/pending",
    "multi_claim_summary": "Summary of ALL/MULTIPLE claims for a member",
    "pharmacy_info": "Dispensing pharmacy name, location, NCPDP for ONE claim",
    "prescriber_info": "Prescribing physician/doctor name, NPI for ONE claim",
    "pricing_info": "Copay, ingredient cost, dispensing fee for ONE claim",
    "reimbursement_info": "Amount paid TO pharmacy, reimbursement rationale",
    "rejection_reasons": "Rejection codes, failed edits, denial reasons, how to resolve",
    "settlement_info": "Settlement codes, pharmacy response codes for ONE claim",
    "rx_details": "RX number, fill number, quantity, days supply, strength",
    "reversal_info": "Claim reversal, R&R, manual adjustments, resubmission",
    "cob_info": "Coordination of Benefits, other insurance, dual coverage",
    "generic_availability": "Generic alternatives, therapeutic equivalents",
    "daw_info": "DAW status, brand vs generic requirement",
    "government_claim_type": "Medicare/Medicaid claim type, government program",
    "mail_order_info": "Mail order/home delivery prescription",
    "medicare_part_d": "Medicare Part D summary, PDE, MEDD pricing",
    "network_info": "Pharmacy network details, payment network",
    "prior_auth_info": "Prior authorization status for ONE claim",
    "approval_info": "Claim approval, plan overrides, transition fill (TF)",
    "audit_info": "Audit trail, change history, modification timestamps",
    "beneficiary_info": "Member benefit phase, coverage tier, accumulations",
    "plan_summary": "Benefit plan overview, active plan snapshot",
    "plan_history": "Plan change log, revision history",
    "plan_finder": "Search for available benefit plans",
    "compound_info": "Compound medication, MIC breakdown, ingredient costs",
    "date_range_claims": "Claims within date range, deductible/accumulation history",
    "drug_info": "Drug name, NDC, GPI, therapeutic class, formulary status",
    "drug_interaction_info": "DUR edits, drug interaction alerts",
    "fill_date_info": "Date prescription was filled, service date",
    "Refills": "Search claims by refill count, remaining refills",
    "DaysSupply": "Filter claims by days supply (30, 60, 90 days)",
    "PriorAuth": "Search claims that required prior authorization",
    "Diagnosis": "Filter claims by ICD-10 diagnosis code",
    "Settlement": "Filter claims by settlement code NUMBER",
    "PharmType": "Filter claims by pharmacy type (retail, mail, specialty)",
    "Plan": "Filter claims by insurance plan code",
    "Pharmacy": "Search claims from a specific pharmacy",
    "Prescriber": "Search claims by prescriber name or NPI",
    "Pricing": "Search pricing across MULTIPLE claims for a drug",
    "Status": "Filter claims by status (paid, rejected, pending)",
    "RejectCode": "Search claims by NCPDP rejection code",
    "DrugLast": "When was a drug last dispensed for a member",
    "Month": "Filter claims by calendar month",
    "ClaimNum": "Look up a specific claim by claim number",
    "NDC": "Search claims by NDC number",
    "Manufacturer": "Filter claims by drug manufacturer",
    "Generic": "Filter for generic drug claims only",
    "Brand": "Filter for brand name drug claims only",
    "greeting": "Hello, hi, welcome",
    "help": "How to submit claims, filing guidance",
    "out_of_scope": "Unrelated to pharmacy — weather, recipes, etc.",
    "member_coverage": "Coverage eligibility windows, enrollment dates",
    "member_hierarchy": "Client/CAG hierarchy, organizational structure",
    "benefit_reset_date": "Benefit year reset date, accumulator reset",
    "family_type": "Individual vs family plan, coverage tier type",
    "family_members": "List family members, dependents on same plan",
    "alternate_insurance": "Other/secondary insurance, dual coverage",
    "medicare_coverage": "Medicare Part D enrollment for a MEMBER",
    "lics_status": "Low Income Subsidy (LICS/LIS) status/level",
    "stcob_linkage": "Short-term COB linkage, STCOB links",
    "cvs_id_lookup": "CVS ID for the member",
    "related_cagm": "Related CAGMs by CVS ID or family ID",
    "alternate_ids": "All alternate IDs on file for the member",
    "member_demographics": "Member name, DOB, gender, person code, relationship code",
    "member_contact_info": "Member email, phone, mailing/postal address",
    "member_eligibility_copay": "Copay fields: copayBrand, copayGeneric, copay3, copay4",
    "member_transition_status": "Member transition fill status and start date",
    "member_dur_config": "DUR review key and process flag configuration",
    "member_mbi_number": "Medicare Beneficiary Identifier (MBI) number",
    "member_caretaker_info": "Caretaker name and address from Part D",
    "member_language_pref": "Member language code/preference",
    "member_discount_program": "Discount program type for the member",
    "member_override_plan": "Member override plan ID from eligibility",
    "pa_summary": "PA summary, key fields, configuration overview",
    "pa_override_reject": "Will PA override reject codes 75/70/76",
    "pa_field_help": "What a specific PA field does (documentation)",
    "pa_copay_pricing": "PA copay override impact on pricing",
    "pa_drug_coverage": "Drugs covered by this PA (GPI/NDC lists)",
    "pa_claim_usage": "How many claims used this PA",
    "pa_reason_code": "PA reason code (U1, LC, OD, OA, US, U3)",
    "pa_effective_dates": "PA effective begin/end dates, expiration",
    "pa_agent_code": "Agent/source code on PA (A, C, 3, H)",
    "pa_ignore_status": "Ignore status code (Y, P, 3)",
    "pa_specialty_rx_override": "Specialty Rx reject override indicator",
    "pa_clinical_admin_code": "Clinical administration code (A, C, blank)",
    "pa_transform_care": "Transform care type on PA",
    "pa_follow_me_logic": "Follow me logic indicator on PA",
    "pa_drug_type_indicator": "Authorized drug type (G=GPI, N=NDC)",
    "pa_modification_history": "PA last modified date/time",
}


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING CLIENT (singleton, lazy-loaded)
# ═════════════════════════════════════════════════════════════════════════════

class _VertexEmbedder:
    """Thin wrapper around Vertex AI text-embedding-005."""

    def __init__(self):
        self.project = os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc")
        self.location = os.getenv("LOCATION", "us-central1")
        from google import genai
        self.client = genai.Client(
            vertexai=True, project=self.project, location=self.location,
        )
        logger.info(f"Vertex AI embedder initialized (project={self.project})")

    def embed(self, text: str) -> List[float]:
        """Embed a single text string → 768-dim vector."""
        from google.genai import types
        backoff = 2.0
        for attempt in range(5):
            try:
                r = self.client.models.embed_content(
                    model="text-embedding-005",
                    contents=[types.Part.from_text(text=text)],
                )
                return r.embeddings[0].values
            except Exception as e:
                if any(k in str(e).lower() for k in ("429", "exhausted", "quota")) and attempt < 4:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise


_embedder_instance = None


def _get_embedder() -> _VertexEmbedder:
    """Singleton embedder — only one gRPC connection per process."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = _VertexEmbedder()
    return _embedder_instance


# ═════════════════════════════════════════════════════════════════════════════
# LLM FALLBACK (Gemini Flash for low-confidence queries)
# ═════════════════════════════════════════════════════════════════════════════

def _llm_classify(query: str, candidates: List[str],
                  ensemble_intent: str = None,
                  ensemble_confidence: float = 0.0) -> str:
    """LLM fallback — sends top-5 candidates to Gemini for disambiguation."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc"),
        location=os.getenv("LOCATION", "us-central1"),
    )

    candidate_desc = "\n".join(
        f"- {n}: {INTENT_DESC.get(n, n)}" for n in candidates
    )

    system_instruction = f"""You are an intent classification system for a Pharmacy Benefit Manager (PBM).
Classify the user query into exactly ONE of these candidate intents:

{candidate_desc}

KEY RULES:
- cap_api intents = details about ONE specific claim (references "this claim" or a claim number)
- claim_history_search intents = SEARCH/FILTER across MULTIPLE claims
- member_domain intents = member demographics, eligibility, coverage
- override_domain intents = PA management, override analysis
- benefits_api intents = benefit plan summary, history, finder

OUTPUT FORMAT (JSON only):
{{"intent": "<one of the candidates>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}
"""

    user_prompt = f"Query: {query}\n"
    if ensemble_intent:
        user_prompt += (
            f"Note: Primary classifier suggested '{ensemble_intent}' "
            f"({ensemble_confidence:.0%} confidence) but was uncertain.\n"
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
                system_instruction=system_instruction,
            ),
        )
        text = response.text.strip()
        # Strip markdown fences
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', text, re.DOTALL)
        if json_match:
            llm_result = json.loads(json_match.group(0))
        else:
            llm_result = json.loads(text)

        predicted = llm_result.get("intent", "")

        # Validate against candidates (case-insensitive)
        for c in candidates:
            if c.lower() == predicted.lower():
                return c
        for c in candidates:
            if c.lower() in predicted.lower() or predicted.lower() in c.lower():
                return c
        if predicted in INTENT_TO_DOMAIN:
            return predicted

        logger.warning(f"LLM returned unknown intent '{predicted}', using ensemble pick")
        return candidates[0]

    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return candidates[0]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CLASSIFIER CLASS
# ═════════════════════════════════════════════════════════════════════════════

class IntentClassifierV3:
    """
    Production intent classifier using the v3 PCA + Ensemble pipeline.
    
    Loads a pre-trained pipeline (v3_pipeline.pkl) and provides:
    - classify(query) → sync classification
    - classify_async(query) → async classification
    - classify_batch(queries) → batch classification
    
    Returns a standardized dict compatible with the existing classifier interface:
    {
        "intent": str,           # e.g. "pricing_info"
        "domain": str,           # e.g. "cap_api"
        "domain_name": str,      # e.g. "Cap-API"
        "api_endpoint": str,     # e.g. "/myclaims/claims/v1/claim/byclaimnumber"
        "confidence": float,     # 0.0 - 1.0
        "margin": float,         # gap between top-1 and top-2
        "source": str,           # "ensemble" or "llm"
        "agreement": bool,       # all 3 sub-classifiers agree
        "top_5": list,           # [(intent, prob), ...]
        "entities": dict,        # extracted claim_number, sequence_number, etc.
        "needs_clarification": bool,
    }
    """

    def __init__(
        self,
        model_path: str = MODEL_PKL,
        confidence_threshold: float = 0.30,
        margin_threshold: float = 0.05,
        use_llm_fallback: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.use_llm_fallback = use_llm_fallback
        self._pipeline = None
        self._model_path = model_path
        self._load_time = None

    def _ensure_loaded(self):
        """Lazy-load the pipeline on first classify() call."""
        if self._pipeline is not None:
            return
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Trained pipeline not found at {self._model_path}. "
                f"Run intent_detection_v3.py first to train and save the model."
            )

        # Import IntentPipeline so the unpickler can resolve it.
        # The pickle stores the class as '__main__.IntentPipeline' because the
        # training script (intent_detection_v3.py) was run directly.  We need
        # that class available in whichever module pickle looks up.
        try:
            from Intent_detection_system.intent_detection_v3 import IntentPipeline
        except ImportError:
            from intent_detection_v3 import IntentPipeline

        # Patch __main__ so pickle.load finds IntentPipeline regardless of
        # which script is currently __main__.
        import __main__
        if not hasattr(__main__, "IntentPipeline"):
            __main__.IntentPipeline = IntentPipeline

        t0 = time.time()
        with open(self._model_path, "rb") as f:
            self._pipeline = pickle.load(f)
        self._load_time = time.time() - t0
        n_intents = len(self._pipeline.label_names)
        logger.info(
            f"Pipeline loaded: {n_intents} intents, "
            f"PCA-{self._pipeline.n_pca}, "
            f"loaded in {self._load_time*1000:.0f}ms"
        )

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a user prompt into intent + domain.
        
        Args:
            query: Raw user prompt (can contain claim numbers, etc.)
            
        Returns:
            Classification result dict (see class docstring for schema)
        """
        self._ensure_loaded()
        t0 = time.time()

        # 1. Extract entities from raw text
        entities = _extract_entities(query)

        # 2. Normalize query (strip numbers) for embedding
        normalized = _normalize_query(query)

        # 3. Embed
        embedder = _get_embedder()
        vec = np.array(embedder.embed(normalized))

        # 4. Predict with ensemble
        pred = self._pipeline.predict_single(vec)

        # 5. Confidence gate → optional LLM fallback
        #    Uses calibrated confidence from pipeline (temperature + disagreement
        #    + confusion-pair penalties already applied).
        confident = (
            pred["confidence"] >= self.confidence_threshold
            and pred["margin"] >= self.margin_threshold
            and pred["agreement"]
        )

        # Stricter gate for known confusion-prone intents
        if confident and pred["intent"] in _CONFUSION_PRONE_INTENTS:
            confident = pred["confidence"] >= 0.55 and pred["margin"] >= 0.20

        # Even stricter when top-2 are a known confusion pair
        if confident and pred.get("is_confusion_pair", False):
            confident = pred["confidence"] >= 0.60 and pred["margin"] >= 0.25

        if confident or not self.use_llm_fallback:
            final_intent = pred["intent"]
            source = "ensemble"
        else:
            logger.info(
                f"Low confidence ({pred['confidence']:.2f}), "
                f"margin ({pred['margin']:.2f}) — calling LLM fallback"
            )
            final_intent = _llm_classify(
                query,
                [name for name, _ in pred["top_5"]],
                ensemble_intent=pred["intent"],
                ensemble_confidence=pred["confidence"],
            )
            source = "llm"

        # 6. Resolve domain
        domain = INTENT_TO_DOMAIN.get(final_intent, "unknown")
        elapsed_ms = (time.time() - t0) * 1000

        result = {
            "intent": final_intent,
            "domain": domain,
            "domain_name": DOMAIN_NAMES.get(domain, domain),
            "api_endpoint": DOMAIN_ENDPOINTS.get(domain),
            "confidence": pred["confidence"],
            "margin": pred["margin"],
            "source": source,
            "agreement": pred["agreement"],
            "top_5": pred["top_5"],
            "entities": entities,
            "needs_clarification": pred["confidence"] < 0.4,
            "latency_ms": round(elapsed_ms, 1),
        }

        logger.info(
            f"Classified: '{query[:60]}...' → "
            f"intent={final_intent}, domain={domain}, "
            f"conf={pred['confidence']:.2f}, src={source}, "
            f"{elapsed_ms:.0f}ms"
        )
        return result

    async def classify_async(self, query: str) -> Dict[str, Any]:
        """Async wrapper — runs classify() in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, query)

    def classify_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple queries sequentially."""
        return [self.classify(q) for q in queries]

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        self._ensure_loaded()
        return {
            "n_intents": len(self._pipeline.label_names),
            "intents": sorted(self._pipeline.label_names),
            "n_domains": len(set(INTENT_TO_DOMAIN.values())),
            "domains": sorted(set(INTENT_TO_DOMAIN.values())),
            "pca_dims": self._pipeline.n_pca,
            "temperature": self._pipeline.temperature,
            "ensemble_weights": self._pipeline.weights,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "use_llm_fallback": self.use_llm_fallback,
            "model_path": self._model_path,
            "load_time_ms": round(self._load_time * 1000, 1) if self._load_time else None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESSOR
# ═════════════════════════════════════════════════════════════════════════════

_classifier_instance: Optional[IntentClassifierV3] = None


def get_classifier(
    confidence_threshold: float = 0.30,
    margin_threshold: float = 0.05,
    use_llm_fallback: bool = True,
) -> IntentClassifierV3:
    """
    Get the singleton IntentClassifierV3 instance.
    
    First call creates the classifier (lazy — pipeline loaded on first classify()).
    Subsequent calls return the same instance.
    
    Args:
        confidence_threshold: Minimum confidence to skip LLM fallback (default 0.30)
        margin_threshold: Minimum margin between top-1 and top-2 (default 0.05)
        use_llm_fallback: Whether to call Gemini when confidence is low (default True)
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifierV3(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            use_llm_fallback=use_llm_fallback,
        )
        logger.info("IntentClassifierV3 singleton created")
    return _classifier_instance


# ═════════════════════════════════════════════════════════════════════════════
# CLI — Quick test / demo
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    classifier = get_classifier()

    # Demo queries covering all 6 domains
    test_queries = [
        # cap_api
        "What is the copay on claim 132435151040074 sequence 001?",
        "Prescriber details for claim 220133725669000 sequence 001.",
        "Show the settlement codes for this claim.",
        "Why was this claim rejected?",
        # benefits_api
        "Show the current benefit plan overview for this member.",
        "Display the audit trail of plan changes.",
        # claim_history_search
        "Show all rejected claims for this member.",
        "How much did the member pay for METFORMIN across all fills?",
        "List claims filled at CVS PHARMACY 00610.",
        "When was ATORVASTATIN last dispensed?",
        "Show claims with reject code 79.",
        "NDC 33342-0395-44",
        # member_domain
        "Does this member have active coverage as of today?",
        "What is the CVS ID for this member?",
        "Is this member LICS?",
        # override_domain
        "Will this PA override a reject 75?",
        "What drugs will this PA cover?",
        "How many claims used this PA?",
        # general
        "Hello",
        "What's the weather today?",
    ]

    print("=" * 80)
    print("  Intent Classifier v3 — Demo")
    print("=" * 80)
    print(f"  Model: {classifier._model_path}")
    print(f"  Confidence threshold: {classifier.confidence_threshold}")
    print(f"  LLM fallback: {classifier.use_llm_fallback}")
    print("=" * 80)
    print()

    for q in test_queries:
        result = classifier.classify(q)
        conf_bar = "█" * int(result["confidence"] * 20) + "░" * (20 - int(result["confidence"] * 20))
        entities_str = ", ".join(f"{k}={v}" for k, v in result["entities"].items()) if result["entities"] else "—"
        print(f"  Query:      {q}")
        print(f"  Intent:     {result['intent']}")
        print(f"  Domain:     {result['domain']} ({result['domain_name']})")
        print(f"  Confidence: {conf_bar} {result['confidence']:.2%}")
        print(f"  Source:     {result['source']} | Agree: {result['agreement']}")
        print(f"  Entities:   {entities_str}")
        print(f"  Endpoint:   {result['api_endpoint'] or '(none)'}")
        print(f"  Latency:    {result['latency_ms']}ms")
        print()

    # Print model info
    info = classifier.model_info
    print("=" * 80)
    print(f"  Model Info: {info['n_intents']} intents, {info['n_domains']} domains, PCA-{info['pca_dims']}")
    print(f"  Domains: {', '.join(info['domains'])}")
    print("=" * 80)
