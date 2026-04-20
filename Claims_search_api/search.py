"""
claims_search_api.search
Reusable functions to filter claims data by various criteria.
"""
import re
from typing import List, Dict, Any
from datetime import datetime

def filter_claims_by_drug_name(claims: List[Dict[str, Any]], drug_name: str) -> List[Dict[str, Any]]:
    """
    Returns claims where the drug productName contains the given drug_name (case-insensitive).
    """
    return [
        claim for claim in claims
        if drug_name.lower() in claim.get('drug', {}).get('productName', '').lower()
    ]

def filter_claims_by_reject_code(claims: List[Dict[str, Any]], reject_code: str) -> List[Dict[str, Any]]:
    """
    Returns claims where any reject code matches the given reject_code.
    """
    filtered = []
    for claim in claims:
        reject_codes = claim.get('messages', {}).get('rejectCodes')
        if reject_codes:
            for code in reject_codes:
                if code.get('code') == reject_code:
                    filtered.append(claim)
                    break
    return filtered

def filter_claims_by_status(claims: List[Dict[str, Any]], status: str) -> List[Dict[str, Any]]:
    """
    Returns claims where claimStatus matches the given status (e.g., 'P', 'R', 'X').
    """
    return [
        claim for claim in claims
        if claim.get('claimInformation', {}).get('claimStatus') == status
    ]

def get_nested_value(d: dict, field_path: str):
    """
    Safely get a nested value from a dict using dot notation (e.g., 'drug.productName').
    """
    keys = field_path.split('.')
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        elif isinstance(d, list):
            # If the current value is a list, return all values for this key in the list
            d = [item.get(key) for item in d if isinstance(item, dict)]
        else:
            return None
    return d

def filter_claims_by_field(claims, field_path, value, partial=True):
    """
    Generalized filter: returns claims where the field at field_path matches value.
    If partial is True, does a case-insensitive substring match for strings.
    Supports nested fields (e.g., 'drug.productName', 'messages.rejectCodes.code').
    """
    filtered = []
    for claim in claims:
        field_val = get_nested_value(claim, field_path)
        if isinstance(field_val, list):
            # If the field is a list (e.g., rejectCodes), check if any match
            for v in field_val:
                if v is not None:
                    if partial and isinstance(v, str) and value.lower() in v.lower():
                        filtered.append(claim)
                        break
                    elif not partial and v == value:
                        filtered.append(claim)
                        break
        else:
            if field_val is not None:
                if partial and isinstance(field_val, str) and value.lower() in field_val.lower():
                    filtered.append(claim)
                elif not partial and field_val == value:
                    filtered.append(claim)
    return filtered

def get_last_claim_for_field_value(claims, field_path, value):
    """
    Returns the most recent claim (by fillDate) where field_path matches value.
    """
    filtered = filter_claims_by_field(claims, field_path, value)
    if not filtered:
        return None
    filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
    return filtered[0]

def filter_claims_by_month(claims, year, month):
    """
    Returns all claims with fillDate in the given year and month.
    """
    result = []
    for claim in claims:
        fill_date = claim.get('claimInformation', {}).get('fillDate')
        if fill_date:
            try:
                dt = datetime.strptime(fill_date, '%Y-%m-%d')
                if dt.year == year and dt.month == month:
                    result.append(claim)
            except Exception:
                continue
    return result

def filter_claims_by_date_range(claims, start_date, end_date):
    """
    Returns all claims with fillDate between start_date and end_date (inclusive).
    Dates should be datetime.date or datetime.datetime objects.
    """
    result = []
    for claim in claims:
        fill_date = claim.get('claimInformation', {}).get('fillDate')
        if fill_date:
            try:
                dt = datetime.strptime(fill_date, '%Y-%m-%d').date()
                if start_date <= dt <= end_date:
                    result.append(claim)
            except Exception:
                continue
    return result

# ---------------------------------------------------------------------------
# Pharmacy / Prescriber / Pricing filters
# ---------------------------------------------------------------------------

def filter_claims_by_pharmacy(claims: List[Dict[str, Any]], pharmacy_name: str) -> List[Dict[str, Any]]:
    """
    Returns claims where pharmacyName contains the given name (case-insensitive).
    """
    return [
        c for c in claims
        if pharmacy_name.lower() in c.get('prescription', {}).get('pharmacyName', '').lower()
    ]


def filter_claims_by_pharmacy_city(claims: List[Dict[str, Any]], city: str) -> List[Dict[str, Any]]:
    """
    Returns claims where pharmacyCity matches (case-insensitive substring).
    """
    return [
        c for c in claims
        if city.lower() in c.get('prescription', {}).get('pharmacyCity', '').lower()
    ]


def filter_claims_by_pharmacy_state(claims: List[Dict[str, Any]], state: str) -> List[Dict[str, Any]]:
    """
    Returns claims where pharmacyState matches (case-insensitive, exact 2-letter or full name).
    """
    state_l = state.lower().strip()
    return [
        c for c in claims
        if c.get('prescription', {}).get('pharmacyState', '').lower().strip() == state_l
    ]


def filter_claims_by_prescriber(claims: List[Dict[str, Any]], prescriber_name: str) -> List[Dict[str, Any]]:
    """
    Returns claims where prescriberFirstName or prescriberLastName
    contains the given name (case-insensitive substring).
    """
    name_l = prescriber_name.lower()
    result = []
    for c in claims:
        rx = c.get('prescription', {})
        first = (rx.get('prescriberFirstName') or '').lower()
        last = (rx.get('prescriberLastName') or '').lower()
        full = f"{first} {last}"
        if name_l in first or name_l in last or name_l in full:
            result.append(c)
    return result


def filter_claims_by_prescriber_npi(claims: List[Dict[str, Any]], npi: str) -> List[Dict[str, Any]]:
    """
    Returns claims where prescriberID matches the NPI exactly.
    """
    return [
        c for c in claims
        if c.get('prescription', {}).get('prescriberID') == npi
    ]


def filter_claims_with_patient_pay(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns claims that have a non-null patientPay value.
    """
    return [
        c for c in claims
        if c.get('pricing', {}).get('patientPay') is not None
    ]


def filter_claims_by_rx_number(claims: List[Dict[str, Any]], rx_number: str) -> List[Dict[str, Any]]:
    """
    Returns claims where rxNumber matches (exact).
    """
    return [
        c for c in claims
        if c.get('prescription', {}).get('rxNumber') == rx_number
    ]


def filter_claims_by_claim_number(claims: List[Dict[str, Any]], claim_number: str) -> List[Dict[str, Any]]:
    """
    Returns claims where claimNumber matches (exact or partial).
    """
    return [
        c for c in claims
        if claim_number in (c.get('claimInformation', {}).get('claimNumber') or '')
    ]


def filter_claims_by_ndc(claims: List[Dict[str, Any]], ndc: str) -> List[Dict[str, Any]]:
    """
    Returns claims where productNdc matches (exact or partial, ignoring dashes).
    """
    ndc_clean = ndc.replace('-', '')
    return [
        c for c in claims
        if ndc_clean in (c.get('drug', {}).get('productNdc') or '').replace('-', '')
    ]


def filter_claims_by_gpi(claims: List[Dict[str, Any]], gpi: str) -> List[Dict[str, Any]]:
    """
    Returns claims where GPI starts with or matches the given value.
    """
    return [
        c for c in claims
        if (c.get('drug', {}).get('gpi') or '').startswith(gpi)
    ]


def filter_claims_by_manufacturer(claims: List[Dict[str, Any]], manufacturer: str) -> List[Dict[str, Any]]:
    """
    Returns claims where manufacturer contains the given name (case-insensitive).
    """
    return [
        c for c in claims
        if manufacturer.lower() in (c.get('drug', {}).get('manufacturer') or '').lower()
    ]


def filter_claims_by_generic_indicator(claims: List[Dict[str, Any]], is_generic: bool) -> List[Dict[str, Any]]:
    """
    Returns claims where genericIndicator is 'Y' (generic) or 'N' (brand).
    """
    target = 'Y' if is_generic else 'N'
    return [
        c for c in claims
        if c.get('drug', {}).get('genericIndicator') == target
    ]


def filter_claims_by_refill_number(claims: List[Dict[str, Any]], refill_num: str) -> List[Dict[str, Any]]:
    """
    Returns claims where refillNumber matches. Use '00' for original fills.
    """
    return [
        c for c in claims
        if c.get('prescription', {}).get('refillNumber') == refill_num
    ]


def filter_claims_by_days_supply(claims: List[Dict[str, Any]], days: str) -> List[Dict[str, Any]]:
    """
    Returns claims where daysSupplied matches the given value.
    """
    return [
        c for c in claims
        if c.get('claimInformation', {}).get('daysSupplied') == days
    ]


def filter_claims_by_fill_date(claims: List[Dict[str, Any]], fill_date: str) -> List[Dict[str, Any]]:
    """
    Returns claims where fillDate matches exactly (YYYY-MM-DD format).
    """
    return [
        c for c in claims
        if c.get('claimInformation', {}).get('fillDate') == fill_date
    ]


def filter_claims_with_prior_auth(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns claims that used prior authorization.
    """
    result = []
    for c in claims:
        overrides = c.get('overrides', {})
        pa = c.get('priorAuthorization', {})
        if (overrides.get('priorAuthorizationUsed') == 'Yes'
                or (pa.get('paIndicator') is not None and pa.get('paIndicator') != '')):
            result.append(c)
    return result


def filter_claims_by_diagnosis_code(claims: List[Dict[str, Any]], code: str) -> List[Dict[str, Any]]:
    """
    Returns claims where submittedDiagnosisCodeIndicator matches (case-insensitive).
    """
    return [
        c for c in claims
        if code.upper() in (c.get('prescription', {}).get('submittedDiagnosisCodeIndicator') or '').upper()
    ]


def filter_claims_by_compound(claims: List[Dict[str, Any]], is_compound: bool = True) -> List[Dict[str, Any]]:
    """
    Returns compound (Y) or non-compound (N) claims.
    """
    target = 'Y' if is_compound else 'N'
    return [
        c for c in claims
        if c.get('claimInformation', {}).get('compound') == target
    ]


def filter_claims_by_specialty(claims: List[Dict[str, Any]], is_specialty: bool = True) -> List[Dict[str, Any]]:
    """
    Returns specialty (Y) or non-specialty (N) claims.
    """
    target = 'Y' if is_specialty else 'N'
    return [
        c for c in claims
        if c.get('claimInformation', {}).get('speciality') == target
    ]


def filter_claims_by_daw(claims: List[Dict[str, Any]], daw_code: str = None) -> List[Dict[str, Any]]:
    """
    Returns claims with non-zero DAW code, or matching a specific DAW code.
    """
    if daw_code:
        return [
            c for c in claims
            if c.get('claimInformation', {}).get('dispenseAsWritten') == daw_code
        ]
    return [
        c for c in claims
        if (c.get('claimInformation', {}).get('dispenseAsWritten') or '0') != '0'
    ]


def filter_claims_with_reversal(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns claims that have a reversalDate (i.e. were reversed).
    """
    return [
        c for c in claims
        if c.get('prescription', {}).get('reversalDate') is not None
    ]


def filter_claims_by_pharmacy_type(claims: List[Dict[str, Any]], pharm_type: str) -> List[Dict[str, Any]]:
    """
    Returns claims where pharmacyType matches (e.g. 'Retail', 'Mail Order').
    """
    return [
        c for c in claims
        if pharm_type.lower() in (c.get('prescription', {}).get('pharmacyType') or '').lower()
    ]


def filter_claims_by_settlement_code(claims: List[Dict[str, Any]], code: str) -> List[Dict[str, Any]]:
    """
    Returns claims where any settlement code matches.
    """
    result = []
    for c in claims:
        codes = c.get('messages', {}).get('settlementCodes') or []
        for sc in codes:
            if sc.get('code') == code:
                result.append(c)
                break
    return result


def filter_claims_by_plan(claims: List[Dict[str, Any]], plan_id: str) -> List[Dict[str, Any]]:
    """
    Returns claims where planId or clientPlanCode matches (case-insensitive).
    """
    plan_l = plan_id.lower()
    return [
        c for c in claims
        if plan_l in (c.get('member', {}).get('planId') or '').lower()
        or plan_l in (c.get('member', {}).get('clientPlanCode') or '').lower()
        or plan_l in (c.get('member', {}).get('finalPlanCode') or '').lower()
    ]


def generalized_claims_query(claims, user_prompt, current_date=None):
    """
    Attempts to answer a wide range of user queries about claims using simple keyword matching.

    IMPORTANT: This function always returns ALL matching claims (sorted newest-first)
    rather than picking a single result.  The downstream LLM is responsible for
    interpreting the user's intent (e.g. "last claim") from the filtered set.

    Supported scenarios (examples):
    - When was <drug-name> taken last?
    - What was the last claim for <drug-name>?
    - Show all claims for reject code <code>
    - Give me all claims for this member in this month/last month/January
    - Show me all rejected / paid / reversed claims
    - ...and more, based on keywords
    """
    import re
    from datetime import datetime, timedelta
    if current_date is None:
        current_date = datetime.now().date()
    prompt = user_prompt.lower()

    # ---------------------------------------------------------------
    # Category-based filters (must be checked BEFORE drug extraction
    # to avoid "generic drug", "brand name", etc. being treated as drug names)
    # ---------------------------------------------------------------

    # Generic vs Brand queries
    if any(kw in prompt for kw in ['generic drug', 'generic claim', 'generic medication', 'all generic']):
        filtered = filter_claims_by_generic_indicator(claims, is_generic=True)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered
    if any(kw in prompt for kw in ['brand name', 'brand drug', 'brand claim', 'non-generic', 'non generic', 'single source']):
        filtered = filter_claims_by_generic_indicator(claims, is_generic=False)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # Compound claims
    if 'compound' in prompt:
        filtered = filter_claims_by_compound(claims, is_compound=True)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # Specialty claims
    if any(kw in prompt for kw in ['specialty', 'speciality']):
        filtered = filter_claims_by_specialty(claims, is_specialty=True)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # DAW (Dispense As Written) claims
    daw_match = re.search(r'daw\s*(?:code)?\s*[:#]?\s*(\d)', prompt)
    if daw_match:
        filtered = filter_claims_by_daw(claims, daw_code=daw_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered
    if 'daw' in prompt or 'dispense as written' in prompt:
        filtered = filter_claims_by_daw(claims)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # Pharmacy type (Retail / Mail Order) — before drug extraction
    phtype_match = re.search(r'(retail|mail\s*order|mail)\s*(?:pharmacy|pharm|claims?)?', prompt)
    if phtype_match:
        ptype = phtype_match.group(1).strip()
        if 'mail' in ptype:
            ptype = 'Mail'
        elif 'retail' in ptype:
            ptype = 'Retail'
        filtered = filter_claims_by_pharmacy_type(claims, ptype)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Drug-name extraction (handles multiple natural phrasing styles)
    # Always return ALL matching claims sorted newest-first so the
    # LLM can decide the answer (e.g. "the last one").
    # ---------------------------------------------------------------
    drug_name = _extract_drug_name(prompt)

    # Any drug-related query → return all claims for that drug, sorted newest first
    if drug_name:
        filtered = filter_claims_by_field(claims, 'drug.productName', drug_name)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Pharmacy-related queries
    # ---------------------------------------------------------------
    pharmacy_match = re.search(
        r'(?:pharmacy|filled at|dispensed at|picked up at|from)'
        + _FILLER + r'\s+((?:cvs|walgreens|rite aid|walmart|costco|kroger|publix|safeway|albertsons|heb|wegmans|sam\'s|target)[\w\s#]*\d*)',
        prompt,
        re.IGNORECASE,
    )
    if not pharmacy_match:
        # fallback: "pharmacy <name>" or "at <name> pharmacy"
        pharmacy_match = re.search(
            r'(?:pharmacy)\s+([\w\s#]+?)(?:\?|$|\.|,|\s+(?:for|in|on|at|with|claim))',
            prompt,
            re.IGNORECASE,
        )
    if pharmacy_match:
        pharm_name = pharmacy_match.group(1).strip().rstrip('?.,!;')
        if pharm_name and pharm_name.lower() not in _STOP_WORDS:
            filtered = filter_claims_by_pharmacy(claims, pharm_name)
            filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
            return filtered

    # Pharmacy city queries
    city_match = re.search(
        r'(?:pharmacy|pharmacies|filled|dispensed)\s+(?:in|at|from)\s+([A-Za-z\s]+?)(?:\?|$|\.|,)',
        prompt,
        re.IGNORECASE,
    )
    if city_match:
        city = city_match.group(1).strip().rstrip('?.,!;')
        # skip if it's a state abbreviation (2 chars) — handled below
        if city and len(city) > 2 and city.lower() not in _STOP_WORDS:
            filtered = filter_claims_by_pharmacy_city(claims, city)
            if filtered:
                filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
                return filtered

    # ---------------------------------------------------------------
    # Prescriber / doctor queries
    # ---------------------------------------------------------------
    npi_match = re.search(r'(?:npi|prescriber\s*id|provider\s*id)\s*[:#]?\s*(\d{10})', prompt)
    if npi_match:
        filtered = filter_claims_by_prescriber_npi(claims, npi_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    prescriber_match = re.search(
        r'(?:prescriber|doctor|dr|physician|provider|prescribed by)'
        + _FILLER + r'\s+([A-Za-z\-\' ]+?)(?:\?|$|\.|,|\s+(?:for|in|on|at|with|claim))',
        prompt,
        re.IGNORECASE,
    )
    if prescriber_match:
        pname = prescriber_match.group(1).strip().rstrip('?.,!;')
        # strip filler words
        words = pname.split()
        while words and words[0].lower() in _STOP_WORDS:
            words.pop(0)
        while words and words[-1].lower() in _STOP_WORDS:
            words.pop()
        pname = ' '.join(words)
        if pname and pname.lower() not in _STOP_WORDS:
            filtered = filter_claims_by_prescriber(claims, pname)
            filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
            return filtered

    # ---------------------------------------------------------------
    # Rx number queries
    # ---------------------------------------------------------------
    rx_match = re.search(r'(?:rx|prescription)\s*(?:#|number|num)?\s*[:#]?\s*(\d{5,})', prompt)
    if rx_match:
        filtered = filter_claims_by_rx_number(claims, rx_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Claim number lookup
    # ---------------------------------------------------------------
    claim_num_match = re.search(r'(?:claim\s*(?:#|number|num|id))\s*[:#]?\s*(\d{10,})', prompt)
    if claim_num_match:
        filtered = filter_claims_by_claim_number(claims, claim_num_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # NDC (National Drug Code) lookup
    # ---------------------------------------------------------------
    ndc_match = re.search(r'(?:ndc|national\s+drug\s+code)\s*[:#]?\s*([\d\-]{8,})', prompt)
    if ndc_match:
        filtered = filter_claims_by_ndc(claims, ndc_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # GPI (Generic Product Identifier) lookup
    # ---------------------------------------------------------------
    gpi_match = re.search(r'(?:gpi)\s*[:#]?\s*(\w{8,})', prompt)
    if gpi_match:
        filtered = filter_claims_by_gpi(claims, gpi_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Manufacturer / brand queries
    # ---------------------------------------------------------------
    mfr_match = re.search(
        r'(?:manufacturer|manufactured\s+by|made\s+by|mfr)\s*[:#]?\s*'
        + _FILLER + r'\s*([A-Za-z\s]+?)(?:\?|$|\.|,)',
        prompt,
        re.IGNORECASE,
    )
    if mfr_match:
        mfr = mfr_match.group(1).strip().rstrip('?.,!;')
        words = mfr.split()
        while words and words[0].lower() in _STOP_WORDS:
            words.pop(0)
        while words and words[-1].lower() in _STOP_WORDS:
            words.pop()
        mfr = ' '.join(words)
        if mfr and mfr.lower() not in _STOP_WORDS:
            filtered = filter_claims_by_manufacturer(claims, mfr)
            filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
            return filtered

    # ---------------------------------------------------------------
    # Refill queries
    # ---------------------------------------------------------------
    refill_match = re.search(r'refill\s*(?:#|number|num)?\s*[:#]?\s*(\d{1,2})', prompt)
    if refill_match:
        filtered = filter_claims_by_refill_number(claims, refill_match.group(1).zfill(2))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered
    if any(kw in prompt for kw in ['original fill', 'first fill', 'new prescription', 'new rx']):
        filtered = filter_claims_by_refill_number(claims, '00')
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered
    if 'refill' in prompt and 'refill too soon' not in prompt:
        # Generic "show refills" → all claims with refillNumber > 00
        filtered = [c for c in claims if (c.get('prescription', {}).get('refillNumber') or '00') != '00']
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered if filtered else claims[:]

    # ---------------------------------------------------------------
    # Days supply queries
    # ---------------------------------------------------------------
    days_match = re.search(r'(\d+)\s*(?:day|days)\s*(?:supply|sup)', prompt)
    if days_match:
        filtered = filter_claims_by_days_supply(claims, days_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Specific fill date queries (YYYY-MM-DD or MM/DD/YYYY)
    # ---------------------------------------------------------------
    date_match = re.search(r'(?:filled|fill date|filled on|date)\s*[:#]?\s*(\d{4}-\d{2}-\d{2})', prompt)
    if not date_match:
        date_match = re.search(r'(?:filled|fill date|filled on|date)\s*[:#]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})', prompt)
    if date_match:
        date_str = date_match.group(1)
        # Normalize to YYYY-MM-DD
        if '/' in date_str or (len(date_str.split('-')[0]) <= 2):
            try:
                parts = re.split(r'[/\-]', date_str)
                date_str = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
            except (IndexError, ValueError):
                pass
        filtered = filter_claims_by_fill_date(claims, date_str)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Prior Authorization queries
    # ---------------------------------------------------------------
    if any(kw in prompt for kw in [
        'prior auth', 'prior authorization', 'pa claim', 'pa used',
        'authorization', 'pre-auth', 'preauth',
    ]):
        filtered = filter_claims_with_prior_auth(claims)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Diagnosis / ICD code queries
    # ---------------------------------------------------------------
    diag_match = re.search(r'(?:diagnosis|diag|icd|icd-?10)\s*(?:code)?\s*[:#]?\s*([A-Za-z]\d{2,}[\w.]*)', prompt)
    if diag_match:
        filtered = filter_claims_by_diagnosis_code(claims, diag_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Reversal queries (by reversalDate, not status)
    # ---------------------------------------------------------------
    if any(kw in prompt for kw in ['reversal date', 'has reversal', 'was reversed']):
        filtered = filter_claims_with_reversal(claims)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Settlement code queries
    # ---------------------------------------------------------------
    settle_match = re.search(r'settlement\s*(?:code)?\s*[:#]?\s*(\w+)', prompt)
    if settle_match:
        filtered = filter_claims_by_settlement_code(claims, settle_match.group(1))
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Plan / benefit queries
    # ---------------------------------------------------------------
    plan_match = re.search(r'(?:plan|benefit|plan\s*id|plan\s*code)\s*[:#]?\s*([A-Za-z0-9#]+)', prompt)
    if plan_match:
        plan_val = plan_match.group(1).strip()
        if plan_val.lower() not in _STOP_WORDS and len(plan_val) > 1:
            filtered = filter_claims_by_plan(claims, plan_val)
            filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
            return filtered

    # ---------------------------------------------------------------
    # Pricing / cost / copay queries  → return all claims that have
    # pricing data so the LLM can answer cost questions
    # ---------------------------------------------------------------
    pricing_keywords = [
        'cost', 'price', 'pricing', 'copay', 'co-pay', 'patient pay',
        'how much', 'amount', 'charge', 'fee', 'expense', 'payment',
        'dispensing fee', 'drug cost', 'total cost', 'out of pocket',
    ]
    if any(kw in prompt for kw in pricing_keywords):
        # If a drug is also mentioned, filter by drug first
        drug_name = _extract_drug_name(prompt)
        if drug_name:
            filtered = filter_claims_by_field(claims, 'drug.productName', drug_name)
        else:
            filtered = claims[:]
        # Keep only claims that have at least one non-null pricing value
        filtered = [c for c in filtered if _has_pricing_data(c)]
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered if filtered else claims[:]

    # ---------------------------------------------------------------
    # Reject / settlement code queries → all matching claims
    # ---------------------------------------------------------------
    reject_match = re.search(r'reject(?:\s*(?:code|codes))?\s+(\w+)', prompt)
    if reject_match:
        code = reject_match.group(1)
        filtered = filter_claims_by_field(claims, 'messages.rejectCodes.code', code, partial=False)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Status-based queries  (paid / rejected / reversed) → all matching
    # ---------------------------------------------------------------
    status_map = {'paid': 'P', 'rejected': 'R', 'reversed': 'X', 'cancelled': 'X', 'canceled': 'X'}
    for word, code in status_map.items():
        if word in prompt:
            filtered = filter_claims_by_status(claims, code)
            filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
            return filtered

    # ---------------------------------------------------------------
    # Month / date-range queries → all matching, sorted
    # ---------------------------------------------------------------
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    for mname, mnum in month_map.items():
        if mname in prompt:
            filtered = filter_claims_by_month(claims, current_date.year, mnum)
            filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
            return filtered
    if 'this month' in prompt:
        filtered = filter_claims_by_month(claims, current_date.year, current_date.month)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered
    if 'last month' in prompt:
        last_month = current_date.month - 1 or 12
        year = current_date.year if current_date.month > 1 else current_date.year - 1
        filtered = filter_claims_by_month(claims, year, last_month)
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # ---------------------------------------------------------------
    # Generic "last claim" for the member (no drug specified)
    # → return ALL claims sorted newest-first; LLM picks the answer
    # ---------------------------------------------------------------
    if 'last claim' in prompt:
        filtered = claims[:]
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return filtered

    # Fallback: return all claims sorted newest-first
    all_claims = claims[:]
    all_claims.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
    return all_claims


# --------------------------------------------------------------------------
# Pricing helper
# --------------------------------------------------------------------------

def _has_pricing_data(claim: dict) -> bool:
    """Return True if the claim has at least one non-null pricing field."""
    pricing = claim.get('pricing', {})
    if not pricing:
        return False
    return any(v is not None for v in pricing.values())


# --------------------------------------------------------------------------
# Drug-name extraction helper
# --------------------------------------------------------------------------
# Filler words that appear between keywords and the actual drug name
_FILLER = r'(?:\s+(?:the|this|a|my|for|of|about|related\s+to|regarding))*'

# Patterns ordered by specificity (most specific first)
_DRUG_PATTERNS = [
    # "when was <DRUG> taken last" / "was <DRUG> taken"
    re.compile(
        r'(?:when\s+was|was)\s+' + _FILLER + r'\s*([\w\-/\' ]+?)\s+taken',
        re.IGNORECASE,
    ),
    # "last claim for <DRUG>" / "claims for <DRUG>" / "claim for <DRUG>"
    re.compile(
        r'claims?\s+(?:for|related\s+to|of|about)' + _FILLER + r'\s+([\w\-/\' ]+)',
        re.IGNORECASE,
    ),
    # "claims related to the drug <DRUG>"
    re.compile(
        r'(?:drug|medication|medicine)' + _FILLER + r'\s+([\w\-/\' ]+)',
        re.IGNORECASE,
    ),
    # "give me <DRUG> claims" / "show <DRUG> claims"
    re.compile(
        r'(?:give|show|get|find|list|fetch)\s+(?:me\s+)?' + _FILLER + r'\s*([\w\-/\' ]+?)\s+claims?',
        re.IGNORECASE,
    ),
]

# Words that should never be treated as a drug name
_STOP_WORDS = frozenset({
    'last', 'first', 'all', 'the', 'this', 'a', 'my', 'me', 'for', 'of',
    'about', 'in', 'on', 'at', 'to', 'from', 'with', 'and', 'or', 'is',
    'was', 'were', 'are', 'been', 'be', 'member', 'claim', 'claims',
    'reject', 'code', 'status', 'paid', 'rejected', 'reversed', 'month',
    'year', 'date', 'pharmacy', 'prescriber',
    # Category words that should not be drug names
    'generic', 'brand', 'compound', 'specialty', 'speciality',
    'retail', 'mail', 'order', 'refill', 'refills', 'supply',
    'prior', 'authorization', 'auth', 'daw', 'dispense', 'written',
    'cost', 'price', 'pricing', 'copay', 'payment', 'amount',
    'diagnosis', 'icd', 'settlement', 'ndc', 'gpi', 'npi',
    'manufacturer', 'manufactured', 'made', 'name', 'number',
    'show', 'give', 'get', 'find', 'list', 'fetch', 'display',
})


def _extract_drug_name(prompt: str) -> str:
    """
    Extract a drug name from a natural-language user prompt.
    Returns the cleaned drug name or empty string if none found.
    """
    for pattern in _DRUG_PATTERNS:
        m = pattern.search(prompt)
        if m:
            raw = m.group(1).strip().rstrip('?.,!;')
            # Strip leading filler / stop words
            words = raw.split()
            while words and words[0].lower() in _STOP_WORDS:
                words.pop(0)
            # Strip trailing stop words
            while words and words[-1].lower() in _STOP_WORDS:
                words.pop()
            cleaned = ' '.join(words)
            if cleaned and cleaned.lower() not in _STOP_WORDS:
                return cleaned
    return ''
