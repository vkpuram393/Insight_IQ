"""
claims_search_api.search
Reusable functions to filter claims data by various criteria.
"""
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

def generalized_claims_query(claims, user_prompt, current_date=None):
    """
    Attempts to answer a wide range of user queries about claims using simple keyword matching.
    Returns filtered claims or relevant claim(s) based on the prompt.
    
    Supported scenarios (examples):
    - When was <drug-name> taken last?
    - What was the last claim for <drug-name>?
    - Show all claims for reject code <code>
    - Give me all claims for this member in this month/last month/January
    - ...and more, based on keywords
    """
    import re
    from datetime import datetime, timedelta
    if current_date is None:
        current_date = datetime.now().date()
    prompt = user_prompt.lower()
    # Drug name queries
    drug_match = re.search(r'(?:drug|claim for|taken) ([\w\- ]+)', prompt)
    if 'last' in prompt and drug_match:
        drug_name = drug_match.group(1).strip()
        return [get_last_claim_for_field_value(claims, 'drug.productName', drug_name)]
    if 'all claims' in prompt and drug_match:
        drug_name = drug_match.group(1).strip()
        return filter_claims_by_field(claims, 'drug.productName', drug_name)
    # Reject code queries
    reject_match = re.search(r'reject code (\d+)', prompt)
    if reject_match:
        code = reject_match.group(1)
        return filter_claims_by_field(claims, 'messages.rejectCodes.code', code, partial=False)
    # Claims by month
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    for mname, mnum in month_map.items():
        if mname in prompt:
            return filter_claims_by_month(claims, current_date.year, mnum)
    if 'this month' in prompt:
        return filter_claims_by_month(claims, current_date.year, current_date.month)
    if 'last month' in prompt:
        last_month = current_date.month - 1 or 12
        year = current_date.year if current_date.month > 1 else current_date.year - 1
        return filter_claims_by_month(claims, year, last_month)
    # Last claim for member
    if 'last claim' in prompt:
        filtered = claims[:]
        filtered.sort(key=lambda c: c.get('claimInformation', {}).get('fillDate', ''), reverse=True)
        return [filtered[0]] if filtered else []
    # Fallback: return all claims
    return claims
