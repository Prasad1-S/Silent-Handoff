"""
fhir_client.py
==============
FHIR R4 data fetcher with clean extraction.
Returns lean, LLM-friendly dicts instead of raw FHIR blobs.
"""

import os
import requests
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(data, *keys, default="N/A"):
    """Safe nested get for dicts and lists."""
    for key in keys:
        if data is None:
            return default
        if isinstance(data, list):
            if isinstance(key, int) and key < len(data):
                data = data[key]
            else:
                return default
        elif isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default


def _age(birth_date_str: str) -> str:
    if not birth_date_str or birth_date_str == "N/A":
        return "N/A"
    try:
        bd = datetime.strptime(birth_date_str[:10], "%Y-%m-%d")
        today = datetime.today()
        age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        return str(age)
    except Exception:
        return "N/A"


def _code_display(resource: dict, path_root="code") -> str:
    """Extract human-readable display from a CodeableConcept."""
    block = resource.get(path_root, {})
    text = block.get("text")
    if text:
        return text
    codings = block.get("coding", [])
    for c in codings:
        if c.get("display"):
            return c["display"]
    return "Unknown"


def _fhir_get(url: str, timeout: int = 15) -> dict | None:
    """Make a FHIR GET request; return parsed JSON or None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"Accept": "application/fhir+json"})
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"[fhir_client] Request failed: {url} — {e}")
    return None


def _bundle_resources(data: dict) -> list:
    """Extract resource list from a FHIR Bundle."""
    if not data:
        return []
    return [entry.get("resource", {}) for entry in data.get("entry", []) if entry.get("resource")]


def validate_fhir_url(url: str) -> bool:
    """
    Validate that the URL is a plausible FHIR endpoint.
    Allows both http (for localhost dev) and https.
    Blocks obviously malformed URLs only.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.hostname:
            return False
        return True
    except Exception:
        return False


# ─── Extraction helpers ────────────────────────────────────────────────────────

def _extract_demographics(resource: dict) -> dict:
    names = resource.get("name", [])
    full_name = "N/A"
    if names:
        given = " ".join(names[0].get("given", []))
        family = names[0].get("family", "")
        full_name = f"{given} {family}".strip() or "N/A"

    birth_date = resource.get("birthDate", "N/A")
    address = resource.get("address", [{}])[0] if resource.get("address") else {}

    return {
        "name": full_name,
        "gender": resource.get("gender", "N/A"),
        "birth_date": birth_date,
        "age": _age(birth_date),
        "marital_status": _get(resource, "maritalStatus", "text", default="N/A"),
        "city": address.get("city", "N/A"),
        "state": address.get("state", "N/A"),
    }


def _extract_condition(resource: dict) -> dict:
    return {
        "condition": _code_display(resource),
        "status": _get(resource, "clinicalStatus", "coding", 0, "code", default="unknown"),
        "onset": resource.get("onsetDateTime", resource.get("onsetPeriod", {}).get("start", "N/A")),
    }


def _extract_medication(resource: dict) -> dict:
    # medicationCodeableConcept or medicationReference
    med_name = _code_display(resource, "medicationCodeableConcept")
    if med_name == "Unknown":
        med_name = _get(resource, "medicationReference", "display", default="Unknown")
    return {
        "medication": med_name,
        "status": resource.get("status", "N/A"),
        "authored": resource.get("authoredOn", "N/A"),
        "reason": _get(resource, "reasonCode", 0, "text",
                       default=_get(resource, "reasonCode", 0, "coding", 0, "display", default="N/A")),
    }


def _extract_observation(resource: dict) -> dict:
    name = _code_display(resource)
    value = resource.get("valueQuantity", {}).get("value", "N/A")
    unit = resource.get("valueQuantity", {}).get("unit", "")
    # handle component observations (e.g. blood pressure)
    components = resource.get("component", [])
    if components and value == "N/A":
        parts = []
        for c in components:
            v = c.get("valueQuantity", {}).get("value", "")
            u = c.get("valueQuantity", {}).get("unit", "")
            n = _code_display(c)
            if v:
                parts.append(f"{n}: {v} {u}".strip())
        value = " | ".join(parts) if parts else "N/A"
        unit = ""
    return {
        "name": name,
        "value": f"{value} {unit}".strip() if unit else str(value),
        "date": resource.get("effectiveDateTime", "N/A"),
        "status": resource.get("status", "N/A"),
    }


def _extract_allergy(resource: dict) -> dict:
    return {
        "substance": _code_display(resource),
        "criticality": resource.get("criticality", "N/A"),
        "status": _get(resource, "clinicalStatus", "coding", 0, "code", default="N/A"),
        "reaction": _get(resource, "reaction", 0, "manifestation", 0, "text",
                         default=_get(resource, "reaction", 0, "manifestation", 0, "coding", 0, "display", default="N/A")),
    }


def _extract_encounter(resource: dict) -> dict:
    return {
        "type": _get(resource, "type", 0, "text",
                     default=_get(resource, "type", 0, "coding", 0, "display", default="N/A")),
        "status": resource.get("status", "N/A"),
        "start": _get(resource, "period", "start", default="N/A"),
        "end": _get(resource, "period", "end", default="N/A"),
        "reason": _get(resource, "reasonCode", 0, "coding", 0, "display", default="N/A"),
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_patient_demographics(patient_id: str, base_url: str) -> dict:
    if not validate_fhir_url(base_url):
        print(f"[fhir_client] Invalid FHIR URL: {base_url}")
        return {}
    data = _fhir_get(f"{base_url}/Patient/{patient_id}")
    return _extract_demographics(data) if data else {}


def fetch_active_conditions(patient_id: str, base_url: str) -> list:
    if not validate_fhir_url(base_url):
        return []
    data = _fhir_get(f"{base_url}/Condition?patient={patient_id}&clinical-status=active&_count=20")
    return [_extract_condition(r) for r in _bundle_resources(data)]


def fetch_medications(patient_id: str, base_url: str) -> list:
    if not validate_fhir_url(base_url):
        return []
    data = _fhir_get(f"{base_url}/MedicationRequest?patient={patient_id}&_count=20")
    results = []
    for r in _bundle_resources(data):
        if r.get("status") not in ("entered-in-error", "cancelled"):
            results.append(_extract_medication(r))
    return results


def fetch_labs(patient_id: str, base_url: str, limit: int = 15) -> list:
    if not validate_fhir_url(base_url):
        return []
    data = _fhir_get(f"{base_url}/Observation?patient={patient_id}&category=laboratory&_sort=-date&_count={limit}")
    return [_extract_observation(r) for r in _bundle_resources(data)]


def fetch_vitals(patient_id: str, base_url: str, limit: int = 10) -> list:
    if not validate_fhir_url(base_url):
        return []
    data = _fhir_get(f"{base_url}/Observation?patient={patient_id}&category=vital-signs&_sort=-date&_count={limit}")
    return [_extract_observation(r) for r in _bundle_resources(data)]


def fetch_allergies(patient_id: str, base_url: str) -> list:
    if not validate_fhir_url(base_url):
        return []
    data = _fhir_get(f"{base_url}/AllergyIntolerance?patient={patient_id}&_count=20")
    return [_extract_allergy(r) for r in _bundle_resources(data)]


def fetch_recent_encounters(patient_id: str, base_url: str, limit: int = 5) -> list:
    if not validate_fhir_url(base_url):
        return []
    data = _fhir_get(f"{base_url}/Encounter?patient={patient_id}&_sort=-date&_count={limit}")
    return [_extract_encounter(r) for r in _bundle_resources(data)]


def get_patient_summary(patient_id: str, base_url: str = None) -> dict:
    """Convenience function — returns a complete summary dict."""
    if not base_url:
        base_url = os.getenv("FHIR_BASE_URL", "https://r4.smarthealthit.org")
    return {
        "demographics": fetch_patient_demographics(patient_id, base_url),
        "conditions": fetch_active_conditions(patient_id, base_url),
        "medications": fetch_medications(patient_id, base_url),
        "labs": fetch_labs(patient_id, base_url),
        "vitals": fetch_vitals(patient_id, base_url),
        "allergies": fetch_allergies(patient_id, base_url),
    }