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

# ─── LOINC Normal-Range Reference Table ───────────────────────────────────────

LOINC_NORMAL_RANGES = {
    "8480-6":  {"name": "Systolic Blood Pressure",  "low": 90,   "high": 140,   "unit": "mm[Hg]"},
    "8462-4":  {"name": "Diastolic Blood Pressure", "low": 60,   "high": 90,    "unit": "mm[Hg]"},
    "8310-5":  {"name": "Body Temperature",         "low": 36.0, "high": 37.5,  "unit": "Cel"},
    "2339-0":  {"name": "Glucose",                  "low": 70,   "high": 140,   "unit": "mg/dL"},
    "718-7":   {"name": "Hemoglobin",               "low": 12.0, "high": 17.5,  "unit": "g/dL"},
    "2160-0":  {"name": "Creatinine",               "low": 0.6,  "high": 1.2,   "unit": "mg/dL"},
    "4548-4":  {"name": "HbA1c",                    "low": 0.0,  "high": 5.7,   "unit": "%"},
    "59408-5": {"name": "SpO2",                     "low": 95.0, "high": 100.0, "unit": "%"},
    "8867-4":  {"name": "Heart Rate",               "low": 60,   "high": 100,   "unit": "/min"},
    "9279-1":  {"name": "Respiratory Rate",         "low": 12,   "high": 20,    "unit": "/min"},
    "29463-7": {"name": "Body Weight",              "low": None, "high": None,  "unit": "kg"},
    "39156-5": {"name": "BMI",                      "low": 18.5, "high": 24.9,  "unit": "kg/m2"},
}

# Acute vitals — right-now emergencies, highest weight
ACUTE_VITAL_LOINCS = {
    "59408-5",  # SpO2
    "8867-4",   # Heart Rate
    "9279-1",   # Respiratory Rate
    "8480-6",   # Systolic BP
    "8462-4",   # Diastolic BP
    "8310-5",   # Body Temperature
}

# Biometric history — chronic background, lowest weight
BIOMETRIC_LOINCS = {
    "29463-7",  # Body Weight
    "39156-5",  # BMI
    "8302-2",   # Body Height
    "8287-5",   # Head Circumference
}

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


# ─── Deterministic Observation Classifier ────────────────────────────────────

def get_observation_weight(loinc_code: str | None) -> str:
    """
    Returns weight category for a LOINC code:
    - "ACUTE"     if in ACUTE_VITAL_LOINCS  (right-now emergencies)
    - "BIOMETRIC" if in BIOMETRIC_LOINCS    (chronic background)
    - "LAB"       for everything else
    """
    if loinc_code is None:
        return "LAB"
    if loinc_code in ACUTE_VITAL_LOINCS:
        return "ACUTE"
    if loinc_code in BIOMETRIC_LOINCS:
        return "BIOMETRIC"
    return "LAB"


def calculate_weighted_acuity(classified_observations: list[dict]) -> tuple[str, str]:
    """
    Weighted acuity algorithm — BIOMETRIC-only findings can never exceed STABLE.

    Returns:
        (acuity: str, basis: str)
        acuity  — "CRITICAL" | "HIGH" | "MODERATE" | "STABLE"
        basis   — "acute_vitals" | "labs" | "biometric_only"
    """
    acute   = [c for c in classified_observations if c.get("weight") == "ACUTE"]
    labs    = [c for c in classified_observations if c.get("weight") == "LAB"]
    bio     = [c for c in classified_observations if c.get("weight") == "BIOMETRIC"]

    acute_critical = [c for c in acute if c.get("status") == "CRITICAL"]
    acute_abnormal = [c for c in acute if c.get("status") == "ABNORMAL"]
    lab_critical   = [c for c in labs  if c.get("status") == "CRITICAL"]
    lab_abnormal   = [c for c in labs  if c.get("status") == "ABNORMAL"]

    # Build a quick value lookup by LOINC code (most-recent / only observation)
    val: dict[str, float | None] = {}
    for c in classified_observations:
        code = c.get("loinc_code")
        if code:
            val[code] = c.get("value")

    def v(code: str) -> float | None:
        return val.get(code)

    # ── CRITICAL thresholds ────────────────────────────────────────────────────
    if acute_critical:
        return "CRITICAL", "acute_vitals"
    spo2 = v("59408-5")
    hr   = v("8867-4")
    rr   = v("9279-1")
    sbp  = v("8480-6")
    temp = v("8310-5")

    if spo2 is not None and spo2 < 90:
        return "CRITICAL", "acute_vitals"
    if hr is not None and (hr > 130 or hr < 40):
        return "CRITICAL", "acute_vitals"
    if rr is not None and rr > 30:
        return "CRITICAL", "acute_vitals"
    if sbp is not None and (sbp > 180 or sbp < 80):
        return "CRITICAL", "acute_vitals"
    if temp is not None and (temp > 39.5 or temp < 35.0):
        return "CRITICAL", "acute_vitals"

    # ── HIGH thresholds ────────────────────────────────────────────────────────
    if acute_abnormal:
        return "HIGH", "acute_vitals"
    if len(lab_critical) >= 2:
        return "HIGH", "labs"
    if sbp is not None and sbp > 160:
        return "HIGH", "acute_vitals"
    if spo2 is not None and spo2 < 94:
        return "HIGH", "acute_vitals"
    if hr is not None and hr > 110:
        return "HIGH", "acute_vitals"

    # ── MODERATE thresholds ────────────────────────────────────────────────────
    if len(lab_critical) >= 1:
        return "MODERATE", "labs"
    if len(lab_abnormal) >= 3:
        return "MODERATE", "labs"
    # Any acute vital outside normal range (but not CRITICAL/ABNORMAL already handled)
    if any(c.get("status") in ("CRITICAL", "ABNORMAL") for c in acute):
        return "MODERATE", "acute_vitals"

    # ── STABLE ─────────────────────────────────────────────────────────────────
    # Only biometric abnormalities, or all clear
    any_bio_abnormal = any(c.get("status") in ("CRITICAL", "ABNORMAL") for c in bio)
    basis = "biometric_only" if any_bio_abnormal else "labs"
    return "STABLE", basis


def classify_observation(observation: dict) -> dict:
    """
    Classify a single FHIR Observation resource against LOINC_NORMAL_RANGES.

    Returns a dict with:
      name, value, unit, loinc_code, normal_range, status, grounded
    """
    # Resolve LOINC code from coding array
    loinc_code: str | None = None
    for coding in observation.get("code", {}).get("coding", []):
        if coding.get("system") == "http://loinc.org" and coding.get("code"):
            loinc_code = str(coding["code"])
            break

    # Extract numeric value from valueQuantity
    vq = observation.get("valueQuantity", {})
    raw_value = vq.get("value")
    try:
        value: float | None = float(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        value = None

    unit: str = vq.get("unit", "")

    # Default / unknown result scaffold
    result = {
        "name": observation.get("code", {}).get("text") or loinc_code or "Unknown",
        "value": value,
        "unit": unit,
        "loinc_code": loinc_code,
        "normal_range": "varies",
        "status": "UNKNOWN",
        "grounded": False,
    }

    # Always set weight — even for ungrounded observations
    result["weight"] = get_observation_weight(loinc_code)

    if loinc_code not in LOINC_NORMAL_RANGES:
        return result  # not in table — ungrounded

    ref = LOINC_NORMAL_RANGES[loinc_code]
    result["name"] = ref["name"]
    result["grounded"] = True
    # Use the table unit if observation didn't carry one
    if not result["unit"]:
        result["unit"] = ref["unit"]

    low, high = ref["low"], ref["high"]
    if low is None or high is None:
        result["normal_range"] = "varies"
        result["status"] = "UNKNOWN"
        return result

    result["normal_range"] = f"{low}-{high} {ref['unit']}"

    if value is None:
        return result  # can't classify without a numeric value

    if value > high * 1.2 or value < low * 0.8:
        result["status"] = "CRITICAL"
    elif value > high or value < low:
        result["status"] = "ABNORMAL"
    else:
        result["status"] = "NORMAL"

    result["weight"] = get_observation_weight(loinc_code)
    return result


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


RXNAV_INTERACTION_URL = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"


def fetch_rxnorm_interactions(rxcui_list: list[str]) -> dict:
    """
    Query the NIH RxNav interaction API for a list of RxCUI codes.

    Args:
        rxcui_list: List of RxNorm CUI strings (e.g. ["161", "1191"]).

    Returns:
        Parsed JSON response dict from RxNav, or {} on any failure.
    """
    if not rxcui_list:
        return {}
    rxcuis_param = " ".join(rxcui_list)
    url = f"{RXNAV_INTERACTION_URL}?rxcuis={requests.utils.quote(rxcuis_param, safe='')}"
    try:
        resp = requests.get(url, timeout=10, headers={"Accept": "application/json"})
        if resp.status_code == 200:
            return resp.json()
        print(f"[fhir_client] RxNav returned HTTP {resp.status_code} for rxcuis={rxcuis_param}")
    except Exception as e:
        print(f"[fhir_client] RxNav request failed: {e}")
    return {}


CLINICALTRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_clinical_trials(condition_name: str, state: str = None) -> list[dict]:
    """
    Query the ClinicalTrials.gov v2 public API for RECRUITING trials.

    Args:
        condition_name: Medical condition to search (e.g. "Hypertension").
        state: Optional US state to note in results (not filterable via v2 API
               query param — included for future use / logging).

    Returns:
        List of up to 3 trial dicts, or [] on any failure or empty results.
    """
    try:
        encoded = urllib.parse.quote(condition_name)
        url = (
            f"{CLINICALTRIALS_URL}"
            f"?query.cond={encoded}"
            f"&filter.overallStatus=RECRUITING"
            f"&pageSize=3"
        )
        resp = requests.get(url, timeout=5, headers={"Accept": "application/json"})
        if resp.status_code != 200:
            print(f"[fhir_client] ClinicalTrials API HTTP {resp.status_code} for '{condition_name}'")
            return []

        data = resp.json()
        studies = data.get("studies", [])
        if not studies:
            return []

        results = []
        for study in studies:
            try:
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                design = proto.get("designModule", {})
                contacts = proto.get("contactsLocationsModule", {})

                nct_id = ident.get("nctId", "")
                phases = design.get("phases", [])
                phase_str = "/".join(phases) if phases else "N/A"
                locations = contacts.get("locations", [])

                results.append({
                    "trial_id": nct_id,
                    "trial_name": ident.get("briefTitle", "N/A"),
                    "status": "RECRUITING",
                    "phase": phase_str,
                    "location_count": len(locations),
                    "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "N/A",
                })
            except Exception as inner_e:
                print(f"[fhir_client] ClinicalTrials study parse error: {inner_e}")
                continue

        return results

    except Exception as e:
        print(f"[fhir_client] ClinicalTrials request failed for '{condition_name}': {e}")
        return []


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