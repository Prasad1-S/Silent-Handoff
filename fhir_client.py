import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_nested(data, keys, default="Not available"):
    """Safely get a value from a nested dictionary."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        elif isinstance(data, list) and isinstance(key, int) and len(data) > key:
            data = data[key]
        else:
            return default
        if data is None:
            return default
    return data

def calculate_age(birth_date_str):
    if not birth_date_str or birth_date_str == "Not available":
        return "Not available"
    try:
        birth_date = datetime.strptime(birth_date_str[:10], "%Y-%m-%d")
        today = datetime.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError):
        return "Not available"

def get_patient_summary(patient_id, base_url=None):
    if not base_url:
        base_url = os.getenv("base_url", "https://r4.smarthealthit.org")
    summary = {
        "name": "Not available",
        "age": "Not available",
        "gender": "Not available",
        "mrn": "Not available",
        "conditions": [],
        "medications": [],
        "labs": [],
        "vitals": []
    }
    
    # 1. Fetch Patient
    patient_url = f"{base_url}/Patient/{patient_id}"
    try:
        response = requests.get(patient_url, timeout=10)
        if response.status_code == 200:
            p_data = response.json()
            
            # Name
            names = p_data.get("name", [])
            if names:
                first_name = " ".join(names[0].get("given", []))
                last_name = names[0].get("family", "")
                summary["name"] = f"{first_name} {last_name}".strip() or "Not available"
                
            # Gender
            summary["gender"] = p_data.get("gender", "Not available")
            
            # Age
            birth_date = p_data.get("birthDate", "Not available")
            if birth_date != "Not available":
                summary["age"] = calculate_age(birth_date)
            
            # MRN
            identifiers = p_data.get("identifier", [])
            for ident in identifiers:
                if get_nested(ident, ["type", "coding", 0, "code"]) == "MR":
                    summary["mrn"] = ident.get("value", "Not available")
                    break
            if summary["mrn"] == "Not available" and identifiers:
                summary["mrn"] = identifiers[0].get("value", "Not available")
    except Exception as e:
        print(f"Error fetching patient: {e}")

    # 2. Fetch active Conditions
    try:
        cond_url = f"{base_url}/Condition?patient={patient_id}&clinical-status=active"
        cond_resp = requests.get(cond_url, timeout=10)
        if cond_resp.status_code == 200:
            for entry in cond_resp.json().get("entry", []):
                resource = entry.get("resource", {})
                display = get_nested(resource, ["code", "coding", 0, "display"], None)
                text = get_nested(resource, ["code", "text"], None)
                if display:
                    summary["conditions"].append(display)
                elif text:
                    summary["conditions"].append(text)
    except Exception as e:
        print(f"Error fetching conditions: {e}")

    # 3. Fetch last 5 Observation resources
    try:
        obs_url = f"{base_url}/Observation?patient={patient_id}&_sort=-date&_count=20"
        obs_resp = requests.get(obs_url, timeout=10)
        if obs_resp.status_code == 200:
            for entry in obs_resp.json().get("entry", []):
                resource = entry.get("resource", {})
                category = get_nested(resource, ["category", 0, "coding", 0, "code"], "unknown")
                obs_name = get_nested(resource, ["code", "coding", 0, "display"], get_nested(resource, ["code", "text"]))
                
                value = get_nested(resource, ["valueQuantity", "value"], "")
                unit = get_nested(resource, ["valueQuantity", "unit"], "")
                date_str = resource.get("effectiveDateTime", "Not available")
                
                obs_str = f"{obs_name}: {value} {unit} ({date_str})"
                
                if "vital-signs" in category.lower() or category == "vital-signs":
                    summary["vitals"].append(obs_str)
                else:
                    summary["labs"].append(obs_str)
    except Exception as e:
        print(f"Error fetching observations: {e}")

    # 4. Fetch active MedicationRequests
    try:
        med_url = f"{base_url}/MedicationRequest?patient={patient_id}&_count=10"
        med_resp = requests.get(med_url, timeout=10)
        if med_resp.status_code == 200:
            for entry in med_resp.json().get("entry", []):
                resource = entry.get("resource", {})
                status = resource.get("status", "")
                if status in ["entered-in-error", "cancelled"]:
                    continue
                med_name = get_nested(resource, ["medicationCodeableConcept", "coding", 0, "display"], 
                                      get_nested(resource, ["medicationCodeableConcept", "text"]))
                if med_name and med_name != "Not available":
                    summary["medications"].append(med_name)
    except Exception as e:
        print(f"Error fetching medications: {e}")

    for key in ["conditions", "medications", "labs", "vitals"]:
        if not summary[key]:
            summary[key] = "Not available"

    return summary
