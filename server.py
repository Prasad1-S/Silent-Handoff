"""
silent-handoff MCP Server
=========================
Provider-agnostic clinical handoff MCP server.
Swap LLM providers by changing LLM_PROVIDER env var — core logic never changes.

Supported providers (set LLM_PROVIDER env var):
  - groq      (default)
  - openai
  - anthropic
  - ollama    (local)
"""

import os
import sys
import json
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp import types
from fhir_client import (
    fetch_patient_demographics,
    fetch_active_conditions,
    fetch_medications,
    fetch_labs,
    fetch_vitals,
    fetch_allergies,
    fetch_recent_encounters,
    fetch_rxnorm_interactions,
    fetch_clinical_trials,
    classify_observation,
    calculate_weighted_acuity,
    get_observation_weight,
    _fhir_get,
    _bundle_resources,
)
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse
import uvicorn

load_dotenv()

# ─── LLM Provider Abstraction ─────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base — implement this to add any new LLM provider."""

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Send a system+user prompt, return the response string."""
        ...


class GroqProvider(LLMProvider):
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    # Token budget: Groq free tier TPM is 12k. Keep each call well under that.
    MAX_USER_CHARS = 5_500  # ~3 500 tokens; leaves headroom for system prompt + response

    def __init__(self):
        from groq import Groq
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        user = user[: self.MAX_USER_CHARS]  # hard-cap to avoid 413s
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.choices[0].message.content


class OpenAIProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.choices[0].message.content


class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._model = os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.content[0].text


class OllamaProvider(LLMProvider):
    """Local Ollama — no API key needed."""
    DEFAULT_MODEL = "llama3"

    def __init__(self):
        import requests as _requests
        self._requests = _requests
        self._model = os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = self._requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def build_provider() -> LLMProvider:
    """Factory — reads LLM_PROVIDER env var, defaults to groq."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    providers = {
        "groq": GroqProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }
    if provider not in providers:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose from: {list(providers)}")
    print(f"[silent-handoff] Using LLM provider: {provider}", file=sys.stderr)
    return providers[provider]()


llm = build_provider()

# ─── Prompt Version Registry ──────────────────────────────────────────────────

PROMPT_VERSIONS = {
    "triage_labs_vitals":     "1.2",
    "check_medication_safety": "1.1",
    "get_pending_actions":     "1.1",
    "compare_patient_shift":   "1.1",
    "generate_handoff_brief":  "1.2",
    "batch_handoff":           "1.0",
}

# ─── Prompt Templates ─────────────────────────────────────────────────────────

TRIAGE_SYSTEM = (
    "You are a clinical decision support system. You have been provided with "
    "pre-classified lab and vital sign data. The deterministic_critical and "
    "deterministic_abnormal lists were classified using LOINC reference ranges — "
    "treat these as verified facts, do not reclassify them. For ungrounded "
    "observations, apply your clinical judgment. Add a confidence score (0-100) "
    "to every flag: use 99 for deterministically classified items, your own "
    "estimate for ungrounded ones. "
    "The pre_calculated_acuity field was determined by a deterministic weighted "
    "algorithm that prioritizes acute vitals over biometric history. You may "
    "upgrade this acuity level based on clinical context but NEVER downgrade it. "
    "A patient with only abnormal BMI must never exceed STABLE acuity. "
    "Respond ONLY with valid JSON, no markdown, no preamble. "
    "Keys: critical_flags (array), abnormal_flags (array), "
    "trending_concern (array), overall_acuity (string: CRITICAL/HIGH/MODERATE/STABLE). "
    "Each flag must have: name, value, unit, normal_range, clinical_significance, "
    "confidence (int 0-100), grounded (bool)."
)

MED_SAFETY_SYSTEM_GROUNDED = (
    "You are a clinical pharmacist AI. Analyze the medication list and allergy profile. "
    "The following drug interactions are verified from the NIH RxNav database. "
    "Use ONLY these verified interactions, do not add any from memory. "
    "Respond ONLY with valid JSON. Keys: interactions (array of pairs with severity "
    "LOW|MODERATE|HIGH), allergy_conflicts (array), high_risk_meds (array with "
    "monitoring_required field). No markdown, no preamble."
)

MED_SAFETY_SYSTEM_UNGROUNDED = (
    "You are a clinical pharmacist AI. Analyze the medication list and allergy profile. "
    "No verified drug interaction data is available for this patient (no RxNorm codes "
    "found in FHIR data). Provide your best LLM-based analysis but note it is unverified. "
    "Respond ONLY with valid JSON. Keys: interactions (array of pairs with severity "
    "LOW|MODERATE|HIGH), allergy_conflicts (array), high_risk_meds (array with "
    "monitoring_required field). No markdown, no preamble."
)

PENDING_ACTIONS_SYSTEM = (
    "You are a clinical coordinator. Review recent encounters, active conditions, and lab data. "
    "Identify what is unresolved or pending. Respond ONLY with valid JSON. Keys: "
    "pending_labs (array), unresolved_conditions (array), monitoring_required (array). "
    "Each item must have name and reason fields. No markdown, no preamble. "
    "Do not comment on clinical trials — that data is appended separately from a verified government source."
)

SHIFT_COMPARE_SYSTEM = (
    "You are interpreting shifted vital trends. Compare recent vitals vs prior 24 hours for: "
    "heart_rate, systolic_bp, diastolic_bp, temperature, spo2, respiratory_rate. "
    "Provide a one-sentence clinical interpretation of the trend. "
    "Respond ONLY with valid JSON. Keys: improved (array), deteriorated (array), "
    "new_findings (array), clinical_interpretation (string), hours_compared (integer)."
)

def handoff_system(fmt: str) -> str:
    return (
        f"You are a senior charge nurse composing a clinical handoff brief. "
        f"Generate a structured {fmt} handoff in plain clinical English. "
        f"Be specific, prioritized, and actionable. The incoming team has 90 seconds to read this. "
        f"Flag anything that cannot wait. Use acuity level to set urgency tone.\n\n"
        f"Format definitions:\n"
        f"- SBAR: Situation / Background / Assessment / Recommendation\n"
        f"- IPASS: Illness severity / Patient summary / Action list / Situation awareness / Synthesis\n"
        f"- SOAP: Subjective / Objective / Assessment / Plan\n"
        f"Follow the {fmt} format strictly."
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    """Safely parse LLM JSON — strips markdown fences if present."""
    cleaned = raw.strip().strip("```").removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"raw_response": raw}


def compact(data, limit: int = 3_000) -> str:
    """JSON-encode and truncate to a safe character limit."""
    return json.dumps(data)[:limit]


def err_res(msg: str, masked_pid: str, e_type: str = "unknown"):
    return [types.TextContent(type="text", text=json.dumps({
        "error": msg,
        "error_type": e_type,
        "patient_id": masked_pid,
        "suggestion": "Check patient_id or FHIR server",
    }))]

# ─── MCP Server ───────────────────────────────────────────────────────────────

server = Server("silent-handoff")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="server_info",
            description="Returns server metadata.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_patient_snapshot",
            description="Returns clean JSON with demographics, conditions, and allergies. Set demo=true for a cached example response without FHIR calls.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "fhir_base_url": {"type": "string"},
                    "demo": {"type": "boolean", "description": "If true, return hardcoded demo data without FHIR calls."},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="triage_labs_vitals",
            description="Analyzes lab and vital sign data; returns triage JSON with acuity level.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "fhir_base_url": {"type": "string"},
                },
                "required": ["patient_id"],
            },
        ),
        types.Tool(
            name="check_medication_safety",
            description="Analyzes medications and allergies for conflicts and interactions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "fhir_base_url": {"type": "string"},
                },
                "required": ["patient_id"],
            },
        ),
        types.Tool(
            name="get_pending_actions",
            description="Identifies unresolved or pending labs, conditions, and monitoring needs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "fhir_base_url": {"type": "string"},
                },
                "required": ["patient_id"],
            },
        ),
        types.Tool(
            name="compare_patient_shift",
            description="Compares patient vitals over the last N hours vs prior 24 hours.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "hours_lookback": {"type": "integer"},
                    "fhir_base_url": {"type": "string"},
                },
                "required": ["patient_id"],
            },
        ),
        types.Tool(
            name="generate_handoff_brief",
            description="Generates a clinical shift handoff brief (SBAR/IPASS/SOAP) from FHIR data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["SBAR", "IPASS", "SOAP"]},
                    "fhir_base_url": {"type": "string"},
                },
                "required": ["patient_id"],
            },
        ),
        types.Tool(
            name="batch_handoff",
            description="Generates prioritized handoff briefs for multiple patients, ranked by acuity. Max 5 patients per call.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of FHIR patient IDs (max 5)",
                    },
                    "fhir_base_url": {"type": "string"},
                    "format": {"type": "string", "enum": ["SBAR", "IPASS", "SOAP"]},
                },
                "required": ["patient_ids"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    patient_id = arguments.get("patient_id", "")
    masked_pid = f"***{patient_id[-4:]}" if len(patient_id) >= 4 else ("***" if patient_id else "N/A")
    print(f"[{datetime.now().isoformat()}] TOOL CALL: {name} | PATIENT: {masked_pid}", file=sys.stderr)

    fhir_base_url = arguments.get("fhir_base_url") or os.getenv("FHIR_BASE_URL", "https://r4.smarthealthit.org")

    try:
        # ── server_info ────────────────────────────────────────────────────────
        if name == "server_info":
            return [types.TextContent(type="text", text=json.dumps({
                "server_name": "silent-handoff",
                "version": "3.0.0",
                "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
                "transport_modes": ["stdio", "sse"],
                "available_tools": [
                    "server_info", "get_patient_snapshot", "triage_labs_vitals",
                    "check_medication_safety", "get_pending_actions",
                    "compare_patient_shift", "generate_handoff_brief", "batch_handoff",
                ],
                "fhir_version": "R4",
                "smart_compliant": True,
                "clinical_purpose": "Generate clinical shift handoff summaries and triage patient data",
                "prompt_versions": PROMPT_VERSIONS,
                "grounding_sources": ["rxnav_nih_gov", "loinc_org"],
                "available_formats": ["SBAR", "IPASS", "SOAP"],
            }))]

        # ── get_patient_snapshot ───────────────────────────────────────────────
        elif name == "get_patient_snapshot":
            if arguments.get("demo", False):
                return [types.TextContent(type="text", text=json.dumps({
                    "demo_mode": True,
                    "demographics": {
                        "name": "Geoffrey Abbott",
                        "gender": "male",
                        "birth_date": "1992-07-03",
                        "age": 33,
                        "city": "Southborough",
                        "state": "Massachusetts",
                    },
                    "conditions": [
                        {
                            "condition": "Hypertension",
                            "status": "active",
                            "onset": "2011-05-27",
                        }
                    ],
                    "allergies": [],
                    "note": "Demo mode — using cached real patient data. Set demo=false to fetch live FHIR data.",
                }))]
            if not patient_id:
                return [types.TextContent(type="text", text=json.dumps({
                    "error": "patient_id is required when demo is false",
                    "error_type": "ValidationError",
                }))]
            demographics = fetch_patient_demographics(patient_id, fhir_base_url)
            conditions = fetch_active_conditions(patient_id, fhir_base_url)
            allergies = fetch_allergies(patient_id, fhir_base_url)
            return [types.TextContent(type="text", text=json.dumps({
                "demographics": demographics,
                "conditions": conditions,
                "allergies": allergies,
            }))]

        # ── triage_labs_vitals ─────────────────────────────────────────────────
        elif name == "triage_labs_vitals":
            # — Step 1: Fetch raw FHIR bundles
            raw_labs_bundle = _fhir_get(
                f"{fhir_base_url}/Observation?patient={patient_id}"
                f"&category=laboratory&_sort=-date&_count=15"
            )
            raw_vitals_bundle = _fhir_get(
                f"{fhir_base_url}/Observation?patient={patient_id}"
                f"&category=vital-signs&_sort=-date&_count=8"
            )
            all_raw = (
                _bundle_resources(raw_labs_bundle or {})
                + _bundle_resources(raw_vitals_bundle or {})
            )

            # — Step 2: Classify and split by status
            classified = [classify_observation(obs) for obs in all_raw]

            det_critical   = [c for c in classified if c["grounded"] and c["status"] == "CRITICAL"]
            det_abnormal   = [c for c in classified if c["grounded"] and c["status"] == "ABNORMAL"]
            ungrounded     = [c for c in classified if not c["grounded"]]
            grounded_count = sum(1 for c in classified if c["grounded"])

            # — Step 3: Deterministic weighted acuity (floor the LLM can only raise)
            pre_acuity, acuity_basis = calculate_weighted_acuity(classified)

            # Collect ACUTE vital names present in data for reporting
            acute_vitals_checked = [
                c["name"] for c in classified
                if c.get("weight") == "ACUTE" and c.get("grounded")
            ]

            print(
                f"[triage_labs_vitals] grounded={grounded_count} "
                f"pre_acuity={pre_acuity} basis={acuity_basis} "
                f"critical={len(det_critical)} abnormal={len(det_abnormal)} "
                f"ungrounded={len(ungrounded)} | patient={masked_pid}",
                file=sys.stderr,
            )

            # — Step 4: Build prompt with pre-calculated acuity as a floor
            user_content = (
                f"pre_calculated_acuity: {pre_acuity}\n\n"
                f"deterministic_critical:\n{compact(det_critical, 1_400)}\n\n"
                f"deterministic_abnormal:\n{compact(det_abnormal, 1_400)}\n\n"
                f"ungrounded_observations:\n{compact(ungrounded, 1_400)}"
            )
            raw = llm.complete(TRIAGE_SYSTEM, user_content)
            result = parse_json_response(raw)

            # — Step 5: Attach metadata; enforce the acuity floor
            ACUITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "STABLE": 3, "UNKNOWN": 4}
            llm_acuity = result.get("overall_acuity", pre_acuity).upper()
            # Never let the LLM downgrade below the deterministic floor
            if ACUITY_ORDER.get(llm_acuity, 4) > ACUITY_ORDER.get(pre_acuity, 4):
                result["overall_acuity"] = pre_acuity

            result["loinc_grounded_count"]  = grounded_count
            result["data_source"]           = "loinc_deterministic+llm_inference"
            result["prompt_version"]        = PROMPT_VERSIONS["triage_labs_vitals"]
            result["acuity_method"]         = "weighted_deterministic+llm_review"
            result["acute_vitals_checked"]  = acute_vitals_checked

            return [types.TextContent(type="text", text=json.dumps(result))]

        # ── check_medication_safety ────────────────────────────────────────────
        elif name == "check_medication_safety":
            # Fetch the lean medication list (for display / LLM context)
            meds = fetch_medications(patient_id, fhir_base_url)
            allergies = fetch_allergies(patient_id, fhir_base_url)

            # Extract RxNorm CUIs from raw FHIR MedicationRequest resources
            rxnorm_system = "http://www.nlm.nih.gov/research/umls/rxnorm"
            raw_bundle = _fhir_get(
                f"{fhir_base_url}/MedicationRequest?patient={patient_id}&_count=20"
            )
            rxcui_list: list[str] = []
            for resource in _bundle_resources(raw_bundle or {}):
                if resource.get("status") in ("entered-in-error", "cancelled"):
                    continue
                for coding in (
                    resource
                    .get("medicationCodeableConcept", {})
                    .get("coding", [])
                ):
                    if coding.get("system") == rxnorm_system and coding.get("code"):
                        rxcui_list.append(str(coding["code"]))

            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_rxcuis = [c for c in rxcui_list if not (c in seen or seen.add(c))]
            print(
                f"[check_medication_safety] RxCUIs found: {unique_rxcuis} | patient={masked_pid}",
                file=sys.stderr,
            )

            if len(unique_rxcuis) >= 2:
                # Ground the interaction check with live NIH RxNav data
                rxnav_data = fetch_rxnorm_interactions(unique_rxcuis)
                user_content = (
                    f"Medications:\n{compact(meds)}\n\n"
                    f"Allergies:\n{compact(allergies)}\n\n"
                    f"verified_interactions (NIH RxNav):\n{compact(rxnav_data, 2_000)}"
                )
                raw = llm.complete(MED_SAFETY_SYSTEM_GROUNDED, user_content)
                result = parse_json_response(raw)
                result["data_source"] = "rxnav_nih_gov"
                result["rxcuis_checked"] = unique_rxcuis
            else:
                # Fall back to LLM-only analysis — note the limitation
                user_content = (
                    f"Medications:\n{compact(meds)}\n\n"
                    f"Allergies:\n{compact(allergies)}\n\n"
                    f"Note: unverified - no RxNorm codes found in FHIR data"
                )
                raw = llm.complete(MED_SAFETY_SYSTEM_UNGROUNDED, user_content)
                result = parse_json_response(raw)
                result["data_source"] = "llm_inference_only"
                result["note"] = "unverified - no RxNorm codes found in FHIR data"

            result["prompt_version"] = PROMPT_VERSIONS["check_medication_safety"]
            return [types.TextContent(type="text", text=json.dumps(result))]

        # ── get_pending_actions ────────────────────────────────────────────────
        elif name == "get_pending_actions":
            encounters = fetch_recent_encounters(patient_id, fhir_base_url, limit=5)
            conditions = fetch_active_conditions(patient_id, fhir_base_url)
            labs = fetch_labs(patient_id, fhir_base_url, limit=15)
            user_content = (
                f"Recent Encounters:\n{compact(encounters, 1_500)}\n\n"
                f"Active Conditions:\n{compact(conditions, 1_500)}\n\n"
                f"Labs:\n{compact(labs, 1_500)}"
            )
            raw = llm.complete(PENDING_ACTIONS_SYSTEM, user_content)
            result = parse_json_response(raw)
            result["prompt_version"] = PROMPT_VERSIONS["get_pending_actions"]

            # — Clinical trials enrichment —
            # Pull patient state for context (best-effort, None is fine)
            patient_state: str | None = None
            try:
                demo_data = fetch_patient_demographics(patient_id, fhir_base_url)
                raw_state = demo_data.get("state", "")
                patient_state = raw_state if raw_state and raw_state != "N/A" else None
            except Exception:
                pass

            # Query ClinicalTrials.gov for up to 2 unresolved conditions
            unresolved = result.get("unresolved_conditions", [])
            trials_by_condition = []
            any_found = False
            for cond in unresolved[:2]:
                cond_name = cond.get("name", "") if isinstance(cond, dict) else str(cond)
                if not cond_name:
                    continue
                trials = fetch_clinical_trials(cond_name, state=patient_state)
                if trials:
                    any_found = True
                trials_by_condition.append({
                    "condition": cond_name,
                    "trials": trials,
                })

            result["clinical_trials_nearby"] = trials_by_condition
            result["trials_data_source"] = "clinicaltrials_gov"
            if not any_found:
                result["trials_note"] = "No recruiting trials found for current conditions"

            return [types.TextContent(type="text", text=json.dumps(result))]

        # ── compare_patient_shift ──────────────────────────────────────────────
        elif name == "compare_patient_shift":
            hours_lookback = int(arguments.get("hours_lookback", 12))
            vitals = fetch_vitals(patient_id, fhir_base_url, limit=50)

            last_n, prior_24 = [], []
            now = datetime.now(timezone.utc)
            for v in vitals:
                dt_str = v.get("effectiveDateTime")
                if not dt_str:
                    continue
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    hrs = (now - dt).total_seconds() / 3600.0
                    if hrs <= hours_lookback:
                        last_n.append(v)
                    elif hrs <= hours_lookback + 24:
                        prior_24.append(v)
                except Exception:
                    pass

            user_content = (
                f"Last {hours_lookback} hours vitals:\n{compact(last_n)}\n\n"
                f"Prior 24 hours vitals:\n{compact(prior_24)}\n\n"
                f"Lookback window: {hours_lookback} hours"
            )
            raw = llm.complete(SHIFT_COMPARE_SYSTEM, user_content)
            parsed = parse_json_response(raw)
            parsed["hours_compared"] = hours_lookback
            parsed["prompt_version"] = PROMPT_VERSIONS["compare_patient_shift"]
            return [types.TextContent(type="text", text=json.dumps(parsed))]

        # ── generate_handoff_brief ─────────────────────────────────────────────
        elif name == "generate_handoff_brief":
            fmt = arguments.get("format", "SBAR").upper()

            # Fetch FHIR data with conservative limits to stay under token budgets
            demographics = fetch_patient_demographics(patient_id, fhir_base_url)
            conditions = fetch_active_conditions(patient_id, fhir_base_url)
            meds = fetch_medications(patient_id, fhir_base_url)
            labs = fetch_labs(patient_id, fhir_base_url, limit=10)
            vitals = fetch_vitals(patient_id, fhir_base_url, limit=5)

            # Step 1: Triage — compact labs+vitals only, single focused call
            triage_user = f"Labs:\n{compact(labs)}\n\nVitals:\n{compact(vitals)}"
            triage_raw = llm.complete(TRIAGE_SYSTEM, triage_user)
            triage = parse_json_response(triage_raw)
            acuity = triage.get("overall_acuity", "UNKNOWN")

            # Step 2: Handoff brief — uses triage SUMMARY (not raw labs/vitals again)
            # This is the key fix: we never duplicate the raw data across two calls
            synth_user = (
                f"Acuity Level: {acuity}\n\n"
                f"Demographics:\n{compact(demographics, 800)}\n\n"
                f"Active Conditions:\n{compact(conditions, 1_000)}\n\n"
                f"Medications:\n{compact(meds, 1_000)}\n\n"
                f"Triage Summary (labs + vitals already analyzed):\n{compact(triage, 1_200)}"
            )
            brief_raw = llm.complete(handoff_system(fmt), synth_user, temperature=0.2)

            version = PROMPT_VERSIONS["generate_handoff_brief"]
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            provider_name = os.getenv("LLM_PROVIDER", "groq")
            footer = f"\n\n---\nPrompt Version: {version} | Model: {model_name} | Provider: {provider_name}"
            output = f"⚠️ ACUITY: {acuity} — Review before rounds\n\n{brief_raw}{footer}"
            return [types.TextContent(type="text", text=output)]

        # ── batch_handoff ──────────────────────────────────────────────────────
        elif name == "batch_handoff":
            patient_ids = arguments.get("patient_ids", [])
            fmt = arguments.get("format", "SBAR").upper()
            print(
                f"[batch_handoff] called for {len(patient_ids)} patients",
                file=sys.stderr,
            )

            if len(patient_ids) > 5:
                return [types.TextContent(type="text", text=json.dumps({
                    "error": "Too many patients — max 5 per call.",
                    "error_type": "ValidationError",
                    "received": len(patient_ids),
                }))]

            # Tiebreaker order within same acuity: acute_vitals < labs < biometric_only
            BASIS_RANK = {"acute_vitals": 0, "labs": 1, "biometric_only": 2}
            ACUITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "STABLE": 3}

            BATCH_SUMMARY_SYSTEM = (
                "You are a clinical charge nurse. Write exactly 3 sentences summarising "
                "this patient's current status for an incoming nurse. Be specific and "
                "clinical. No bullet points. Plain text only."
            )

            patient_records = []
            for pid in patient_ids:
                try:
                    demo = fetch_patient_demographics(pid, fhir_base_url)
                    labs_b = _fhir_get(
                        f"{fhir_base_url}/Observation?patient={pid}"
                        f"&category=laboratory&_sort=-date&_count=10"
                    )
                    vitals_b = _fhir_get(
                        f"{fhir_base_url}/Observation?patient={pid}"
                        f"&category=vital-signs&_sort=-date&_count=6"
                    )
                    all_obs = (
                        _bundle_resources(labs_b or {})
                        + _bundle_resources(vitals_b or {})
                    )
                    classified = [classify_observation(obs) for obs in all_obs]

                    # Deterministic weighted acuity — no LLM call needed for ranking
                    acuity, basis = calculate_weighted_acuity(classified)

                    det_critical = [c for c in classified if c["grounded"] and c["status"] == "CRITICAL"]
                    det_abnormal = [c for c in classified if c["grounded"] and c["status"] == "ABNORMAL"]

                    summary_user = (
                        f"Patient: {demo.get('name', 'Unknown')}, "
                        f"age {demo.get('age', '?')}, {demo.get('gender', '?')}\n"
                        f"Acuity: {acuity} (basis: {basis})\n"
                        f"Critical findings: {compact(det_critical, 400)}\n"
                        f"Abnormal findings: {compact(det_abnormal, 400)}"
                    )
                    summary = llm.complete(BATCH_SUMMARY_SYSTEM, summary_user, temperature=0.2)

                    patient_records.append({
                        "patient_id": pid,
                        "name": demo.get("name", "Unknown"),
                        "age": demo.get("age", "N/A"),
                        "acuity": acuity,
                        "acuity_basis": basis,
                        "summary": summary.strip(),
                        "prompt_version": PROMPT_VERSIONS["batch_handoff"],
                        "_acuity_rank": ACUITY_RANK.get(acuity, 99),
                        "_basis_rank": BASIS_RANK.get(basis, 1),
                    })
                except Exception as pe:
                    patient_records.append({
                        "patient_id": pid,
                        "error": str(pe),
                        "acuity": "UNKNOWN",
                        "acuity_basis": "unknown",
                        "_acuity_rank": 99,
                        "_basis_rank": 99,
                    })

            # Sort: primary acuity level, tiebreaker by basis
            patient_records.sort(key=lambda p: (p["_acuity_rank"], p["_basis_rank"]))
            for rank, rec in enumerate(patient_records, start=1):
                rec["rank"] = rank
                rec.pop("_acuity_rank", None)
                rec.pop("_basis_rank", None)

            counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "STABLE": 0}
            for rec in patient_records:
                acuity_key = rec.get("acuity", "STABLE")
                if acuity_key in counts:
                    counts[acuity_key] += 1

            return [types.TextContent(type="text", text=json.dumps({
                "total_patients": len(patient_records),
                "critical_count": counts["CRITICAL"],
                "high_count": counts["HIGH"],
                "moderate_count": counts["MODERATE"],
                "stable_count": counts["STABLE"],
                "patients": patient_records,
                "data_source": "fhir_r4_live",
                "ranking_note": (
                    "Acuity weighted toward acute vitals (SpO2, HR, RR, BP, Temp). "
                    "Chronic biometric findings ranked lower to reduce alert fatigue."
                ),
            }))]

        return err_res(f"Unknown tool: {name}", masked_pid, "InvalidToolError")

    except Exception as e:
        return err_res(str(e), masked_pid, type(e).__name__)


# ─── SSE Transport ────────────────────────────────────────────────────────────

sse_transport = SseServerTransport("/messages/")

async def handle_sse(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

async def health(request: Request):
    return JSONResponse({"status": "ok", "provider": os.getenv("LLM_PROVIDER", "groq")})

starlette_app = Starlette(
    routes=[
        Route("/health", health),
        Route("/sse", handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)

# ─── Entrypoint ───────────────────────────────────────────────────────────────

async def main_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    mode = os.getenv("TRANSPORT", "stdio")
    if mode == "sse":
        uvicorn.run(starlette_app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    else:
        asyncio.run(main_stdio())