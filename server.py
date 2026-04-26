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

# ─── Prompt Templates ─────────────────────────────────────────────────────────

TRIAGE_SYSTEM = (
    "You are a clinical decision support system. Analyze lab and vital sign data "
    "for a hospitalized patient. Respond ONLY with a valid JSON object — no markdown, "
    "no explanation. Keys: critical_flags (array), abnormal_flags (array), "
    "trending_concern (array), overall_acuity (string: CRITICAL|HIGH|MODERATE|STABLE). "
    "Each flag item: name, value, unit, normal_range, clinical_significance."
)

MED_SAFETY_SYSTEM = (
    "You are a clinical pharmacist AI. Analyze the medication list and allergy profile. "
    "Respond ONLY with valid JSON. Keys: interactions (array of pairs with severity "
    "LOW|MODERATE|HIGH), allergy_conflicts (array), high_risk_meds (array with "
    "monitoring_required field). No markdown, no preamble."
)

PENDING_ACTIONS_SYSTEM = (
    "You are a clinical coordinator. Review recent encounters, active conditions, and lab data. "
    "Identify what is unresolved or pending. Respond ONLY with valid JSON. Keys: "
    "pending_labs (array), unresolved_conditions (array), monitoring_required (array). "
    "Each item must have name and reason fields. No markdown, no preamble."
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
            description="Returns clean JSON with demographics, conditions, and allergies.",
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
                    "compare_patient_shift", "generate_handoff_brief",
                ],
                "fhir_version": "R4",
                "smart_compliant": True,
                "clinical_purpose": "Generate clinical shift handoff summaries and triage patient data",
            }))]

        # ── get_patient_snapshot ───────────────────────────────────────────────
        elif name == "get_patient_snapshot":
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
            labs = fetch_labs(patient_id, fhir_base_url, limit=15)
            vitals = fetch_vitals(patient_id, fhir_base_url, limit=8)
            user_content = f"Labs:\n{compact(labs)}\n\nVitals:\n{compact(vitals)}"
            raw = llm.complete(TRIAGE_SYSTEM, user_content)
            return [types.TextContent(type="text", text=json.dumps(parse_json_response(raw)))]

        # ── check_medication_safety ────────────────────────────────────────────
        elif name == "check_medication_safety":
            meds = fetch_medications(patient_id, fhir_base_url)
            allergies = fetch_allergies(patient_id, fhir_base_url)
            user_content = f"Medications:\n{compact(meds)}\n\nAllergies:\n{compact(allergies)}"
            raw = llm.complete(MED_SAFETY_SYSTEM, user_content)
            return [types.TextContent(type="text", text=json.dumps(parse_json_response(raw)))]

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
            return [types.TextContent(type="text", text=json.dumps(parse_json_response(raw)))]

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

            output = f"⚠️ ACUITY: {acuity} — Review before rounds\n\n{brief_raw}"
            return [types.TextContent(type="text", text=output)]

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