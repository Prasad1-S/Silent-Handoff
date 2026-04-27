# Silent Handoff MCP Server

Silent Handoff is a Model Context Protocol (MCP) server designed to automate and augment the generation of clinical shift handoff summaries. By directly interfacing with a FHIR (Fast Healthcare Interoperability Resources) server, this tool fetches live patient data (including demographics, conditions, medications, labs, vitals, allergies, and recent encounters), performs deterministic weighted triage, and synthesizes a structured, highly actionable brief for incoming medical personnel.

This project implements the MCP `stdio` and `sse` (Server-Sent Events) transports. It seamlessly integrates into any MCP-compatible client (like Claude Desktop) as a local extension via `stdio`, or can run as an independent web service over HTTP via `sse`.

---

## Architecture & Workflow

The core logic resides in `server.py`, `fhir_client.py`, and `prompts.py`. When the host client invokes the tools, the system executes the following sequence:

1. **Data Ingestion (`fhir_client.py`)**: Uses a lightweight wrapper built on `requests` to pull Patient, Condition, Observation (Labs/Vitals), MedicationRequest, AllergyIntolerance, and Encounter resources via FHIR REST APIs. Compresses nested FHIR JSON down to flat, token-efficient structures designed for LLM consumption.

2. **Deterministic Weighted Triage**: Before any LLM call, observations are classified using LOINC reference ranges and weighted by clinical urgency. Acute vitals (SpO2, HR, RR, BP, Temp) are evaluated first with hard-coded critical thresholds. Chronic biometric measurements (BMI, weight) can never drive acuity above STABLE on their own. This eliminates alert fatigue from chronic background noise.

3. **LLM Abstraction Layer**: An `LLMProvider` base class makes it trivial to swap AI backends. Natively supports **Groq**, **OpenAI**, **Anthropic**, and local **Ollama**.

4. **Clinical Triage & Actionable Synthesis**: Zero-temperature parsing for determinism on all analytical tools. The LLM may only raise — never lower — the pre-calculated acuity floor. Final narrative synthesis uses `temperature=0.2`.

5. **Final Handoff Delivery**: Synthesizes structured patient data and triage analysis into a prioritized narrative formatted strictly as SBAR, IPASS, or SOAP. Delivered back to the host client via MCP protocol.

---

## Available Tools

The server exposes **8 tools** to the client:

| Tool | Description | Inputs | LLM Call | Output |
|------|-------------|--------|----------|--------|
| `server_info` | Returns server metadata, capabilities, and grounding sources | none | No | JSON manifest |
| `get_patient_snapshot` | Returns demographics, conditions, and allergies. Supports `demo=true` for keyless testing | `patient_id`, `fhir_base_url`, `demo` | No | JSON |
| `triage_labs_vitals` | Deterministic LOINC-grounded triage with confidence scores and weighted acuity | `patient_id`, `fhir_base_url` | Yes (temp 0.0) | JSON |
| `check_medication_safety` | NIH RxNav-grounded drug interaction and allergy conflict check | `patient_id`, `fhir_base_url` | Yes (temp 0.0) | JSON |
| `get_pending_actions` | Unresolved conditions, pending labs, monitoring needs + recruiting clinical trials | `patient_id`, `fhir_base_url` | Yes (temp 0.0) | JSON |
| `compare_patient_shift` | Vital sign delta across shift window — what improved, deteriorated, or is new | `patient_id`, `hours_lookback`, `fhir_base_url` | Yes (temp 0.0) | JSON |
| `generate_handoff_brief` | Full SBAR / IPASS / SOAP handoff brief with acuity banner | `patient_id`, `format`, `fhir_base_url` | Yes (temp 0.2) | Text |
| `batch_handoff` | Ward-level triage — runs handoff for up to 5 patients, ranked by acuity | `patient_ids`, `format`, `fhir_base_url` | Yes (temp 0.2) | JSON |

---

## Clinical Intelligence Features

### Deterministic Weighted Triage (triage_labs_vitals, batch_handoff)

Observations are classified into three weight tiers before any LLM involvement:

| Tier | LOINC Codes | Examples | Can Drive CRITICAL? |
|------|-------------|----------|---------------------|
| ACUTE | SpO2, HR, RR, SBP, DBP, Temp | Real-time vitals | Yes |
| LAB | Glucose, HbA1c, Creatinine, Hemoglobin | Blood work | Yes (2+ critical) |
| BIOMETRIC | BMI, Weight, Height | Chronic measurements | Never above STABLE |

Hard-coded critical thresholds applied deterministically:
- SpO2 < 90% → **CRITICAL**
- Heart Rate > 130 or < 40 bpm → **CRITICAL**
- Systolic BP > 180 or < 80 mmHg → **CRITICAL**
- Respiratory Rate > 30/min → **CRITICAL**
- Temperature > 39.5°C or < 35.0°C → **CRITICAL**

The LLM receives `pre_calculated_acuity` as a floor it may only raise, never lower. Every flag includes a `confidence` score (0–100) and a `grounded` boolean indicating whether the range came from LOINC reference data or LLM inference.

### NIH RxNav Drug Interaction Grounding (check_medication_safety)

RxCUI codes are extracted directly from FHIR MedicationRequest resources and submitted to the [NIH RxNav API](https://rxnav.nlm.nih.gov) for verified interaction checking. Results include `data_source: "rxnav_nih_gov"` and `rxcuis_checked` for full auditability. The LLM is instructed to use only verified interactions — no inference from training data.

### ClinicalTrials.gov Integration (get_pending_actions)

For each unresolved condition identified, the tool queries [ClinicalTrials.gov](https://clinicaltrials.gov) for actively recruiting trials. Results include trial ID, phase, location count, and a direct URL. Degrades gracefully if the API is unavailable.

### Batch Ward Triage (batch_handoff)

Acuity ranking for multi-patient batches uses the same deterministic `calculate_weighted_acuity` function — no LLM call needed per patient for ranking. Within the same acuity level, patients are sorted: acute_vitals first → labs → biometric_only last, minimizing alert fatigue.

---

## Supported LLM Providers

Set `LLM_PROVIDER` in your `.env` to switch models dynamically:

| Provider | Default Model | API Key Env Var |
|----------|---------------|-----------------|
| `groq` (default) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `ollama` | `llama3` | `OLLAMA_BASE_URL` |

---

## SHARP-on-MCP Compliance & Clinical Safety Design

| Safety Property | Implementation |
|-----------------|----------------|
| No Data Retention | Patient data is never stored locally or logged to stdout |
| Traceable Logs | All tool calls logged to stderr with masked patient IDs (e.g. `***1234`) |
| Hallucination Mitigation | All analytical tools run at `temperature=0.0` with enforced JSON schemas |
| Deterministic Acuity Floor | LLM may only raise pre-calculated acuity, never lower it |
| LOINC-Grounded Ranges | Normal ranges sourced from LOINC reference table, not LLM memory |
| RxNorm-Grounded Interactions | Drug interactions verified via NIH RxNav API, not LLM inference |
| SSRF Protection | FHIR URLs validated against private IP ranges before any request |
| Graceful Fallbacks | Missing FHIR resources return empty subsets without crashing |
| Prompt Versioning | Every LLM-powered tool output includes `prompt_version` for auditability |

---

## Prerequisites

- **Python 3.10+**
- **Git**
- An API key for your LLM of choice (e.g., [Groq](https://console.groq.com/)), or a local [Ollama](https://ollama.com/) instance.

---

## Environment Setup & Installation

### 1. Clone & Navigate
```bash
git clone <repository_url>
cd silent-handoff
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
```

Activate:
- **Linux/macOS:** `source venv/bin/activate`
- **Windows:** `.\venv\Scripts\activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Key dependencies: `mcp`, `starlette`, `uvicorn`, `requests`, `python-dotenv`, `groq`, `openai`, `anthropic`, `pydantic`

### 4. Configure Environment Variables

Create a `.env` file:
```bash
touch .env
```

```env
# Choose LLM Provider: 'groq', 'openai', 'anthropic', or 'ollama'
LLM_PROVIDER="groq"
GROQ_API_KEY="gsk_your_api_key_here"

# (Optional) OpenAI
# OPENAI_API_KEY="sk-..."
# OPENAI_MODEL="gpt-4o"

# (Optional) Anthropic
# ANTHROPIC_API_KEY="sk-ant-..."
# ANTHROPIC_MODEL="claude-haiku-4-5-20251001"

# (Optional) Ollama — fully local, no API key needed
# OLLAMA_BASE_URL="http://localhost:11434"
# OLLAMA_MODEL="llama3"

# FHIR Server (defaults to public R4 sandbox)
FHIR_BASE_URL="https://r4.smarthealthit.org"

# Transport mode: "stdio" or "sse" (defaults to stdio)
TRANSPORT="stdio"

# Port for SSE server (defaults to 8000)
PORT=8000
```

---

## Usage

### Running in SSE Mode
Set `TRANSPORT=sse` in `.env`, then:
```bash
python server.py
```
- SSE Endpoint: `http://localhost:8000/sse`
- Health Check: `http://localhost:8000/health`

### Running in STDIO Mode
```bash
python server.py
```
The server waits for JSON-RPC messages via stdin. Use `Ctrl+C` to terminate.

### Testing with MCP Inspector (Recommended)
The fastest way to test all tools interactively:
```bash
npx @modelcontextprotocol/inspector
```
Connect to `http://localhost:8000/sse`. All 8 tools appear as forms — no manual curl or handshake needed.

### Configuring Claude Desktop
Update `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "silent-handoff": {
      "command": "/absolute/path/to/silent-handoff/venv/bin/python",
      "args": ["/absolute/path/to/silent-handoff/server.py"],
      "env": {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "gsk_your_api_key_here",
        "FHIR_BASE_URL": "https://r4.smarthealthit.org"
      }
    }
  }
}
```

Restart Claude Desktop, then try: *"Generate an IPASS handoff brief for patient ID [XYZ]"*

---

## Finding Critical Patients in the Public Sandbox

For a meaningful demo, search the R4 sandbox for genuinely acute patients:

```bash
# Patients with low oxygen (SpO2 < 90)
curl "https://r4.smarthealthit.org/Observation?code=2708-6&value-quantity=lt90&_count=5"

# Patients with high heart rate (> 110 bpm)
curl "https://r4.smarthealthit.org/Observation?code=8867-4&value-quantity=gt110&_count=5"

# Patients with Sepsis or Pneumonia
curl "https://r4.smarthealthit.org/Condition?code=sepsis&_count=5"
```

Extract a patient `id` from the results and use it across tools for a clinically realistic demo.

---

## Quick Demo (No FHIR Needed)

Use `get_patient_snapshot` with `demo=true` to verify the server is working without any patient ID or FHIR connection:

```json
{
  "name": "get_patient_snapshot",
  "arguments": { "demo": true }
}
```