# Silent Handoff MCP Server

Silent Handoff is a Model Context Protocol (MCP) server designed to automate and augment the generation of clinical shift handoff summaries. By directly interfacing with a FHIR (Fast Healthcare Interoperability Resources) server, this tool fetches live patient data (including demographics, conditions, medications, labs, vitals, allergies, and recent encounters), performs an LLM-powered triage, and synthesizes a structured, highly actionable brief for incoming medical personnel.

This project implements the MCP `stdio` and `sse` (Server-Sent Events) transports. It seamlessly integrates into any MCP-compatible client (like Claude Desktop) as a local extension via `stdio`, or can run as an independent web service over HTTP via `sse`.

## Architecture & Workflow

The core logic resides in `server.py`, `fhir_client.py`, and `prompts.py`. When the host client invokes the tools, the system executes the following sequence:
1. **Data Ingestion (`fhir_client.py`)**: Uses a lightweight wrapper built on `requests` to pull Patient, Condition, Observation (Labs/Vitals), MedicationRequest, AllergyIntolerance, and Encounter resources via FHIR REST APIs. It compresses nested FHIR JSON down to flat, token-efficient structures specifically designed for LLMs.
2. **LLM Abstraction Layer**: An `LLMProvider` base class makes it trivial to swap AI backends. It natively supports **Groq**, **OpenAI**, **Anthropic**, and local **Ollama**.
3. **Clinical Triage & Actionable Synthesis**: Uses zero-temperature parsing for determinism. The LLMs perform tasks like medication safety checks, vital sign trend comparisons, and lab anomaly detection.
4. **Final Handoff Delivery**: Synthesizes structured patient data and triage analysis using a `0.2` temperature model to generate a natural, prioritized narrative formatted strictly as SBAR, IPASS, or SOAP. Delivery goes back to the host client.

## 🛠 Available Tools

Currently, the server exposes 7 powerful tools to the client:

| Tool | Description | Inputs | LLM Call? |
|------|-------------|--------|-----------|
| `server_info` | Returns server metadata and capabilities. | none | No |
| `get_patient_snapshot` | Returns clean JSON with demographics, conditions, and allergies. | `patient_id`, `fhir_base_url` | No |
| `triage_labs_vitals` | Analyzes lab and vital sign data; returns triage JSON with acuity level. | `patient_id`, `fhir_base_url` | Yes (0.0 temp) |
| `check_medication_safety` | Analyzes medications and allergies for conflicts and interactions. | `patient_id`, `fhir_base_url` | Yes (0.0 temp) |
| `get_pending_actions` | Identifies unresolved or pending labs, conditions, and monitoring needs. | `patient_id`, `fhir_base_url` | Yes (0.0 temp) |
| `compare_patient_shift` | Compares patient vitals over the last N hours vs prior 24 hours. | `patient_id`, `hours_lookback`, `fhir_base_url` | Yes (0.0 temp) |
| `generate_handoff_brief` | Generates a clinical shift handoff brief (SBAR/IPASS/SOAP) from FHIR data. | `patient_id`, `format`, `fhir_base_url` | Yes (0.2 temp) |

### Supported LLM Providers

The server uses `LLM_PROVIDER` environment variable to switch models dynamically.
- `groq` (Default) - Models: `llama-3.3-70b-versatile`. Needs `GROQ_API_KEY`. Fast, free-tier tokens managed effectively.
- `openai` - Models: `gpt-4o-mini` (default). Needs `OPENAI_API_KEY`.
- `anthropic` - Models: `claude-haiku-4-5-20251001` (default). Needs `ANTHROPIC_API_KEY`.
- `ollama` - Fully local. Model: `llama3` (default). Needs `OLLAMA_BASE_URL` (usually `http://localhost:11434`).

### SHARP-on-MCP Compliance & Clinical Safety Design
This server adheres to healthcare AI safety standards:
- **No Data Retention**: Patient data is never stored locally or logged to stdout.
- **Traceable Logs**: All tool invocations are logged to stderr with masked patient identifiers (e.g. `***1234`).
- **Hallucination Mitigation**: All analytical tools (triage, med safety) run strictly at `temperature=0.0` and enforce JSON structures. Only the final narrative synthesis uses `temperature=0.2`.
- **SSRF Protection**: Built-in validators (`urllib.parse`) verify FHIR URLs safely before sending request, to prevent rogue arbitrary endpoints.
- **Graceful Fallbacks**: Missing resources safely return empty subsets and default strings without crashing.

## Prerequisites

- **Python 3.10+**
- **Git**
- An API Key for your LLM of choice (e.g., [Groq](https://console.groq.com/)), or a local [Ollama](https://ollama.com/) instance.

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

Activate the virtual environment:
- **Linux/macOS:** `source venv/bin/activate`
- **Windows:** `.\venv\Scripts\activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Key dependencies: `mcp`, `starlette`, `uvicorn`, `requests`, `python-dotenv`, `groq`, `openai`, `anthropic`)*

### 4. Configure Environment Variables
Create a `.env` file in the root of the project:
```bash
touch .env
```
Populate the `.env` file:
```env
# Choose LLM Provider: 'groq', 'openai', 'anthropic', or 'ollama'
LLM_PROVIDER="groq"
GROQ_API_KEY="gsk_your_api_key_here"

# (Optional) Alternatively use OpenAI or Anthropic
# OPENAI_API_KEY="sk-..."
# OPENAI_MODEL="gpt-4o"
# ANTHROPIC_API_KEY="sk-ant-..."
# ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"

# (Optional) For Ollama
# OLLAMA_BASE_URL="http://localhost:11434"
# OLLAMA_MODEL="llama3"

# FHIR Server fallback configuration
FHIR_BASE_URL="https://r4.smarthealthit.org"

# Transport mode ("stdio" or "sse"). Defaults to "stdio".
TRANSPORT="stdio"

# Port for SSE server. Defaults to 8000.
PORT=8000
```

## 💻 Usage

### Running in SSE Mode
To run the server as a continuous web service over HTTP with Server-Sent Events, set `TRANSPORT=sse` in your `.env` file or environment, then run:
```bash
python server.py
```
The server will start using Starlette/Uvicorn.
- SSE Endpoint: `http://localhost:8000/sse`
- Health Check: `http://localhost:8000/health`

### Testing STDIO via Command Line
Because `stdio` is the default mode, it must be executed by an MCP host. You can test if it initializes without errors by running:
```bash
python server.py
```
*(Note: It will sit waiting for JSON-RPC messages via stdin. Terminate using `Ctrl+C`)*

### Configuring Claude Desktop (Example Host)
To make this server available inside Claude Desktop, update your `claude_desktop_config.json` file (usually located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "silent-handoff": {
      "command": "/absolute/path/to/silent-handoff/venv/bin/python",
      "args": [
        "/absolute/path/to/silent-handoff/server.py"
      ],
      "env": {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "gsk_your_api_key_here",
        "FHIR_BASE_URL": "https://r4.smarthealthit.org"
      }
    }
  }
}
```

Restart Claude Desktop, and you will now be able to ask Claude to interact with patients from your FHIR server securely!
For instance: `Generate a shift handoff brief for patient ID [XYZ] using the IPASS format!`