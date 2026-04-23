# Silent Handoff MCP Server

Silent Handoff is a Model Context Protocol (MCP) server designed to automate the generation of clinical shift handoff summaries. By directly interfacing with a FHIR (Fast Healthcare Interoperability Resources) server, this tool fetches live patient data (including conditions, medications, labs, and vitals), performs an LLM-powered triage, and synthesizes a structured, highly actionable brief for incoming medical personnel.

This project implements the MCP `stdio` transport, meaning it seamlessly integrates into any MCP-compatible client (like Claude Desktop) as a local extension.

## 🏗 Architecture & Workflow

The core logic resides in `server.py` and `fhir_client.py`. When the host client invokes the `generate_handoff_brief` tool, the system executes the following sequence:
1. **Data Ingestion**: Uses `requests` to pull Patient, Condition, Observation (Labs/Vitals), and MedicationRequest resources via FHIR REST APIs.
2. **Clinical Triage**: Uses the **Groq API** (running the `llama-3.3-70b-versatile` model) with a temperature of `0.0` to perform a deterministic, clinical triage on the gathered labs and vitals.
3. **Synthesis**: Combines the structural patient data, the triage analysis, and a structured system prompt, sending it back to Groq at a `0.2` temperature to generate a natural, prioritized handoff narrative.
4. **Delivery**: Returns the final string through the MCP protocol back to the host client.

## 🛠 Available Tools

Currently, the server exposes a single, powerful tool to the client:
- `generate_handoff_brief`: Fetches live patient info via `patient_id` and returns a prioritized handoff summary.
  - **Inputs**: 
    - `patient_id` (string, required) - The FHIR patient ID.
    - `fhir_base_url` (string, optional) - The FHIR server base URL. Overrides the `.env` variable if provided.

## 📋 Prerequisites

- **Python 3.10+**
- **Git**
- A [Groq Console](https://console.groq.com/) account and API key.

## 🚀 Environment Setup & Installation

### 1. Clone & Navigate
If you haven't already, ensure you are in the project directory:
```bash
cd silent-handoff
```

### 2. Create a Virtual Environment
It is heavily recommended to use a virtual environment to manage project dependencies:
```bash
python3 -m venv venv
```

Activate the virtual environment:
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```cmd
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```
*(Key dependencies include: `mcp`, `groq`, `requests`, `python-dotenv`)*

### 4. Configure Environment Variables
Create a `.env` file in the root of the project:
```bash
touch .env
```
Populate the `.env` file with the following variables:
```env
# Required: Your Groq API Key for LLM Inference
GROQ_API_KEY="gsk_your_api_key_here"

# Optional: Defaults to "https://r4.smarthealthit.org" if omitted
FHIR_BASE_URL="https://your.fhir.server/r4" 
```

## 💻 Usage

Because this is a `stdio` based MCP server, you do not "run" it like a standard web server for direct user interaction. It must be executed by an MCP host. 

### Testing via Command Line
You can test if the server initializes without errors by running:
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
        "GROQ_API_KEY": "gsk_your_api_key_here",
        "FHIR_BASE_URL": "https://r4.smarthealthit.org"
      }
    }
  }
}
```

Restart Claude Desktop, and you will now be able to ask Claude to "generate a shift handoff brief for patient ID [XYZ]!"
