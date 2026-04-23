import os
import asyncio
from dotenv import load_dotenv
from groq import Groq
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from fhir_client import get_patient_summary
from prompts import SYSTEM_PROMPT, USER_PROMPT, TRIAGE_PROMPT

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
server = Server("silent-handoff")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="generate_handoff_brief",
            description="Generates a clinical shift handoff brief from FHIR patient data. Fetches live patient info and returns a prioritized summary for incoming doctors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The FHIR patient ID"
                    },
                    "fhir_base_url": {
                        "type": "string", 
                        "description": "Optional: FHIR server base URL. Defaults to env variable."
                    }
                },
                "required": ["patient_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "generate_handoff_brief":

        patient_id = arguments.get("patient_id")
        fhir_base_url = arguments.get(
            "fhir_base_url", 
            os.getenv("FHIR_BASE_URL", "https://r4.smarthealthit.org")
        )

        # 1. Fetch FHIR data
        summary = get_patient_summary(patient_id, fhir_base_url)

        def to_str(val):
            return "\n".join(val) if isinstance(val, list) else str(val)

        observations = f"Labs:\n{to_str(summary['labs'])}\n\nVitals:\n{to_str(summary['vitals'])}"

        # 2. Triage
        triage_result = ""
        try:
            triage_completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": TRIAGE_PROMPT.format(observations=observations)}
                ],
                temperature=0.0
            )
            triage_result = triage_completion.choices[0].message.content
        except Exception as e:
            triage_result = "Triage unavailable"

        # 3. Generate handoff brief
        user_content = USER_PROMPT.format(
            name=summary.get("name", "Not available"),
            age=summary.get("age", "Not available"),
            gender=summary.get("gender", "Not available"),
            mrn=summary.get("mrn", "Not available"),
            conditions=to_str(summary.get("conditions", "Not available")),
            medications=to_str(summary.get("medications", "Not available")),
            labs=to_str(summary.get("labs", "Not available")),
            vitals=to_str(summary.get("vitals", "Not available"))
        )

        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Triage:\n{triage_result}\n\n{user_content}"}
                ],
                temperature=0.2
            )
            return [types.TextContent(
                type="text",
                text=completion.choices[0].message.content
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    return [types.TextContent(type="text", text="Unknown tool")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())