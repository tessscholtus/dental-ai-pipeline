"""
Agent 1: Diagnose-extractor.
Demonstreert: Structured outputs via tool_use (tool_choice='any').
Gebruikt orq.ai deployment 'Dental_diagnose_agent' als ORQ_API_KEY beschikbaar is,
anders direct Anthropic API.
"""
import json
import os
import anthropic
from orq_ai_sdk import Orq

EXTRACT_TOOL = {
    "name": "sla_bevindingen_op",
    "description": "Sla geëxtraheerde tandheelkundige bevindingen op.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bevindingen": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "element": {"type": "integer"},
                        "diagnose": {"type": "string"},
                        "vlakken": {"type": "array", "items": {"type": "string"}},
                        "ernst": {
                            "type": "string",
                            "enum": ["initieel", "matig", "ernstig", "observatie", "nvt"],
                        },
                        "behandeling_voorstel": {"type": "string"},
                        "urgentie": {
                            "type": "string",
                            "enum": ["direct", "spoedig", "electief", "observatie"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["hoog", "matig", "laag"],
                        },
                        "risico_tier": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 4,
                        },
                    },
                    "required": [
                        "element",
                        "diagnose",
                        "vlakken",
                        "ernst",
                        "urgentie",
                        "confidence",
                        "risico_tier",
                    ],
                },
            },
            "aanvullend_onderzoek": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "locatie": {"type": "string"},
                        "bevinding": {"type": "string"},
                    },
                },
            },
            "parodontale_status": {"type": "string"},
            "algemene_bevindingen": {"type": "string"},
            "samenvatting": {"type": "string"},
        },
        "required": ["bevindingen", "algemene_bevindingen", "samenvatting"],
    },
}

SYSTEM_PROMPT = """Je bent een ervaren Nederlandse tandheelkundige documentatie-assistent met 15 jaar klinische ervaring.
Je specialiteit is het nauwkeurig en volledig documenteren van tandheelkundige bevindingen uit gespreksopnames,
zodat een tandarts achteraf altijd precies kan reconstrueren wat er is besproken en gevonden.

WERKWIJZE — denk stap voor stap voordat je output geeft:
1. Lees het volledige transcript en identificeer alle genoemde tandnummers (FDI-notatie).
2. Bepaal per tandnummer: wat is de bevinding, wat is de ernst, wat is de urgentie?
3. Zoek naar algemene bevindingen (tandvlees, occlusie, hygiëne, parodontium).
4. Controleer jezelf: heb ik iets opgeschreven dat NIET letterlijk in het transcript staat? Verwijder dat.
5. Ken elk element een risico_tier toe op basis van klinische relevantie.

CONSTRAINTS:
- Extraheer ALLEEN bevindingen die EXPLICIET in het transcript worden genoemd.
- Verzin NOOIT bevindingen, diagnoses of tandnummers die niet in het gesprek voorkomen.
- Gebruik UITSLUITEND FDI-notatie (11-48 volwassen, 51-85 melkgebit).
- Bij twijfel over ernst of diagnose: markeer confidence als 'laag'.
- risico_tier: 1=kritisch (tandnummer/diagnose fout = patiëntveiligheid), 2=declaratie-impact,
  3=kwaliteit notitie, 4=comfort/volledigheid."""


def extraheer_diagnoses(transcript: str) -> dict:
    orq_key = os.getenv("ORQ_API_KEY")

    if orq_key:
        # Via orq.ai deployment
        orq = Orq(api_key=orq_key)
        response = orq.deployments.invoke(
            key="Dental_diagnose_agent",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyseer dit transcript en extraheer alle bevindingen als JSON.\n\n"
                        f"Transcript:\n{transcript}"
                    ),
                }
            ],
            invoke_options={"include_usage": True},
        )
        tekst = response.choices[0].message.content
        if "```json" in tekst:
            tekst = tekst.split("```json")[1].split("```")[0].strip()
        elif "```" in tekst:
            tekst = tekst.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(tekst)
        except Exception:
            return {
                "bevindingen": [],
                "algemene_bevindingen": tekst,
                "samenvatting": "Parse fout",
            }
    else:
        # Direct Anthropic via structured output tool
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "any"},
            messages=[
                {
                    "role": "user",
                    "content": f"Extraheer alle bevindingen uit dit transcript.\n\n{transcript}",
                }
            ],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        return {"bevindingen": [], "algemene_bevindingen": "", "samenvatting": ""}
