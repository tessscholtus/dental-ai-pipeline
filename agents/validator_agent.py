"""
Agent 4: LLM-as-Judge validator.
Demonstreert: Cross-model validatie — bewust een ANDER (goedkoper) model dan de hoofdagents.
Gebruikt claude-haiku-4-5-20251001 direct via Anthropic API, nooit via orq.ai.
Dit voorkomt dat één model zijn eigen fouten goedkeurt.
Structured output via tool_choice garandeert valid JSON.
"""
import json
import anthropic

VALIDATOR_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """Je bent een onafhankelijke medische kwaliteitsreviewer voor tandheelkundige AI-output.
Wees kritisch maar objectief. Gebruik altijd de tool sla_validatie_op om je beoordeling te registreren."""

VALIDATIE_TOOL = {
    "name": "sla_validatie_op",
    "description": "Sla de kwaliteitsbeoordeling van de tandheelkundige AI-output op.",
    "input_schema": {
        "type": "object",
        "properties": {
            "goedgekeurd": {"type": "boolean", "description": "True als de output voldoende kwaliteit heeft"},
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "bevindingen_ok": {"type": "boolean", "description": "FDI-nummers klinisch plausibel en compleet"},
            "nza_codes_ok": {"type": "boolean", "description": "NZa-codes kloppen bij de diagnoses"},
            "soep_ok": {"type": "boolean", "description": "SOEP bevat alle vier secties met inhoud"},
            "issues": {"type": "array", "items": {"type": "string"}, "description": "Concrete problemen gevonden"},
            "aanbevelingen": {"type": "array", "items": {"type": "string"}, "description": "Verbeterpunten"},
        },
        "required": ["goedgekeurd", "score", "bevindingen_ok", "nza_codes_ok", "soep_ok", "issues", "aanbevelingen"],
    },
}


def valideer_pipeline_output(
    bevindingen: list, gecodeerde: list, soep: str, transcript: str = ""
) -> dict:
    """
    Valideert de volledige pipeline-output met een onafhankelijk model.
    Het transcript wordt meegegeven zodat faithfulness gecheckt kan worden:
    bevat de SOEP claims die NIET in het transcript staan? (hallucinatie-detectie)
    """
    client = anthropic.Anthropic()

    transcript_sectie = f"ORIGINEEL TRANSCRIPT:\n{transcript}\n\n" if transcript else ""
    context = (
        f"{transcript_sectie}"
        f"BEVINDINGEN:\n{json.dumps(bevindingen, ensure_ascii=False, indent=2)}\n\n"
        f"NZa-CODES:\n{json.dumps(gecodeerde, ensure_ascii=False, indent=2)}\n\n"
        f"SOEP-NOTITIE:\n{soep}"
    )

    response = client.messages.create(
        model=VALIDATOR_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        tools=[VALIDATIE_TOOL],
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Beoordeel deze tandheelkundige AI-output kritisch:\n\n"
                    "1. Zijn FDI-tandnummers klinisch plausibel (11-48)?\n"
                    "2. Kloppen NZa-codes bij de diagnoses?\n"
                    "3. Bevat SOEP alle vier secties met inhoud?\n"
                    "4. Bevat de SOEP claims die NIET in het transcript staan? (faithfulness)\n"
                    "5. Zijn er tegenstrijdigheden?\n\n"
                    f"{context}"
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return {"goedgekeurd": None, "score": None, "bevindingen_ok": None, "nza_codes_ok": None, "soep_ok": None, "issues": [], "aanbevelingen": []}
