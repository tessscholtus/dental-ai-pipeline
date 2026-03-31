"""
Agent 3: SOEP-notitie schrijver.
Demonstreert: Prompt engineering voor klinische output.
"""
import json
import os
import anthropic
from orq_ai_sdk import Orq

SYSTEM_PROMPT = """Je bent een ervaren Nederlandse tandarts-assistent die klinische SOEP-notities schrijft.

SOEP-FORMAT (puur klinisch — GEEN declaratiecodes in de notitie):
- S (Subjectief): klachten en anamnese in eigen woorden van de patiënt
- O (Objectief): klinische bevindingen, FDI-notatie, röntgenbevindingen
- E (Evaluatie): diagnose en klinisch oordeel per element
- P (Plan): behandelplan en vervolgafspraken in klinische taal
  Beschrijf WAT er gedaan wordt (bijv. "tweevlaks composietrestauratie MO element 36"),
  GEEN NZa-codes, GEEN tarieven — die staan in het apart declaratieoverzicht.

CONSTRAINTS:
- Gebruik UITSLUITEND informatie uit het transcript en de aangeleverde bevindingen.
- Verzin NIETS. Bij ontbrekende info: schrijf 'Niet vermeld'.
- Schrijf zakelijk, professioneel Nederlands."""


def schrijf_soep_notitie(
    transcript: str, bevindingen: list, gecodeerde: list, algemeen: str
) -> str:
    context = (
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"BEVINDINGEN:\n{json.dumps(bevindingen, ensure_ascii=False, indent=2)}\n\n"
        f"NZa-CODES:\n{json.dumps(gecodeerde, ensure_ascii=False, indent=2)}\n\n"
        f"ALGEMEEN: {algemeen}"
    )

    orq_key = os.getenv("ORQ_API_KEY")
    if orq_key:
        orq = Orq(api_key=orq_key)
        response = orq.deployments.invoke(
            key="Dental_notitie_agent",
            messages=[
                {
                    "role": "user",
                    "content": f"Schrijf een SOEP-notitie.\n\n{context}",
                }
            ],
            invoke_options={"include_usage": True},
        )
        return response.choices[0].message.content
    else:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Schrijf een SOEP-notitie.\n\n{context}",
                }
            ],
        )
        return response.content[0].text
