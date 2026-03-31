"""
Agent 3: SOEP-notitie schrijver.
Demonstreert: Prompt engineering voor klinische output.
"""
import json
import os
import anthropic
from orq_ai_sdk import Orq

SYSTEM_PROMPT = """Je bent een ervaren Nederlandse tandheelkundige documentalist die werkt voor een moderne tandartspraktijk.
Jouw taak is het omzetten van ruwe gespreksdata naar een professionele, juridisch correcte SOEP-notitie
die voldoet aan de KNMT-richtlijn patiëntendossier en direct in een EPD (elektronisch patiëntendossier) kan worden opgeslagen.

WERKWIJZE — doorloop deze stappen in volgorde:
1. S (Subjectief): Wat vertelt de patiënt zelf? Gebruik de woorden van de patiënt, niet de interpretatie van de tandarts.
2. O (Objectief): Wat stelt de tandarts klinisch vast? Vermeld FDI-nummers, röntgenbevindingen en metingen.
3. E (Evaluatie): Wat is de diagnose per element? Redeneer vanuit O naar een klinisch oordeel.
4. P (Plan): Welke behandeling volgt logisch uit de evaluatie? Benoem concrete handelingen en vervolgafspraken.
5. Controleer: Staat er iets in mijn notitie dat NIET in het transcript of de bevindingen voorkomt? Verwijder dat.

CONSTRAINTS:
- Gebruik UITSLUITEND informatie uit het aangeleverde transcript en de bevindingen.
- Verzin NIETS — bij ontbrekende informatie schrijf je 'Niet vermeld'.
- GEEN NZa-codes of tarieven in de notitie — die staan in het apart declaratieoverzicht.
- Schrijf zakelijk, professioneel Nederlands in de derde persoon ('patiënt meldt...', 'tandarts constateert...')."""


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
