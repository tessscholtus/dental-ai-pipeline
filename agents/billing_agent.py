"""
Agent 2: NZa-code koppelaar met function calling.
Demonstreert: Agentic tool-use loop, max 10 iteraties als guardrail.

Fix: zoek_codes_op_trefwoord() toegevoegd zodat het model op diagnose kan zoeken
     in plaats van een code te moeten raden. NZa-referentie staat ook in de system
     prompt zodat het model al een overzicht heeft vóór de tools aanroept.
"""
import json
import anthropic
from knowledge.rag_engine import get_rag_engine
from pathlib import Path

_NZA_PATH = Path(__file__).parent.parent / "data" / "nza_codes.json"
NZA_DB: dict = json.loads(_NZA_PATH.read_text()) if _NZA_PATH.exists() else {}

COMBINATIEREGELS_DB = [
    (["C002", "C003"], "C002 en C003 niet combineerbaar in dezelfde zitting"),
    (["V91", "V92"], "V91 en V92 niet combineerbaar per element"),
    (["V91", "V93"], "V91 en V93 niet combineerbaar per element"),
    (["V91", "V94"], "V91 en V94 niet combineerbaar per element"),
    (["V92", "V93"], "V92 en V93 niet combineerbaar per element"),
    (["V92", "V94"], "V92 en V94 niet combineerbaar per element"),
    (["V93", "V94"], "V93 en V94 niet combineerbaar per element"),
    (["A10", "A15"], "A10 en A15 niet combineerbaar per kaakhelft"),
    (["M02", "M03"], "M02 en M03 niet combineerbaar"),
]


# ------------------------------------------------------------------ #
# Tool implementaties
# ------------------------------------------------------------------ #

def zoek_codes_op_trefwoord(trefwoord: str) -> dict:
    """
    Zoek NZa-codes op trefwoord in omschrijving of categorie.
    Gebruik dit EERST als je niet weet welke code bij een behandeling hoort.
    """
    t = trefwoord.lower()
    resultaten = [
        {"code": code, **info}
        for code, info in NZA_DB.items()
        if t in info["omschrijving"].lower() or t in info["categorie"].lower()
    ]
    return {"gevonden": len(resultaten), "resultaten": resultaten}


def zoek_nza_code(code: str) -> dict:
    """Haal tarief en omschrijving op voor een bekende NZa-code."""
    if code in NZA_DB:
        return {"gevonden": True, "code": code, **NZA_DB[code]}
    return {"gevonden": False, "code": code,
            "tip": "Gebruik zoek_codes_op_trefwoord() om de juiste code te vinden."}


def zoek_restauratiecode(aantal_vlakken: int) -> dict:
    """Geef direct de juiste composietvulling-code (1→V91, 2→V92, 3+→V93)."""
    mapping = {1: "V91", 2: "V92", 3: "V93"}
    code = mapping.get(aantal_vlakken, "V93")
    return {"code": code, "aantal_vlakken": aantal_vlakken, **NZA_DB.get(code, {})}


def check_combinatieregels(codes: list) -> dict:
    """Controleer of de opgegeven codes combineerbaar zijn in één zitting."""
    codes_set = set(codes)
    waarschuwingen = []
    for combo, bericht in COMBINATIEREGELS_DB:
        if set(combo).issubset(codes_set):
            waarschuwingen.append(bericht)
    return {"toegestaan": len(waarschuwingen) == 0, "waarschuwingen": waarschuwingen}


def zoek_rag_context(query: str) -> dict:
    """Zoek relevante NZa-regels en KNMT/KIMO richtlijnen in de kennisbank."""
    return get_rag_engine().zoek(query)


TOOL_EXECUTORS = {
    "zoek_codes_op_trefwoord": zoek_codes_op_trefwoord,
    "zoek_nza_code": zoek_nza_code,
    "zoek_restauratiecode": zoek_restauratiecode,
    "check_combinatieregels": check_combinatieregels,
    "zoek_rag_context": zoek_rag_context,
}

TOOLS_SCHEMA = [
    {
        "name": "zoek_codes_op_trefwoord",
        "description": (
            "Zoek NZa-codes op trefwoord (bijv. 'composiet', 'wortelkanaal', 'extractie', "
            "'bitewing', 'verdoving', 'controle'). Gebruik dit EERST als je niet weet "
            "welke code bij een behandeling hoort."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"trefwoord": {"type": "string"}},
            "required": ["trefwoord"],
        },
    },
    {
        "name": "zoek_nza_code",
        "description": "Haal tarief en omschrijving op voor een specifieke NZa-code die je al kent.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "zoek_restauratiecode",
        "description": "Geef direct de juiste NZa-code voor een composietvulling op basis van aantal vlakken (1, 2 of 3+).",
        "input_schema": {
            "type": "object",
            "properties": {"aantal_vlakken": {"type": "integer"}},
            "required": ["aantal_vlakken"],
        },
    },
    {
        "name": "check_combinatieregels",
        "description": (
            "Controleer of een lijst NZa-codes combineerbaar is in dezelfde zitting. "
            "Roep dit ALTIJD aan vóór je de eindlijst teruggeeft."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"codes": {"type": "array", "items": {"type": "string"}}},
            "required": ["codes"],
        },
    },
    {
        "name": "zoek_rag_context",
        "description": "Zoek relevante NZa-regels en KNMT/KIMO richtlijnen bij twijfel.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

# Compacte NZa-referentie in de system prompt — model hoeft niet te raden
_NZA_REFERENTIE = "\n".join(
    f"  {code}: {info['omschrijving']} (€{info['tarief']}, {info['categorie']})"
    for code, info in NZA_DB.items()
)

SYSTEM_PROMPT = f"""Je bent een gespecialiseerde Nederlandse tandheelkundige declaratie-expert met diepgaande kennis
van de NZa-prestatiecode systematiek 2026. Jouw taak is het correct en volledig koppelen van klinische
bevindingen aan de juiste declaratiecodes, zodat de praktijk correct kan factureren en een audit altijd
herleidbaar is naar het klinische transcript.

BESCHIKBARE NZa-CODES 2026:
{_NZA_REFERENTIE}

WERKWIJZE — volg deze stappen strikt in volgorde per bevinding:
1. Lees de bevinding en het transcript: wat is er gedaan? (behandeling, röntgen, verdoving, isolatie?)
2. Zoek kandidaat-codes met zoek_codes_op_trefwoord(trefwoord) — raad NOOIT een code zonder te zoeken.
3. Voor composietvullingen: gebruik zoek_restauratiecode(aantal_vlakken) voor de exacte code.
4. Bevestig het tarief van elke gevonden code met zoek_nza_code(code).
5. Bij twijfel over richtlijnen: gebruik zoek_rag_context(query) voor NZa-regels en KNMT-richtlijnen.
6. Controleer de volledige codelijst met check_combinatieregels(alle_codes) vóór je afrondt.
7. Stel jezelf de vraag: heb ik een code toegevoegd die NIET in het transcript staat? Verwijder die.
8. Bij urgentie 'observatie': declareer GEEN behandelcode — alleen diagnostiek indien uitgevoerd.

VASTE DOMEINREGELS (strikt toepassen, NOOIT afwijken):
- CONSULTATIECODES: Gebruik C002 voor routinecontrole (geen klacht), C003 voor probleemgericht
  consult (specifieke klacht/pijn). Nooit C11 of C13 — die zijn vervallen in 2026.
- BITEWING: Bitewings worden gefactureerd als losse X10 codes (€21,00 per foto).
  Bitewings worden standaard beiderzijds gemaakt (links én rechts) = declareer 2x X10.
  Als het transcript expliciet "bitewing links" OF "bitewing rechts" vermeldt = 1x X10.
  Als het transcript "bitewings" (meervoud), "beiderzijds" of geen zijde vermeldt = 2x X10.
  Gebruik NOOIT X21 voor bitewings — X21 is uitsluitend voor panoramische opname (OPT, €90,02),
  alleen bij verstandskiezen of andere indicaties waarvoor een volledige kaakopname nodig is.
- COFFERDAM: Rubberdamisolatie (cofferdam, C022, €15,00) wordt vaak gebruikt bij composietvullingen
  en wortelkanaalbehandelingen. Declareer C022 APART naast de restauratie- of endodontiecode
  als het transcript cofferdam, rubberdam of isolatie vermeldt. Vermeld het NIET als het transcript
  er niets over zegt — hallucinate geen codes.
- WORTELKANAAL: Gebruik E13 (1 kanaal/voortanden), E14 (2 kanalen/premolaren), E16 (3 kanalen),
  E17 (4+ kanalen/molaren). Nooit E01/E02/E04 — die zijn vervallen in 2026.
- COMPOSIET: 1 vlak=V91, 2 vlakken=V92 (bijv. MO/DO), 3 vlakken=V93 (bijv. MOD), 4+=V94.
- HALLUCINATE NIET: Gebruik uitsluitend codes uit de bovenstaande BESCHIKBARE NZa-CODES lijst.
  Als een behandeling geen overeenkomende code heeft, laat het veld leeg — verzin geen codes.

OUTPUT: JSON-lijst, per element:
  element (int), diagnose (str), nza_codes (lijst van {{code, omschrijving, tarief}}), totaal_tarief (float)"""


# ------------------------------------------------------------------ #
# Agentic loop
# ------------------------------------------------------------------ #

def _voer_tool_uit(naam: str, args: dict) -> str:
    func = TOOL_EXECUTORS.get(naam)
    if func:
        return json.dumps(func(**args), ensure_ascii=False)
    return json.dumps({"fout": f"Onbekende tool: {naam}"})


def koppel_nza_codes(bevindingen: list[dict], transcript: str = "") -> list[dict]:
    MAX_ITER = 10
    bev_json = json.dumps(bevindingen, ensure_ascii=False, indent=2)
    transcript_sectie = f"\n\nTRANSCRIPT (voor procedures zoals verdoving, röntgen):\n{transcript}" if transcript else ""
    messages = [
        {
            "role": "user",
            "content": (
                "Koppel NZa-codes aan deze bevindingen. "
                "Voeg ook codes toe voor procedures uit het transcript (verdoving, röntgenfoto's) als die expliciet worden vermeld. "
                "Gebruik zoek_codes_op_trefwoord() om codes te vinden — raad nooit.\n\n"
                f"BEVINDINGEN:\n{bev_json}"
                f"{transcript_sectie}"
            ),
        }
    ]

    client = anthropic.Anthropic()
    tekst = ""

    for _ in range(MAX_ITER):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SCHEMA,
            messages=messages,
        )
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    resultaat = _voer_tool_uit(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": resultaat,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            for block in resp.content:
                if hasattr(block, "text"):
                    tekst = block.text
            break

    if "```json" in tekst:
        tekst = tekst.split("```json")[1].split("```")[0].strip()
    elif "```" in tekst:
        tekst = tekst.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(tekst)
    except Exception:
        return []
