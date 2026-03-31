"""
Output guardrails: FDI-validatie, NZa-code check, combinatieregels, SOEP-compleetheid, PII-scan.
"""
import json
import re
from pathlib import Path
from knowledge.fdi_notatie import is_geldig_fdi

DATA_DIR = Path(__file__).parent.parent / "data"


def _laad_nza_codes() -> dict:
    path = DATA_DIR / "nza_codes.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


NZA_CODES = _laad_nza_codes()

COMBINATIEREGELS = [
    ({"C002", "C003"}, "C002 en C003 niet combineerbaar in dezelfde zitting"),
    ({"M02", "M03"}, "M02 en M03 niet combineerbaar in dezelfde zitting"),
    ({"A10", "A15"}, "A10 en A15 niet combineerbaar per kaakhelft"),
]


def validate_fdi_in_bevindingen(bevindingen: list[dict]) -> dict:
    fouten, oke = [], []
    for bev in bevindingen:
        el = bev.get("element")
        if el is None:
            continue
        if is_geldig_fdi(int(el)):
            oke.append(el)
        else:
            fouten.append(f"Ongeldig FDI-nummer: {el}")
    return {"geldig": len(fouten) == 0, "fouten": fouten, "oke": oke}


def validate_nza_codes(codes: list[str]) -> dict:
    onbekend = [c for c in codes if c not in NZA_CODES]
    return {"geldig": len(onbekend) == 0, "onbekende_codes": onbekend}


def validate_combinatieregels(codes: list[str]) -> dict:
    codes_set = set(codes)
    waarschuwingen = []
    # Check restauratiecodes per element (simplified: check globally)
    restauratie_codes = {"V91", "V92", "V93", "V94"} & codes_set
    if len(restauratie_codes) > 1:
        waarschuwingen.append(
            f"Meerdere restauratiecodes: {restauratie_codes} — max 1 per element"
        )
    for combo, bericht in COMBINATIEREGELS:
        if combo.issubset(codes_set):
            waarschuwingen.append(bericht)
    return {"geldig": len(waarschuwingen) == 0, "waarschuwingen": waarschuwingen}


def validate_soep_compleetheid(notitie: str) -> dict:
    secties = {"S": False, "O": False, "E": False, "P": False}
    for sectie in secties:
        if re.search(rf'\b{sectie}\s*[:\-\)]', notitie, re.IGNORECASE):
            secties[sectie] = True
        elif sectie == "S" and "subjectief" in notitie.lower():
            secties[sectie] = True
        elif sectie == "O" and "objectief" in notitie.lower():
            secties[sectie] = True
        elif sectie == "E" and "evaluatie" in notitie.lower():
            secties[sectie] = True
        elif sectie == "P" and "plan" in notitie.lower():
            secties[sectie] = True
    ontbrekend = [s for s, aanwezig in secties.items() if not aanwezig]
    return {
        "compleet": len(ontbrekend) == 0,
        "ontbrekende_secties": ontbrekend,
        "secties": secties,
    }


def check_output_pii(tekst: str) -> dict:
    issues = []
    if re.search(r'\b\d{9}\b', tekst):
        issues.append("Mogelijk BSN in output")
    return {"schoon": len(issues) == 0, "issues": issues}
