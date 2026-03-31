"""
Input guardrails: PII-detectie, prompt injection check, consent check.
Demonstreert: veiligheidslaag vóór de AI-pipeline.
"""
import re
from dataclasses import dataclass, field


@dataclass
class InputGuardResultaat:
    veilig: bool
    waarschuwingen: list[str] = field(default_factory=list)
    geblokkeerd: bool = False
    reden: str = ""


BSN_PATROON = re.compile(r'\b\d{9}\b')
DATUM_PATROON = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')
INJECTION_PATRONEN = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'forget\s+(everything|all)',
    r'you\s+are\s+now\s+a',
    r'jailbreak',
    r'act\s+as\s+if',
]


def check_pii(tekst: str) -> InputGuardResultaat:
    waarschuwingen = []
    if BSN_PATROON.search(tekst):
        waarschuwingen.append("Mogelijk BSN-nummer gedetecteerd in transcript")
    if len(DATUM_PATROON.findall(tekst)) > 3:
        waarschuwingen.append("Meerdere datums gevonden — controleer op geboortedatum PII")
    return InputGuardResultaat(veilig=True, waarschuwingen=waarschuwingen)


def check_prompt_injection(tekst: str) -> InputGuardResultaat:
    tekst_lower = tekst.lower()
    for patroon in INJECTION_PATRONEN:
        if re.search(patroon, tekst_lower):
            return InputGuardResultaat(
                veilig=False,
                geblokkeerd=True,
                reden=f"Prompt injection gedetecteerd: '{patroon}'",
                waarschuwingen=["GEBLOKKEERD: Mogelijke prompt injection aanval"],
            )
    return InputGuardResultaat(veilig=True)


def check_consent(session: dict) -> InputGuardResultaat:
    if not session.get("opname_toestemming", False):
        return InputGuardResultaat(
            veilig=False,
            waarschuwingen=["Geen opname-toestemming geregistreerd in sessie"],
        )
    return InputGuardResultaat(veilig=True)


def valideer_input(tekst: str, session: dict | None = None) -> InputGuardResultaat:
    """Combineer alle input checks."""
    injection = check_prompt_injection(tekst)
    if injection.geblokkeerd:
        return injection
    pii = check_pii(tekst)
    consent = check_consent(session or {"opname_toestemming": True})
    alle_waarschuwingen = pii.waarschuwingen + consent.waarschuwingen
    return InputGuardResultaat(
        veilig=consent.veilig,
        waarschuwingen=alle_waarschuwingen,
    )
