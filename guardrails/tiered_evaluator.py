"""
Risicogestuurde evaluatie: tier 1-4 op basis van soort bevinding.
Tier 1 (kritisch): tandnummers, medicatie → altijd review
Tier 2 (declaratie): NZa-codes → steekproef
Tier 3 (kwaliteit): SOEP-tekst → periodieke audit
Tier 4 (comfort): afspraken → geen review
"""
from dataclasses import dataclass, field


@dataclass
class TierResultaat:
    tier: int
    label: str
    drempel: float
    score: float
    vereist_review: bool
    toelichting: str


def evalueer_bevindingen(bevindingen: list[dict]) -> TierResultaat:
    """Tier 1: zijn alle tandnummers aanwezig en plausibel?"""
    totaal = len(bevindingen)
    met_element = sum(1 for b in bevindingen if b.get("element") is not None)
    score = met_element / totaal if totaal > 0 else 0.0
    return TierResultaat(
        tier=1,
        label="Tandnummer compleetheid",
        drempel=0.99,
        score=score,
        vereist_review=score < 0.99,
        toelichting=f"{met_element}/{totaal} bevindingen hebben een FDI-nummer",
    )


def evalueer_nza_codes(gecodeerde: list[dict], nza_database: dict) -> TierResultaat:
    """Tier 2: zijn NZa-codes correct en compleet?"""
    totaal_codes = sum(len(item.get("nza_codes", [])) for item in gecodeerde)
    geldige_codes = sum(
        1
        for item in gecodeerde
        for c in item.get("nza_codes", [])
        if c.get("code") in nza_database
    )
    score = geldige_codes / totaal_codes if totaal_codes > 0 else 1.0
    return TierResultaat(
        tier=2,
        label="NZa-code correctheid",
        drempel=0.95,
        score=score,
        vereist_review=score < 0.95,
        toelichting=f"{geldige_codes}/{totaal_codes} codes geldig in database",
    )


def evalueer_soep(notitie: str) -> TierResultaat:
    """Tier 3: bevat de SOEP-notitie alle vier secties met minimale inhoud?"""
    secties = {
        "S": ["s (subjectief)", "subjectief"],
        "O": ["o (objectief)", "objectief"],
        "E": ["e (evaluatie)", "evaluatie"],
        "P": ["p (plan)", "plan"],
    }
    notitie_lower = notitie.lower()
    aanwezig = []
    ontbrekend = []
    for label, zoektermen in secties.items():
        gevonden = any(term in notitie_lower for term in zoektermen)
        # Sectie telt alleen mee als er ook inhoud na de header staat (>10 tekens na de term)
        if gevonden:
            for term in zoektermen:
                idx = notitie_lower.find(term)
                if idx != -1 and len(notitie[idx + len(term):].strip()) > 10:
                    aanwezig.append(label)
                    break
            else:
                ontbrekend.append(label)
        else:
            ontbrekend.append(label)

    score = len(aanwezig) / 4
    toelichting = (
        f"Secties aanwezig: {', '.join(aanwezig) or 'geen'}"
        + (f" | Ontbreekt: {', '.join(ontbrekend)}" if ontbrekend else "")
    )
    return TierResultaat(
        tier=3,
        label="SOEP kwaliteit",
        drempel=0.90,
        score=score,
        vereist_review=score < 0.90,
        toelichting=toelichting,
    )


def evalueer_alles(
    bevindingen, gecodeerde, notitie, nza_database
) -> list[TierResultaat]:
    return [
        evalueer_bevindingen(bevindingen),
        evalueer_nza_codes(gecodeerde, nza_database),
        evalueer_soep(notitie),
    ]
