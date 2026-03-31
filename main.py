#!/usr/bin/env python3
"""
Dental AI Pipeline — Orchestrator.
Demonstreert: Multi-agent orchestratie, guardrails, RAG, LLM-as-judge validatie.
Draait met alleen ANTHROPIC_API_KEY. Voeg ORQ_API_KEY toe voor orq.ai routing.
"""
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

from data.transcript_voorbeeld import TRANSCRIPT
from agents.diagnose_agent import extraheer_diagnoses
from agents.billing_agent import koppel_nza_codes, NZA_DB
from agents.notitie_agent import schrijf_soep_notitie
from agents.validator_agent import valideer_pipeline_output
from guardrails.input_guard import valideer_input
from guardrails.output_guard import (
    validate_fdi_in_bevindingen,
    validate_nza_codes,
    validate_combinatieregels,
    validate_soep_compleetheid,
    check_output_pii,
)
from guardrails.tiered_evaluator import evalueer_alles

# ANSI kleuren
GROEN = "\033[92m"
ROOD = "\033[91m"
GEEL = "\033[93m"
BLAUW = "\033[94m"
RESET = "\033[0m"
VET = "\033[1m"

USE_ORQ = bool(os.getenv("ORQ_API_KEY"))


def sectie(titel: str):
    print(f"\n{BLAUW}{'=' * 70}{RESET}")
    print(f"{VET}  {titel}{RESET}")
    print(f"{BLAUW}{'=' * 70}{RESET}")


def ok(msg):
    print(f"  {GROEN}[OK]{RESET} {msg}")


def warn(msg):
    print(f"  {GEEL}[WAARSCHUWING]{RESET}  {msg}")


def fout(msg):
    print(f"  {ROOD}[FOUT]{RESET} {msg}")


def main():
    print(f"\n{VET}DENTAL AI PIPELINE{RESET}")
    route = (
        f"{GROEN}orq.ai{RESET}"
        if USE_ORQ
        else f"{GEEL}Anthropic direct{RESET}"
    )
    print(
        f"    Routing: {route}  |  "
        f"Validator: {BLAUW}claude-haiku-4-5-20251001 (direct){RESET}\n"
    )

    sectie("INPUT: Transcript")
    print(TRANSCRIPT)

    # === INPUT GUARDS ===
    sectie("INPUT GUARDRAILS")
    input_check = valideer_input(TRANSCRIPT, {"opname_toestemming": True})
    if input_check.geblokkeerd:
        fout(f"GEBLOKKEERD: {input_check.reden}")
        return
    ok("Geen prompt injection gedetecteerd")
    for w in input_check.waarschuwingen:
        warn(w)
    if not input_check.waarschuwingen:
        ok("Geen PII-waarschuwingen")

    # === AGENT 1 ===
    agent1_label = "[orq.ai]" if USE_ORQ else "[Anthropic]"
    sectie(f"AGENT 1 — Diagnose-extractor {agent1_label}")
    print("Analyseert transcript...")
    t0 = time.time()
    agent1_output = extraheer_diagnoses(TRANSCRIPT)
    t1 = time.time()
    bevindingen = agent1_output.get("bevindingen", [])
    algemeen = agent1_output.get("algemene_bevindingen", "")

    if not bevindingen:
        fout("Geen bevindingen geextraheerd — pipeline gestopt (graceful fail)")
        return

    ok(f"{len(bevindingen)} bevindingen in {t1 - t0:.1f}s")
    for b in bevindingen:
        tier_label = f"[tier {b.get('risico_tier', '?')}]"
        element = str(b.get("element", "?")).rjust(3)
        diagnose = b.get("diagnose", "")[:38].ljust(38)
        ernst = b.get("ernst", "").ljust(12)
        confidence = b.get("confidence", "")
        print(f"    {tier_label} Element {element} | {diagnose} | {ernst} | {confidence}")

    # Output guard: FDI
    fdi_check = validate_fdi_in_bevindingen(bevindingen)
    if fdi_check["geldig"]:
        ok("FDI-validatie geslaagd")
    else:
        for f_ in fdi_check["fouten"]:
            warn(f_)

    # === AGENT 2 ===
    agent2_label = "[orq.ai]" if USE_ORQ else "[Anthropic]"
    sectie(f"AGENT 2 — NZa-code koppelaar {agent2_label} + RAG")
    print("Function calling loop...")
    t2 = time.time()
    gecodeerd = koppel_nza_codes(bevindingen, TRANSCRIPT)
    t3 = time.time()

    if not gecodeerd:
        warn(
            "Agent 2 leverde geen output — doorgaan zonder declaratiecodes "
            "(graceful degradation)"
        )
        gecodeerd = []
    else:
        ok(f"{len(gecodeerd)} elementen gecodeerd in {t3 - t2:.1f}s")
        alle_codes = [
            c.get("code", "")
            for item in gecodeerd
            for c in item.get("nza_codes", [])
        ]
        totaal = sum(NZA_DB.get(c, {}).get("tarief", 0) for c in alle_codes)
        for item in gecodeerd:
            codes_str = ", ".join(c.get("code", "") for c in item.get("nza_codes", []))
            element = str(item.get("element", "?")).rjust(3)
            diagnose = item.get("diagnose", "")[:38].ljust(38)
            print(f"    Element {element} | {diagnose} | {codes_str}")
        print(f"\n    {VET}Geschat declaratiebedrag: EUR {totaal:.2f}{RESET}")
        sectie("DECLARATIEOVERZICHT (Agent 2 output)")
        for item in gecodeerd:
            element = str(item.get("element", "?"))
            diagnose = item.get("diagnose", "")
            nza_codes = item.get("nza_codes", [])
            if not nza_codes:
                print(f"  Element {element}: {diagnose} — geen declaratiecode (observatie)")
                continue
            print(f"  Element {element}: {diagnose}")
            for c in nza_codes:
                print(f"    {c.get('code','')}: {c.get('omschrijving','')} — EUR {c.get('tarief', 0):.2f}")
        print(f"\n  {VET}Totaal: EUR {totaal:.2f}{RESET}")

        # Output guards: NZa + combinatieregels
        nza_check = validate_nza_codes(alle_codes)
        if nza_check["geldig"]:
            ok("Alle NZa-codes geldig")
        else:
            for c in nza_check["onbekende_codes"]:
                warn(f"Onbekende NZa-code: {c}")

        combo_check = validate_combinatieregels(alle_codes)
        if combo_check["geldig"]:
            ok("Combinatieregels OK")
        else:
            for w in combo_check["waarschuwingen"]:
                warn(w)

    # === AGENT 3 ===
    agent3_label = "[orq.ai]" if USE_ORQ else "[Anthropic]"
    sectie(f"AGENT 3 — SOEP-notitie schrijver {agent3_label}")
    print("Schrijft SOEP-notitie...")
    t4 = time.time()
    soep = schrijf_soep_notitie(TRANSCRIPT, bevindingen, gecodeerd, algemeen)
    t5 = time.time()
    ok(f"Notitie gegenereerd in {t5 - t4:.1f}s")

    # Output guards: SOEP + PII
    soep_check = validate_soep_compleetheid(soep)
    if soep_check["compleet"]:
        ok("SOEP volledig (alle 4 secties aanwezig)")
    else:
        warn(f"Ontbrekende secties: {soep_check['ontbrekende_secties']}")

    pii_check = check_output_pii(soep)
    if pii_check["schoon"]:
        ok("Output PII-check: schoon")
    else:
        warn(f"PII in output: {pii_check['issues']}")

    sectie("UITVOER: SOEP-notitie")
    print(soep)

    # === AGENT 4 — VALIDATOR (altijd Anthropic haiku) ===
    sectie(
        "AGENT 4 — LLM-as-Judge validator "
        "[claude-haiku-4-5-20251001, altijd direct Anthropic]"
    )
    print("Onafhankelijk model valideert pipeline-output...")
    t6 = time.time()
    validatie = valideer_pipeline_output(bevindingen, gecodeerd, soep, transcript=TRANSCRIPT)
    t7 = time.time()

    score = validatie.get("score", "?")
    goedgekeurd = validatie.get("goedgekeurd")
    if goedgekeurd is True:
        symbool = f"{GROEN}[GOEDGEKEURD]{RESET}"
    elif goedgekeurd is False:
        symbool = f"{ROOD}[AFGEKEURD]{RESET}"
    else:
        symbool = "[ONBEKEND]"

    print(
        f"\n  {symbool} Score: {VET}{score}/10{RESET}  |  "
        f"Goedgekeurd: {goedgekeurd}  |  {t7 - t6:.1f}s"
    )
    for label, key in [
        ("Bevindingen", "bevindingen_ok"),
        ("NZa-codes", "nza_codes_ok"),
        ("SOEP", "soep_ok"),
    ]:
        val = validatie.get(key)
        vlag = "OK" if val else "NOK"
        print(f"    [{vlag}] {label}: {val}")
    for issue in validatie.get("issues", []):
        warn(issue)
    for tip in validatie.get("aanbevelingen", []):
        print(f"    -> {tip}")

    # === TIERED EVALUATOR ===
    sectie("TIERED EVALUATOR")
    tier_resultaten = evalueer_alles(bevindingen, gecodeerd, soep, NZA_DB)
    for tr in tier_resultaten:
        if tr.vereist_review:
            review_label = f"{ROOD}[REVIEW NODIG]{RESET}"
        else:
            review_label = f"{GROEN}[OK]{RESET}"
        print(
            f"  Tier {tr.tier} | {tr.label:<30} | "
            f"score: {tr.score:.2f} (drempel: {tr.drempel}) {review_label}"
        )
        print(f"         {tr.toelichting}")

    # === SAMENVATTING ===
    sectie("PIPELINE VOLTOOID")
    totaal_tijd = t7 - t0
    ok(f"Agent 1: {len(bevindingen)} bevindingen ({t1 - t0:.1f}s)")
    ok(f"Agent 2: {len(gecodeerd)} elementen gecodeerd ({t3 - t2:.1f}s)")
    ok(f"Agent 3: SOEP-notitie gegenereerd ({t5 - t4:.1f}s)")
    ok(f"Agent 4: validatiescore {score}/10 ({t7 - t6:.1f}s)")
    print(f"\n  {VET}Totale pipeline-tijd: {totaal_tijd:.1f}s{RESET}")
    if USE_ORQ:
        print(
            f"  {GROEN}Bekijk traces op: app.orq.ai -> Observability{RESET}\n"
        )


if __name__ == "__main__":
    main()
