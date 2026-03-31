#!/usr/bin/env python3
"""
Dental AI Pipeline — Endo scenario.
Patiënt: vrouw met pijnklacht element 17, irreversibele pulpitis, begroting gevraagd.
Demonstreert: echte RAG (geen platte codelijst in prompt), begroting-output.
"""
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

from data.transcript_endo import TRANSCRIPT_ENDO as TRANSCRIPT
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

GROEN = "\033[92m"
ROOD  = "\033[91m"
GEEL  = "\033[93m"
BLAUW = "\033[94m"
RESET = "\033[0m"
VET   = "\033[1m"

USE_ORQ = bool(os.getenv("ORQ_API_KEY"))


def sectie(titel: str):
    print(f"\n{BLAUW}{'=' * 70}{RESET}")
    print(f"{VET}  {titel}{RESET}")
    print(f"{BLAUW}{'=' * 70}{RESET}")


def ok(msg):   print(f"  {GROEN}[OK]{RESET} {msg}")
def warn(msg): print(f"  {GEEL}[WAARSCHUWING]{RESET}  {msg}")
def fout(msg): print(f"  {ROOD}[FOUT]{RESET} {msg}")


def druk_begroting_af(gecodeerd: list, nza_db: dict):
    """Gestructureerde begroting per behandelbezoek."""
    sectie("BEGROTING — Endodontische behandeling element 17")
    print(f"  {'─' * 66}")

    # Splits codes in diagnostiek (huidig bezoek) vs behandeling (vervolgbezoek)
    diagnostiek_cats = {"Consultatie", "Radiologie"}
    behandeling_cats = {"Endodontie", "Verdoving", "Vulling"}

    diagnostiek, behandeling, overig = [], [], []
    for item in gecodeerd:
        for code_item in item.get("nza_codes", []):
            code = code_item.get("code", "")
            info = nza_db.get(code, {})
            cat  = info.get("categorie", "")
            rij  = {
                "code": code,
                "omschrijving": code_item.get("omschrijving", info.get("omschrijving", "")),
                "tarief": code_item.get("tarief", info.get("tarief", 0)),
            }
            if cat in diagnostiek_cats:
                diagnostiek.append(rij)
            elif cat in behandeling_cats:
                behandeling.append(rij)
            else:
                overig.append(rij)

    def druk_blok(titel, rijen):
        if not rijen:
            return 0.0
        print(f"\n  {VET}{titel}{RESET}")
        totaal = 0.0
        for r in rijen:
            print(f"    {r['code']:<6}  {r['omschrijving']:<45}  € {r['tarief']:>7.2f}")
            totaal += r["tarief"]
        print(f"    {'─' * 60}")
        print(f"    {'Subtotaal':<51}  € {totaal:>7.2f}")
        return totaal

    t1 = druk_blok("Bezoek 1 — Diagnostiek (vandaag)", diagnostiek)
    t2 = druk_blok("Bezoek 2 & 3 — Endodontische behandeling", behandeling)
    t3 = druk_blok("Overig", overig)

    totaal = t1 + t2 + t3
    print(f"\n  {'═' * 66}")
    print(f"  {VET}  TOTAAL BEGROTING (excl. definitieve restauratie)   € {totaal:>7.2f}{RESET}")
    print(f"  {'═' * 66}")
    print(f"\n  {GEEL}Let op: na de wortelkanaalbehandeling is een definitieve restauratie{RESET}")
    print(f"  {GEEL}nodig (kroon of opbouw). Dit wordt apart begroot.{RESET}")


def main():
    print(f"\n{VET}DENTAL AI PIPELINE — ENDO SCENARIO{RESET}")
    route = f"{GROEN}orq.ai{RESET}" if USE_ORQ else f"{GEEL}Anthropic direct{RESET}"
    print(f"    Routing: {route}  |  Validator: {BLAUW}claude-haiku-4-5-20251001 (direct){RESET}")
    print(f"    RAG: {GROEN}ChromaDB — geen platte codelijst in prompt{RESET}\n")

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
    sectie(f"AGENT 1 — Diagnose-extractor")
    print("Analyseert transcript...")
    t0 = time.time()
    agent1_output = extraheer_diagnoses(TRANSCRIPT)
    t1 = time.time()
    bevindingen = agent1_output.get("bevindingen", [])
    algemeen    = agent1_output.get("algemene_bevindingen", "")

    if not bevindingen:
        fout("Geen bevindingen — pipeline gestopt")
        return

    ok(f"{len(bevindingen)} bevindingen in {t1 - t0:.1f}s")
    for b in bevindingen:
        tier_label = f"[tier {b.get('risico_tier', '?')}]"
        element    = str(b.get("element", "?")).rjust(3)
        diagnose   = b.get("diagnose", "")[:45].ljust(45)
        ernst      = b.get("ernst", "").ljust(12)
        confidence = b.get("confidence", "")
        print(f"    {tier_label} Element {element} | {diagnose} | {ernst} | {confidence}")

    fdi_check = validate_fdi_in_bevindingen(bevindingen)
    if fdi_check["geldig"]:
        ok("FDI-validatie geslaagd")
    else:
        for f_ in fdi_check["fouten"]:
            warn(f_)

    # === AGENT 2 — RAG actief ===
    sectie("AGENT 2 — NZa-code koppelaar (RAG — codes via retrieval, niet via prompt)")
    print("Function calling loop — agent zoekt codes op via RAG en tools...")
    t2 = time.time()
    gecodeerd = koppel_nza_codes(bevindingen, TRANSCRIPT)
    t3 = time.time()

    if not gecodeerd:
        warn("Agent 2 leverde geen output — doorgaan zonder codes (graceful degradation)")
        gecodeerd = []
    else:
        ok(f"{len(gecodeerd)} elementen gecodeerd in {t3 - t2:.1f}s")
        alle_codes = [c.get("code", "") for item in gecodeerd for c in item.get("nza_codes", [])]
        totaal = sum(NZA_DB.get(c, {}).get("tarief", 0) for c in alle_codes)
        for item in gecodeerd:
            codes_str = ", ".join(c.get("code", "") for c in item.get("nza_codes", []))
            element   = str(item.get("element", "?")).rjust(3)
            diagnose  = item.get("diagnose", "")[:45].ljust(45)
            if codes_str:
                print(f"    Element {element} | {diagnose} | {codes_str}")
            else:
                print(f"    Element {element} | {diagnose} | — (observatie)")

        nza_check   = validate_nza_codes(alle_codes)
        combo_check = validate_combinatieregels(alle_codes)
        if nza_check["geldig"]:
            ok("Alle NZa-codes geldig (geverifieerd via RAG/tools)")
        else:
            for c in nza_check["onbekende_codes"]:
                warn(f"Onbekende NZa-code: {c}")
        if combo_check["geldig"]:
            ok("Combinatieregels OK")
        else:
            for w in combo_check["waarschuwingen"]:
                warn(w)

        # Begroting afdrukken
        druk_begroting_af(gecodeerd, NZA_DB)

    # === AGENT 3 ===
    sectie("AGENT 3 — SOEP-notitie schrijver")
    print("Schrijft SOEP-notitie...")
    t4 = time.time()
    soep = schrijf_soep_notitie(TRANSCRIPT, bevindingen, gecodeerd, algemeen)
    t5 = time.time()
    ok(f"Notitie gegenereerd in {t5 - t4:.1f}s")

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

    # === AGENT 4 ===
    sectie("AGENT 4 — LLM-as-Judge validator [claude-haiku-4-5-20251001, altijd direct Anthropic]")
    print("Onafhankelijk model valideert pipeline-output...")
    t6 = time.time()
    validatie = valideer_pipeline_output(bevindingen, gecodeerd, soep, transcript=TRANSCRIPT)
    t7 = time.time()

    score      = validatie.get("score", "?")
    goedgekeurd = validatie.get("goedgekeurd")
    symbool = (f"{GROEN}[GOEDGEKEURD]{RESET}" if goedgekeurd is True
               else f"{ROOD}[AFGEKEURD]{RESET}" if goedgekeurd is False
               else "[ONBEKEND]")

    print(f"\n  {symbool} Score: {VET}{score}/10{RESET}  |  Goedgekeurd: {goedgekeurd}  |  {t7 - t6:.1f}s")
    for label, key in [("Bevindingen", "bevindingen_ok"), ("NZa-codes", "nza_codes_ok"), ("SOEP", "soep_ok")]:
        val  = validatie.get(key)
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
        review_label = f"{ROOD}[REVIEW NODIG]{RESET}" if tr.vereist_review else f"{GROEN}[OK]{RESET}"
        print(f"  Tier {tr.tier} | {tr.label:<30} | score: {tr.score:.2f} (drempel: {tr.drempel}) {review_label}")
        print(f"         {tr.toelichting}")

    # === SAMENVATTING ===
    sectie("PIPELINE VOLTOOID")
    totaal_tijd = t7 - t0
    ok(f"Agent 1: {len(bevindingen)} bevindingen ({t1 - t0:.1f}s)")
    ok(f"Agent 2: {len(gecodeerd)} elementen gecodeerd via RAG ({t3 - t2:.1f}s)")
    ok(f"Agent 3: SOEP-notitie gegenereerd ({t5 - t4:.1f}s)")
    ok(f"Agent 4: validatiescore {score}/10 ({t7 - t6:.1f}s)")
    print(f"\n  {VET}Totale pipeline-tijd: {totaal_tijd:.1f}s{RESET}")
    if USE_ORQ:
        print(f"  {GROEN}Bekijk RAG-traces op: app.orq.ai -> Observability{RESET}\n")


if __name__ == "__main__":
    main()
