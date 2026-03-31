#!/usr/bin/env python3
"""
Draait 4 scenario's uit de golden set door de pipeline en slaat output op.
Resultaten komen in outputs/scenario_<id>_<scenario>.json

Gebruik:
    python run_scenarios.py              # alle 4 scenario's
    python run_scenarios.py --id 2       # alleen scenario 2
"""
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agents.diagnose_agent import extraheer_diagnoses
from agents.billing_agent import koppel_nza_codes, NZA_DB
from agents.notitie_agent import schrijf_soep_notitie
from agents.validator_agent import valideer_pipeline_output
from guardrails.output_guard import (
    validate_fdi_in_bevindingen, validate_nza_codes,
    validate_combinatieregels, validate_soep_compleetheid,
)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GROEN = "\033[92m"
ROOD  = "\033[91m"
GEEL  = "\033[93m"
BLAUW = "\033[94m"
VET   = "\033[1m"
RESET = "\033[0m"


def run_scenario(scenario: dict) -> dict:
    sid = scenario["id"]
    naam = scenario["scenario"]
    transcript = scenario["transcript"]

    print(f"\n{BLAUW}{'─'*65}{RESET}")
    print(f"{VET}Scenario {sid}: {naam}{RESET}")
    print(f"{BLAUW}{'─'*65}{RESET}")

    resultaat = {
        "id": sid,
        "scenario": naam,
        "beschrijving": scenario.get("beschrijving", ""),
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript,
        "pipeline": {},
        "guardrails": {},
        "validatie": {},
        "metrics": {},
        "fouten": [],
    }

    # ── Agent 1: diagnose-extractie ──────────────────────────────────
    print("  ⏳ Agent 1 — diagnose-extractie...")
    t0 = time.time()
    try:
        agent1 = extraheer_diagnoses(transcript)
        bevindingen = agent1.get("bevindingen", [])
        algemeen = agent1.get("algemene_bevindingen", "")
    except Exception as e:
        resultaat["fouten"].append(f"Agent 1 fout: {e}")
        bevindingen, algemeen = [], ""
    t1 = time.time()

    resultaat["pipeline"]["agent1"] = {
        "bevindingen": bevindingen,
        "algemene_bevindingen": algemeen,
        "latency_s": round(t1 - t0, 2),
    }
    print(f"  {GROEN}✓{RESET} {len(bevindingen)} bevindingen ({t1-t0:.1f}s)")
    for b in bevindingen:
        print(f"     El.{b.get('element','?'):>3} | {b.get('diagnose','')[:45]:<45} | {b.get('urgentie','')}")

    # FDI-validatie
    fdi = validate_fdi_in_bevindingen(bevindingen)
    resultaat["guardrails"]["fdi"] = fdi
    if not fdi["geldig"]:
        for f in fdi["fouten"]:
            print(f"  {GEEL}⚠{RESET}  {f}")

    # ── Agent 2: NZa-code koppeling ──────────────────────────────────
    print("  ⏳ Agent 2 — NZa-codes koppelen...")
    t2 = time.time()
    try:
        gecodeerd = koppel_nza_codes(bevindingen)
    except Exception as e:
        resultaat["fouten"].append(f"Agent 2 fout: {e}")
        gecodeerd = []
    t3 = time.time()

    alle_codes = [c.get("code", "") for item in gecodeerd for c in item.get("nza_codes", [])]
    totaal_tarief = sum(NZA_DB.get(c, {}).get("tarief", 0) for c in alle_codes)

    resultaat["pipeline"]["agent2"] = {
        "gecodeerde_bevindingen": gecodeerd,
        "alle_codes": alle_codes,
        "totaal_tarief": round(totaal_tarief, 2),
        "latency_s": round(t3 - t2, 2),
    }
    print(f"  {GROEN}✓{RESET} {len(gecodeerd)} elementen | codes: {', '.join(alle_codes) or '—'} | €{totaal_tarief:.2f} ({t3-t2:.1f}s)")

    # NZa + combinatieregels validatie
    nza_check = validate_nza_codes(alle_codes)
    combo_check = validate_combinatieregels(alle_codes)
    resultaat["guardrails"]["nza"] = nza_check
    resultaat["guardrails"]["combinatieregels"] = combo_check
    if not nza_check["geldig"]:
        print(f"  {GEEL}⚠{RESET}  Onbekende codes: {nza_check['onbekende_codes']}")
    if not combo_check["geldig"]:
        for w in combo_check["waarschuwingen"]:
            print(f"  {GEEL}⚠{RESET}  {w}")

    # ── Agent 3: SOEP-notitie ────────────────────────────────────────
    print("  ⏳ Agent 3 — SOEP-notitie schrijven...")
    t4 = time.time()
    try:
        soep = schrijf_soep_notitie(transcript, bevindingen, gecodeerd, algemeen)
    except Exception as e:
        resultaat["fouten"].append(f"Agent 3 fout: {e}")
        soep = ""
    t5 = time.time()

    soep_check = validate_soep_compleetheid(soep)
    resultaat["pipeline"]["agent3"] = {
        "soep_notitie": soep,
        "latency_s": round(t5 - t4, 2),
    }
    resultaat["guardrails"]["soep"] = soep_check
    soep_icon = f"{GROEN}✓{RESET}" if soep_check["compleet"] else f"{GEEL}⚠{RESET}"
    print(f"  {soep_icon} SOEP gegenereerd ({t5-t4:.1f}s) | compleet: {soep_check['compleet']}")

    # ── Agent 4: LLM-as-Judge validatie ─────────────────────────────
    print("  ⏳ Agent 4 — LLM-as-judge (haiku)...")
    t6 = time.time()
    try:
        validatie = valideer_pipeline_output(bevindingen, gecodeerd, soep)
    except Exception as e:
        resultaat["fouten"].append(f"Agent 4 fout: {e}")
        validatie = {}
    t7 = time.time()

    resultaat["validatie"] = {**validatie, "latency_s": round(t7 - t6, 2)}
    score = validatie.get("score", "?")
    goed = validatie.get("goedgekeurd")
    icon = f"{GROEN}✓{RESET}" if goed else f"{ROOD}✗{RESET}" if goed is False else "?"
    print(f"  {icon} Validatiescore: {VET}{score}/10{RESET} ({t7-t6:.1f}s)")
    for issue in validatie.get("issues", []):
        print(f"     {ROOD}✗{RESET} {issue}")

    # ── Precision / Recall / F1 vs golden set ───────────────────────
    def prf1(verwacht: set, gevonden: set) -> dict:
        tp = len(verwacht & gevonden)
        fp = len(gevonden - verwacht)
        fn = len(verwacht - gevonden)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2),
                "tp": tp, "fp": fp, "fn": fn}

    verwachte_els = set(scenario.get("verwachte_elementen", []))
    gevonden_els  = {b.get("element") for b in bevindingen if b.get("element")}
    fdi_metrics   = prf1(verwachte_els, gevonden_els)

    verwachte_codes = set(scenario.get("verwachte_nza_codes", []))
    gevonden_codes  = set(alle_codes)
    nza_metrics     = prf1(verwachte_codes, gevonden_codes)

    resultaat["metrics"] = {
        "fdi":  fdi_metrics,
        "nza":  nza_metrics,
        "soep_compleet": soep_check["compleet"],
        "validatie_score": score,
        "totaal_latency_s": round(t7 - t0, 2),
        "verwachte_nza": sorted(verwachte_codes),
        "gevonden_nza":  sorted(gevonden_codes),
        "gemiste_codes": sorted(verwachte_codes - gevonden_codes),
        "extra_codes":   sorted(gevonden_codes - verwachte_codes),
    }

    nza_f1 = nza_metrics["f1"]
    fdi_f1 = fdi_metrics["f1"]
    f1_icon = GROEN if nza_f1 >= 0.8 else GEEL if nza_f1 >= 0.5 else ROOD
    print(f"  📊 FDI  — P:{fdi_metrics['precision']:.0%} R:{fdi_metrics['recall']:.0%} F1:{fdi_f1:.0%}")
    print(f"  📊 NZa  — P:{nza_metrics['precision']:.0%} R:{nza_metrics['recall']:.0%} {f1_icon}F1:{nza_f1:.0%}{RESET}")
    if resultaat["metrics"]["gemiste_codes"]:
        print(f"     {ROOD}Gemist:{RESET}  {', '.join(resultaat['metrics']['gemiste_codes'])}")
    if resultaat["metrics"]["extra_codes"]:
        print(f"     {GEEL}Extra:{RESET}   {', '.join(resultaat['metrics']['extra_codes'])}")
    print(f"  ⏱  Totaal: {t7-t0:.1f}s")

    # ── Opslaan ──────────────────────────────────────────────────────
    bestandsnaam = OUTPUT_DIR / f"scenario_{sid}_{naam}.json"
    with open(bestandsnaam, "w", encoding="utf-8") as f:
        json.dump(resultaat, f, ensure_ascii=False, indent=2)
    print(f"  💾 Opgeslagen: outputs/scenario_{sid}_{naam}.json")

    return resultaat


def print_samenvatting(resultaten: list[dict]) -> None:
    print(f"\n{BLAUW}{'='*65}{RESET}")
    print(f"{VET}  SAMENVATTING — {len(resultaten)} scenario's{RESET}")
    print(f"{BLAUW}{'='*65}{RESET}")
    print(f"  {'Scenario':<22} {'FDI-F1':>7} {'NZa-P':>7} {'NZa-R':>7} {'NZa-F1':>7} {'Score':>6} {'Tijd':>7}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*7}")
    for r in resultaten:
        m = r["metrics"]
        soep = "✓" if m["soep_compleet"] else "✗"
        fdi_f1 = m["fdi"]["f1"]
        nza_p  = m["nza"]["precision"]
        nza_r  = m["nza"]["recall"]
        nza_f1 = m["nza"]["f1"]
        print(
            f"  {r['scenario']:<22} {fdi_f1:>6.0%} {nza_p:>7.0%} {nza_r:>7.0%} "
            f"{nza_f1:>7.0%} {str(m['validatie_score']):>6} {m['totaal_latency_s']:>6.1f}s"
        )
    gem_fdi_f1 = sum(r["metrics"]["fdi"]["f1"] for r in resultaten) / len(resultaten)
    gem_nza_f1 = sum(r["metrics"]["nza"]["f1"] for r in resultaten) / len(resultaten)
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'Gemiddelde':<22} {gem_fdi_f1:>6.0%} {'':>7} {'':>7} {gem_nza_f1:>7.0%}")
    print(f"\n  Alle outputs opgeslagen in: outputs/\n")


def main():
    parser = argparse.ArgumentParser(description="Draai dental AI scenario's")
    parser.add_argument("--id", type=int, help="Draai alleen dit scenario-id")
    args = parser.parse_args()

    golden_path = DATA_DIR / "golden_set.json"
    with open(golden_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    # Max 4 scenario's
    scenarios = golden_set[:4]
    if args.id:
        scenarios = [s for s in scenarios if s["id"] == args.id]
        if not scenarios:
            print(f"Scenario {args.id} niet gevonden.")
            return

    print(f"\n{VET}🦷  DENTAL AI — Scenario runner{RESET}")
    print(f"    {len(scenarios)} scenario('s) te draaien\n")

    resultaten = [run_scenario(s) for s in scenarios]

    if len(resultaten) > 1:
        print_samenvatting(resultaten)


if __name__ == "__main__":
    main()
