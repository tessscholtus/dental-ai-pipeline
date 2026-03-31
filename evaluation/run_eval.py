"""
Golden set evaluatie: meet pipeline accuracy op 5 geverifieerde scenario's.
Metrics: FDI accuracy, NZa code accuracy, SOEP compleetheid, faithfulness, latency.
"""
import json
import time
from pathlib import Path
from agents.diagnose_agent import extraheer_diagnoses
from agents.billing_agent import koppel_nza_codes
from agents.notitie_agent import schrijf_soep_notitie
from guardrails.output_guard import validate_soep_compleetheid

DATA_DIR = Path(__file__).parent.parent / "data"


def bereken_fdi_accuracy(gevonden: list, verwacht: list) -> float:
    gevonden_els = {b.get("element") for b in gevonden if b.get("element")}
    verwacht_els = set(verwacht)
    if not verwacht_els:
        return 1.0
    return len(gevonden_els & verwacht_els) / len(verwacht_els)


def bereken_nza_accuracy(gevonden_gecodeerd: list, verwachte_codes: list) -> float:
    gevonden_codes = {
        c.get("code")
        for item in gevonden_gecodeerd
        for c in item.get("nza_codes", [])
    }
    verwacht_set = set(verwachte_codes)
    if not verwacht_set:
        return 1.0
    return len(gevonden_codes & verwacht_set) / len(verwacht_set)


def bereken_faithfulness(soep: str, transcript: str) -> float:
    # Simpele keyword check: hoeveel unieke woorden (>4 tekens) uit transcript staan in SOEP
    transcript_woorden = {w.lower() for w in transcript.split() if len(w) > 4}
    soep_woorden = {w.lower() for w in soep.split()}
    if not transcript_woorden:
        return 1.0
    return len(transcript_woorden & soep_woorden) / len(transcript_woorden)


def run_evaluatie():
    golden_path = DATA_DIR / "golden_set.json"
    with open(golden_path) as f:
        golden_set = json.load(f)

    resultaten = []
    print("\n=== GOLDEN SET EVALUATIE ===\n")

    for geval in golden_set:
        print(f"Scenario: {geval['scenario']} (id: {geval['id']})")
        transcript = geval["transcript"]

        t0 = time.time()
        bevindingen_output = extraheer_diagnoses(transcript)
        t1 = time.time()
        bevindingen = bevindingen_output.get("bevindingen", [])
        gecodeerd = koppel_nza_codes(bevindingen, transcript)
        t2 = time.time()
        soep = schrijf_soep_notitie(
            transcript,
            bevindingen,
            gecodeerd,
            bevindingen_output.get("algemene_bevindingen", ""),
        )
        t3 = time.time()

        fdi_acc = bereken_fdi_accuracy(
            bevindingen, geval.get("verwachte_elementen", [])
        )
        nza_acc = bereken_nza_accuracy(gecodeerd, geval.get("verwachte_nza_codes", []))
        soep_check = validate_soep_compleetheid(soep)
        faith = bereken_faithfulness(soep, transcript)

        r = {
            "scenario": geval["scenario"],
            "fdi_accuracy": round(fdi_acc, 2),
            "nza_accuracy": round(nza_acc, 2),
            "soep_compleet": soep_check["compleet"],
            "faithfulness": round(faith, 2),
            "latency_agent1_s": round(t1 - t0, 1),
            "latency_agent2_s": round(t2 - t1, 1),
            "latency_agent3_s": round(t3 - t2, 1),
        }
        resultaten.append(r)
        totaal_latency = (
            r["latency_agent1_s"] + r["latency_agent2_s"] + r["latency_agent3_s"]
        )
        soep_symbool = "OK" if r["soep_compleet"] else "INCOMPLEET"
        print(
            f"  FDI: {r['fdi_accuracy']:.0%}  "
            f"NZa: {r['nza_accuracy']:.0%}  "
            f"SOEP: {soep_symbool}  "
            f"Faith: {r['faithfulness']:.0%}  "
            f"Latency: {totaal_latency:.1f}s"
        )

    print("\n--- GEMIDDELDEN ---")
    for metric in ["fdi_accuracy", "nza_accuracy", "faithfulness"]:
        gem = sum(r[metric] for r in resultaten) / len(resultaten)
        print(f"  {metric}: {gem:.0%}")
    soep_ok = sum(1 for r in resultaten if r["soep_compleet"])
    print(f"  soep_compleetheid: {soep_ok}/{len(resultaten)}")


if __name__ == "__main__":
    run_evaluatie()
