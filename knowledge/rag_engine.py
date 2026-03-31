"""
RAG engine voor NZa-codes en KNMT/KIMO richtlijnen.
Demonstreert: Retrieval-Augmented Generation met ChromaDB vector store.
"""
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

KNMT_CHUNKS = [
    "KNMT Richtlijn Patiëntendossier: het dossier bevat minimaal datum, klacht, bevinding, diagnose, behandeling en NZa-code.",
    "KIMO Richtlijn Cariës: bij ICDAS 1-3 is non-operatief beleid geïndiceerd. Pas operatief behandelen vanaf ICDAS 4.",
    "NZa Combinatieregel 2026: C002 (periodiek) en C003 (probleemgericht) zijn niet combineerbaar in dezelfde zitting.",
    "NZa Combinatieregel 2026: V91, V92, V93 en V94 zijn niet combineerbaar per element — kies de code die het aantal vlakken dekt (1→V91, 2→V92, 3→V93, 4+→V94).",
    "NZa Röntgenregel 2026: X10 is de kleine intra-orale röntgenfoto (€21,00 per foto). Bitewings worden gefactureerd als losse X10 codes: standaard beiderzijds (links én rechts) = 2x X10. Alleen 1x X10 als transcript expliciet één zijde noemt. X21 is UITSLUITEND voor panoramische opname (OPT, €90,02) — nooit voor bitewings.",
    "NZa Combinatieregel 2026: A10 (geleidings) en A15 (infiltratie) — maximaal 1 verdovingscode per kaakhelft per zitting.",
    "NZa Wijziging 2026: Consultatiecodes zijn C002 (periodiek, €28,51) en C003 (niet-periodiek/probleemgericht, €28,51). De oude C11/C13 codes zijn vervallen.",
    "NZa Wijziging 2026: Wortelkanaalcodes zijn E13 (1 kanaal), E14 (2 kanalen), E16 (3 kanalen), E17 (4+ kanalen). Gebruik E17 voor molaren met 4+ kanalen.",
    "NZa Wijziging 2026: Rubberdamisolatie (cofferdam) heeft eigen code C022 (€15,00) — declareer dit apart naast de restauratie- of endodontiecode.",
    "NZa Wijziging 2026: Implantologie-tarieven stijgen 17,17%, gelijkgetrokken met algemene tandheelkunde.",
    "KNMT Richtlijn: Bij wortelkanaalbehandeling molaren (4+ kanalen) gebruik E17. Bij premolaren met 2 kanalen gebruik E14. Bij voortanden gebruik E13.",
]


class RAGEngine:
    def __init__(self):
        self.beschikbaar = False
        self.chunks = []
        self.chunk_ids = []
        self._laad_chunks()
        self._init_chromadb()

    def _laad_chunks(self):
        """
        Laad NZa-codes als rijke chunks + KNMT/KIMO teksten.
        Elke chunk bevat: code, omschrijving, tarief, categorie, wanneer te gebruiken,
        en relevante combinatieregels. Rijkere chunks = betere RAG-resultaten.
        """
        # Indicaties en combinatieregels per code (aanvullend op nza_codes.json)
        extra_context = {
            "C002": "Gebruik bij halfjaarlijkse/jaarlijkse routinecontrole zonder specifieke klacht. NIET combineerbaar met C003.",
            "C003": "Gebruik bij consult met specifieke klacht of probleem (pijn, breuk, zwelling). NIET combineerbaar met C002.",
            "X10": "Kleine intra-orale röntgenfoto (periapicaal of bitewing), €21,00 per foto. Standaard bitewings = meerdere X10 codes. Gebruik NOOIT X21 voor bitewings.",
            "X21": "UITSLUITEND panoramische opname (OPT), €90,02. Indicaties: verstandskiezen, kaakafwijkingen, orthodontie. NOOIT voor gewone bitewings.",
            "V91": "Eénvlaks composiet. NIET combineerbaar met V92, V93 of V94 voor hetzelfde element.",
            "V92": "Tweevlaks composiet (bijv. MO of DO). NIET combineerbaar met V91, V93 of V94 voor hetzelfde element.",
            "V93": "Drievlaks composiet (bijv. MOD). NIET combineerbaar met V91, V92 of V94 voor hetzelfde element.",
            "V94": "Viervlaks of meer composiet. NIET combineerbaar met V91, V92 of V93 voor hetzelfde element.",
            "C022": "Rubberdamisolatie (cofferdam) — eigen declaratiecode, apart van restauratie of endodontie.",
            "A10": "Geleidingsanesthesie — voor onderkaak molaren/premolaren. Max 1 verdovingscode per kaakhelft. NIET combineerbaar met A15 per kaakhelft.",
            "A15": "Infiltratieanesthesie — voor bovenkaak elementen en onderkaak voortanden. Max 1 verdovingscode per kaakhelft. NIET combineerbaar met A10 per kaakhelft.",
            "E13": "Wortelkanaal 1 kanaal — voortanden (11-13, 21-23, 31-33, 41-43) en hoektanden. Gebruik bij pulpanecrose of irreversibele pulpitis.",
            "E14": "Wortelkanaal 2 kanalen — premolaren met 2 kanalen (14, 15, 24, 25).",
            "E16": "Wortelkanaal 3 kanalen — premolaren met 3 kanalen of kleine molaren.",
            "E17": "Wortelkanaal 4+ kanalen — molaren (16, 17, 26, 27, 36, 37, 46, 47). Meest gebruikt bij molaar endodontie.",
            "H11": "Eenvoudige extractie — gebruik bij losse tanden, erupteerde elementen.",
            "H16": "Chirurgische verwijdering — gebruik bij geïmpacteerde elementen, verstandskiezen, gespleten wortels.",
            "M03": "Gebitsreiniging — NIET combineerbaar met M02 in dezelfde zitting.",
            "M02": "Mondhygiëne-instructie — NIET combineerbaar met M03 in dezelfde zitting.",
        }

        nza_path = DATA_DIR / "nza_codes.json"
        if nza_path.exists():
            with open(nza_path) as f:
                codes = json.load(f)
            for code, info in codes.items():
                extra = extra_context.get(code, "")
                chunk = (
                    f"NZa code {code}: {info['omschrijving']}. "
                    f"Tarief: €{info['tarief']}. "
                    f"Categorie: {info['categorie']}."
                    + (f" {extra}" if extra else "")
                )
                self.chunks.append(chunk)
                self.chunk_ids.append(f"nza_{code}")
        for i, chunk in enumerate(KNMT_CHUNKS):
            self.chunks.append(chunk)
            self.chunk_ids.append(f"knmt_{i}")

    def _init_chromadb(self):
        try:
            import chromadb
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("dental_knowledge")
            if self.collection.count() == 0:
                self.collection.add(documents=self.chunks, ids=self.chunk_ids)
            self.beschikbaar = True
        except Exception:
            self.beschikbaar = False

    def zoek(self, query: str, n: int = 3) -> dict:
        """Zoek top-n relevante chunks voor een query."""
        if self.beschikbaar:
            try:
                resultaat = self.collection.query(
                    query_texts=[query], n_results=min(n, len(self.chunks))
                )
                docs = resultaat["documents"][0]
                return {
                    "chunks": docs,
                    "bronnen": self.chunk_ids[: len(docs)],
                    "methode": "chromadb",
                }
            except Exception:
                pass
        # Fallback: keyword matching
        query_woorden = set(query.lower().split())
        gesorteerd = sorted(
            self.chunks,
            key=lambda c: len(query_woorden & set(c.lower().split())),
            reverse=True,
        )
        return {
            "chunks": gesorteerd[:n],
            "bronnen": ["keyword_match"] * n,
            "methode": "keyword_fallback",
        }


# Singleton
_engine = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
