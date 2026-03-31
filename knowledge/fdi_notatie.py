"""FDI tandnummering validatie."""

PERMANENTE_TANDEN = {k for k in range(11, 49) if 1 <= k % 10 <= 8 and k // 10 in (1, 2, 3, 4)}
MELKTANDEN = {k for k in range(51, 86) if 1 <= k % 10 <= 5 and k // 10 in (5, 6, 7, 8)}
ALLE_GELDIGE_FDI = PERMANENTE_TANDEN | MELKTANDEN


def is_geldig_fdi(nummer: int) -> bool:
    return nummer in ALLE_GELDIGE_FDI


def kwadrant(nummer: int) -> int:
    return nummer // 10


KWADRANT_NAMEN = {
    1: "rechtsboven",
    2: "linksboven",
    3: "linksonder",
    4: "rechtsonder",
    5: "rechtsboven melk",
    6: "linksboven melk",
    7: "linksonder melk",
    8: "rechtsonder melk",
}
