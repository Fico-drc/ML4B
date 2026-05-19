# StepSense
### XAI-gestützte Bewegungsklassifikation für energieeffiziente Wearables
ML4B SoSe 2026 | FAU Erlangen-Nürnberg | Gruppe: NoNames

---

## Projektübersicht

StepSense entwickelt einen Bewegungsklassifikationsalgorithmus auf Basis von Smartphone-Sensordaten, der durch den Einsatz von Explainable AI (SHAP) systematisch auf seine wesentlichsten Features reduziert wird. Ziel ist ein schlankes, wissenschaftlich begründetes Modell, das theoretisch auf ressourcenbeschränkten Geräten wie Fitness-Armbändern oder Smartwatches einsetzbar ist.

Das Projekt folgt dem CRISP-DM Prozessmodell und mündet in einer Streamlit-Webanwendung, die auf der ML4B 2026 Conference bei Schaeffler in Nürnberg präsentiert wird.

---

## Forschungsfrage

Kann der Einsatz von XAI-Methoden (SHAP) die Bewegungsklassifikation durch gezielte Feature-Reduktion effizienter gestalten als ein klassischer Ansatz – gemessen an Modellgröße, Feature-Anzahl und Klassifikationsgenauigkeit?

---

## Team

| Name | GitHub | Aufgabe |
|------|--------|---------|
| _Name 1_ | [@username](https://github.com/username) | _z.B. Data Preparation_ |
| _Name 2_ | [@username](https://github.com/username) | _z.B. Modellierung_ |
| _Name 3_ | [@username](https://github.com/username) | _z.B. Streamlit App_ |

---

## Klassifikationsklassen

Das Modell klassifiziert sechs menschliche Bewegungszustände:

- Stehen
- Gehen
- Laufen
- Sprinten
- Treppe rauf / runter
- Liegen

Aufnahmeprotokoll: Handy stets in der rechten Hosentasche, kein aktives Festhalten des Geräts.

---

## Projektstruktur

```
ML4B/
├── data/
│   ├── raw/                  # Rohdaten (unveraendert)
│   │   ├── Gehen_1_DATUM/
│   │   │   ├── Accelerometer.csv
│   │   │   ├── Gyroscope.csv
│   │   │   ├── Orientation.csv
│   │   │   └── Annotation.csv
│   │   └── ...
│   └── processed/            # Gefensterte, engineerte Features
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_modeling_baseline.ipynb
│   └── 04_modeling_shap.ipynb
├── app.py                    # Streamlit Web-Applikation
├── pyproject.toml            # Abhaengigkeiten (uv)
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Voraussetzungen

- [uv](https://docs.astral.sh/uv/) installiert
- VS Code mit den Extensions **Python** und **Jupyter**

### Lokale Einrichtung

```bash
# Repository klonen
git clone https://github.com/Fico-drc/ML4B
cd ML4B

# Abhaengigkeiten installieren
uv sync

# Streamlit App starten
uv run streamlit run app.py
```

### Pakete verwalten

```bash
uv add <paketname>     # Paket hinzufuegen
uv sync                # Umgebung synchronisieren
```

---

## Verwendete Libraries

Alle Abhaengigkeiten sind in `pyproject.toml` dokumentiert und werden automatisch mit `uv sync` installiert.

| Paket | Verwendung |
|-------|-----------|
| `pandas` | Datenverarbeitung |
| `scikit-learn` | Modelltraining & Evaluation |
| `shap` | XAI / Feature-Analyse |
| `streamlit` | Web-Applikation |
| `ipykernel` | Jupyter Notebooks |

---

## Methodik

### Stufe 1 – Baseline-Klassifizierer

Training eines Random Forest mit allen verfuegbaren Features als Referenzmodell.

### Stufe 2 – SHAP-gestuetzte Optimierung

SHAP-Analyse auf dem Baseline-Modell, systematische Feature-Reduktion, Re-Training und Vergleich von Accuracy, F1-Score, Feature-Anzahl und Modellgroesse.

---

## Git Workflow

```bash
# Vor dem Arbeiten
git pull

# Nach dem Arbeiten
git add .
git commit -m "Kurze Beschreibung der Aenderung"
git push
```

---

## Kurs

**Veranstaltung:** Machine Learning for Business (ML4B)  
**Semester:** SoSe 2026  
**Hochschule:** FAU Erlangen-Nuernberg  
**Praesentation:** ML4B Conference 2026, Schaeffler Nuernberg