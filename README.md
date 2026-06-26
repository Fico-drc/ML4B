# NoNames – Bewegungsklassifikation aus Smartphone-Sensordaten

ML4B SoSe 2026 | FAU Erlangen-Nürnberg | Gruppe: NoNames

---

## Projektübersicht

Dieses Projekt untersucht die Klassifikation menschlicher Bewegungsaktivitäten auf Basis von Smartphone-Sensordaten mittels Machine-Learning-Verfahren. Ziel ist die Entwicklung eines Modells, das sechs Alltagsaktivitäten aus Accelerometer-, Gyroscope- und Orientierungsdaten eines in der Hosentasche getragenen Smartphones erkennt. Das Projekt folgt dem CRISP-DM Prozessmodell und mündet in einer Streamlit-Webanwendung, die auf der ML4B 2026 Conference bei Schaeffler in Nürnberg präsentiert wird.

---

## Forschungsfrage

Wie genau lassen sich menschliche Bewegungsklassen aus Smartphone-Sensordaten mittels Machine Learning klassifizieren, wenn das Gerät in der Hosentasche getragen wird? Kann ein auf wenigen Probanden trainiertes Modell auf unbekannte Sessions generalisieren?

---

## Team

| Name | GitHub | Aufgabe |
|------|--------|---------|
| Yann Lawrenz | [@Erdmannboy](https://github.com/Erdmannboy) | Programmierung, Datenerhebung |
| _Name 2_ | [@Fico-drc](https://github.com/Fico-drc/ML4B) | Datenerhebung |
| _Name 3_ | [@username](https://github.com/username) | _Aufgabe_ |

---

## Klassifikationsklassen

Das Modell klassifiziert sechs Bewegungsklassen:

| Klasse | Beschreibung |
|--------|-------------|
| Gehen | Rhythmische Bewegung, moderate Schrittfrequenz (~1–2 Hz) |
| Laufen | Erhöhte Schrittfrequenz (~2–3 Hz), höhere Beschleunigungsamplitude |
| Liegen | Nahezu keine Bewegung, horizontale Geräteausrichtung |
| Stehen | Minimale Bewegung, vertikale Geräteausrichtung |
| Treppe_hoch | Aufwärtsbewegung, charakteristische Orientierungsänderung |
| Treppe_runter | Abwärtsbewegung, charakteristische Orientierungsänderung |

Aufnahmeprotokoll: Smartphone in der rechten Hosentasche, vertikal ausgerichtet, kein aktives Festhalten.

---

## Projektstruktur

```
ML4B/
├── data/
│   ├── raw/                        # Rohdaten, unveraendert
│   │   ├── Gehen/
│   │   │   ├── Gehen_1_Yann/
│   │   │   │   ├── Accelerometer.csv
│   │   │   │   ├── Gyroscope.csv
│   │   │   │   ├── Orientation.csv
│   │   │   │   └── Annotation.csv
│   │   │   └── ...
│   │   ├── Laufen/
│   │   ├── Liegen/
│   │   ├── Stehen/
│   │   ├── Treppe_hoch/
│   │   ├── Treppe_runter/
│   │   └── Mixed/                  # Zusammengesetzte Aufnahmen fuer Evaluation
│   └── processed/                  # Generierte Feature-Matrizen, Modell-Artefakte
├── notebooks/
│   ├── 01_data_understanding_preparation.ipynb
│   └── 02_modeling_baseline.ipynb
├── app.py                          # Streamlit Web-Applikation
├── pyproject.toml                  # Abhaengigkeiten (uv)
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Voraussetzungen

- [uv](https://docs.astral.sh/uv/) installiert (Python wird automatisch verwaltet)
- VS Code mit den Extensions **Python** und **Jupyter**

### Lokale Einrichtung

```bash
# 1. Repository klonen
git clone https://github.com/Fico-drc/ML4B
cd ML4B

# 2. Abhaengigkeiten installieren (erstellt .venv automatisch)
uv sync

# 3. Streamlit App starten
uv run streamlit run app.py
```

### Notebooks ausfuehren

1. VS Code oeffnen
2. Notebook oeffnen
3. Kernel auf `.venv\Scripts\python.exe` setzen
4. Notebooks in Reihenfolge ausfuehren: `01_...` dann `02_...`

Die Notebooks generieren alle Artefakte in `data/processed/` die die App benoetigt.

### Pakete verwalten

```bash
uv add <paketname>     # Paket hinzufuegen
uv sync                # Umgebung synchronisieren
uv sync --dry-run      # Pruefe ob Abhaengigkeiten aktuell sind
```

---

## Verwendete Libraries

Alle Abhaengigkeiten sind in `pyproject.toml` versioniert und werden automatisch mit `uv sync` installiert.

| Paket | Verwendung |
|-------|-----------|
| `pandas` | Datenverarbeitung, Zeitreihen-Merge |
| `numpy` | Numerische Berechnungen, FFT |
| `scikit-learn` | Modelltraining, Evaluation, Preprocessing |
| `scipy` | Resampling auf einheitliche Sampling-Rate |
| `matplotlib` / `seaborn` | Visualisierungen |
| `streamlit` | Web-Applikation |
| `shap` | Feature-Importance-Analyse |
| `ipykernel` | Jupyter Notebook Kernel |

---

## Datensatz

| Klasse | Sessions | Fenster |
|--------|----------|---------|
| Gehen | 9 | 1.006 |
| Laufen | 7 | 838 |
| Liegen | 8 | 828 |
| Stehen | 7 | 400 |
| Treppe_hoch | 7 | 243 |
| Treppe_runter | 9 | 306 |
| **Gesamt** | **47** | **3.621** |

Sensoren: Accelerometer (x,y,z), Gyroscope (x,y,z), Orientation (roll, pitch, yaw)
Sampling-Raten: 61 Hz und 100 Hz (Resampling auf 61 Hz im Preprocessing)

---

## Methodik (Kurzuebersicht)

**Feature Engineering:** Sliding Window (2s, 50% Ueberlappung), 81 Features je Fenster (statistische Basisfeatures, Magnitude, FFT-Dominanzfrequenz, Orientierungs-Delta)

**Modellvergleich:** Decision Tree, Random Forest, Extra Trees, SVM, Gradient Boosting, HistGradientBoosting, KNN, Voting Ensemble

**Bestes Modell:** Gradient Boosting (tuned) – Mixed-Eval F1 = 0.87 | CV F1 = 0.88 | Test F1 = 0.99*

**Evaluation:** Session-basierter stratifizierter Split (Train/Val/Test), GroupKFold Cross-Validation (k=3, session-separiert), gewichtetes F1 als Hauptmetrik. Primäre Vergleichsmetrik ist die **Mixed-Evaluation** (F1 = 0.87 auf ungesehenen Aufnahmen mit Aktivitätswechseln) – nicht das Test-Set.

> *Test F1 = 0.99 auf 6 isolierten Einzelaktivitäts-Sessions (sehr kleines Set, strukturell optimistisch). CV F1 und Mixed F1 stimmen eng überein → kein Overfitting.

Detaillierte Beschreibung: siehe `project.md`

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
