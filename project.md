# StepSense – Project Documentation
ML4B SoSe 2026 | FAU Erlangen-Nürnberg | Gruppe: NoNames

---

## 1 Introduction

### 1.1 Project Overview

_Kurze Beschreibung des Projekts in 3-5 Sätzen. Was wird gemacht, warum, und für wen?_

StepSense ist ein Machine-Learning-Projekt das Bewegungsklassifikation mittels Smartphone-Sensordaten mit Methoden der Explainable AI (XAI) verbindet. Ziel ist es, einen Klassifikationsalgorithmus zu entwickeln der menschliche Fortbewegungsarten präzise erkennt – und diesen durch XAI-gestützte Feature-Selektion so weit zu minimieren, dass er theoretisch auf ressourcenbeschränkten Geräten wie Fitness-Armbändern oder Smartwatches einsetzbar wäre.

### 1.2 Research Question

Kann der Einsatz von XAI-Methoden (SHAP) die Bewegungsklassifikation durch gezielte Feature-Reduktion effizienter gestalten als ein klassischer Ansatz – gemessen an Modellgröße, Feature-Anzahl und Klassifikationsgenauigkeit?

### 1.3 Motivation & Business Case

_Warum ist dieses Problem relevant? Wer profitiert von der Lösung?_

- Wearable-Hersteller (Fitbit, Garmin, Apple Watch): Reduktion des Energieverbrauchs durch kleinere Klassifizierungsmodelle
- Automobilindustrie (z.B. Schaeffler): Minimierung von Sensorik in eingebetteten Systemen
- Industrie 4.0 / Predictive Maintenance: Effiziente Anomalieerkennung mit minimalem Sensoraufwand
- Medizintechnik: Akkuschonende Bewegungsanalyse in Patientenmonitoring-Geräten

### 1.4 Team

| Name | Aufgabe |
|------|---------|
| _Name 1_ | _Aufgabe_ |
| _Name 2_ | _Aufgabe_ |
| _Name 3_ | _Aufgabe_ |

---

## 2 Related Work

_Welche bestehenden Arbeiten gibt es zu diesem Thema? Wie grenzt sich euer Ansatz ab?_

### 2.1 Human Activity Recognition (HAR)

_TODO: 2-3 relevante Paper oder Projekte kurz beschreiben_

- Referenz 1: ...
- Referenz 2: ...

### 2.2 XAI in ressourcenbeschränkten Systemen

_TODO: Bestehende Ansätze zu SHAP / Feature-Selektion für Wearables_

- Referenz 1: ...
- Referenz 2: ...

### 2.3 Abgrenzung des eigenen Ansatzes

_Was macht StepSense anders als bestehende Arbeiten?_

_TODO: Ausfüllen nach Related Work Recherche_

---

## 3 Methodology

### 3.1 General Methodology

Das Projekt folgt dem CRISP-DM Prozessmodell mit folgenden Phasen:

| Phase | Inhalt | Status |
|-------|--------|--------|
| Business Understanding | Projektziel, Forschungsfrage, Business Case | Abgeschlossen |
| Data Understanding | Datenerhebung, explorative Analyse, Qualitätsprüfung | In Bearbeitung |
| Data Preparation | Windowing, Feature Engineering, Train/Test-Split | Offen |
| Modeling | Baseline Random Forest + SHAP-optimiertes Modell | Offen |
| Evaluation | Vergleich Accuracy, F1, Modellgröße, Feature-Anzahl | Offen |
| Deployment | Streamlit Web-App, Präsentation bei Schaeffler | Offen |

#### Zweistufige Modellierungsstrategie

**Stufe 1 – Baseline-Klassifizierer:**
Training eines Random Forest mit allen verfügbaren Features als Referenzmodell. Evaluation mit Accuracy, F1-Score und Confusion Matrix.

**Stufe 2 – SHAP-gestützte Optimierung:**
SHAP-Analyse auf dem Baseline-Modell. Identifikation von Features mit geringem Beitrag. Systematische Feature-Reduktion und Re-Training. Vergleich beider Modelle anhand definierter Metriken.

### 3.2 Data Understanding and Preparation

#### 3.2.1 Datenbeschreibung

**Datentyp:** Zeitreihendaten (multivariate Sensordaten)
**Erhebungsmethode:** Sensor Logger App (iOS/Android)
**Geräteposition:** Rechte Hosentasche, kein aktives Festhalten

**Klassifikationsklassen:**

| Klasse | Beschreibung | Aufnahmen |
|--------|-------------|-----------|
| Gehen | Rhythmische Beschleunigung, moderate Schrittfrequenz | _TODO: Anzahl_ |
| Stehen | Minimale Bewegung, vertikale Ausrichtung | _TODO: Anzahl_ |
| Liegen | Nahezu keine Bewegung, horizontale Ausrichtung | _TODO: Anzahl_ |
| Joggen | Erhöhte Schrittfrequenz, stärkere Beschleunigung | _TODO: Anzahl_ |
| Treppe hoch | Charakteristische Neigungsänderungen, aufwärts | _TODO: Anzahl_ |
| Treppe runter | Charakteristische Neigungsänderungen, abwärts | _TODO: Anzahl_ |
| Mixed | Gemischte Aufnahmen für spätere Klassifikation | _TODO: Anzahl_ |

#### 3.2.2 Verwendete Sensoren

| Sensor | Datei | Begründung |
|--------|-------|-----------|
| Linearer Beschleunigungssensor | `Accelerometer.csv` | Bewegungsintensität & Rhythmus |
| Gyroskop | `Gyroscope.csv` | Drehbewegungen, Treppenerkennung |
| Geräteorientierung | `Orientation.csv` | Liegen vs. aufrechte Bewegung |
| Annotation | `Annotation.csv` | Labels / Zeitstempel der Aufnahme |

**Bewusst ausgeschlossen:** AccelerometerUncalibrated, GyroscopeUncalibrated, TotalAcceleration, Metadata

#### 3.2.3 Datenstruktur

```
data/
├── raw/
│   ├── Gehen/
│   │   ├── Gehen_1_Yann/
│   │   │   ├── Accelerometer.csv
│   │   │   ├── Gyroscope.csv
│   │   │   ├── Orientation.csv
│   │   │   └── Annotation.csv
│   │   └── ...
│   ├── Stehen/
│   ├── Liegen/
│   ├── Joggen/
│   ├── Treppe_hoch/
│   ├── Treppe_runter/
│   └── Mixed/
└── processed/            # Wird nach Data Preparation befüllt
```

#### 3.2.4 Bekannte Probleme & offene Fragen

_TODO: Nach erster explorativer Analyse ausfüllen_

- [ ] Sind die Sampling-Raten über alle Aufnahmen konsistent?
- [ ] Gibt es Ausreißer oder fehlerhafte Aufnahmen?
- [ ] Ist die Klassenverteilung ausgewogen?
- [ ] Gibt es Überschneidungen zwischen ähnlichen Klassen (z.B. Gehen vs. Treppe)?

#### 3.2.5 Data Preparation Strategie

_TODO: Nach Notebook 01 ausfüllen_

- Fenstergröße: _z.B. 2 Sekunden / 100 Samples_
- Überlappung: _z.B. 50%_
- Features je Fenster: _Mittelwert, Std, Energie, Peaks, ..._
- Train/Test-Split: _z.B. 80/20_
- Normalisierung: _z.B. StandardScaler_

---

## 4 Results

_TODO: Nach Modellierung ausfüllen_

### 4.1 Baseline Modell

| Metrik | Wert |
|--------|------|
| Accuracy | _TODO_ |
| F1-Score (macro) | _TODO_ |
| Anzahl Features | _TODO_ |
| Modellgröße | _TODO_ |

### 4.2 SHAP-optimiertes Modell

| Metrik | Wert | Veränderung zu Baseline |
|--------|------|------------------------|
| Accuracy | _TODO_ | _TODO_ |
| F1-Score (macro) | _TODO_ | _TODO_ |
| Anzahl Features | _TODO_ | _TODO_ |
| Modellgröße | _TODO_ | _TODO_ |

### 4.3 Wichtigste SHAP-Features

_TODO: Top Features nach SHAP-Analyse_

---

## 5 Discussion

_TODO: Nach Evaluation ausfüllen_

### 5.1 Interpretation der Ergebnisse

_Was bedeuten die Ergebnisse? Wurde die Forschungsfrage beantwortet?_

### 5.2 Limitationen

_Was sind die Grenzen des Ansatzes?_

- Aufnahmen nur von _N_ Personen → begrenzte Generalisierbarkeit
- Handy immer in der Hosentasche → nicht repräsentativ für alle Trageweisen
- _TODO: weitere Limitationen_

### 5.3 Ausblick

_Was könnte in zukünftigen Arbeiten verbessert werden?_

---

## 6 Appendix

### 6.1 Aufnahmeprotokoll

- Gerät: Smartphone in der rechten Hosentasche
- Kein aktives Festhalten des Geräts
- Mindestaufnahmedauer: 2-3 Minuten pro Klasse und Session
- Mindestanzahl Sessions: 3 pro Person und Klasse
- Labeling: Manuell direkt nach jeder Aufnahme

### 6.2 Änderungshistorie

| Datum | Änderung | Person |
|-------|----------|--------|
| _Datum_ | Dokument erstellt | _Name_ |
| | | |
