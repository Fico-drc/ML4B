# NoNames – Project Documentation
ML4B SoSe 2026 | FAU Erlangen-Nürnberg | Gruppe: NoNames

---

## 1 Introduction

### 1.1 Motivation

Die automatische Erkennung menschlicher Bewegungsaktivitäten (Human Activity Recognition, HAR) aus Inertialsensordaten ist ein etabliertes Forschungsfeld mit Anwendungen in der Medizintechnik, im Sport und in der industriellen Fertigung. Mit der zunehmenden Verbreitung von Smartphones und Wearables stehen kostengünstige Sensoren zur Verfügung, die eine kontinuierliche Bewegungserfassung ermöglichen. Gleichzeitig stellt die Rechenkapazität ressourcenbeschränkter Geräte wie Fitness-Armbänder oder Smartwatches eine technische Limitierung dar, die effiziente Klassifikationsmodelle erfordert.

Dieses Projekt untersucht, ob ein statistisches Machine-Learning-Modell auf Basis eines einzelnen Smartphones in der Hosentasche sechs Alltagsaktivitäten zuverlässig unterscheiden kann. Der gewählte Ansatz – Sliding-Window-Feature-Extraktion mit klassischen Klassifikatoren – orientiert sich an etablierten Methoden der HAR-Literatur und ist bewusst auf Reproduzierbarkeit und Nachvollziehbarkeit ausgelegt.

### 1.2 Forschungsfragen

**Primäre Forschungsfrage:**
Wie genau lassen sich menschliche Bewegungsklassen aus Smartphone-Sensordaten mittels Machine Learning klassifizieren, wenn das Gerät in der Hosentasche getragen wird?

**Sekundäre Forschungsfrage:**
Kann ein auf wenigen Probanden trainiertes Modell auf unbekannte Aufnahmesessions generalisieren, und welche Faktoren limitieren die Generalisierungsfähigkeit?

### 1.3 Zielsystem & Business Case

| Zielgruppe | Anwendung |
|-----------|-----------|
| Wearable-Hersteller | Bewegungsklassifikation auf ressourcenbeschränkten Geräten ohne Cloud-Anbindung |
| Medizintechnik | Akkuschonende Aktivitätsüberwachung in Patientenmonitoring-Systemen |
| Sportanalytik | Automatische Aktivitätserkennung ohne manuelle Annotation |

### 1.4 Team

| Name | Aufgabe |
|------|---------|
| Yann Lawrenz | Programmierung, Datenerhebung, App-Entwicklung |
| _Name 2_ | _Aufgabe_ |
| _Name 3_ | _Aufgabe_ |

### 1.5 Dokumentstruktur

Abschnitt 2 gibt einen Überblick über verwandte Arbeiten. Abschnitt 3 beschreibt die Methodik einschließlich Datenerhebung, Feature Engineering, Modellauswahl und Evaluationsstrategie. Abschnitt 4 dokumentiert die erzielten Ergebnisse. Abschnitt 5 diskutiert die Befunde, Limitierungen und ethische Aspekte. Abschnitt 6 fasst die Erkenntnisse zusammen.

---

## 2 Related Work

### 2.1 Suchprozess

**Suchdatenbanken:** Google Scholar, IEEE Xplore, ACM Digital Library, GitHub

**Suchbegriffe:**
- `"Human Activity Recognition" smartphone accelerometer gyroscope`
- `"HAR" sliding window feature extraction classification`
- `"activity recognition" inertial sensor random forest`
- `"pocket placement" activity recognition smartphone`

**Zeitraum:** 2013–2024

### 2.2 Relevante Arbeiten

#### Anguita et al. (2013) – UCI HAR Dataset

> D. Anguita, A. Ghio, L. Oneto, X. Parra, J.L. Reyes-Ortiz. *"Training Computationally Efficient Smartphone-Based Human Activity Recognition Models."* ICANN, 2013.

Das meistzitierte Referenzpaper im HAR-Bereich. Smartphone in der Hosentasche, Accelerometer und Gyroscope, 6 Aktivitätsklassen (Gehen, Treppe hoch/runter, Sitzen, Stehen, Liegen), SVM-basierte Klassifikation. Das Setup ist nahezu identisch zu diesem Projekt und dient als direkter Vergleichspunkt. Die Autoren verwenden ebenfalls ein Sliding-Window-Verfahren mit statistischen Features.

**Relevanz:** Identisches Sensorsetup und ähnliche Aktivitätsklassen. Bestätigt die Kombination von Accelerometer und Gyroscope als sinnvolles Sensorsetup.

---

#### Bayat et al. (2014) – HAR mit Smartphone-Accelerometer

> A. Bayat, M. Pomplun, D.A. Tran. *"A Study on Human Activity Recognition Using Accelerometer Data from Smartphones."* Procedia Computer Science, 2014.

Untersucht HAR ausschließlich auf Basis von Accelerometerdaten unter realen Bedingungen. Durch Kombination von fünf Klassifikatoren wird eine Gesamtgenauigkeit von 91,15% erreicht.

**Relevanz:** Bestätigt den Ansatz statistischer Features aus Accelerometerdaten. Dieses Projekt erweitert den Ansatz um Gyroscope und Orientation als zusätzliche Signalquellen.

---

#### Balli et al. (2019) – Random Forest für Smartwatch-Daten

> S. Balli, E.A. Sagbas, M. Peker. *"Human Activity Recognition from Smart Watch Sensor Data Using a Hybrid of Principal Component Analysis and Random Forest Algorithm."* Measurement and Control, 2019.

Klassifiziert Bewegungen aus Smartwatch-Sensordaten (Accelerometer, Gyroscope, Schrittzähler, Herzrate). Rohdaten werden in 2-Sekunden-Fenster aufgeteilt. Random Forest erzielt die beste Klassifikationsleistung gegenüber SVM, C4.5 und KNN.

**Relevanz:** Bestätigt Random Forest als leistungsstärksten Klassifikator für sensorbasierte Bewegungsdaten. Fenstergröße von 2 Sekunden entspricht exakt der Wahl in diesem Projekt.

---

#### Malekzadeh et al. (2019) – MotionSense Dataset

> M. Malekzadeh, R.G. Clegg, A. Cavallaro, H. Haddadi. *"Mobile Sensor Data Anonymization."* GitHub: mmalekzadeh/motion-sense.

Öffentlicher Datensatz mit iPhone-Sensordaten (Accelerometer, Gyroscope), 50 Hz, 24 Probanden, 6 Aktivitäten, Gerät in der vorderen Hosentasche.

**Relevanz:** Nahezu identisches Erhebungsprotokoll. Mit 24 Probanden deutlich größer als der vorliegende Datensatz, was die Limitierung durch die geringe Probandenanzahl in diesem Projekt verdeutlicht.

### 2.3 Abgrenzung des eigenen Ansatzes

| Aspekt | Anguita (2013) | Bayat (2014) | Balli (2019) | MotionSense (2019) | Dieses Projekt |
|--------|---------------|-------------|-------------|-------------------|---------------|
| Sensoren | Acc + Gyro | Acc | Acc + Gyro + HR | Acc + Gyro | Acc + Gyro + Orientation |
| Fenstergrösse | 2,56s | k.A. | 2s | k.A. | 2s |
| Klassifikator | SVM | Ensemble | Random Forest | – | Random Forest + Vergleich |
| Probanden | 30 | k.A. | k.A. | 24 | 2–3 |
| Klassen | 6 | k.A. | k.A. | 6 | 6 |
| Split-Strategie | k.A. | k.A. | k.A. | k.A. | Session-basiert (kein Leakage) |

---

## 3 Methodology

### 3.1 General Methodology

Das Projekt folgt dem CRISP-DM Prozessmodell. Aufgrund iterativer Erkenntnisse wurden einzelne Phasen mehrfach durchlaufen – insbesondere Data Preparation und Modeling wurden nach Identifikation von Data Leakage und Klassenungleichgewicht grundlegend überarbeitet.

| Phase | Inhalt | Status |
|-------|--------|--------|
| Business Understanding | Forschungsfrage, Zielsystem, Business Case | Abgeschlossen |
| Data Understanding | Datenerhebung, explorative Analyse, Qualitätsprüfung | Abgeschlossen |
| Data Preparation | Windowing, Feature Engineering, session-basierter Split | Abgeschlossen |
| Modeling | Modellvergleich, GroupKFold CV, Hyperparameter-Tuning | Abgeschlossen |
| Evaluation | Test-Set, Mixed-Evaluation, Bias-Analyse | Abgeschlossen |
| Deployment | Streamlit Web-App | Abgeschlossen |

### 3.2 Data Understanding and Preparation

#### 3.2.1 Datensatzbeschreibung

**Datentyp:** Multivariate Zeitreihendaten aus Inertialsensoren
**Erhebungsmethode:** Sensor Logger App (iOS/Android)
**Geräteposition:** Rechte Hosentasche, vertikal, kein aktives Festhalten
**Sampling-Raten:** 61 Hz (Gerät A) und 100 Hz (Gerät B)

| Klasse | Sessions | Fenster | Durchschn. Dauer |
|--------|----------|---------|-----------------|
| Gehen | 9 | 1.006 | 115s |
| Laufen | 7 | 838 | 101s |
| Liegen | 8 | 828 | 88s |
| Stehen | 7 | 400 | 61s |
| Treppe_hoch | 7 | 243 | 38s |
| Treppe_runter | 9 | 306 | 36s |
| **Gesamt** | **47** | **3.621** | |

**Verwendete Sensordateien:** `Accelerometer.csv` (x,y,z), `Gyroscope.csv` (x,y,z), `Orientation.csv` (roll, pitch, yaw)

**Bewusst ausgeschlossen:** AccelerometerUncalibrated, GyroscopeUncalibrated, TotalAcceleration, Metadata (redundant oder unkalibriert)

#### 3.2.2 Datenstruktur

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
│   ├── Laufen/ | Liegen/ | Stehen/ | Treppe_hoch/ | Treppe_runter/
│   └── Mixed/              # Zusammengesetzte Aufnahmen mit manueller Annotation
└── processed/
    ├── X_train.csv / X_val.csv / X_test.csv
    ├── y_train.csv / y_val.csv / y_test.csv
    ├── features_all.csv
    ├── feature_names.txt
    ├── scaler.pkl
    ├── model.pkl
    ├── model_metadata.json
    └── groups_trainval.csv
```

#### 3.2.3 Preprocessing-Pipeline

Die folgende Pipeline ist identisch in Notebook 01 und `app.py` implementiert:

1. **Sensor-Merge:** Zusammenführung von Accelerometer, Gyroscope und Orientation über `merge_asof` mit 50ms Toleranz
2. **Resampling auf 61 Hz:** `scipy.signal.resample` – normalisiert ZCR und FFT-Features über unterschiedliche Sampling-Raten hinweg. Sessions bei 100 Hz werden auf 61 Hz downsampelt; Sessions innerhalb ±5 Hz bleiben unverändert.
3. **Trim:** Erste und letzte 2 Sekunden jeder Session werden entfernt (Taschenrauschen beim Ein-/Ausstecken des Geräts)
4. **Gap-Check:** Sessions mit Zeitlücken > 5s werden verworfen oder am Gap gekürzt (unterbrochene Aufnahmen durch App-Hintergrund oder Display-Sperre)
5. **Deduplizierung:** Doppelte Zeitstempel werden entfernt
6. **Missing-Value-Imputation:** Forward-Fill, dann Backward-Fill für fehlende Werte nach dem Merge

#### 3.2.4 Feature Engineering

**Windowing:** 2-Sekunden-Fenster, 50% Überlappung (1s Schritt), Mindestanzahl 10 Samples pro Fenster

**81 Features gesamt:**

| Gruppe | Features | Anzahl | Begründung |
|--------|----------|--------|-----------|
| Basis-Statistiken | mean, std, min, max, range, energy, IQR, ZCR je Signal | 9 × 8 = 72 | Standardansatz aus HAR-Literatur |
| Accelerometer-Magnitude | mean, std, energy, FFT-Dominanzfrequenz | 4 | Rotationsinvariant; dom_freq erfasst Schrittfrequenz (Gehen ~1–2 Hz, Laufen ~2–3 Hz) |
| Gyroscope-Magnitude | mean, std, energy | 3 | Gesamtrotationsrate; Stehen ≈ 0 rad/s |
| Orientierungs-Delta | pitch_delta, roll_delta | 2 | Treppensteigen erzeugt stärkere Neigungsänderung als flaches Gehen |

Die FFT-Dominanzfrequenz wird dynamisch aus der tatsächlichen Fensteranzahl berechnet (`actual_hz = len(mag) / WINDOW_SIZE_S`), nicht aus einer fixen angenommenen Rate.

#### 3.2.5 Datensplit

**Problem:** 50% Fenster-Überlappung führt dazu, dass benachbarte Fenster Rohdaten teilen. Ein zufälliger Fensterschnitt würde Data Leakage erzeugen und die Evaluationsmetriken künstlich erhöhen (im Projektverlauf beobachtet: CV F1 ~0.99 bei zufälligem Split).

**Lösung: Session-basierter stratifizierter Split**

- Ganze Sessions werden exklusiv einem Split zugeordnet – kein Fenster einer Session erscheint in mehreren Splits
- Pro Klasse wird die Session mit medianer Fensterzahl dem Test-Set zugeordnet (verhindert Dominanz atypisch langer oder kurzer Sessions)
- Assertions prüfen: alle 6 Klassen in Test und Validation vorhanden
- Verhältnis: Train ~62% / Val ~19% / Test ~20%
- Normalisierung: `StandardScaler` gefittet ausschließlich auf Trainingsdaten, dann auf Val/Test transformiert

### 3.3 Modeling and Evaluation

#### 3.3.1 Modellauswahl

Sieben Klassifikatoren sowie ein Voting-Ensemble wurden verglichen:

| Modell | Klassengewichtung |
|--------|-----------------|
| Decision Tree | class_weight='balanced' |
| Random Forest (200 Bäume) | class_weight='balanced' |
| Extra Trees (200 Bäume) | class_weight='balanced' |
| SVM (RBF, C=10) | class_weight='balanced' |
| Gradient Boosting (200, lr=0.05) | sample_weight |
| HistGradientBoosting (200, lr=0.05) | sample_weight |
| KNN (k=5) | keine (nicht unterstützt) |
| Voting Ensemble (Top-3, soft) | je nach Sub-Modell |

Alle Modelle sind als `sklearn.Pipeline` (StandardScaler + Classifier) in `model.pkl` gespeichert. Vorhersagen werden mit unskalierter Feature-Matrix aufgerufen – der Scaler ist Teil der Pipeline.

#### 3.3.2 Cross-Validation

- **GroupKFold (k=3):** Hält ganze Sessions zusammen (kein Fenster-Leakage). k=3 statt k=5 aufgrund der geringen Anzahl an Stehen-Sessions (7 Sessions; k=5 würde einzelne Folds ohne Stehen-Trainingsbeispiele erzeugen).
- **Metrik:** Gewichtetes F1 (Hauptmetrik), Accuracy (Sekundärmetrik)
- **Gruppen:** Aus `groups_trainval.csv` geladen (von Notebook 01 exportiert, alignment-garantiert)

#### 3.3.3 Hyperparameter-Tuning

`RandomizedSearchCV` (n_iter=30) für das beste Einzelmodell aus dem CV-Vergleich. Das getunete Modell wird nur übernommen, wenn der CV-F1 tatsächlich höher ist als das Basismodell (kein Test-Set-Shopping).

#### 3.3.4 Evaluationsmetriken

| Metrik | Begründung |
|--------|-----------|
| Gewichtetes F1 | Berücksichtigt Klassenungleichgewicht; Hauptmetrik |
| Accuracy | Vergleichbarkeit mit Literatur |
| CV F1 (mean ± std) | Stabilitätsindikator über Folds |
| Konfusionsmatrix (absolut + normalisiert) | Identifikation spezifischer Verwechslungsmuster |
| Per-Klasse Precision/Recall/F1/Support | Separate Beurteilung von Minderheitsklassen |
| Binäre CM: Gehen vs. Treppe_runter | Gezielte Analyse des Hauptverwechslungspaares |
| Mixed-Evaluation | Evaluation auf zusammengesetzten Aufnahmen mit Ground-Truth-Annotation |

---

## 4 Results

### 4.1 Erzielte Modellleistung

**Primäre Evaluationsmetrik: Mixed-Evaluation F1 = 0.87**

Die Mixed-Evaluation ist die aussagekräftigste Kennzahl: Sie misst das Modell auf vollständig ungesehenen Aufnahmen mit realistischen Aktivitätswechseln – identisch zur Zielsituation in der App. Das Test-Set (isolierte Einzelaktivitäten, sehr klein) liefert einen strukturell überhöhten Wert und dient nur als Kontrollpunkt.

| Evaluation | F1 | Accuracy | Datenbasis | Aussagekraft |
|-----------|-----|----------|-----------|-------------|
| **Mixed-Evaluation** | **0.8695** | **0.8660** | **642 Fenster, 4 Sessions mit Aktivitätswechseln** | **Primär – realistische Bedingungen** |
| Cross-Validation (CV) | 0.8752 | 0.8466 | 3 166 Fenster, 41 Sessions, GroupKFold k=3 | Modellvergleich und Selektion |
| Test-Set (isoliert) | 0.9912 | 0.9912 | 455 Fenster, 6 Sessions – je eine pro Klasse | Kontrollpunkt (kleines Set, optimistisch) |

> **Kein Overfitting:** CV F1 (0.875) und Mixed F1 (0.870) stimmen eng überein. Das Modell generalisiert auf unbekannte gemischte Aufnahmen ähnlich gut wie in der CV – der hohe Test-F1 (0.991) ist ein Artefakt des kleinen, per Design typischen Test-Sets und nicht Overfitting.

**Modellvergleich – 3-fache GroupKFold Cross-Validation (Train+Val, session-separiert):**

| Modell | CV F1 (mean ± std) | CV Accuracy |
|--------|-------------------|-------------|
| Decision Tree | 0.8275 ± 0.0364 | 0.8292 |
| Random Forest | 0.8103 ± 0.0111 | 0.8161 |
| Extra Trees | 0.8367 ± 0.0208 | 0.8405 |
| SVM | 0.7854 ± 0.0194 | 0.7940 |
| Gradient Boosting | 0.8413 ± 0.0526 | 0.8466 |
| HistGradientBoosting | 0.8136 ± 0.0284 | 0.8210 |
| KNN | 0.7721 ± 0.0296 | 0.7757 |
| Voting Ensemble (Top-3) | 0.8404 ± 0.0411 | 0.8453 |
| **Gradient Boosting (tuned)** | **0.8752** | **–** |

Das Gradient-Boosting-Modell wurde nach Modellselektion per `RandomizedSearchCV` (n_iter=30) getuned (beste Parameter: n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.7) und als finale Pipeline in `model.pkl` gespeichert.

### 4.2 Mixed-Evaluation – Per-Session Metriken

Evaluation auf vier manuell annotierten zusammengesetzten Aufnahmen aus `data/mixed/`. Die Mixed-Sessions wurden **nicht** für Training, Validation oder Hyperparameter-Tuning verwendet.

| Session | Aktivitäten (Ground Truth) | Fenster |
|---------|---------------------------|---------|
| mixed_1 | Treppe_runter, Treppe_hoch, Gehen | – |
| mixed_2 | Gehen, Laufen, Stehen, Gehen, Treppe_hoch, Gehen, Liegen | – |
| mixed_3 | Gehen, Stehen (wechselnd) | – |
| mixed_4 | Stehen, Treppe_runter, Gehen | – |

**Gesamt: 642 annotierte Fenster, Accuracy 0.8660, F1 0.8695**

Visualisierungen: `data/processed/mixed_confusion_matrix.png` (absolut + normalisiert), `mixed_evaluation.png` (Zeitverlauf je Session), `mixed_per_session_metrics.png`.

### 4.3 Per-Klasse Metriken (Test-Set – isolierte Einzelaktivitäten)

Test-Set: 6 Sessions (je 1 pro Klasse), 455 Fenster. Nur als Kontrollpunkt – nicht primäre Metrik.

| Klasse | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Gehen | 0.98 | 0.98 | 0.98 | 122 |
| Laufen | 0.98 | 1.00 | 0.99 | 109 |
| Liegen | 1.00 | 1.00 | 1.00 | 118 |
| Stehen | 1.00 | 1.00 | 1.00 | 32 |
| Treppe_hoch | 1.00 | 0.97 | 0.99 | 38 |
| Treppe_runter | 1.00 | 0.97 | 0.99 | 36 |

### 4.4 Feature Importance

Die wichtigsten Features nach Modell-Feature-Importance (Gradient Boosting):

1. `orie_pitch_energy` – Orientierungsenergie der Pitch-Achse
2. `orie_pitch_range` – Orientierungsbereich der Pitch-Achse
3. `acc_mag_dom_freq` – Dominante Schrittfrequenz (FFT)

Die Dominanz von Orientierungs-Features bestätigt, dass das Modell die Geräteausrichtung als primäres Unterscheidungsmerkmal verwendet. Dies hat direkte Auswirkungen auf die Generalisierbarkeit bei anderen Trageweisen (siehe Abschnitt 5.2).

### 4.5 App-Konzept

Die Streamlit-Applikation besteht aus drei Bereichen:

**Klassifikation:** Nutzer laden alle CSV-Dateien eines Sensor-Logger-Aufnahmeordners hoch. Die App erkennt automatisch Accelerometer, Gyroscope und Orientation, führt die identische Preprocessing-Pipeline wie das Training aus (Resampling, Trim, Gap-Check, Windowing, Feature-Extraktion, Temporal Smoothing via Rolling Majority Vote über 3 Fenster) und gibt den zeitlichen Aktivitätsverlauf als Segmentdiagramm sowie Zeitanteile je Klasse aus.

**Modell-Evaluation:** Darstellung der Trainingsmetriken, Konfusionsmatrix, Per-Klasse Metriken und Feature Importance auf Basis des gespeicherten Test-Sets.

**Über das Projekt:** Projektbeschreibung, CRISP-DM Phasen, Bias-Analyse, Related Work, Sensorsetup.

---

## 5 Discussion

### 5.1 Interpretation der Ergebnisse

_Nach finalem Notebook-Durchlauf ausfüllen._

Die Modellleistung auf dem session-separierten Test-Set zeigt, dass einfache statistische Features aus drei Sensoren eine Unterscheidung von Ruheaktivitäten (Liegen, Stehen) und Hochenergie-Aktivitäten (Laufen) mit hoher Genauigkeit ermöglichen. Die Hauptschwierigkeit liegt in der Unterscheidung von Gehen und Treppe_runter, da beide Aktivitäten ähnliche Schrittfrequenzen und Beschleunigungsamplituden aufweisen. Die ergänzten Orientierungs-Delta-Features zeigen eine messbare Verbesserung für diese Klasse.

Der beobachtete Gap zwischen CV F1 und Test F1 quantifiziert den Personenbias: Das Modell generalisiert innerhalb von Aufnahmen einer Person gut, jedoch schlechter auf vollständig unbekannte Sessions. Dies entspricht einem bekannten Problem in der HAR-Literatur bei kleinen Probandengruppen.

### 5.2 Limitierungen

| Limitation | Schwere | Beschreibung |
|-----------|---------|-------------|
| Personenbias | Hoch | Aufnahmen von 2–3 Personen. Das Modell lernt individuelle Gangmuster und Körperhaltungen, nicht allgemeingültige Bewegungscharakteristika. Der Gap zwischen CV F1 (session-intern) und Test F1 (session-separiert) quantifiziert diesen Effekt. |
| Positionsbias | Hoch | Gerät ausschließlich in der rechten Hosentasche, vertikal. Feature Importance zeigt orie_pitch als dominantes Feature – dieser ist stark positionsabhängig. Andere Trageweisen würden abweichende Orientierungswerte erzeugen. |
| Kleine Test-Datenbasis | Mittel | Je Klasse nur eine Test-Session. Evaluationsmetriken sind sensitiv gegenüber der Qualität dieser einzelnen Session. |
| Klassenungleichgewicht | Mittel | Treppe_hoch und Treppe_runter haben ~40% weniger Fenster als Gehen. Mit class_weight='balanced' und stratifiziertem Split abgemildert, aber nicht vollständig kompensiert. |
| Sampling-Rate-Mismatch | Mittel | Zwei Geräte mit 61 Hz und 100 Hz. Durch Resampling auf 61 Hz weitgehend eliminiert. |
| Umgebungsbias | Niedrig | Aufnahmen ausschließlich in Universitätsgebäuden. Andere Treppenkonstruktionen oder Untergründe können die Klassifikation beeinflussen. |

### 5.3 Ethische Betrachtung

**Diskriminierungspotenzial:** Das Modell wurde auf einem demographisch homogenen Datensatz (2–3 Personen, FAU-Umfeld) trainiert. Bei einer Ausweitung auf weitere Nutzergruppen ist zu prüfen, ob die Klassifikationsleistung für Personen mit anderen Körpermassen, Gangmustern oder Beeinträchtigungen vergleichbar ist. Eine ungleiche Leistung über Bevölkerungsgruppen hinweg wäre aus ethischer Sicht problematisch, insbesondere bei Einsatz in medizinischen Anwendungen.

**Transparenz:** Das Modell und seine Limitierungen sind in der App explizit dokumentiert. Nutzer werden darauf hingewiesen, dass die Klassifikation für andere Trageweisen oder Personengruppen nicht validiert ist.

**Datenschutz:** Die erhobenen Sensordaten ermöglichen Rückschlüsse auf individuelle Bewegungsmuster und sind damit als personenbezogene Daten einzustufen. Im Rahmen dieses Projekts wurden ausschließlich Daten von Mitgliedern der Projektgruppe erhoben. Bei einer Ausweitung auf externe Probanden wäre eine formelle Einwilligungserklärung erforderlich.

**Umwelt:** Das Training klassischer Machine-Learning-Modelle auf einem Datensatz dieser Größe hat einen vernachlässigbaren CO2-Fußabdruck.

### 5.4 Weiterer Forschungsbedarf

- Erhebung eines größeren und demographisch diverseren Datensatzes (>20 Probanden)
- Evaluation der Modellleistung unter verschiedenen Trageweisen und Gerätepositionen
- Untersuchung von Deep-Learning-Ansätzen (CNN, LSTM) auf demselben session-basierten Split zum direkten Vergleich
- Systematische SHAP-basierte Feature-Reduktion und Quantifizierung des Tradeoffs zwischen Modellgröße und Klassifikationsleistung

---

## 6 Conclusion

Dieses Projekt zeigt, dass die Klassifikation von sechs Alltagsaktivitäten aus Smartphone-Sensordaten mit statistischen Features und klassischen Klassifikatoren grundsätzlich möglich ist. Ruheaktivitäten (Liegen, Stehen) und Hochenergie-Aktivitäten (Laufen) werden zuverlässig erkannt. Die Unterscheidung ähnlicher Aktivitäten (Gehen vs. Treppe_runter) stellt eine verbleibende Herausforderung dar.

Ein zentraler methodischer Befund ist die Notwendigkeit eines session-basierten Datensplits: Der ursprünglich verwendete zufällige Fensterschnitt erzeugte durch die 50%-Fenster-Überlappung Data Leakage und führte zu nicht validen Metriken (CV F1 ~0.99). Nach Umstellung auf session-basierten GroupShuffleSplit und GroupKFold sind die Metriken interpretierbar und spiegeln die tatsächliche Generalisierungsfähigkeit wider.

Die geringe Probandenanzahl ist die wesentliche Limitierung des Projekts und limitiert die Übertragbarkeit der Ergebnisse auf neue Nutzer. Für eine produktive Anwendung wäre eine deutlich größere und diversere Datenbasis erforderlich.

---

## 7 Appendix

### 7.1 Aufnahmeprotokoll

- Gerät: Smartphone in der rechten Hosentasche
- Kein aktives Festhalten des Geräts
- Display sollte nicht gesperrt sein, App muss im Vordergrund bleiben
- Mindestaufnahmedauer: 30 Sekunden pro Session
- Labeling: Klassenordner-Name dient als Label (kein manuelles Tagging nötig)

### 7.2 Bekannte Fehler und deren Behebung im Projektverlauf

| Problem | Ursache | Behebung |
|---------|---------|---------|
| CV F1 ~0.99, Test F1 ~0.99 (initial) | Zufälliger Fensterschnitt, Data Leakage durch 50% Überlappung | Session-basierter GroupShuffleSplit + GroupKFold |
| Stehen Recall = 0% | Nur 1 Trainings-Session mit 22s (21 Fenster) | 4 neue Stehen-Sessions aufgenommen |
| Gehen wird als Laufen klassifiziert | Fehlende Frequenz-Features | FFT-Dominanzfrequenz und Magnitude-Features hinzugefügt |
| Double-Scaling in app.py | model.pkl ist Pipeline; zusätzlicher scaler.transform()-Aufruf | Rohe Features direkt an model.predict() übergeben |
| Test-Set enthält nicht alle Klassen | GroupShuffleSplit ohne Stratifizierung | Session-stratifizierter Split mit Median-Session-Auswahl |
| Sampling-Rate-abhängige Features | 61 Hz vs. 100 Hz Geräte | Resampling auf 61 Hz vor Windowing |

### 7.3 Reproduzierbarkeit

| Aspekt | Status |
|--------|--------|
| Preprocessing | Vollständig in Notebook 01, identisch in app.py |
| Feature Engineering | Identisch in NB01, NB02 und app.py |
| Modelltraining | Deterministisch (RANDOM_STATE=42) |
| Split | Session-stratifiziert, deterministisch, mit Assertions |
| Modell-Export | model.pkl (Pipeline), feature_names.txt, model_metadata.json |
| Abhaengigkeiten | pyproject.toml mit Versionsangaben, uv.lock für exakte Reproduktion |