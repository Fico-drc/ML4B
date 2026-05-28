import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import resample as scipy_resample

st.set_page_config(
    page_title="NoNames – StepSense",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
.metric-card {
    background: #0f0f0f; border: 1px solid #2a2a2a;
    border-radius: 4px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #e8ff4a; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #555;
    text-transform: uppercase; letter-spacing: 0.15em;
    border-bottom: 1px solid #1e1e1e; padding-bottom: 0.5rem;
    margin-bottom: 1.5rem; margin-top: 2rem;
}
.explain-box {
    background: #111; border-left: 2px solid #e8ff4a;
    padding: 0.8rem 1rem; border-radius: 0 4px 4px 0;
    font-size: 0.82rem; color: #999; line-height: 1.6; margin-bottom: 1rem;
}
.warn-box {
    background: #1a1000; border-left: 2px solid #ff9800;
    padding: 0.8rem 1rem; border-radius: 0 4px 4px 0;
    font-size: 0.82rem; color: #cc7a00; line-height: 1.6; margin-bottom: 1rem;
}
.stApp { background-color: #080808; color: #e0e0e0; }
.stSidebar { background-color: #0d0d0d; border-right: 1px solid #1e1e1e; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed"
MODEL_PATH     = os.path.join(PROCESSED_PATH, "model.pkl")
SCALER_PATH    = os.path.join(PROCESSED_PATH, "scaler.pkl")
FEATURES_PATH  = os.path.join(PROCESSED_PATH, "feature_names.txt")
METADATA_PATH  = os.path.join(PROCESSED_PATH, "model_metadata.json")
WINDOW_SIZE_S  = 2.0
STEP_SIZE_S    = 1.0
TARGET_HZ      = 61   # niedrigste vorhandene Sampling-Rate als Ziel

SENSOR_PREFIX = {
    "Accelerometer": "acc",
    "Gyroscope":     "gyro",
    "Orientation":   "orie",
}
SENSOR_COLUMNS = {
    "Accelerometer": ["x","y","z"],
    "Gyroscope":     ["x","y","z"],
    "Orientation":   ["roll","pitch","yaw"]
}

CLASS_COLORS = {
    "Gehen":         "#4CAF50",
    "Laufen":        "#2196F3",
    "Liegen":        "#9C27B0",
    "Stehen":        "#FF9800",
    "Treppe_hoch":   "#F44336",
    "Treppe_runter": "#00BCD4",
}

FEATURE_COLS_RAW = [
    "acc_x","acc_y","acc_z",
    "gyro_x","gyro_y","gyro_z",
    "orie_roll","orie_pitch","orie_yaw"
]

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH) as f:
            feature_cols = [l.strip() for l in f.readlines()]
        metadata = {}
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
        return model, scaler, feature_cols, metadata
    except FileNotFoundError:
        return None, None, None, {}

# ── Feature engineering ────────────────────────────────────────────────────────
def compute_features(window):
    features = {}
    for col in FEATURE_COLS_RAW:
        if col not in window.columns:
            continue
        s = window[col].dropna()
        if len(s) < 5:
            continue
        features[f"{col}_mean"]   = s.mean()
        features[f"{col}_std"]    = s.std()
        features[f"{col}_min"]    = s.min()
        features[f"{col}_max"]    = s.max()
        features[f"{col}_range"]  = s.max() - s.min()
        features[f"{col}_energy"] = (s**2).mean()
        features[f"{col}_iqr"]    = s.quantile(0.75) - s.quantile(0.25)
        features[f"{col}_zcr"]    = ((s.iloc[:-1].values * s.iloc[1:].values) < 0).sum() / len(s)

    # Magnitude (rotationsinvariant – kompensiert unterschiedliche Handylage)
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    if all(c in window.columns for c in acc_cols):
        mag = np.sqrt(window["acc_x"]**2 + window["acc_y"]**2 + window["acc_z"]**2).dropna()
        if len(mag) >= 5:
            features["acc_mag_mean"]   = mag.mean()
            features["acc_mag_std"]    = mag.std()
            features["acc_mag_energy"] = (mag**2).mean()
            # Dominante Frequenz via FFT (erfasst Schrittfrequenz-Unterschied Gehen vs. Laufen)
            actual_hz = len(mag) / WINDOW_SIZE_S
            fft_vals  = np.abs(np.fft.rfft(mag - mag.mean()))
            freqs     = np.fft.rfftfreq(len(mag), d=1.0 / actual_hz)
            features["acc_mag_dom_freq"] = float(freqs[np.argmax(fft_vals[1:]) + 1]) if len(fft_vals) > 1 else 0.0

    # Orientierungsänderung je Fenster (Treppe kippt den Körper, flaches Gehen nicht)
    if "orie_pitch" in window.columns:
        features["orie_pitch_delta"] = float(window["orie_pitch"].max() - window["orie_pitch"].min())
    if "orie_roll" in window.columns:
        features["orie_roll_delta"]  = float(window["orie_roll"].max()  - window["orie_roll"].min())

    return features

def process_csv(uploaded_file, sensor_name):
    df = pd.read_csv(uploaded_file)
    df["time_s"] = (df["time"] - df["time"].iloc[0]) / 1e9
    df = df.drop(columns=["time","seconds_elapsed"], errors="ignore")
    prefix = SENSOR_PREFIX.get(sensor_name, sensor_name[:4].lower())
    cols = SENSOR_COLUMNS.get(sensor_name, [])
    df = df.rename(columns={c: f"{prefix}_{c}" for c in cols if c in df.columns})
    df = df[["time_s"] + [f"{prefix}_{c}" for c in cols if f"{prefix}_{c}" in df.columns]]
    return df

def find_sensor_file(uploaded_files, sensor_name):
    for f in uploaded_files:
        name = f.name.replace(".csv","").strip().lower()
        if name == sensor_name.lower():
            return f
    return None

def merge_sensors(dfs):
    df_merged = dfs["Accelerometer"].sort_values("time_s")
    for name in ["Gyroscope","Orientation"]:
        if name in dfs:
            df_merged = pd.merge_asof(
                df_merged, dfs[name].sort_values("time_s"),
                on="time_s", direction="nearest", tolerance=0.05
            )
    return df_merged

TRIM_S   = 2.0   # Sekunden am Anfang/Ende abschneiden (Taschenrauschen)
GAP_S    = 5.0   # Maximale erlaubte Lücke zwischen Samples


def resample_to_target_hz(df, target_hz=TARGET_HZ):
    """Resampled alle Sensorspalten auf einheitliche Sampling-Rate."""
    sensor_cols = [c for c in df.columns if c != "time_s"]
    median_dt = df["time_s"].diff().median()
    if pd.isna(median_dt) or median_dt <= 0:
        return df
    actual_hz = 1.0 / median_dt
    if abs(actual_hz - target_hz) <= 5:
        return df
    n_target = int(len(df) * target_hz / actual_hz)
    if n_target < 10:
        return df
    t_new = np.linspace(df["time_s"].iloc[0], df["time_s"].iloc[-1], n_target)
    df_out = pd.DataFrame({"time_s": t_new})
    for col in sensor_cols:
        df_out[col] = scipy_resample(df[col].values, n_target)
    return df_out


def prepare_dataframe(df):
    """
    Identische Vorverarbeitung wie in Notebook 01:
    0. Resampling auf TARGET_HZ (eliminiert Sampling-Rate-Abhängigkeit von FFT/ZCR)
    1. Erste und letzte 2s trimmen (Taschenrauschen beim Rein-/Rausnehmen)
    2. Sessions mit Lücken > 5s verwerfen (unterbrochene Aufnahmen)
    3. Duplikate entfernen
    4. Fehlende Werte per ffill/bfill auffüllen
    """
    sensor_cols = [c for c in df.columns if c != "time_s"]

    # 0. Resampling auf einheitliche Rate (vor Trim, damit Randsekunden korrekt sind)
    df = resample_to_target_hz(df)

    # 1. Trim: erste und letzte 2s entfernen
    t_min = df["time_s"].min() + TRIM_S
    t_max = df["time_s"].max() - TRIM_S
    df = df[(df["time_s"] >= t_min) & (df["time_s"] <= t_max)].copy()

    if len(df) < 20:
        return None, "Aufnahme zu kurz nach Trim (mindestens ~6 Sekunden nötig)."

    # 2. Gap-Check: Lücken > GAP_S erkennen
    df = df.sort_values("time_s").reset_index(drop=True)
    gaps = df["time_s"].diff()
    max_gap = gaps.max()
    if max_gap > GAP_S:
        # Nur den Teil vor der ersten großen Lücke verwenden
        gap_idx = gaps[gaps > GAP_S].index[0]
        df = df.iloc[:gap_idx].copy()
        if len(df) < 20:
            return None, f"Aufnahme enthält eine Lücke von {max_gap:.1f}s – zu wenig Daten vor der Lücke."

    # 3. Duplikate entfernen
    df = df.drop_duplicates(subset=["time_s"]).reset_index(drop=True)

    # 4. Fehlende Werte auffüllen
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    return df, None


def classify_dataframe(df, model, feature_cols):
    # Vorverarbeitung identisch zu Notebook 01
    df, error = prepare_dataframe(df)
    if df is None:
        return None, error

    t_start = df["time_s"].min()
    t_end   = df["time_s"].max()
    rows = []
    t = t_start
    while t + WINDOW_SIZE_S <= t_end:
        window = df[(df["time_s"] >= t) & (df["time_s"] < t + WINDOW_SIZE_S)]
        if len(window) >= 10:
            feats = compute_features(window)
            feats["window_mid"] = t + WINDOW_SIZE_S / 2
            rows.append(feats)
        t += STEP_SIZE_S
    if not rows:
        return None, "Zu wenige Samples nach Vorverarbeitung."

    df_feat = pd.DataFrame(rows).fillna(0)
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
    # model is a Pipeline (scaler + clf) – pass raw features directly
    df_feat["predicted"] = model.predict(df_feat[feature_cols].values)
    # Temporal smoothing: rolling majority vote (window=3) reduces single-window flicker
    _preds = df_feat["predicted"].values.copy()
    for _i in range(len(_preds)):
        _lo = max(0, _i - 1)
        _hi = min(len(_preds), _i + 2)
        _vals, _cnts = np.unique(_preds[_lo:_hi], return_counts=True)
        _preds[_i] = _vals[_cnts.argmax()]
    df_feat["predicted"] = _preds
    return df_feat, None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# NoNames")
    st.markdown("<div style='color:#555;font-size:0.75rem;font-family:monospace'>StepSense · ML4B SoSe 2026</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["Klassifikation","Modell-Evaluation","Über das Projekt"], label_visibility="collapsed")
    st.markdown("---")
    model, scaler, feature_cols, metadata = load_artifacts()
    if model is not None:
        st.markdown("<div class='section-header'>Modell</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:monospace;font-size:0.8rem;color:#888'>{metadata.get('model_name','–')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:monospace;font-size:0.8rem;color:#e8ff4a'>Test F1: {metadata.get('test_f1','–')}</div>", unsafe_allow_html=True)
    else:
        st.warning("Kein Modell gefunden.\nBitte zuerst Notebook 02 ausführen.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – KLASSIFIKATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Klassifikation":
    st.markdown("# Aktivitätsklassifikation")

    if model is None:
        st.error("Modell nicht gefunden. Bitte zuerst Notebook 02 ausführen.")
        st.stop()

    # Aufnahmeanleitung
    st.markdown("<div class='section-header'>Aufnahmeanleitung</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col_ui, num, title, content in [
        (col1, "01", "POSITION",
         "📱 Smartphone in die <strong style='color:#ddd'>rechte Hosentasche</strong><br>Gerät vertikal ausgerichtet<br>Nicht aktiv festhalten<br>Display kann aus sein"),
        (col2, "02", "AUFNAHME",
         "📲 Sensor Logger App öffnen<br>Aufnahme starten<br>Aktivität <strong style='color:#ddd'>mindestens 10 Sekunden</strong> ausführen<br>Aufnahme beenden"),
        (col3, "03", "UPLOAD",
         "📁 Aufnahmeordner öffnen<br><strong style='color:#ddd'>Strg+A</strong> → alle Dateien auswählen<br>Hochladen – App erkennt automatisch<br>Accelerometer, Gyroscope & Orientation"),
    ]:
        with col_ui:
            st.markdown(f"""
            <div style='background:#0d0d0d;border:1px solid #1e1e1e;border-radius:4px;padding:1rem'>
            <div style='font-family:monospace;font-size:0.7rem;color:#e8ff4a;margin-bottom:0.8rem'>{num} · {title}</div>
            <div style='font-size:0.85rem;color:#aaa;line-height:1.7'>{content}</div>
            </div>""", unsafe_allow_html=True)

    # Upload
    st.markdown("<div class='section-header'>Aufnahmeordner hochladen</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explain-box'>
    Wähle <strong>alle Dateien</strong> aus deinem Sensor-Logger-Aufnahmeordner aus
    (Strg+A im Ordner). Der Ordner kann beliebig viele Dateien enthalten –
    die App erkennt automatisch <strong>Accelerometer.csv</strong>, <strong>Gyroscope.csv</strong>
    und <strong>Orientation.csv</strong> und ignoriert den Rest.
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Alle Dateien aus dem Aufnahmeordner auswählen (Strg+A)",
        type="csv",
        accept_multiple_files=True,
        help="Öffne den Aufnahmeordner, drücke Strg+A und lade alle Dateien hoch"
    )

    if uploaded_files:
        acc_file  = find_sensor_file(uploaded_files, "Accelerometer")
        gyro_file = find_sensor_file(uploaded_files, "Gyroscope")
        ori_file  = find_sensor_file(uploaded_files, "Orientation")

        # Status
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col_ui, label, found in [
            (c1, "Accelerometer.csv", acc_file is not None),
            (c2, "Gyroscope.csv",     gyro_file is not None),
            (c3, "Orientation.csv",   ori_file is not None),
        ]:
            with col_ui:
                color = "#4CAF50" if found else "#F44336"
                icon  = "✓" if found else "✗"
                st.markdown(f"""
                <div style='background:#0d0d0d;border:1px solid {color}55;border-radius:4px;
                           padding:0.6rem 1rem;text-align:center'>
                    <span style='color:{color};font-family:monospace;font-size:1.1rem'>{icon}</span>
                    <span style='font-size:0.8rem;color:#aaa;margin-left:0.5rem'>{label}</span>
                </div>""", unsafe_allow_html=True)

        ignored = [f.name for f in uploaded_files
                  if f.name.replace(".csv","").strip().lower()
                  not in ["accelerometer","gyroscope","orientation"]]
        if ignored:
            st.markdown(f"<div style='font-size:0.72rem;color:#333;margin-top:0.4rem'>Ignoriert: {', '.join(ignored)}</div>",
                       unsafe_allow_html=True)

        if acc_file and gyro_file and ori_file:
            try:
                dfs = {
                    "Accelerometer": process_csv(acc_file,  "Accelerometer"),
                    "Gyroscope":     process_csv(gyro_file, "Gyroscope"),
                    "Orientation":   process_csv(ori_file,  "Orientation"),
                }
                df_merged = merge_sensors(dfs)
                df_result, prep_error = classify_dataframe(df_merged, model, feature_cols)

                if prep_error:
                    st.warning(f"⚠️ Vorverarbeitung: {prep_error}")

                if df_result is not None and len(df_result) > 0:
                    top_class = df_result["predicted"].value_counts().idxmax()
                    top_pct   = df_result["predicted"].value_counts(normalize=True).max() * 100
                    n_windows = len(df_result)
                    duration  = df_merged["time_s"].max() - df_merged["time_s"].min()

                    st.markdown("<div class='section-header'>Zusammenfassung</div>", unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    for col_ui, val, label in [
                        (c1, top_class,          "Hauptaktivität"),
                        (c2, f"{top_pct:.0f}%",  "Anteil Hauptaktivität"),
                        (c3, str(n_windows),     "Analysierte Fenster"),
                        (c4, f"{duration:.1f}s", "Aufnahmedauer"),
                    ]:
                        with col_ui:
                            st.markdown(f"""<div class='metric-card'>
                                <div class='metric-value'>{val}</div>
                                <div class='metric-label'>{label}</div>
                            </div>""", unsafe_allow_html=True)

                    # ── Aktivitätsverlauf Zeitreihe ──────────────────────────
                    st.markdown("<div class='section-header'>Aktivitätsverlauf über Zeit</div>",
                            unsafe_allow_html=True)
                    st.markdown("""
                    <div class='explain-box'>
                    Jedes farbige Segment zeigt eine erkannte Aktivitätsphase. Die Breite entspricht der Dauer.
                    Beschriftung zeigt Aktivitätsname und Dauer in Sekunden.
                    </div>
                    """, unsafe_allow_html=True)

                    # Phasen berechnen
                    times_arr  = df_result["window_mid"].values
                    preds_arr  = df_result["predicted"].values
                    phases = []
                    prev_cls, phase_start = None, None
                    for t, cls in zip(times_arr, preds_arr):
                        if cls != prev_cls:
                            if prev_cls is not None:
                                phases.append((prev_cls, phase_start, t))
                            phase_start = t
                            prev_cls    = cls
                    if prev_cls is not None:
                        phases.append((prev_cls, phase_start, times_arr[-1] + STEP_SIZE_S))

                    fig, ax = plt.subplots(figsize=(14, 2.2))
                    fig.patch.set_facecolor("#0f0f0f")
                    ax.set_facecolor("#0f0f0f")

                    BAR_H = 0.55
                    for cls, t_start, t_end in phases:
                        color    = CLASS_COLORS.get(cls, "#888")
                        duration = t_end - t_start
                        ax.barh(0, duration, left=t_start, height=BAR_H,
                            color=color, alpha=0.85,
                            edgecolor="#0f0f0f", linewidth=0.8)
                        if duration >= 3:
                            ax.text(t_start + duration / 2, 0,
                                f"{cls}\n{duration:.0f}s",
                                ha="center", va="center",
                                fontsize=7, color="white",
                                fontfamily="monospace", fontweight="600")

                    total = times_arr[-1] + STEP_SIZE_S - times_arr[0]
                    ax.set_xlim(times_arr[0], times_arr[-1] + STEP_SIZE_S)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks([])
                    ax.set_xlabel("Zeit (s)", color="#555", fontsize=8)
                    ax.tick_params(colors="#555", labelsize=7)
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#1e1e1e")
                    ax.grid(axis="x", alpha=0.08, color="#fff", linestyle="--")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.markdown("""
                    <div class='explain-box'>
                    Der Zeitstrahl zeigt welche Aktivität zu welchem Zeitpunkt erkannt wurde.
                    Die Y-Achse zeigt die Aktivitätsklasse, die X-Achse die Zeit in Sekunden.
                    Farbige Hintergrundbereiche markieren zusammenhängende Aktivitätsphasen –
                    so sieht man auf einen Blick wie lange welche Aktivität ausgeführt wurde.
                    </div>
                    """, unsafe_allow_html=True)

                    # Alle bekannten Klassen auf Y-Achse
                    all_classes   = sorted(CLASS_COLORS.keys())
                    class_to_int  = {c: i for i, c in enumerate(all_classes)}
                    y_vals        = df_result["predicted"].map(class_to_int)
                    times         = df_result["window_mid"].values
                    predictions   = df_result["predicted"].values

                    fig, ax = plt.subplots(figsize=(14, 5))
                    fig.patch.set_facecolor("#0f0f0f")
                    ax.set_facecolor("#0f0f0f")

                    # Hintergrundbereiche je zusammenhängende Phase
                    prev_cls    = None
                    phase_start = None
                    for i, (t, cls) in enumerate(zip(times, predictions)):
                        if cls != prev_cls:
                            if prev_cls is not None:
                                ax.axvspan(phase_start, t,
                                          color=CLASS_COLORS.get(prev_cls, "#888"),
                                          alpha=0.12, linewidth=0)
                            phase_start = t
                            prev_cls    = cls
                    # letzte Phase
                    if prev_cls is not None:
                        ax.axvspan(phase_start, times[-1],
                                  color=CLASS_COLORS.get(prev_cls, "#888"),
                                  alpha=0.12, linewidth=0)

                    # Treppenlinie
                    ax.step(times, y_vals,
                           color="#e8ff4a", linewidth=1.8,
                           where="post", zorder=3)

                    # Punkte farbig je Klasse
                    point_colors = [CLASS_COLORS.get(c, "#888") for c in predictions]
                    ax.scatter(times, y_vals,
                              c=point_colors, s=25, zorder=4, linewidths=0)

                    # Achsen
                    ax.set_yticks(range(len(all_classes)))
                    ax.set_yticklabels(all_classes, color="#aaa", fontsize=9)
                    ax.set_xlabel("Zeit (s)", color="#555", fontsize=9)
                    ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
                    ax.set_ylim(-0.5, len(all_classes) - 0.5)
                    ax.tick_params(colors="#555", labelsize=8)
                    ax.grid(axis="x", alpha=0.08, color="#fff", linestyle="--")
                    ax.grid(axis="y", alpha=0.05, color="#fff")
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#1e1e1e")

                    # Legende nur für vorhandene Klassen
                    unique_classes = df_result["predicted"].unique()
                    patches = [mpatches.Patch(color=CLASS_COLORS.get(c,"#888"), label=c)
                              for c in sorted(unique_classes)]
                    ax.legend(handles=patches, loc="upper right", fontsize=8,
                             facecolor="#1a1a1a", edgecolor="#333", labelcolor="#aaa",
                             framealpha=0.9)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # ── Zeitanteile je Aktivität ─────────────────────────────
                    st.markdown("<div class='section-header'>Zeitanteile je Aktivität</div>",
                               unsafe_allow_html=True)
                    st.markdown("""
                    <div class='explain-box'>
                    Wie viel Zeit wurde mit welcher Aktivität verbracht?
                    Jedes Fenster entspricht 2 Sekunden.
                    </div>
                    """, unsafe_allow_html=True)

                    counts = df_result["predicted"].value_counts().sort_values(ascending=False)
                    for cls, cnt in counts.items():
                        pct      = cnt / len(df_result) * 100
                        secs     = cnt * STEP_SIZE_S
                        color    = CLASS_COLORS.get(cls, "#888")
                        bar_pct  = int(pct)
                        st.markdown(f"""
                        <div style='display:flex;align-items:center;margin-bottom:0.5rem;gap:0.8rem'>
                            <div style='min-width:120px;font-family:monospace;font-size:0.82rem;color:#ccc'>{cls}</div>
                            <div style='flex:1;background:#111;border-radius:2px;height:16px;overflow:hidden'>
                                <div style='width:{bar_pct}%;background:{color};height:100%;border-radius:2px;opacity:0.8'></div>
                            </div>
                            <div style='min-width:80px;font-family:monospace;font-size:0.8rem;color:#e8ff4a;text-align:right'>
                                {secs:.0f}s · {pct:.0f}%
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # ── Signalplot ───────────────────────────────────────────
                    st.markdown("<div class='section-header'>Rohdaten der Sensoren</div>",
                               unsafe_allow_html=True)
                    st.markdown("""
                    <div class='explain-box'>
                    <strong>Accelerometer</strong> misst lineare Beschleunigung (m/s²) – bei Bewegung stark ausgeschlagen.
                    <strong>Gyroscope</strong> misst Rotationsrate (rad/s) – bei Drehungen des Geräts aktiv.
                    <strong>Orientation</strong> zeigt die Ausrichtung des Geräts im Raum (Roll, Pitch, Yaw in Radiant).
                    </div>
                    """, unsafe_allow_html=True)

                    fig2, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
                    fig2.patch.set_facecolor("#0f0f0f")
                    palette = ["#e8ff4a","#4a9eff","#ff6b6b"]
                    sensor_groups = [
                        (["acc_x","acc_y","acc_z"],           "Acc (m/s²)"),
                        (["gyro_x","gyro_y","gyro_z"],         "Gyro (rad/s)"),
                        (["orie_roll","orie_pitch","orie_yaw"], "Orientation (rad)"),
                    ]
                    for ax2, (cols, ylabel) in zip(axes, sensor_groups):
                        ax2.set_facecolor("#0f0f0f")
                        for col, color in zip(cols, palette):
                            if col in df_merged.columns:
                                ax2.plot(df_merged["time_s"], df_merged[col],
                                        color=color, linewidth=0.6, alpha=0.9,
                                        label=col.split("_")[-1])
                        ax2.set_ylabel(ylabel, color="#555", fontsize=7)
                        ax2.tick_params(colors="#555", labelsize=7)
                        ax2.legend(loc="upper right", fontsize=6,
                                  facecolor="#1a1a1a", edgecolor="#333", labelcolor="#aaa")
                        ax2.grid(True, alpha=0.05, color="#fff")
                        for sp in ax2.spines.values():
                            sp.set_edgecolor("#1e1e1e")
                    axes[-1].set_xlabel("Zeit (s)", color="#555", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                else:
                    st.warning("Zu wenige Samples. Aufnahme mindestens 10 Sekunden lang machen.")

            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung: {e}")

        elif uploaded_files:
            missing = []
            if not acc_file:  missing.append("Accelerometer.csv")
            if not gyro_file: missing.append("Gyroscope.csv")
            if not ori_file:  missing.append("Orientation.csv")
            st.warning(f"Folgende Dateien fehlen: {', '.join(missing)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MODELL-EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Modell-Evaluation":
    st.markdown("# Modell-Evaluation")
    st.markdown("""
    <div class='explain-box'>
    Diese Seite zeigt wie gut das trainierte Modell auf <strong>ungesehenen Testdaten</strong> abschneidet.
    Der Test-Datensatz wurde während des Trainings vollständig zurückgehalten (15% der Gesamtdaten)
    und erst nach Abschluss des Trainings zur Evaluation verwendet.
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Modell nicht gefunden. Bitte zuerst Notebook 02 ausführen.")
        st.stop()

    if metadata:
        st.markdown("<div class='section-header'>Gesamtmetriken (Test-Set)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='explain-box'>
        <strong>Accuracy:</strong> Anteil aller korrekt klassifizierten Fenster.<br>
        <strong>F1 gewichtet:</strong> Harmonisches Mittel aus Precision und Recall,
        gewichtet nach Klassenhäufigkeit. Besser geeignet als Accuracy bei ungleich verteilten Klassen.<br>
        <strong>CV F1:</strong> F1-Score aus 5-facher Kreuzvalidierung – zeigt wie stabil das Modell ist.
        </div>
        """, unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col_ui, val, label in [
            (c1, metadata.get("test_accuracy","–"), "Accuracy"),
            (c2, metadata.get("test_f1","–"),       "F1 gewichtet"),
            (c3, f"{metadata.get('cv_f1_mean','–')} ± {metadata.get('cv_f1_std','–')}", "CV F1 (5-fold)"),
            (c4, metadata.get("n_features","–"),    "Features"),
        ]:
            with col_ui:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

    features_all_path  = os.path.join(PROCESSED_PATH, "features_all.csv")
    session_split_path = os.path.join(PROCESSED_PATH, "session_split.json")
    y_test_path        = os.path.join(PROCESSED_PATH, "y_test.csv")

    can_eval = os.path.exists(features_all_path) and os.path.exists(session_split_path) and os.path.exists(y_test_path)
    if can_eval:
        with open(session_split_path) as _f:
            _sp = json.load(_f)
        _df_all  = pd.read_csv(features_all_path)
        _mask    = _df_all["session"].isin(_sp["test"])
        X_test   = _df_all.loc[_mask, feature_cols].reset_index(drop=True)
        y_test   = _df_all.loc[_mask, "label"].reset_index(drop=True)
        y_pred   = model.predict(X_test.values)
        present_classes = sorted(y_test.unique())

        st.markdown("<div class='section-header'>Konfusionsmatrix</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='explain-box'>
        Zeilen = tatsächliche Klasse, Spalten = vorhergesagte Klasse.
        Die Diagonale zeigt korrekte Vorhersagen. Außerdiagonale Einträge sind Verwechslungen.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='warn-box'>
        ⚠️ <strong>Overfitting-Hinweis:</strong> Die normalisierte Matrix kann täuschend gut aussehen
        wenn eine Klasse sehr selten im Test-Set vorkommt.
        Immer auch die absoluten Zahlen und den Support beachten.
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        for col_ui, normalize, title in [
            (col_l, None,   "Absolut – echte Fensterzahlen"),
            (col_r, "true", "Normalisiert – Recall je Klasse"),
        ]:
            cm  = confusion_matrix(y_test, y_pred, labels=present_classes, normalize=normalize)
            fmt = ".2f" if normalize else "d"
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor("#0f0f0f"); ax.set_facecolor("#0f0f0f")
            sns.heatmap(cm, annot=True, fmt=fmt, cmap="YlOrRd",
                       xticklabels=present_classes, yticklabels=present_classes,
                       ax=ax, linewidths=0.5, linecolor="#1a1a1a", cbar_kws={"shrink":0.8})
            ax.set_title(title, color="#888", fontsize=9, pad=10)
            ax.set_ylabel("Tatsächliche Klasse", color="#555", fontsize=8)
            ax.set_xlabel("Vorhergesagte Klasse", color="#555", fontsize=8)
            ax.tick_params(colors="#aaa", labelsize=7, rotation=45)
            with col_ui:
                st.pyplot(fig)
            plt.close()

        st.markdown("<div class='section-header'>Per-Klasse Metriken</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='explain-box'>
        <strong>Precision:</strong> Wie viele der als X vorhergesagten Fenster waren wirklich X? (wenig Fehlalarme)<br>
        <strong>Recall:</strong> Wie viele der echten X-Fenster wurden erkannt? (wenig verpasste Aktivitäten)<br>
        <strong>F1:</strong> Harmonisches Mittel aus Precision und Recall.<br>
        <strong>Support:</strong> Anzahl Testfenster – kleine Werte = weniger zuverlässige Schätzung.
        </div>
        """, unsafe_allow_html=True)

        report = classification_report(y_test, y_pred, labels=present_classes,
                                      output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(report).T.loc[present_classes,
                              ["precision","recall","f1-score","support"]].round(3)

        fig, ax = plt.subplots(figsize=(10,4))
        fig.patch.set_facecolor("#0f0f0f"); ax.set_facecolor("#0f0f0f")
        x = np.arange(len(present_classes)); w = 0.25
        ax.bar(x-w, df_rep["precision"], w, label="Precision", color="#e8ff4a", alpha=0.9)
        ax.bar(x,   df_rep["recall"],    w, label="Recall",    color="#4a9eff", alpha=0.9)
        ax.bar(x+w, df_rep["f1-score"],  w, label="F1",        color="#ff6b6b", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(present_classes, color="#aaa", fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.axhline(0.8, color="#555", linestyle="--", linewidth=0.8, label="0.8 Referenzlinie")
        ax.set_ylabel("Score", color="#555", fontsize=8)
        ax.tick_params(colors="#555", labelsize=7)
        ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#aaa")
        ax.grid(axis="y", alpha=0.05, color="#fff")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e1e1e")
        st.pyplot(fig); plt.close()

        st.dataframe(df_rep.style.background_gradient(
            cmap="RdYlGn", vmin=0, vmax=1,
            subset=["precision","recall","f1-score"]),
            use_container_width=True)

        _clf_step = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
        if hasattr(_clf_step, "feature_importances_"):
            st.markdown("<div class='section-header'>Feature Importance – Top 15</div>",
                       unsafe_allow_html=True)
            st.markdown("""
            <div class='explain-box'>
            Welche der 72 Features beeinflussen die Klassifikation am stärksten?
            Schema: <code>sensor_achse_statistik</code> – z.B. <code>acc_z_std</code> =
            Standardabweichung der Z-Achse des Accelerometers.
            </div>
            """, unsafe_allow_html=True)
            fi = pd.DataFrame({"Feature": feature_cols,
                              "Importance": _clf_step.feature_importances_}
                             ).sort_values("Importance", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10,5))
            fig.patch.set_facecolor("#0f0f0f"); ax.set_facecolor("#0f0f0f")
            colors = ["#e8ff4a" if i<3 else "#4a9eff" if i<8 else "#555"
                     for i in range(len(fi))]
            fi.sort_values("Importance").plot(kind="barh", x="Feature", y="Importance",
                ax=ax, color=colors[::-1], edgecolor="none", legend=False)
            ax.set_xlabel("Importance", color="#555", fontsize=8)
            ax.tick_params(colors="#aaa", labelsize=7)
            ax.grid(axis="x", alpha=0.05, color="#fff")
            for sp in ax.spines.values(): sp.set_edgecolor("#1e1e1e")
            st.pyplot(fig); plt.close()
    else:
        st.info("Test-Set nicht gefunden. Bitte Notebook 01 & 02 ausführen.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ÜBER DAS PROJEKT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Über das Projekt":
    st.markdown("# NoNames – StepSense")
    st.markdown("<div style='color:#666;font-size:1rem;margin-bottom:2rem'>Bewegungsklassifikation aus Smartphone-Sensordaten · ML4B SoSe 2026 · FAU Erlangen-Nürnberg</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Forschungsfrage</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1.1rem;line-height:1.8;color:#ccc;border-left:3px solid #e8ff4a;
               padding-left:1.2rem;margin-bottom:2rem'>
    Wie genau lassen sich menschliche Bewegungsklassen aus Smartphone-Sensordaten
    mittels Machine Learning klassifizieren, wenn das Gerät in der Hosentasche getragen wird?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Methodik – CRISP-DM Prozess</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explain-box'>
    Das Projekt folgt dem <strong>CRISP-DM Prozessmodell</strong> (Cross-Industry Standard Process
    for Data Mining), einem bewährten iterativen Rahmenwerk für Data-Science-Projekte.
    </div>
    """, unsafe_allow_html=True)

    for num, title, desc in [
        ("01","Business Understanding",
         "Projektziel und Forschungsfrage definiert. StepSense soll Bewegungsklassen aus Smartphone-Sensordaten erkennen – motiviert durch den Einsatz in ressourcenbeschränkten Wearables wie Fitness-Armbändern."),
        ("02","Data Understanding",
         "Sensordaten mit der Sensor Logger App aufgenommen. Drei Sensoren: Accelerometer, Gyroscope, Orientation. Sampling-Raten 61–100 Hz. Klassenverteilung und Signalqualität analysiert."),
        ("03","Data Preparation",
         "Sliding-Window-Verfahren: 2-Sekunden-Fenster mit 50% Überlappung. 8 statistische Features je Signal = 72 Features gesamt. Train/Val/Test-Split 70/15/15% stratifiziert. StandardScaler-Normalisierung."),
        ("04","Modeling",
         "Vergleich von 5 Klassifikatoren (Decision Tree, Random Forest, Gradient Boosting, KNN, SVM) mittels 5-facher Kreuzvalidierung. Gewichtetes F1 als Hauptmetrik. class_weight='balanced' für Minderheitsklassen."),
        ("05","Evaluation",
         "Finale Evaluation auf zurückgehaltenem Test-Set. Konfusionsmatrix, Per-Klasse Precision/Recall/F1. Mixed-Evaluation auf annotierten zusammengesetzten Aktivitätsaufnahmen."),
        ("06","Deployment",
         "Streamlit-App zur interaktiven Demonstration. Nutzer können eigene Sensor-CSV-Dateien hochladen und erhalten Aktivitätsverlauf, Zeitanteile und Signalvisualisierung."),
    ]:
        highlight = num == "06"
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:0.8rem;background:#0d0d0d;
                   border:1px solid {"#e8ff4a" if highlight else "#1a1a1a"};
                   border-radius:4px;padding:1rem 1.2rem'>
            <div style='background:{"#e8ff4a" if highlight else "#1e1e1e"};
                       color:{"#0f0f0f" if highlight else "#e8ff4a"};
                       font-family:monospace;font-weight:600;font-size:1rem;
                       padding:0.4rem 0.7rem;border-radius:3px;
                       min-width:2.8rem;text-align:center;height:fit-content'>{num}</div>
            <div>
                <div style='font-family:monospace;font-size:0.9rem;color:#ddd;margin-bottom:0.3rem'>{title}</div>
                <div style='font-size:0.82rem;color:#777;line-height:1.6'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Bias-Analyse</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explain-box'>
    Bias bezeichnet systematische Verzerrungen in Daten oder Modell die zu ungenauen Vorhersagen führen.
    Eine ehrliche Bias-Analyse ist Teil guter wissenschaftlicher Praxis.
    </div>
    """, unsafe_allow_html=True)

    for bias_name, severity, color, desc in [
        ("Personenbias",         "HOCH",    "#F44336",
         "Aufnahmen von nur 2–3 Personen. Modell lernt individuelle Gangmuster – Generalisierung nicht garantiert."),
        ("Positionsbias",        "HOCH",    "#F44336",
         "Handy immer rechte Hosentasche. Andere Trageweisen (Jackentasche, Hand, Rucksack) nicht abgedeckt."),
        ("Klassenungleichgewicht","MITTEL",  "#FF9800",
         "Gehen/Stehen haben mehr Trainingsdaten als Treppe_hoch/runter. Gegenmaßnahme: class_weight='balanced'."),
        ("Gerätebias",           "MITTEL",  "#FF9800",
         "Zwei Smartphones mit unterschiedlichen Sampling-Raten (61 Hz / 100 Hz) können Features beeinflussen."),
        ("Umgebungsbias",        "NIEDRIG", "#4CAF50",
         "Aufnahmen hauptsächlich in Uni-Gebäuden. Andere Treppen oder Untergründe könnten Erkennung beeinflussen."),
    ]:
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:0.6rem;background:#0d0d0d;
                   border:1px solid #1a1a1a;border-radius:4px;padding:0.8rem 1rem'>
            <div style='min-width:160px'>
                <div style='font-family:monospace;font-size:0.8rem;color:#ddd'>{bias_name}</div>
                <div style='background:{color};color:#0f0f0f;font-size:0.65rem;font-family:monospace;
                           font-weight:600;padding:0.15rem 0.5rem;border-radius:2px;
                           display:inline-block;margin-top:0.3rem'>{severity}</div>
            </div>
            <div style='font-size:0.8rem;color:#777;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Sensorsetup & Team</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div style='background:#0d0d0d;border:1px solid #1a1a1a;border-radius:4px;padding:1rem'>
        <div style='font-family:monospace;font-size:0.7rem;color:#e8ff4a;margin-bottom:0.8rem'>SENSORSETUP</div>
        <div style='font-size:0.82rem;color:#777;line-height:1.9'>
        📱 Smartphone · rechte Hosentasche · kein Festhalten<br>
        📡 Accelerometer · Gyroscope · Orientation<br>
        ⏱ 61 Hz und 100 Hz Abtastrate<br>
        🏃 6 Klassen: Gehen, Laufen, Liegen, Stehen, Treppe hoch/runter<br>
        👥 2–3 Probanden · selbst erhobene Daten
        </div></div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div style='background:#0d0d0d;border:1px solid #1a1a1a;border-radius:4px;padding:1rem'>
        <div style='font-family:monospace;font-size:0.7rem;color:#e8ff4a;margin-bottom:0.8rem'>TEAM & KURS</div>
        <div style='font-size:0.82rem;color:#777;line-height:1.9'>
        👥 Gruppe: <strong style='color:#ccc'>NoNames</strong><br>
        🎓 Kurs: ML4B SoSe 2026<br>
        🏛 FAU Erlangen-Nürnberg<br>
        🎤 Präsentation: ML4B Conference 2026, Schaeffler Nürnberg
        </div></div>""", unsafe_allow_html=True)