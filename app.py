import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.signal import resample as scipy_resample

st.set_page_config(
    page_title="NoNames – Bewegungsklassifikation",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design-System ──────────────────────────────────────────────────────────────
# Palette  (WCAG AA auf weißen Karten und auf Hintergrund):
#   BG App:    #eef4ff  (helles Blau – nicht klinisch weiß)
#   BG Sidebar:#e2edfb  (etwas satter)
#   BG Card:   #ffffff  (weiße Karten heben sich vom BG ab)
#   Accent:    #1d4ed8  (Blau)
#   Warn:      #c2410c  (Orange)
#   Text:      #111827 / #374151 / #6b7280
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ═══════════════════════════════════════════════════════
   HINTERGRUND
   Alle Containerebenen auf #d4e4f7. Der Trick:
   .block-container hat intern white gesetzt → explizit überschreiben.
   ═══════════════════════════════════════════════════════ */
html, body { background-color: #d4e4f7 !important; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
.main { background-color: #d4e4f7 !important; }

/* DAS ist die weiße Karte – NUR Main-Content, nicht Sidebar */
[data-testid="stMain"] .block-container,
[data-testid="stMain"] > div > .block-container,
.main .block-container {
    background-color: #d4e4f7 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding-top: 1.8rem !important;
}

/* Alle internen verschachtelten Blöcke transparent lassen */
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"],
[data-testid="stElementContainer"],
.element-container,
[data-testid="stMarkdownContainer"] {
    background-color: transparent !important;
}

/* ─ Header-Leiste ─ */
[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background-color: #d4e4f7 !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
/* Deploy-Button und Deko ausblenden */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stDeployButton"] { display: none !important; }

/* Sidebar-Toggle ausblenden – Sidebar ist CSS-seitig immer sichtbar */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* ═══════════════════════════════════════════════════════
   SIDEBAR – immer sichtbar, dunkler als Hauptfläche
   transform:none überschreibt Streamlit-JS-Collapse (inline style)
   ═══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #b4cbe8 !important;
    border-right: 1px solid #96b5d8 !important;
    min-width: 260px !important;
    max-width: 320px !important;
    transform: none !important;
    transition: none !important;
    display: block !important;
    visibility: visible !important;
}
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] .block-container {
    background-color: #b4cbe8 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.6rem 1.2rem !important;
}
[data-testid="stSidebar"] h1 { font-size: 1.55rem !important; color: #0f172a !important; }

/* Alle Sidebar-Texte dunkel */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #0f172a !important;
}
/* Radio-Optionen explizit */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] span {
    color: #0f172a !important;
    font-size: 0.95rem !important;
}
[data-testid="stSidebar"] hr { border-color: #96b5d8 !important; }

/* ═══════════════════════════════════════════════════════
   TEXTFARBE – globale Grundregel
   ═══════════════════════════════════════════════════════ */
html, body { font-family: 'IBM Plex Sans', sans-serif !important; color: #111827; }
p, li, ol, ul                              { color: #111827 !important; }
h1, h2, h3, h4, h5, h6                    { color: #111827 !important; font-family: 'IBM Plex Mono', monospace !important; }
label                                      { color: #111827 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em    { color: #111827 !important; }
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span        { color: #374151 !important; }

/* ═══════════════════════════════════════════════════════
   FILE-UPLOADER – in Hintergrundfarbe eingebettet
   ═══════════════════════════════════════════════════════ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div {
    background-color: transparent !important;
}
/* Dropzone: leicht heller als Seite, subtile Linie */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {
    background-color: #e2eef8 !important;
    border: 1.5px dashed #7aacd4 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    padding: 1.2rem !important;
}
/* SVG-Ikone: selbe Farbe wie Rand, nicht grell */
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploader"] section svg {
    color: #4a7fa8 !important;
    fill: #4a7fa8 !important;
    opacity: 0.75;
}
/* Label-Text über der Dropzone */
[data-testid="stFileUploader"] label,
[data-testid="stWidgetLabel"] p {
    color: #1e3a5f !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
/* Hinweistext und Dateiformat-Info */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #3d5f7a !important;
    font-size: 0.85rem !important;
}
/* "Browse files"-Button: weiß mit blauem Rand – klar erkennbar */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzoneButton"],
[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    color: #1d4ed8 !important;
    border: 2px solid #1d4ed8 !important;
    border-radius: 6px !important;
    padding: 0.35rem 1.1rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}
/* Hochgeladene Dateien – Liste */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] {
    color: #111827 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg {
    color: #6b7280 !important;
    fill: #6b7280 !important;
}

/* ═══════════════════════════════════════════════════════
   WEITERE WIDGETS
   ═══════════════════════════════════════════════════════ */

/* Buttons allgemein */
.stButton > button {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 0.45rem 1.2rem !important;
}
.stButton > button:hover { background-color: #1e40af !important; color: #ffffff !important; }

/* Radio */
[data-testid="stRadio"] label,
[data-testid="stRadio"] p { color: #111827 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background-color: #c8dcf0 !important;
    border-color: #5a8fd6 !important;
    border-radius: 7px !important;
    color: #111827 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #c8dcf0 !important;
    border-radius: 8px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #d4e4f7; }
::-webkit-scrollbar-thumb { background: #7aaee8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a85d0; }

/* ═══════════════════════════════════════════════════════
   CUSTOM KOMPONENTEN
   ═══════════════════════════════════════════════════════ */

/* Weiße Karten – heben sich vom blauen BG ab */
.metric-card {
    background: #ffffff;
    border: 1px solid #96b5d8;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(29,78,216,0.10);
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #1d4ed8;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.73rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
}

/* Abschnittsüberschriften */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1d4ed8;
    border-bottom: 2px solid #1d4ed8;
    padding-bottom: 0.45rem;
    margin-bottom: 1.3rem;
    margin-top: 2rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

/* Info-Box blau */
.explain-box {
    background: #ffffff;
    border-left: 4px solid #1d4ed8;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.75;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 4px rgba(29,78,216,0.08);
}

/* Warn-Box orange */
.warn-box {
    background: #ffffff;
    border-left: 4px solid #c2410c;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.75;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 4px rgba(194,65,12,0.08);
}

/* Learning-Box grün */
.learning-box {
    background: #ffffff;
    border-left: 4px solid #15803d;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.75;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 4px rgba(21,128,61,0.07);
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib-Helfer ──────────────────────────────────────────────────────────
def _style_ax(ax, xlabel=None, ylabel=None):
    ax.set_facecolor("#f8faff")
    for sp in ax.spines.values():
        sp.set_edgecolor("#c7d8f5")
    ax.tick_params(colors="#6b7280", labelsize=8)
    ax.grid(color="#e2edfb", linewidth=0.9, linestyle="-")
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color="#374151", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="#374151", fontsize=8, labelpad=6)

def _style_fig(fig):
    fig.patch.set_facecolor("#eef4ff")

def _legend(ax, **kwargs):
    ax.legend(facecolor="#ffffff", edgecolor="#c7d8f5",
              labelcolor="#111827", fontsize=8, **kwargs)

# ── Konstanten ─────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed"
MODEL_PATH     = os.path.join(PROCESSED_PATH, "model.pkl")
SCALER_PATH    = os.path.join(PROCESSED_PATH, "scaler.pkl")
FEATURES_PATH  = os.path.join(PROCESSED_PATH, "feature_names.txt")
METADATA_PATH  = os.path.join(PROCESSED_PATH, "model_metadata.json")
WINDOW_SIZE_S  = 2.0
STEP_SIZE_S    = 1.0
TARGET_HZ      = 61

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
    "Gehen":         "#16a34a",
    "Laufen":        "#1d4ed8",
    "Liegen":        "#7c3aed",
    "Stehen":        "#c2410c",
    "Treppe_hoch":   "#b91c1c",
    "Treppe_runter": "#0e7490",
}

FEATURE_COLS_RAW = [
    "acc_x","acc_y","acc_z",
    "gyro_x","gyro_y","gyro_z",
    "orie_roll","orie_pitch","orie_yaw"
]

# ── Artefakte laden ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

# ── Feature Engineering ────────────────────────────────────────────────────────
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

    acc_cols = ["acc_x", "acc_y", "acc_z"]
    if all(c in window.columns for c in acc_cols):
        mag = np.sqrt(window["acc_x"]**2 + window["acc_y"]**2 + window["acc_z"]**2).dropna()
        if len(mag) >= 5:
            features["acc_mag_mean"]   = mag.mean()
            features["acc_mag_std"]    = mag.std()
            features["acc_mag_energy"] = (mag**2).mean()
            actual_hz = len(mag) / WINDOW_SIZE_S
            fft_vals  = np.abs(np.fft.rfft(mag - mag.mean()))
            freqs     = np.fft.rfftfreq(len(mag), d=1.0 / actual_hz)
            features["acc_mag_dom_freq"] = float(freqs[np.argmax(fft_vals[1:]) + 1]) if len(fft_vals) > 1 else 0.0

    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
    if all(c in window.columns for c in gyro_cols):
        gyro_mag = np.sqrt(window["gyro_x"]**2 + window["gyro_y"]**2 + window["gyro_z"]**2).dropna()
        if len(gyro_mag) >= 5:
            features["gyro_mag_mean"]   = gyro_mag.mean()
            features["gyro_mag_std"]    = gyro_mag.std()
            features["gyro_mag_energy"] = (gyro_mag**2).mean()

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

TRIM_S = 2.0
GAP_S  = 5.0

def resample_to_target_hz(df, target_hz=TARGET_HZ):
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
    sensor_cols = [c for c in df.columns if c != "time_s"]
    df = resample_to_target_hz(df)
    t_min = df["time_s"].min() + TRIM_S
    t_max = df["time_s"].max() - TRIM_S
    df = df[(df["time_s"] >= t_min) & (df["time_s"] <= t_max)].copy()
    if len(df) < 20:
        return None, "Aufnahme zu kurz nach Trim (mindestens ~6 Sekunden nötig)."
    df = df.sort_values("time_s").reset_index(drop=True)
    gaps = df["time_s"].diff()
    max_gap = gaps.max()
    if max_gap > GAP_S:
        gap_idx = gaps[gaps > GAP_S].index[0]
        df = df.iloc[:gap_idx].copy()
        if len(df) < 20:
            return None, f"Aufnahme enthält eine Lücke von {max_gap:.1f}s – zu wenig Daten vor der Lücke."
    df = df.drop_duplicates(subset=["time_s"]).reset_index(drop=True)
    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    return df, None

def classify_dataframe(df, model, feature_cols):
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
    df_feat["predicted"] = model.predict(df_feat[feature_cols])
    _preds = df_feat["predicted"].values.copy()
    for _i in range(len(_preds)):
        _lo = max(0, _i - 1)
        _hi = min(len(_preds), _i + 2)
        _vals, _cnts = np.unique(_preds[_lo:_hi], return_counts=True)
        _preds[_i] = _vals[_cnts.argmax()]
    df_feat["predicted"] = _preds
    return df_feat, None

# ── Mixed-Evaluation ────────────────────────────────────────────────────────
MIXED_DATA_PATH = "data/mixed"
TRIM_S_MIXED    = 2.0

@st.cache_data
def load_mixed_evaluation(_model, _feature_cols):
    """Replicate notebook Section 9 exactly:
    trim 2s from the whole recording (not per segment), slide windows over the
    full recording, assign labels via annotation midpoint lookup, drop unlabeled."""
    if not os.path.exists(MIXED_DATA_PATH):
        return None
    session_dirs = sorted([
        d for d in os.listdir(MIXED_DATA_PATH)
        if os.path.isdir(os.path.join(MIXED_DATA_PATH, d))
    ])

    def _get_label(t_mid, ann):
        for _, row in ann.iterrows():
            if row["start_s"] <= t_mid < row["end_s"]:
                return row["label"]
        return None

    all_rows = []
    for sess_name in session_dirs:
        sess_path = os.path.join(MIXED_DATA_PATH, sess_name)
        ann_path  = os.path.join(sess_path, "Annotation.csv")
        if not os.path.exists(ann_path):
            continue
        ann = pd.read_csv(ann_path)
        dfs_local = {}
        for s_name, prefix, cols in [
            ("Accelerometer", "acc",  ["x", "y", "z"]),
            ("Gyroscope",     "gyro", ["x", "y", "z"]),
            ("Orientation",   "orie", ["roll", "pitch", "yaw"]),
        ]:
            fp = os.path.join(sess_path, f"{s_name}.csv")
            if not os.path.exists(fp):
                continue
            df_s = pd.read_csv(fp)
            df_s["time_s"] = (df_s["time"] - df_s["time"].iloc[0]) / 1e9
            df_s = df_s.drop(columns=["time", "seconds_elapsed"], errors="ignore")
            df_s = df_s.rename(columns={c: f"{prefix}_{c}" for c in cols if c in df_s.columns})
            keep = ["time_s"] + [f"{prefix}_{c}" for c in cols if f"{prefix}_{c}" in df_s.columns]
            dfs_local[s_name] = df_s[keep]
        if "Accelerometer" not in dfs_local:
            continue
        df_m = dfs_local["Accelerometer"].sort_values("time_s")
        for sn in ["Gyroscope", "Orientation"]:
            if sn in dfs_local:
                df_m = pd.merge_asof(
                    df_m, dfs_local[sn].sort_values("time_s"),
                    on="time_s", direction="nearest", tolerance=0.05
                )
        df_m = resample_to_target_hz(df_m)
        df_m = df_m.drop_duplicates(subset=["time_s"]).reset_index(drop=True)
        s_cols = [c for c in df_m.columns if c != "time_s"]
        df_m[s_cols] = df_m[s_cols].ffill().bfill()

        # Trim 2s from the whole recording (identical to notebook Section 9)
        t_min = df_m["time_s"].min() + TRIM_S_MIXED
        t_max = df_m["time_s"].max() - TRIM_S_MIXED

        t = t_min
        while t + WINDOW_SIZE_S <= t_max:
            win = df_m[(df_m["time_s"] >= t) & (df_m["time_s"] < t + WINDOW_SIZE_S)]
            if len(win) >= 10:
                mid   = t + WINDOW_SIZE_S / 2
                label = _get_label(mid, ann)
                if label is not None:  # drop windows between annotation segments
                    feats = compute_features(win)
                    feats["window_mid"] = mid
                    feats["true_label"] = label
                    feats["session"]    = sess_name
                    all_rows.append(feats)
            t += STEP_SIZE_S

    if not all_rows:
        return None
    df_feat = pd.DataFrame(all_rows).fillna(0)
    for col in _feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
    df_feat["predicted"] = _model.predict(df_feat[_feature_cols])
    for sess in df_feat["session"].unique():
        mask  = df_feat["session"] == sess
        preds = df_feat.loc[mask, "predicted"].values.copy()
        for i in range(len(preds)):
            lo = max(0, i - 1)
            hi = min(len(preds), i + 2)
            vals, cnts = np.unique(preds[lo:hi], return_counts=True)
            preds[i] = vals[cnts.argmax()]
        df_feat.loc[mask, "predicted"] = preds
    return df_feat[["session", "window_mid", "true_label", "predicted"]]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# NoNames")
    st.markdown(
        "<p style='font-family:monospace;font-size:0.78rem;color:#6b7280;margin-top:-0.4rem'>"
        "ML4B SoSe 2026 · FAU Erlangen-Nürnberg</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Klassifikation", "Modell-Evaluation", "Über das Projekt"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    model, scaler, feature_cols, metadata = load_artifacts()
    if model is not None:
        st.markdown(
            "<p style='font-family:monospace;font-size:0.68rem;color:#6b7280;"
            "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>"
            "Aktives Modell</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-family:monospace;font-size:0.9rem;color:#111827;"
            f"font-weight:600;margin-bottom:0.2rem'>{metadata.get('model_name','–')}</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-family:monospace;font-size:0.95rem;color:#1d4ed8;font-weight:700'>"
            f"CV F1: {metadata.get('cv_f1_mean','–')} ± {metadata.get('cv_f1_std','–')}</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-family:monospace;font-size:0.8rem;color:#6b7280'>"
            f"{metadata.get('n_features','–')} Features · 3 GroupKFold-Folds</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Kein Modell gefunden. Notebook 02 ausführen.")

    st.markdown("---")
    st.markdown(
        "<p style='font-family:monospace;font-size:0.68rem;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>"
        "Datensatz</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:0.82rem;color:#374151;line-height:1.85'>"
        "47 Sessions · 3 621 Fenster<br>"
        "Smartphone · rechte Hosentasche<br>"
        "Acc · Gyro · Orientation<br>"
        "61 Hz / 100 Hz · 6 Klassen"
        "</p>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – KLASSIFIKATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Klassifikation":
    st.markdown("# Aktivitätsklassifikation")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.95rem;margin-bottom:1.5rem'>"
        "Lade Sensordaten einer Aufnahme hoch. Das Modell klassifiziert die enthaltenen "
        "Bewegungssequenzen fensterweise und gibt den zeitlichen Aktivitätsverlauf aus.</p>",
        unsafe_allow_html=True
    )

    if model is None:
        st.error("Modell nicht gefunden. Bitte zuerst Notebook 02 ausführen.")
        st.stop()

    st.markdown("<div class='section-header'>Aufnahmeanleitung</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col_ui, num, title, content in [
        (col1, "01", "Position",
         "Smartphone in die <strong>rechte Hosentasche</strong> stecken, vertikal ausgerichtet. "
         "Gerät nicht aktiv festhalten."),
        (col2, "02", "Aufnahme",
         "Sensor Logger App öffnen, Aufnahme starten. "
         "Aktivität <strong>mindestens 10 Sekunden</strong> durchführen, App im Vordergrund lassen."),
        (col3, "03", "Upload",
         "Aufnahmeordner öffnen, <strong>Strg+A</strong> alle Dateien auswählen und hochladen. "
         "Die App erkennt automatisch Accelerometer, Gyroscope und Orientation."),
    ]:
        with col_ui:
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #c7d8f5;border-radius:8px;
                        padding:1.1rem;box-shadow:0 1px 3px rgba(29,78,216,0.06)'>
              <div style='font-family:monospace;font-size:0.68rem;color:#1d4ed8;
                          font-weight:700;letter-spacing:0.07em;margin-bottom:0.65rem'>
                {num} · {title.upper()}
              </div>
              <div style='font-size:0.85rem;color:#374151;line-height:1.65'>{content}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Aufnahmeordner hochladen</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explain-box'>
    Wähle <strong>alle Dateien</strong> aus dem Sensor-Logger-Aufnahmeordner (Strg+A im Ordner).
    Die App erkennt automatisch <strong>Accelerometer.csv</strong>, <strong>Gyroscope.csv</strong>
    und <strong>Orientation.csv</strong> und ignoriert alle anderen Dateien.
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Dateien auswählen (Strg+A im Aufnahmeordner)",
        type="csv",
        accept_multiple_files=True,
        help="Öffne den Aufnahmeordner, drücke Strg+A und lade alle Dateien hoch"
    )

    if uploaded_files:
        acc_file  = find_sensor_file(uploaded_files, "Accelerometer")
        gyro_file = find_sensor_file(uploaded_files, "Gyroscope")
        ori_file  = find_sensor_file(uploaded_files, "Orientation")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col_ui, label, found in [
            (c1, "Accelerometer.csv", acc_file is not None),
            (c2, "Gyroscope.csv",     gyro_file is not None),
            (c3, "Orientation.csv",   ori_file is not None),
        ]:
            with col_ui:
                bg   = "#f0fdf4" if found else "#fef2f2"
                bc   = "#16a34a" if found else "#b91c1c"
                tc   = "#15803d" if found else "#b91c1c"
                icon = "✓" if found else "✗"
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {bc};border-radius:6px;
                            padding:0.6rem 1rem;text-align:center'>
                  <span style='color:{tc};font-family:monospace;font-weight:700;font-size:1rem'>{icon}</span>
                  <span style='font-size:0.82rem;color:#374151;margin-left:0.5rem'>{label}</span>
                </div>""", unsafe_allow_html=True)

        ignored = [f.name for f in uploaded_files
                   if f.name.replace(".csv","").strip().lower()
                   not in ["accelerometer","gyroscope","orientation"]]
        if ignored:
            st.markdown(
                f"<p style='font-size:0.72rem;color:#9ca3af;margin-top:0.4rem'>"
                f"Ignoriert: {', '.join(ignored)}</p>",
                unsafe_allow_html=True
            )

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
                    st.warning(f"Vorverarbeitung: {prep_error}")

                if df_result is not None and len(df_result) > 0:
                    top_class = df_result["predicted"].value_counts().idxmax()
                    top_pct   = df_result["predicted"].value_counts(normalize=True).max() * 100
                    n_windows = len(df_result)
                    duration  = df_merged["time_s"].max() - df_merged["time_s"].min()

                    st.markdown("<div class='section-header'>Zusammenfassung</div>", unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    for col_ui, val, label in [
                        (c1, top_class,          "Hauptaktivität"),
                        (c2, f"{top_pct:.0f}%",  "Anteil"),
                        (c3, str(n_windows),     "Fenster"),
                        (c4, f"{duration:.1f}s", "Dauer"),
                    ]:
                        with col_ui:
                            st.markdown(f"""
                            <div class='metric-card'>
                              <div class='metric-value'>{val}</div>
                              <div class='metric-label'>{label}</div>
                            </div>""", unsafe_allow_html=True)

                    # ── Aktivitätsverlauf (Gantt) ────────────────────────────
                    st.markdown(
                        "<div class='section-header'>Aktivitätsverlauf</div>",
                        unsafe_allow_html=True
                    )

                    times_arr = df_result["window_mid"].values
                    preds_arr = df_result["predicted"].values
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

                    fig, ax = plt.subplots(figsize=(14, 2.0))
                    _style_fig(fig)
                    ax.set_facecolor("#f8faff")
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#c7d8f5")

                    for cls, t_s, t_e in phases:
                        color = CLASS_COLORS.get(cls, "#6b7280")
                        dur   = t_e - t_s
                        ax.barh(0, dur, left=t_s, height=0.55,
                                color=color, edgecolor="#eef4ff", linewidth=0.6)
                        if dur >= 2.5:
                            ax.text(t_s + dur / 2, 0,
                                    f"{cls}\n{dur:.0f}s",
                                    ha="center", va="center",
                                    fontsize=7, color="white",
                                    fontfamily="monospace", fontweight="600")

                    ax.set_xlim(times_arr[0], times_arr[-1] + STEP_SIZE_S)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks([])
                    _style_ax(ax, xlabel="Zeit (s)")
                    ax.grid(axis="x")
                    ax.grid(axis="y", visible=False)
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig)
                    plt.close()

                    # ── Treppenlinie ────────────────────────────────────────
                    all_classes  = sorted(CLASS_COLORS.keys())
                    class_to_int = {c: i for i, c in enumerate(all_classes)}
                    y_vals       = df_result["predicted"].map(class_to_int)
                    times        = df_result["window_mid"].values
                    predictions  = df_result["predicted"].values

                    fig, ax = plt.subplots(figsize=(14, 4.5))
                    _style_fig(fig)
                    _style_ax(ax, xlabel="Zeit (s)")

                    prev_cls, phase_start = None, None
                    for t, cls in zip(times, predictions):
                        if cls != prev_cls:
                            if prev_cls is not None:
                                ax.axvspan(phase_start, t,
                                           color=CLASS_COLORS.get(prev_cls, "#aaa"),
                                           alpha=0.1, linewidth=0)
                            phase_start = t
                            prev_cls    = cls
                    if prev_cls is not None:
                        ax.axvspan(phase_start, times[-1],
                                   color=CLASS_COLORS.get(prev_cls, "#aaa"),
                                   alpha=0.1, linewidth=0)

                    ax.step(times, y_vals, color="#1d4ed8", linewidth=2.0, where="post", zorder=3)
                    point_colors = [CLASS_COLORS.get(c, "#6b7280") for c in predictions]
                    ax.scatter(times, y_vals, c=point_colors, s=35, zorder=4, linewidths=0)

                    ax.set_yticks(range(len(all_classes)))
                    ax.set_yticklabels(all_classes, color="#111827", fontsize=9)
                    ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
                    ax.set_ylim(-0.5, len(all_classes) - 0.5)

                    unique_cls = df_result["predicted"].unique()
                    patches = [mpatches.Patch(color=CLASS_COLORS.get(c,"#aaa"), label=c)
                               for c in sorted(unique_cls)]
                    _legend(ax, handles=patches, loc="upper right")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # ── Zeitanteile ─────────────────────────────────────────
                    st.markdown(
                        "<div class='section-header'>Zeitanteile je Aktivität</div>",
                        unsafe_allow_html=True
                    )
                    counts = df_result["predicted"].value_counts().sort_values(ascending=False)
                    for cls, cnt in counts.items():
                        pct   = cnt / len(df_result) * 100
                        secs  = cnt * STEP_SIZE_S
                        color = CLASS_COLORS.get(cls, "#6b7280")
                        st.markdown(f"""
                        <div style='display:flex;align-items:center;margin-bottom:0.45rem;gap:0.8rem'>
                          <div style='min-width:130px;font-family:monospace;font-size:0.82rem;
                                      color:#111827;font-weight:600'>{cls}</div>
                          <div style='flex:1;background:#e2edfb;border-radius:4px;
                                      height:16px;overflow:hidden'>
                            <div style='width:{int(pct)}%;background:{color};
                                        height:100%;border-radius:4px'></div>
                          </div>
                          <div style='min-width:90px;font-family:monospace;font-size:0.8rem;
                                      color:#1d4ed8;font-weight:700;text-align:right'>
                            {secs:.0f}s · {pct:.0f}%
                          </div>
                        </div>""", unsafe_allow_html=True)

                    # ── Rohdaten ────────────────────────────────────────────
                    st.markdown(
                        "<div class='section-header'>Rohdaten der Sensoren</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                    <div class='explain-box'>
                    <strong>Accelerometer</strong> (m/s²) misst lineare Beschleunigung.
                    <strong>Gyroscope</strong> (rad/s) misst Rotationsrate.
                    <strong>Orientation</strong> zeigt Roll, Pitch und Yaw des Geräts in Radiant.
                    </div>""", unsafe_allow_html=True)

                    fig2, axes = plt.subplots(3, 1, figsize=(14, 6.5), sharex=True)
                    _style_fig(fig2)
                    palette = ["#1d4ed8", "#b91c1c", "#15803d"]
                    sensor_groups = [
                        (["acc_x","acc_y","acc_z"],           "Acc (m/s²)"),
                        (["gyro_x","gyro_y","gyro_z"],         "Gyro (rad/s)"),
                        (["orie_roll","orie_pitch","orie_yaw"], "Orientation (rad)"),
                    ]
                    for ax2, (cols, ylabel) in zip(axes, sensor_groups):
                        _style_ax(ax2, ylabel=ylabel)
                        for col, color in zip(cols, palette):
                            if col in df_merged.columns:
                                ax2.plot(df_merged["time_s"], df_merged[col],
                                         color=color, linewidth=0.9, alpha=0.9,
                                         label=col.split("_")[-1])
                        _legend(ax2, loc="upper right", fontsize=7)
                    _style_ax(axes[-1], xlabel="Zeit (s)")
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
    st.markdown(
        "<p style='color:#6b7280;font-size:0.95rem;margin-bottom:1.2rem'>"
        "Primärmetrik: Mixed-Evaluation auf zusammengesetzten Aufnahmen mit Aktivitätswechseln. "
        "Ergänzend: Cross-Validation (k=3) und Test-Set als Kontrollmetriken.</p>",
        unsafe_allow_html=True
    )

    if model is None:
        st.error("Modell nicht gefunden. Bitte zuerst Notebook 02 ausführen.")
        st.stop()

    # ── MIXED EVALUATION – PRIMÄR ────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Mixed-Evaluation · Primärmetrik</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class='explain-box'>
    <strong>Was ist die Mixed-Evaluation?</strong> Das Modell wird auf Aufnahmen getestet,
    die mehrere Aktivitäten in einem kontinuierlichen Ablauf enthalten – mit echten Übergängen
    zwischen Klassen. Diese Sessions wurden <em>nicht</em> beim Training oder der CV verwendet
    und repräsentieren reale Nutzungsbedingungen.<br><br>
    <strong>Warum Primärmetrik?</strong> Das Test-Set besteht aus isolierten Einzelklassen-Sessions,
    die strukturell dem Trainingsformat entsprechen (optimistisch). Die Mixed-Evaluation stellt
    eine härtere, realistischere Anforderung dar.
    CV F1 (0.88) ≈ Mixed F1 (0.87) → <strong>kein Overfitting</strong>.
    </div>""", unsafe_allow_html=True)

    df_mixed = load_mixed_evaluation(model, feature_cols)

    if df_mixed is not None and len(df_mixed) > 0:
        mixed_acc  = accuracy_score(df_mixed["true_label"], df_mixed["predicted"])
        mixed_f1   = f1_score(df_mixed["true_label"], df_mixed["predicted"],
                              average="weighted", zero_division=0)
        n_win_mix  = len(df_mixed)
        n_sess_mix = df_mixed["session"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        for col_ui, val, label in [
            (c1, f"{mixed_f1:.3f}",  "Mixed F1 (gewichtet)"),
            (c2, f"{mixed_acc:.3f}", "Mixed Accuracy"),
            (c3, str(n_win_mix),     "Annotierte Fenster"),
            (c4, str(n_sess_mix),    "Mixed-Sessions"),
        ]:
            with col_ui:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value'>{val}</div>
                  <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Konfusionsmatrizen (absolut + normalisiert)
        all_labels_mix = sorted(df_mixed["true_label"].unique())
        col_l, col_r = st.columns(2)
        for col_ui, normalize, title, fmt in [
            (col_l, None,   "Absolut – Fensterzahlen",         "d"),
            (col_r, "true", "Normalisiert – Recall je Klasse", ".2f"),
        ]:
            cm_m = confusion_matrix(df_mixed["true_label"], df_mixed["predicted"],
                                    labels=all_labels_mix, normalize=normalize)
            vmax = 1.0 if normalize else None
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig)
            sns.heatmap(cm_m, annot=True, fmt=fmt, cmap="Blues",
                        xticklabels=all_labels_mix, yticklabels=all_labels_mix,
                        ax=ax, linewidths=0.5, linecolor="#c7d8f5",
                        cbar_kws={"shrink": 0.8}, vmin=0, vmax=vmax)
            ax.set_title(title, color="#111827", fontsize=9, fontweight="600", pad=10)
            ax.set_ylabel("Tatsächliche Klasse", color="#374151", fontsize=8)
            ax.set_xlabel("Vorhergesagte Klasse", color="#374151", fontsize=8)
            ax.tick_params(colors="#374151", labelsize=7, rotation=45)
            ax.set_facecolor("#f8faff")
            with col_ui:
                st.pyplot(fig)
            plt.close()

        # Per-Session Aufschlüsselung
        st.markdown(
            "<div class='section-header'>Mixed-Evaluation · Aufschlüsselung je Session</div>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class='explain-box'>
        Jede Mixed-Session enthält andere Aktivitätskombinationen. Die Tabelle zeigt,
        wie gut das Modell auf jeder einzelnen zusammengesetzten Aufnahme abschneidet.
        </div>""", unsafe_allow_html=True)

        sess_rows = []
        for sess in sorted(df_mixed["session"].unique()):
            m  = df_mixed["session"] == sess
            yt = df_mixed.loc[m, "true_label"]
            yp = df_mixed.loc[m, "predicted"]
            sess_rows.append({
                "Session":   sess,
                "Fenster":   int(m.sum()),
                "Accuracy":  round(accuracy_score(yt, yp), 3),
                "F1 (gew.)": round(f1_score(yt, yp, average="weighted", zero_division=0), 3),
                "Klassen":   ", ".join(sorted(yt.unique())),
            })
        df_sess = pd.DataFrame(sess_rows)
        st.dataframe(
            df_sess.set_index("Session").style.background_gradient(
                cmap="RdYlGn", vmin=0.5, vmax=1.0,
                subset=["Accuracy", "F1 (gew.)"]
            ),
            use_container_width=True
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        _style_fig(fig)
        sessions_list = df_sess["Session"].values
        x = np.arange(len(sessions_list))
        for ax_i, metric, color, overall in [
            (axes[0], "Accuracy",  "#4c7fd4", mixed_acc),
            (axes[1], "F1 (gew.)", "#e07b39", mixed_f1),
        ]:
            _style_ax(ax_i, ylabel="Score")
            ax_i.bar(x, df_sess[metric].values, color=color, alpha=0.85, edgecolor="none")
            ax_i.axhline(overall, color="#1d4ed8", linestyle="--", linewidth=1.3, label="Gesamt")
            ax_i.set_xticks(x)
            ax_i.set_xticklabels(sessions_list, fontsize=7.5, rotation=30, ha="right", color="#374151")
            ax_i.set_ylim(0, 1.1)
            ax_i.set_title(f"{metric} je Session", fontsize=9, color="#111827", fontweight="600")
            _legend(ax_i)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info(
            "Mixed-Session-Daten nicht gefunden unter `data/mixed/`. "
            "Annotation.csv und Sensordateien je Session prüfen."
        )

    # ── CROSS-VALIDATION – SEKUNDÄR ──────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Cross-Validation · k=3 GroupKFold</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class='explain-box'>
    3-fache session-basierte CV: Jede Session erscheint ausschließlich in einem Fold –
    kein Datenleck durch überlappende Fenster. CV F1 = 0.88 ≈ Mixed F1 = 0.87 bestätigt:
    das Modell generalisiert, nicht nur memoriert.
    </div>""", unsafe_allow_html=True)

    if metadata:
        c1, c2, c3 = st.columns(3)
        for col_ui, val, label in [
            (c1, f"{metadata.get('cv_f1_mean','–')} ± {metadata.get('cv_f1_std','–')}",
                 "CV F1 – 3-fold GroupKFold"),
            (c2, metadata.get("model_name", "–"), "Modell"),
            (c3, metadata.get("n_features", "–"),  "Features"),
        ]:
            with col_ui:
                fs = "1.25rem" if len(str(val)) > 14 else "2rem"
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value' style='font-size:{fs}'>{val}</div>
                  <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

    # ── TEST-SET – KONTROLLE ─────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Test-Set · Kontrollmetrik (6 Einzelklassen-Sessions)</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class='warn-box'>
    <strong>Einschränkung:</strong> Das Test-Set besteht aus 6 isolierten Einzelklassen-Sessions
    (je 1 pro Klasse, ~455 Fenster). Jede Session enthält nur <em>eine</em> Aktivität –
    strukturell ähnlich zu den Trainingsdaten. Der hohe Test-F1 (0.99) ist deshalb
    <strong>kein geeigneter Indikator für die Praxistauglichkeit</strong>.
    Die Mixed-Evaluation (F1 = 0.87) ist die realistischere Kennzahl.
    </div>""", unsafe_allow_html=True)

    features_all_path  = os.path.join(PROCESSED_PATH, "features_all.csv")
    session_split_path = os.path.join(PROCESSED_PATH, "session_split.json")

    can_eval = (os.path.exists(features_all_path) and os.path.exists(session_split_path))

    if can_eval:
        with open(session_split_path) as _f:
            _sp = json.load(_f)
        _df_all = pd.read_csv(features_all_path)
        _mask   = _df_all["session"].isin(_sp["test"])
        X_test  = _df_all.loc[_mask, feature_cols].reset_index(drop=True)
        y_test  = _df_all.loc[_mask, "label"].reset_index(drop=True)
        y_pred  = model.predict(X_test)
        present_classes = sorted(y_test.unique())

        col_l, col_r = st.columns(2)
        for col_ui, normalize, title in [
            (col_l, None,   "Absolut – Fensterzahlen"),
            (col_r, "true", "Normalisiert – Recall je Klasse"),
        ]:
            cm  = confusion_matrix(y_test, y_pred, labels=present_classes, normalize=normalize)
            fmt = ".2f" if normalize else "d"
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig)
            sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                        xticklabels=present_classes, yticklabels=present_classes,
                        ax=ax, linewidths=0.5, linecolor="#c7d8f5",
                        cbar_kws={"shrink": 0.8})
            ax.set_title(title, color="#111827", fontsize=9, fontweight="600", pad=10)
            ax.set_ylabel("Tatsächliche Klasse", color="#374151", fontsize=8)
            ax.set_xlabel("Vorhergesagte Klasse", color="#374151", fontsize=8)
            ax.tick_params(colors="#374151", labelsize=7, rotation=45)
            ax.set_facecolor("#f8faff")
            with col_ui:
                st.pyplot(fig)
            plt.close()

        # Per-Klasse Metriken
        st.markdown(
            "<div class='section-header'>Per-Klasse Metriken · Test-Set (Kontrolle)</div>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class='explain-box'>
        <strong>Precision</strong> – Anteil korrekt vorhergesagter Fenster je Klasse.<br>
        <strong>Recall</strong> – Anteil erkannter echter Fenster je Klasse.<br>
        <strong>Support</strong> – Anzahl Testfenster; kleine Werte → weniger verlässliche Schätzung.
        Hinweis: Diese Werte basieren auf isolierten Sessions ohne Aktivitätswechsel.
        </div>""", unsafe_allow_html=True)

        report = classification_report(y_test, y_pred, labels=present_classes,
                                       output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(report).T.loc[
            present_classes, ["precision", "recall", "f1-score", "support"]
        ].round(3)

        fig, ax = plt.subplots(figsize=(9, 3.2))
        _style_fig(fig)
        _style_ax(ax, ylabel="Score")
        x = np.arange(len(present_classes))
        w = 0.25
        ax.bar(x - w, df_rep["precision"], w, label="Precision", color="#1d4ed8", alpha=0.9)
        ax.bar(x,     df_rep["recall"],    w, label="Recall",    color="#15803d", alpha=0.9)
        ax.bar(x + w, df_rep["f1-score"],  w, label="F1",        color="#b91c1c", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(present_classes, color="#111827", fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.axhline(0.8, color="#9ca3af", linestyle="--", linewidth=1.0, label="0.8 Referenz")
        _legend(ax)
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            df_rep.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1,
                                             subset=["precision", "recall", "f1-score"]),
            use_container_width=True
        )

        # Feature Importance
        _clf_step = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
        if hasattr(_clf_step, "feature_importances_"):
            st.markdown(
                "<div class='section-header'>Feature Importance – Top 15</div>",
                unsafe_allow_html=True
            )
            st.markdown("""
            <div class='explain-box'>
            Schema: <code>sensor_achse_statistik</code> – z.B. <code>acc_z_std</code> =
            Standardabweichung der Z-Achse des Accelerometers.
            Die Dominanz von <code>orie_pitch</code>-Features erklärt den Positionsbias:
            Das Modell nutzt primär die Geräteausrichtung, nicht die Bewegungsdynamik selbst.
            </div>""", unsafe_allow_html=True)

            fi = pd.DataFrame({
                "Feature":    feature_cols,
                "Importance": _clf_step.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(9, 4.5))
            _style_fig(fig)
            _style_ax(ax, xlabel="Importance")
            bar_colors = ["#1d4ed8" if i < 3 else "#6ea8fe" if i < 8 else "#c7d8f5"
                          for i in range(len(fi))]
            fi.sort_values("Importance").plot(
                kind="barh", x="Feature", y="Importance",
                ax=ax, color=bar_colors[::-1], edgecolor="none", legend=False
            )
            ax.tick_params(colors="#374151", labelsize=8)
            st.pyplot(fig)
            plt.close()

    else:
        st.info("Test-Set nicht gefunden. Bitte Notebook 01 & 02 ausführen.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ÜBER DAS PROJEKT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Über das Projekt":
    st.markdown("# NoNames")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.95rem;margin-bottom:1.8rem'>"
        "Bewegungsklassifikation aus Smartphone-Sensordaten · ML4B SoSe 2026 · FAU Erlangen-Nürnberg</p>",
        unsafe_allow_html=True
    )

    # ── Forschungsfrage ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Forschungsfrage</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1.05rem;line-height:1.9;color:#111827;font-weight:500;
                border-left:4px solid #1d4ed8;padding-left:1.2rem;
                background:#ffffff;padding:1rem 1.2rem;border-radius:0 8px 8px 0;
                margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(29,78,216,0.06)'>
    Wie genau lassen sich menschliche Bewegungsklassen aus Smartphone-Sensordaten
    mittels Machine Learning klassifizieren, wenn das Gerät in der Hosentasche getragen wird?
    Kann ein auf wenigen Probanden trainiertes Modell auf unbekannte Sessions generalisieren?
    </div>""", unsafe_allow_html=True)

    # ── Datensatz ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Datensatz</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explain-box'>
    Selbst erhobene Sensordaten mit der <strong>Sensor Logger App</strong>. Gerät in der
    rechten Hosentasche, vertikal ausgerichtet, kein aktives Festhalten.
    Drei Sensoren: Accelerometer (x,y,z), Gyroscope (x,y,z), Orientation (roll, pitch, yaw).
    Sampling-Raten 61 Hz (Gerät A) und 100 Hz (Gerät B) – im Preprocessing auf 61 Hz normalisiert.
    </div>""", unsafe_allow_html=True)

    dataset_rows = [
        ("Gehen",         9,  "1 006", "~115s"),
        ("Laufen",        7,    "838", "~101s"),
        ("Liegen",        8,    "828",  "~88s"),
        ("Stehen",        7,    "400",  "~61s"),
        ("Treppe hoch",   7,    "243",  "~38s"),
        ("Treppe runter", 9,    "306",  "~36s"),
    ]
    header = "<div style='display:grid;grid-template-columns:1fr 60px 80px 70px;" \
             "gap:0;background:#1d4ed8;border-radius:8px 8px 0 0;padding:0.55rem 1rem'>" \
             "<span style='font-family:monospace;font-size:0.75rem;color:#fff;font-weight:600'>KLASSE</span>" \
             "<span style='font-family:monospace;font-size:0.75rem;color:#fff;font-weight:600;text-align:right'>SESS.</span>" \
             "<span style='font-family:monospace;font-size:0.75rem;color:#fff;font-weight:600;text-align:right'>FENSTER</span>" \
             "<span style='font-family:monospace;font-size:0.75rem;color:#fff;font-weight:600;text-align:right'>Ø DAUER</span>" \
             "</div>"
    rows_html = ""
    for i, (cls, sess, fen, dur) in enumerate(dataset_rows):
        bg = "#ffffff" if i % 2 == 0 else "#f5f8ff"
        dot_color = CLASS_COLORS.get(cls.replace(" ","_"), CLASS_COLORS.get(cls.split()[0], "#6b7280"))
        rows_html += f"<div style='display:grid;grid-template-columns:1fr 60px 80px 70px;" \
                     f"background:{bg};padding:0.5rem 1rem;border-bottom:1px solid #e2edfb'>" \
                     f"<span style='font-size:0.88rem;color:#111827;display:flex;align-items:center;gap:0.5rem'>" \
                     f"<span style='width:8px;height:8px;border-radius:50%;background:{dot_color};display:inline-block'></span>{cls}</span>" \
                     f"<span style='font-family:monospace;font-size:0.85rem;color:#374151;text-align:right'>{sess}</span>" \
                     f"<span style='font-family:monospace;font-size:0.85rem;color:#374151;text-align:right'>{fen}</span>" \
                     f"<span style='font-family:monospace;font-size:0.85rem;color:#374151;text-align:right'>{dur}</span>" \
                     f"</div>"
    total = "<div style='display:grid;grid-template-columns:1fr 60px 80px 70px;" \
            "background:#e2edfb;border-radius:0 0 8px 8px;padding:0.55rem 1rem'>" \
            "<span style='font-family:monospace;font-size:0.85rem;color:#111827;font-weight:700'>GESAMT</span>" \
            "<span style='font-family:monospace;font-size:0.85rem;color:#1d4ed8;font-weight:700;text-align:right'>47</span>" \
            "<span style='font-family:monospace;font-size:0.85rem;color:#1d4ed8;font-weight:700;text-align:right'>3 621</span>" \
            "<span style='font-family:monospace;font-size:0.85rem;color:#6b7280;text-align:right'>–</span>" \
            "</div>"
    st.markdown(header + rows_html + total, unsafe_allow_html=True)

    # ── Methodik – CRISP-DM ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Methodik – CRISP-DM</div>", unsafe_allow_html=True)
    for num, title, desc in [
        ("01", "Business Understanding",
         "Forschungsfrage und Zielsystem definiert: Klassifikation von 6 Bewegungsklassen "
         "aus einem einzelnen Smartphone ohne Cloud-Anbindung – motiviert durch Anwendungen "
         "in ressourcenbeschränkten Wearables (Fitness, Medizin, Sportanalytik)."),
        ("02", "Data Understanding",
         "47 Sessions von 2–3 Personen mit der Sensor Logger App erhoben. "
         "Klassenverteilung, Signalqualität und unterbrochene Aufnahmen analysiert. "
         "Sampling-Raten 61 Hz (Gerät A) und 100 Hz (Gerät B) identifiziert."),
        ("03", "Data Preparation",
         "Preprocessing-Pipeline: Sensor-Merge (merge_asof, 50ms Toleranz) → Resampling auf 61 Hz "
         "→ Trim (erste/letzte 2s) → Gap-Check (>5s) → Deduplizierung → ffill/bfill. "
         "Sliding Window 2s / 50% Overlap → 81 Features (Statistiken, Magnitude, FFT, Orientierungs-Delta). "
         "Session-stratifizierter Split: Train 62% / Val 19% / Test 20%."),
        ("04", "Modeling",
         "8 Klassifikatoren verglichen (Decision Tree, Random Forest, Extra Trees, SVM, "
         "Gradient Boosting, HistGB, KNN, Voting Ensemble) via 3-facher session-basierter "
         "GroupKFold-CV. class_weight='balanced' gegen Klassenungleichgewicht. "
         "RandomizedSearchCV (n_iter=30) für das beste Modell."),
        ("05", "Evaluation",
         "Gradient Boosting erzielte CV F1 = 0.84 ± 0.05 auf session-separierten Folds – "
         "solide für 2–3 Probanden und rein statistische Features ohne Deep Learning. "
         "Ruheaktivitäten (Liegen, Stehen) und Laufen werden zuverlässig erkannt; "
         "Gehen vs. Treppe_runter bleibt die schwerste Verwechslung."),
        ("06", "Deployment",
         "Streamlit-App mit identischer Preprocessing-Pipeline wie das Training "
         "(Trim, Gap-Filter, Resampling, Windowing, Feature-Extraktion, Rolling Majority Vote). "
         "Drei Seiten: Klassifikation (Upload & Ergebnis), Evaluation (Metriken), Projekt."),
    ]:
        highlight = (num == "06")
        bg     = "#eff6ff" if highlight else "#ffffff"
        border = "#1d4ed8" if highlight else "#c7d8f5"
        badge_bg = "#1d4ed8" if highlight else "#e2edfb"
        badge_tc = "#ffffff" if highlight else "#1d4ed8"
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:0.65rem;background:{bg};
                    border:1px solid {border};border-radius:8px;padding:0.9rem 1.1rem;
                    box-shadow:0 1px 3px rgba(29,78,216,0.04)'>
          <div style='background:{badge_bg};color:{badge_tc};font-family:monospace;
                      font-weight:700;font-size:0.88rem;padding:0.3rem 0.6rem;
                      border-radius:5px;min-width:2.5rem;text-align:center;
                      height:fit-content;flex-shrink:0'>{num}</div>
          <div>
            <div style='font-family:monospace;font-size:0.88rem;color:#111827;
                        font-weight:700;margin-bottom:0.28rem'>{title}</div>
            <div style='font-size:0.85rem;color:#374151;line-height:1.65'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Key Learnings ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Key Learnings</div>", unsafe_allow_html=True)
    for icon, title, desc in [
        ("🔍", "Data Leakage entdeckt und behoben",
         "Ein zufälliger Fensterschnitt bei 50 % Fenster-Überlappung erzeugt Data Leakage – "
         "benachbarte Fenster teilen Rohdaten. Das führte initial zu CV F1 ≈ 0.99 und nicht validen Metriken. "
         "Die Lösung: session-basierter GroupShuffleSplit + GroupKFold, der ganze Sessions exklusiv "
         "einem Split zuordnet. Nach der Umstellung CV F1 = 0.84 – realistisch und interpretierbar."),
        ("👤", "Personenbias quantifiziert",
         "Das Modell lernt individuelle Gangmuster (Schrittlänge, Körperhaltung) statt genereller "
         "Bewegungscharakteristika. Innerhalb derselben Person: hohe Genauigkeit. "
         "Zwischen Personen: der session-basierte CV F1 = 0.84 zeigt die tatsächliche Generalisierung. "
         "Abhilfe: deutlich mehr Probanden (Referenz: MotionSense Dataset mit 24 Personen)."),
        ("📐", "Feature Engineering als Schlüssel",
         "Erst nach Hinzufügen der FFT-Dominanzfrequenz (erfasst Schrittfrequenz 1–2 Hz Gehen vs. "
         "2–3 Hz Laufen) und der Orientierungs-Delta-Features (Treppenneigung) wurde eine stabile "
         "Unterscheidung aller 6 Klassen möglich. Feature Importance bestätigt: orie_pitch_energy "
         "ist das wichtigste Feature – was gleichzeitig den Positionsbias erklärt."),
        ("📡", "Sampling-Rate-Normalisierung notwendig",
         "Zwei Geräte mit 61 Hz und 100 Hz erzeugen ohne Normalisierung systematisch unterschiedliche "
         "ZCR- und FFT-Werte für dieselbe Aktivität. Resampling aller Sessions auf 61 Hz vor dem "
         "Windowing behebt dies. Die FFT-Dominanzfrequenz wird zusätzlich dynamisch aus der "
         "tatsächlichen Fensteranzahl berechnet."),
    ]:
        st.markdown(f"""
        <div class='learning-box'>
          <strong>{icon} {title}</strong><br>
          {desc}
        </div>""", unsafe_allow_html=True)

    # ── Limitierungen ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Limitierungen</div>", unsafe_allow_html=True)
    BADGE = {
        "Hoch":    ("#b91c1c", "#ffffff"),
        "Mittel":  ("#c2410c", "#ffffff"),
        "Gering":  ("#15803d", "#ffffff"),
        "Behoben": ("#15803d", "#ffffff"),
    }
    for bias_name, severity, desc in [
        ("Personenbias", "Hoch",
         "Aufnahmen von 2–3 Personen. Für eine robuste Generalisierung wäre ein Datensatz "
         "mit >10 Probanden notwendig (Referenz: UCI HAR 30 Personen, MotionSense 24 Personen)."),
        ("Positionsbias", "Hoch",
         "Gerät ausschließlich in der rechten Hosentasche, vertikal. Die Feature Importance zeigt "
         "orie_pitch als dominantes Merkmal – stark positionsabhängig. Andere Trageweisen würden abweichen."),
        ("Kleines Test-Set", "Mittel",
         "Nur 6 Test-Sessions (je 1 pro Klasse, 455 Fenster gesamt). "
         "Die Per-Klasse-Metriken sind entsprechend mit Vorsicht zu interpretieren."),
        ("Klassenungleichgewicht", "Mittel",
         "Treppe_hoch (243 Fenster) und Treppe_runter (306 Fenster) haben deutlich weniger Daten "
         "als Gehen (1006) und Laufen (838). Abgemildert durch class_weight='balanced'."),
        ("Data Leakage", "Behoben",
         "Zufälliger Fensterschnitt bei 50 % Overlap erzeugte Leakage (CV F1 ≈ 0.99). "
         "Behoben durch session-basierten GroupShuffleSplit + GroupKFold."),
    ]:
        bbg, btc = BADGE.get(severity, ("#6b7280", "#ffffff"))
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-bottom:0.55rem;background:#ffffff;
                    border:1px solid #c7d8f5;border-radius:8px;padding:0.8rem 1rem;
                    box-shadow:0 1px 3px rgba(29,78,216,0.04)'>
          <div style='min-width:155px;flex-shrink:0'>
            <div style='font-family:monospace;font-size:0.85rem;color:#111827;
                        font-weight:700'>{bias_name}</div>
            <div style='background:{bbg};color:{btc};font-size:0.66rem;font-family:monospace;
                        font-weight:700;padding:0.12rem 0.4rem;border-radius:3px;
                        display:inline-block;margin-top:0.28rem;letter-spacing:0.04em'>
              {severity.upper()}
            </div>
          </div>
          <div style='font-size:0.85rem;color:#374151;line-height:1.65'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Verwandte Arbeiten ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Verwandte Arbeiten</div>", unsafe_allow_html=True)
    for authors, title, relevance in [
        ("Anguita et al. (2013)", "UCI HAR Dataset – Smartphone Hosentasche, Acc+Gyro, 6 Klassen, SVM, 30 Probanden",
         "Identisches Sensorsetup und Aktivitätsklassen. Direkter Benchmark-Vergleich."),
        ("Bayat et al. (2014)", "HAR mit Smartphone-Accelerometer – Ensemble, ~91 % Accuracy",
         "Bestätigt statistische Features aus Accelerometerdaten. Dieses Projekt ergänzt Gyro und Orientation."),
        ("Balli et al. (2019)", "PCA + Random Forest auf Smartwatch-Daten – 2-Sekunden-Fenster",
         "Identische Fenstergröße, Random Forest als bestes Modell – bestätigt unsere Modellwahl."),
        ("Malekzadeh et al. (2019)", "MotionSense Dataset – iPhone Hosentasche, 24 Probanden, 6 Aktivitäten, 50 Hz",
         "Nahezu identisches Protokoll, aber mit 24 Probanden. Verdeutlicht die Limitierung durch wenige Personen."),
    ]:
        st.markdown(f"""
        <div style='background:#ffffff;border:1px solid #c7d8f5;border-radius:8px;
                    padding:0.85rem 1.1rem;margin-bottom:0.55rem;
                    box-shadow:0 1px 3px rgba(29,78,216,0.04)'>
          <div style='font-family:monospace;font-size:0.82rem;color:#1d4ed8;
                      font-weight:700;margin-bottom:0.25rem'>{authors}</div>
          <div style='font-size:0.88rem;color:#111827;margin-bottom:0.2rem'>{title}</div>
          <div style='font-size:0.82rem;color:#6b7280;line-height:1.55'>{relevance}</div>
        </div>""", unsafe_allow_html=True)

    # ── Sensorsetup & Team ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Sensorsetup & Team</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #c7d8f5;border-radius:8px;padding:1.2rem;
                    box-shadow:0 1px 3px rgba(29,78,216,0.06)'>
          <div style='font-family:monospace;font-size:0.68rem;color:#1d4ed8;font-weight:700;
                      letter-spacing:0.1em;margin-bottom:0.8rem'>SENSORSETUP</div>
          <div style='font-size:0.88rem;color:#374151;line-height:2.0'>
            Smartphone · rechte Hosentasche · kein Festhalten<br>
            Accelerometer · Gyroscope · Orientation<br>
            61 Hz und 100 Hz · Resampling auf 61 Hz<br>
            6 Klassen · 2–3 Probanden · 47 Sessions · 3 621 Fenster
          </div>
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #c7d8f5;border-radius:8px;padding:1.2rem;
                    box-shadow:0 1px 3px rgba(29,78,216,0.06)'>
          <div style='font-family:monospace;font-size:0.68rem;color:#1d4ed8;font-weight:700;
                      letter-spacing:0.1em;margin-bottom:0.8rem'>TEAM & KURS</div>
          <div style='font-size:0.88rem;color:#374151;line-height:2.0'>
            Gruppe: <strong style='color:#111827'>NoNames</strong><br>
            Kurs: Machine Learning for Business (ML4B) · SoSe 2026<br>
            Hochschule: FAU Erlangen-Nürnberg<br>
            Präsentation: ML4B Conference 2026, Schaeffler Nürnberg
          </div>
        </div>""", unsafe_allow_html=True)
