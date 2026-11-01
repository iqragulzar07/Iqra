# ============================================================
# AUTOMATED THREAT PRIORITIZATION — SINGLE FILE APPLICATION
# ============================================================
# This file combines:
#   1. Package installation/checking
#   2. API-key loading
#   3. Dataset loading and cleaning
#   4. Threat classification
#   5. Feature engineering
#   6. Random Forest
#   7. Neural Network
#   8. LSTM
#   9. K-Means
#  10. Model saving
#  11. Streamlit website
#
# VS Code folder expected:
# D:\Desktop\final thesis\
#
# Raw datasets expected:
#   AWS_Honeypot_marx-geo.csv
#   london.csv
#   singapore.csv
#
# API keys are read from environment variables or a local .env file.
# No API keys are hard-coded into this file.
# ============================================================

import os
import sys
import re
import json
import time
import random
import ipaddress
import hashlib
import warnings
import subprocess
from pathlib import Path
from getpass import getpass
from datetime import datetime

# ------------------------------------------------------------
# BASE DIRECTORY
# ------------------------------------------------------------

BASE_DIR = Path(r"D:\Desktop\final thesis")
if not BASE_DIR.exists():
    BASE_DIR = Path(__file__).resolve().parent

BASE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_DIR)

# ------------------------------------------------------------
# CELL 1 — INSTALLATION (VS CODE VERSION)
# ------------------------------------------------------------

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "pydeck": "pydeck",
    "requests": "requests",
    "streamlit": "streamlit",
    "tensorflow": "tensorflow",
    "joblib": "joblib",
}


def install_missing_packages():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)

    if missing:
        print("Installing missing packages:")
        print("  " + ", ".join(missing))
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            *missing,
        ])

    print("Installation check complete.")


install_missing_packages()

# ------------------------------------------------------------
# API KEY LOADER — NO python-dotenv REQUIRED
# ------------------------------------------------------------


def load_simple_env_file(env_path: Path):
    """Load KEY=value pairs from .env without requiring python-dotenv."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_simple_env_file(BASE_DIR / ".env")

# Colab used getpass(). For VS Code, use the .env/environment first.
# If this script is executed directly from a real terminal and keys are
# missing, securely prompt once without displaying the typed values.
if not os.getenv("VIRUSTOTAL_KEY") and sys.stdin is not None and sys.stdin.isatty():
    try:
        vt_key = getpass("VirusTotal API key (press Enter to skip): ").strip()
        if vt_key:
            os.environ["VIRUSTOTAL_KEY"] = vt_key
    except (EOFError, KeyboardInterrupt):
        pass

if not os.getenv("ABUSEIPDB_KEY") and sys.stdin is not None and sys.stdin.isatty():
    try:
        abuse_key = getpass("AbuseIPDB API key (press Enter to skip): ").strip()
        if abuse_key:
            os.environ["ABUSEIPDB_KEY"] = abuse_key
    except (EOFError, KeyboardInterrupt):
        pass

print("API key configuration loaded.")
print("VirusTotal key:", "loaded" if os.getenv("VIRUSTOTAL_KEY") else "not configured")
print("AbuseIPDB key:", "loaded" if os.getenv("ABUSEIPDB_KEY") else "not configured")

# ------------------------------------------------------------
# CELL 3 — IMPORTS
# ------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import requests
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("Libraries loaded.")
print("TensorFlow:", tf.__version__)

# Website uses these same scores as the original final dashboard.
SEVERITY_SCORE = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100,
}


# ============================================================
# DATASET + MODEL PIPELINE
# ============================================================
# This is the Colab data/model sequence moved into one function so
# Streamlit does not retrain the models on every page interaction.
# ============================================================

PIPELINE_OUTPUTS = [
    BASE_DIR / "combined_threats.csv",
    BASE_DIR / "random_forest.pkl",
    BASE_DIR / "neural_network.pkl",
    BASE_DIR / "neural_network_scaler.pkl",
    BASE_DIR / "lstm_model.keras",
    BASE_DIR / "lstm_label_encoder.pkl",
    BASE_DIR / "lstm_scaler.pkl",
    BASE_DIR / "kmeans.pkl",
    BASE_DIR / "cluster_scaler.pkl",
    BASE_DIR / "model_results.csv",
]

REQUIRED_FILES = [
    "AWS_Honeypot_marx-geo.csv",
    "london.csv",
    "singapore.csv",
]


def pipeline_outputs_exist():
    return all(path.exists() for path in PIPELINE_OUTPUTS)


PREPARATION_LOG = BASE_DIR / "pipeline_status.txt"


def prepare_pipeline():
    print("=" * 80)
    print("AUTOMATED THREAT PRIORITIZATION — DATA / MODEL PIPELINE")
    print("=" * 80)
    print("Working folder:", BASE_DIR)

    missing = [name for name in REQUIRED_FILES if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "The following dataset(s) are missing from D:\\Desktop\\final thesis:\n"
            + "\n".join(f" - {name}" for name in missing)
            + "\n\nCopy all three CSV files into the same folder as app.py."
        )

    if pipeline_outputs_exist():
        print("Existing combined dataset and model files found.")
        print("Skipping retraining.")
        PREPARATION_LOG.write_text(
            "Existing pipeline outputs found. Training was skipped.\n",
            encoding="utf-8",
        )
        return

    print("WARNING: Pipeline artifacts are incomplete. Running the full Colab pipeline once...")

    # ============================================================
    # CELL 6 — THREAT DEFINITIONS
    # ============================================================

    THREAT_TYPES = [
        "XSS Attempt",
        "Credential Stuffing",
        "Log4j Probe",
        "Port Scan",
        "DDoS Attack",
        "Brute Force",
        "SQL Injection",
        "Malware Link"
    ]

    BASE_SEVERITY = {

        "DDoS Attack":
            "Critical",

        "Malware Link":
            "Critical",

        "Credential Stuffing":
            "High",

        "SQL Injection":
            "High",

        "Brute Force":
            "High",

        "XSS Attempt":
            "Medium",

        "Log4j Probe":
            "Medium",

        "Port Scan":
            "Low"
    }

    SEVERITY_WEIGHT = {

        "Low": 0.25,

        "Medium": 0.50,

        "High": 0.75,

        "Critical": 1.00
    }

    print("Threat definitions loaded.")
    # ================= CELL 7 =================
    # ============================================================
    # CELL 7 — PAYLOAD CLASSIFICATION
    # ============================================================

    def classify_payload(raw):

        p = str(raw).lower()

        if (
            "jndi:" in p
            or "${jndi" in p
        ):
            return "Log4j Probe"

        if (
            "<script" in p
            or "onerror=" in p
            or "javascript:" in p
        ):
            return "XSS Attempt"

        if (
            "union select" in p
            or (
                "select " in p
                and " from " in p
            )
        ):
            return "SQL Injection"

        if any(
            x in p
            for x in [
                "shell_exec",
                "new-application",
                "/etc/passwd",
                "base64_decode",
                "wget ",
                "curl "
            ]
        ):
            return "Malware Link"

        if (
            "eth_blocknumber" in p
            or "jsonrpc" in p
        ):
            return "Malware Link"

        if "mstshash=" in p:
            return "Brute Force"

        if any(
            x in p
            for x in [
                "login",
                "wp-login",
                "admin",
                "/xmlrpc"
            ]
        ):
            return "Credential Stuffing"

        if (
            "get " in p
            or "post " in p
        ):
            return "Port Scan"

        return "Port Scan"


    PORT_MAP = {

        22:
            "Brute Force",

        23:
            "Brute Force",

        3389:
            "Brute Force",

        445:
            "Malware Link",

        135:
            "Malware Link",

        6666:
            "Malware Link",

        1433:
            "SQL Injection",

        3306:
            "SQL Injection",

        53:
            "DDoS Attack",

        5060:
            "DDoS Attack"
    }


    def classify_port(port):

        try:

            port = int(float(port))

        except:

            return "Port Scan"

        return PORT_MAP.get(
            port,
            "Port Scan"
        )


    print("Classifiers ready.")
    # ================= CELL 8 =================
    # ============================================================
    # CELL 8 — LOAD DATASETS + MULTI-CLOUD + LOCATION
    # ============================================================

    import numpy as np
    import pandas as pd


    # ============================================================
    # COMMON COLUMNS
    # ============================================================

    COMMON_COLUMNS = [
        "src",
        "payload_text",
        "type",
        "severity",
        "target",
        "country_code",
        "city",
        "region",
        "latitude",
        "longitude",
        "event_time"
    ]


    # ============================================================
    # LONDON / SINGAPORE
    # ============================================================

    def load_payload_dataset(
        filename,
        target_cloud
    ):

        df = pd.read_csv(
            filename,
            low_memory=False
        )


        required = [
            "from",
            "payload"
        ]


        for column in required:

            if column not in df.columns:

                raise ValueError(
                    f"{filename} is missing '{column}'"
                )


        df = df.copy()


        # --------------------------------------------------------
        # Source IP
        # --------------------------------------------------------

        df["src"] = (
            df["from"]
            .astype(str)
            .str.strip()
            .str.strip("'")
            .str.strip('"')
        )


        # --------------------------------------------------------
        # Payload
        # --------------------------------------------------------

        df["payload_text"] = (
            df["payload"]
            .astype(str)
        )


        # --------------------------------------------------------
        # Threat
        # --------------------------------------------------------

        df["type"] = (
            df["payload"]
            .apply(
                classify_payload
            )
        )


        # --------------------------------------------------------
        # Severity
        # --------------------------------------------------------

        df["severity"] = (
            df["type"]
            .map(BASE_SEVERITY)
        )


        # --------------------------------------------------------
        # CLOUD
        # --------------------------------------------------------

        df["target"] = target_cloud


        # --------------------------------------------------------
        # Country / location
        #
        # London/Singapore CSVs identify the honeypot location,
        # NOT necessarily the attacker's physical location.
        # We therefore leave attacker coordinates empty here.
        # They will be obtained from IP geolocation in the website.
        # --------------------------------------------------------

        df["country_code"] = (
            df["country"]
            .astype(str)
        )


        df["city"] = "Unknown"

        df["region"] = "Unknown"

        df["latitude"] = np.nan

        df["longitude"] = np.nan


        # --------------------------------------------------------
        # Event time
        # --------------------------------------------------------

        if "time" in df.columns:

            df["event_time"] = (
                df["time"]
                .astype(str)
            )

        else:

            df["event_time"] = ""


        return df[
            COMMON_COLUMNS
        ].copy()


    # ============================================================
    # AWS HONEYPOT
    # ============================================================

    def load_aws_dataset(
        filename
    ):

        df = pd.read_csv(
            filename,
            on_bad_lines="skip",
            low_memory=False
        )


        required = [
            "srcstr",
            "dpt",
            "cc",
            "country",
            "latitude",
            "longitude"
        ]


        for column in required:

            if column not in df.columns:

                raise ValueError(
                    f"{filename} is missing '{column}'"
                )


        df = df.dropna(
            subset=[
                "srcstr",
                "dpt"
            ]
        ).copy()


        # --------------------------------------------------------
        # Source IP
        # --------------------------------------------------------

        df["src"] = (
            df["srcstr"]
            .astype(str)
            .str.strip()
        )


        # --------------------------------------------------------
        # Threat
        # --------------------------------------------------------

        df["type"] = (
            df["dpt"]
            .apply(
                classify_port
            )
        )


        # --------------------------------------------------------
        # Severity
        # --------------------------------------------------------

        df["severity"] = (
            df["type"]
            .map(BASE_SEVERITY)
        )


        # --------------------------------------------------------
        # CLOUD
        # --------------------------------------------------------

        df["target"] = "AWS"


        # --------------------------------------------------------
        # REAL DATASET LOCATION
        #
        # These latitude/longitude values are already present
        # in your AWS honeypot dataset.
        # --------------------------------------------------------

        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )


        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )


        df["country_code"] = (
            df["cc"]
            .astype(str)
        )


        df["city"] = (
            df["locale"]
            .fillna("Unknown")
            .astype(str)
        )


        df["region"] = (
            df["locale"]
            .fillna("Unknown")
            .astype(str)
        )


        # --------------------------------------------------------
        # Payload representation
        # --------------------------------------------------------

        df["payload_text"] = (
            "NETWORK PORT: "
            +
            df["dpt"]
            .astype(str)
        )


        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------

        if "datetime" in df.columns:

            df["event_time"] = (
                df["datetime"]
                .astype(str)
            )

        else:

            df["event_time"] = ""


        return df[
            COMMON_COLUMNS
        ].copy()


    # ============================================================
    # LOAD DATASETS
    #
    # MULTI-CLOUD PROTOTYPE MAPPING:
    #
    # AWS_Honeypot_marx-geo.csv → AWS
    # london.csv                → Azure
    # singapore.csv             → GCP
    #
    # This mapping must be described in the thesis as a
    # prototype/experimental cloud assignment unless the source
    # logs themselves contain cloud-provider metadata.
    # ============================================================

    aws = load_aws_dataset(
        "AWS_Honeypot_marx-geo.csv"
    )


    london = load_payload_dataset(
        "london.csv",
        "Azure"
    )


    singapore = load_payload_dataset(
        "singapore.csv",
        "GCP"
    )


    # ============================================================
    # COMBINE
    # ============================================================

    combined = pd.concat(

        [
            aws,
            london,
            singapore
        ],

        ignore_index=True

    )


    # ============================================================
    # CLEAN
    # ============================================================

    combined = combined.dropna(

        subset=[
            "src",
            "payload_text",
            "type",
            "severity",
            "target"
        ]

    ).copy()


    combined["src"] = (
        combined["src"]
        .astype(str)
        .str.strip()
    )


    # ============================================================
    # SAVE
    # ============================================================

    combined.to_csv(

        "combined_threats.csv",

        index=False

    )


    # ============================================================
    # OUTPUT
    # ============================================================

    print(
        "=" * 70
    )

    print(
        "MULTI-CLOUD DATASET CREATED"
    )

    print(
        "=" * 70
    )


    print(
        "\nCloud distribution:"
    )

    print(
        combined[
            "target"
        ].value_counts()
    )


    print(
        "\nThreat distribution:"
    )

    print(
        combined[
            "type"
        ].value_counts()
    )


    print(
        "\nCoordinates available:"
    )

    print(
        combined[
            [
                "latitude",
                "longitude"
            ]
        ]
        .notna()
        .all(axis=1)
        .sum()
    )


    print(
        "\nTotal records:",
        len(combined)
    )


    print(
        "\nUnique IPs:",
        combined[
            "src"
        ].nunique()
    )


    print(
        combined.head()
    )
    # ================= CELL 9 =================
    # ============================================================
    # CELL 9 — VALIDATE DATA
    # ============================================================

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        "\nTotal records:",
        len(combined)
    )

    print(
        "Unique IP addresses:",
        combined["src"].nunique()
    )

    print(
        "\nThreat distribution:"
    )

    print(
        combined["type"]
        .value_counts()
    )

    print(
        "\nSeverity distribution:"
    )

    print(
        combined["severity"]
        .value_counts()
    )

    print(
        "\nMissing values:"
    )

    print(
        combined.isnull().sum()
    )


    combined.to_csv(
        "combined_threats.csv",
        index=False
    )

    print(
        "\ncombined_threats.csv saved."
    )
    # ================= CELL 10 =================
    # ============================================================
    # CELL 10 — FEATURE ENGINEERING
    # ============================================================

    def ip_to_features(ip):

        try:

            parts = str(ip).split(".")

            if len(parts) != 4:
                return [0, 0, 0, 0]

            values = [
                int(x)
                for x in parts
            ]

            if not all(
                0 <= x <= 255
                for x in values
            ):
                return [0, 0, 0, 0]

            return values

        except:

            return [0, 0, 0, 0]


    def payload_to_features(payload):

        p = str(
            payload
        ).lower()

        return [

            int(
                "<script" in p
            ),

            int(
                "jndi" in p
            ),

            int(
                "union select" in p
            ),

            int(
                "login" in p
                or "wp-login" in p
            ),

            int(
                "mstshash" in p
            ),

            int(
                "wget" in p
                or "curl" in p
            ),

            int(
                "jsonrpc" in p
                or "eth_blocknumber" in p
            ),

            int(
                "select " in p
                and " from " in p
            ),

            int(
                "network port" in p
            )
        ]


    def encode_row(row):

        ip_features = ip_to_features(
            row["src"]
        )

        payload_features = payload_to_features(
            row["payload_text"]
        )

        cloud_features = [

            int(
                row["target"] == "AWS"
            ),

            int(
                row["target"] == "Azure"
            ),

            int(
                row["target"] == "GCP"
            )
        ]

        return (
            ip_features
            +
            payload_features
            +
            cloud_features
        )


    X = np.array(
        [
            encode_row(row)
            for _, row in combined.iterrows()
        ],
        dtype=np.float32
    )

    # Convert the target labels to a plain NumPy array.
    # This avoids the pandas/pyarrow ExtensionArray indexing error
    # that can occur with newer pandas versions on Windows.
    y = np.asarray(
        combined["type"].astype(str).tolist(),
        dtype=object
    )


    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )
    # ================= CELL 11 =================
    # ============================================================
    # CELL 11 — TRAIN/TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    print(
        "Training records:",
        len(X_train)
    )

    print(
        "Testing records:",
        len(X_test)
    )
    # ================= CELL 12 =================
    # ============================================================
    # CELL 12 — MODEL 1: RANDOM FOREST
    # ============================================================

    rf_model = RandomForestClassifier(

        n_estimators=250,

        max_features="sqrt",

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )


    rf_model.fit(
        X_train,
        y_train
    )


    rf_pred = rf_model.predict(
        X_test
    )


    rf_accuracy = accuracy_score(
        y_test,
        rf_pred
    )

    rf_precision = precision_score(
        y_test,
        rf_pred,
        average="weighted",
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_pred,
        average="weighted",
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_pred,
        average="weighted",
        zero_division=0
    )


    print("RANDOM FOREST")
    print("=" * 50)

    print(
        f"Accuracy : {rf_accuracy:.4f}"
    )

    print(
        f"Precision: {rf_precision:.4f}"
    )

    print(
        f"Recall   : {rf_recall:.4f}"
    )

    print(
        f"F1 Score : {rf_f1:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            rf_pred,
            zero_division=0
        )
    )
    # ================= CELL 13 =================
    # ============================================================
    # CELL 13 — RANDOM FOREST CONFUSION MATRIX
    # ============================================================

    labels = sorted(
        combined["type"].unique()
    )

    cm = confusion_matrix(
        y_test,
        rf_pred,
        labels=labels
    )


    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )


    disp.plot(
        ax=ax,
        colorbar=False,
        xticks_rotation=45
    )


    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.close(fig)
    # ================= CELL 14 =================
    # ============================================================
    # CELL 14 — MODEL 2: NEURAL NETWORK (FIXED)
    # ============================================================

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    # ------------------------------------------------------------
    # 1. Make absolutely sure X is numeric
    # ------------------------------------------------------------

    X_train = np.asarray(
        X_train,
        dtype=np.float64
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64
    )

    # Check for invalid values
    if not np.isfinite(X_train).all():
        raise ValueError(
            "X_train contains NaN or infinite values."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "X_test contains NaN or infinite values."
        )


    # ------------------------------------------------------------
    # 2. Scale the input features
    # ------------------------------------------------------------

    nn_scaler = StandardScaler()

    X_train_nn = nn_scaler.fit_transform(
        X_train
    )

    X_test_nn = nn_scaler.transform(
        X_test
    )


    # ------------------------------------------------------------
    # 3. Make y explicitly a clean 1-D string array
    # ------------------------------------------------------------

    y_train_nn = np.asarray(
        y_train,
        dtype=str
    ).ravel()

    y_test_nn = np.asarray(
        y_test,
        dtype=str
    ).ravel()


    print("X_train dtype:", X_train_nn.dtype)
    print("X_test dtype :", X_test_nn.dtype)
    print("y_train dtype:", y_train_nn.dtype)

    print(
        "\nNumber of classes:",
        len(np.unique(y_train_nn))
    )


    # ------------------------------------------------------------
    # 4. Neural Network
    #
    # IMPORTANT:
    # early_stopping=False
    #
    # This avoids the np.isnan/object-label problem that
    # occurred in your current Colab environment.
    # ------------------------------------------------------------

    nn_model = MLPClassifier(

        hidden_layer_sizes=(
            64,
            32,
            16
        ),

        activation="relu",

        solver="adam",

        alpha=0.0001,

        learning_rate_init=0.001,

        max_iter=500,

        early_stopping=False,

        random_state=42
    )


    # ------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------

    nn_model.fit(
        X_train_nn,
        y_train_nn
    )


    # ------------------------------------------------------------
    # 6. Predictions
    # ------------------------------------------------------------

    nn_pred = nn_model.predict(
        X_test_nn
    )


    # ------------------------------------------------------------
    # 7. Metrics
    # ------------------------------------------------------------

    nn_accuracy = accuracy_score(
        y_test_nn,
        nn_pred
    )

    nn_precision = precision_score(
        y_test_nn,
        nn_pred,
        average="weighted",
        zero_division=0
    )

    nn_recall = recall_score(
        y_test_nn,
        nn_pred,
        average="weighted",
        zero_division=0
    )

    nn_f1 = f1_score(
        y_test_nn,
        nn_pred,
        average="weighted",
        zero_division=0
    )


    # ------------------------------------------------------------
    # 8. Results
    # ------------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("NEURAL NETWORK RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {nn_accuracy:.4f}"
    )

    print(
        f"Precision: {nn_precision:.4f}"
    )

    print(
        f"Recall   : {nn_recall:.4f}"
    )

    print(
        f"F1 Score : {nn_f1:.4f}"
    )


    print("\nClassification Report:")
    print(
        classification_report(
            y_test_nn,
            nn_pred,
            zero_division=0
        )
    )
    # ================= CELL 15 =================
    # ============================================================
    # CHECK THE ACTUAL THREAT CLASSES
    # ============================================================

    print("Threat classes in COMPLETE dataset:")
    print(
        sorted(
            combined["type"].unique()
        )
    )

    print(
        "\nNumber of classes in complete dataset:",
        combined["type"].nunique()
    )

    print("\nClass distribution:")
    print(
        combined["type"].value_counts()
    )

    print("\nClasses in training data:")
    print(
        sorted(
            np.unique(y_train)
        )
    )

    print(
        "\nClasses in test data:"
    )

    print(
        sorted(
            np.unique(y_test)
        )
    )
    # ================= CELL 16 =================
    # ============================================================
    # CELL 14 — MODEL 2: NEURAL NETWORK
    # OPTIMIZED FOR YOUR 1.1M+ RECORD DATASET
    # ============================================================

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    # ------------------------------------------------------------
    # 1. Ensure numeric data
    # ------------------------------------------------------------

    X_train = np.asarray(
        X_train,
        dtype=np.float64
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64
    )


    # ------------------------------------------------------------
    # 2. Check for invalid values
    # ------------------------------------------------------------

    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "X_train contains NaN or infinite values."
        )


    if not np.isfinite(
        X_test
    ).all():

        raise ValueError(
            "X_test contains NaN or infinite values."
        )


    # ------------------------------------------------------------
    # 3. Clean labels
    # ------------------------------------------------------------

    y_train_nn = np.asarray(
        y_train,
        dtype=str
    ).ravel()

    y_test_nn = np.asarray(
        y_test,
        dtype=str
    ).ravel()


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print(
        "Classes:",
        np.unique(y_train_nn)
    )


    # ------------------------------------------------------------
    # 4. Scale features
    # ------------------------------------------------------------

    nn_scaler = StandardScaler()

    X_train_nn = (
        nn_scaler
        .fit_transform(
            X_train
        )
    )

    X_test_nn = (
        nn_scaler
        .transform(
            X_test
        )
    )


    # ------------------------------------------------------------
    # 5. Neural Network
    # ------------------------------------------------------------

    nn_model = MLPClassifier(

        hidden_layer_sizes=(
            32,
            16
        ),

        activation="relu",

        solver="adam",

        alpha=0.0001,

        learning_rate_init=0.001,

        batch_size=2048,

        max_iter=100,

        early_stopping=False,

        random_state=42,

        verbose=True
    )


    # ------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------

    print(
        "\nStarting Neural Network training..."
    )

    nn_model.fit(
        X_train_nn,
        y_train_nn
    )


    # ------------------------------------------------------------
    # 7. Predictions
    # ------------------------------------------------------------

    nn_pred = nn_model.predict(
        X_test_nn
    )


    # ------------------------------------------------------------
    # 8. Metrics
    # ------------------------------------------------------------

    nn_accuracy = accuracy_score(
        y_test_nn,
        nn_pred
    )

    nn_precision = precision_score(
        y_test_nn,
        nn_pred,
        average="weighted",
        zero_division=0
    )

    nn_recall = recall_score(
        y_test_nn,
        nn_pred,
        average="weighted",
        zero_division=0
    )

    nn_f1 = f1_score(
        y_test_nn,
        nn_pred,
        average="weighted",
        zero_division=0
    )


    # ------------------------------------------------------------
    # 9. Results
    # ------------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("NEURAL NETWORK RESULTS")
    print("=" * 65)

    print(
        f"Accuracy : {nn_accuracy:.4f}"
    )

    print(
        f"Precision: {nn_precision:.4f}"
    )

    print(
        f"Recall   : {nn_recall:.4f}"
    )

    print(
        f"F1 Score : {nn_f1:.4f}"
    )


    print(
        "\nClassification Report:"
    )


    print(
        classification_report(
            y_test_nn,
            nn_pred,
            zero_division=0
        )
    )
    # ================= CELL 17 =================
    # ============================================================
    # CELL 15 — PREPARE LSTM DATA
    # ============================================================

    lstm_encoder = LabelEncoder()

    combined["threat_encoded"] = (
        lstm_encoder.fit_transform(
            combined["type"]
        )
    )


    combined["severity_numeric"] = (
        combined["severity"]
        .map(SEVERITY_WEIGHT)
    )


    combined["cloud_numeric"] = (
        combined["target"]
        .map({
            "AWS": 0,
            "Azure": 1,
            "GCP": 2
        })
    )


    # Convert all LSTM input columns to ordinary numeric NumPy data.
    # This also prevents pandas/pyarrow ExtensionArray indexing issues.
    lstm_frame = combined[
        [
            "threat_encoded",
            "severity_numeric",
            "cloud_numeric"
        ]
    ].copy()

    for column in lstm_frame.columns:
        lstm_frame[column] = pd.to_numeric(
            lstm_frame[column],
            errors="coerce"
        )

    lstm_frame = lstm_frame.fillna(0.0)

    lstm_data = lstm_frame.to_numpy(
        dtype=np.float32
    )


    lstm_scaler = StandardScaler()


    lstm_data_scaled = (
        lstm_scaler
        .fit_transform(
            lstm_data
        )
    )


    TIMESTEPS = 8


    X_lstm = []
    y_lstm = []


    for i in range(
        TIMESTEPS,
        len(lstm_data_scaled)
    ):

        X_lstm.append(

            lstm_data_scaled[
                i - TIMESTEPS:i
            ]

        )

        y_lstm.append(

            combined[
                "threat_encoded"
            ].iloc[i]

        )


    X_lstm = np.array(
        X_lstm,
        dtype=np.float32
    )


    y_lstm = np.array(
        y_lstm,
        dtype=np.int32
    )


    print(
        "LSTM input:",
        X_lstm.shape
    )

    print(
        "LSTM target:",
        y_lstm.shape
    )
    # ================= CELL 18 =================
    # ============================================================
    # CELL 16 — LSTM TRAIN/TEST
    # ============================================================

    lstm_split = int(
        len(X_lstm) * 0.80
    )


    X_lstm_train = (
        X_lstm[
            :lstm_split
        ]
    )


    X_lstm_test = (
        X_lstm[
            lstm_split:
        ]
    )


    y_lstm_train = (
        y_lstm[
            :lstm_split
        ]
    )


    y_lstm_test = (
        y_lstm[
            lstm_split:
        ]
    )


    print(
        "LSTM training:",
        X_lstm_train.shape
    )

    print(
        "LSTM testing:",
        X_lstm_test.shape
    )
    # ================= CELL 19 =================
    # ============================================================
    # CELL 17 — BUILD REAL LSTM
    # ============================================================

    num_classes = len(
        lstm_encoder.classes_
    )


    lstm_model = Sequential([

        Input(
            shape=(
                TIMESTEPS,
                X_lstm_train.shape[2]
            )
        ),

        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(
            0.25
        ),

        LSTM(
            32
        ),

        Dropout(
            0.25
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])


    lstm_model.compile(

        optimizer="adam",

        loss=
            "sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    lstm_model.summary()
    # ================= CELL 20 =================
    # ============================================================
    # CELL 18 — TRAIN LSTM
    # ============================================================

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    )


    history = lstm_model.fit(

        X_lstm_train,

        y_lstm_train,

        validation_split=0.15,

        epochs=30,

        batch_size=64,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )


    print(
        "LSTM training finished."
    )
    # ================= CELL 21 =================
    # ============================================================
    # CELL 19 — LSTM EVALUATION
    # ============================================================

    lstm_loss, lstm_accuracy = (
        lstm_model.evaluate(
            X_lstm_test,
            y_lstm_test,
            verbose=0
        )
    )


    lstm_probabilities = (
        lstm_model.predict(
            X_lstm_test,
            verbose=0
        )
    )


    lstm_prediction_encoded = (
        np.argmax(
            lstm_probabilities,
            axis=1
        )
    )


    lstm_prediction = (
        lstm_encoder
        .inverse_transform(
            lstm_prediction_encoded
        )
    )


    lstm_actual = (
        lstm_encoder
        .inverse_transform(
            y_lstm_test
        )
    )


    lstm_precision = precision_score(
        lstm_actual,
        lstm_prediction,
        average="weighted",
        zero_division=0
    )


    lstm_recall = recall_score(
        lstm_actual,
        lstm_prediction,
        average="weighted",
        zero_division=0
    )


    lstm_f1 = f1_score(
        lstm_actual,
        lstm_prediction,
        average="weighted",
        zero_division=0
    )


    print("LSTM")
    print("=" * 50)

    print(
        f"Accuracy : {lstm_accuracy:.4f}"
    )

    print(
        f"Precision: {lstm_precision:.4f}"
    )

    print(
        f"Recall   : {lstm_recall:.4f}"
    )

    print(
        f"F1 Score : {lstm_f1:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            lstm_actual,
            lstm_prediction,
            zero_division=0
        )
    )
    # ================= CELL 22 =================
    # ============================================================
    # CELL 20 — THREE MODEL COMPARISON
    # ============================================================

    model_results = pd.DataFrame({

        "Model": [

            "Random Forest",

            "Neural Network",

            "LSTM"

        ],

        "Accuracy": [

            rf_accuracy,

            nn_accuracy,

            lstm_accuracy

        ],

        "Precision": [

            rf_precision,

            nn_precision,

            lstm_precision

        ],

        "Recall": [

            rf_recall,

            nn_recall,

            lstm_recall

        ],

        "F1 Score": [

            rf_f1,

            nn_f1,

            lstm_f1

        ]

    })


    print(
        model_results
    )


    model_results.to_csv(
        "model_results.csv",
        index=False
    )
    # ================= CELL 23 =================
    # ============================================================
    # CELL 21 — K-MEANS IP BEHAVIOUR
    # ============================================================

    combined["severity_numeric"] = (
        combined["severity"]
        .map(SEVERITY_WEIGHT)
    )


    ip_stats = (
        combined
        .groupby("src")
        .agg(

            attack_count=(
                "src",
                "count"
            ),

            average_severity=(
                "severity_numeric",
                "mean"
            )

        )
        .reset_index()
    )


    cluster_scaler = StandardScaler()


    cluster_X = (
        cluster_scaler
        .fit_transform(
            ip_stats[
                [
                    "attack_count",
                    "average_severity"
                ]
            ]
        )
    )


    kmeans = KMeans(

        n_clusters=3,

        random_state=42,

        n_init=10
    )


    ip_stats["cluster"] = (
        kmeans
        .fit_predict(
            cluster_X
        )
    )


    print(
        "K-Means trained."
    )


    print(
        ip_stats.head()
    )
    # ================= CELL 24 =================
    # ============================================================
    # CELL 23 — ABUSEIPDB API
    # ============================================================

    def abuseipdb_lookup(ip):

        api_key = os.getenv(
            "ABUSEIPDB_KEY"
        )

        if not api_key:

            return {
                "success": False,
                "error":
                    "AbuseIPDB API key missing."
            }


        if not valid_public_ip(ip):

            return {
                "success": False,
                "error":
                    "Not a valid public IP."
            }


        url = (
            "https://api.abuseipdb.com/api/v2/check"
        )


        try:

            response = requests.get(

                url,

                headers={

                    "Accept":
                        "application/json",

                    "Key":
                        api_key

                },

                params={

                    "ipAddress":
                        ip,

                    "maxAgeInDays":
                        90

                },

                timeout=15
            )


            if response.status_code != 200:

                return {

                    "success": False,

                    "status_code":
                        response.status_code,

                    "error":
                        response.text[:500]

                }


            data = (
                response
                .json()
                .get(
                    "data",
                    {}
                )
            )


            return {

                "success": True,

                "ip":
                    data.get(
                        "ipAddress",
                        ip
                    ),

                "abuse_confidence":
                    data.get(
                        "abuseConfidenceScore",
                        0
                    ),

                "country":
                    data.get(
                        "countryCode",
                        "Unknown"
                    ),

                "isp":
                    data.get(
                        "isp",
                        "Unknown"
                    ),

                "domain":
                    data.get(
                        "domain",
                        "Unknown"
                    ),

                "total_reports":
                    data.get(
                        "totalReports",
                        0
                    ),

                "last_reported":
                    data.get(
                        "lastReportedAt",
                        None
                    )

            }


        except Exception as e:

            return {

                "success": False,

                "error":
                    str(e)

            }


    print(
        "AbuseIPDB function ready."
    )
    # ================= CELL 25 =================
    # ============================================================
    # FINAL THREAT PRIORITY
    # ============================================================

    def calculate_priority(
        rf_score,
        nn_score,
        lstm_score,
        severity,
        abuse_result=None
    ):

        # --------------------------------------------------------
        # AI MODEL SCORE
        # --------------------------------------------------------

        ai_score = (

            rf_score * 25

            +

            nn_score * 20

            +

            lstm_score * 20

            +

            SEVERITY_SCORE.get(
                severity,
                50
            ) * 0.35

        )


        final_score = ai_score


        # --------------------------------------------------------
        # ABUSEIPDB
        # --------------------------------------------------------

        if (
            abuse_result
            and
            abuse_result.get("success")
        ):

            abuse_confidence = float(
                abuse_result.get(
                    "abuse_confidence",
                    0
                )
            )

            # Add up to 15 points
            final_score += (
                abuse_confidence * 0.15
            )


        # --------------------------------------------------------
        # LIMIT SCORE TO 0-100
        # --------------------------------------------------------

        final_score = max(
            0,
            min(
                final_score,
                100
            )
        )


        # --------------------------------------------------------
        # PRIORITY
        # --------------------------------------------------------

        if final_score >= 80:

            priority = "CRITICAL"

        elif final_score >= 60:

            priority = "HIGH"

        elif final_score >= 35:

            priority = "MEDIUM"

        else:

            priority = "LOW"


        return (
            round(final_score, 2),
            priority,
            round(ai_score, 2)
        )
    # ================= CELL 26 =================
    # ============================================================
    # CELL 25 — FINAL THREAT PRIORITY
    # ============================================================

    def calculate_final_score(

        ai_risk,

        vt_result,

        abuse_result

    ):

        score = float(
            ai_risk
        )


        # ----------------------------------------
        # VirusTotal evidence
        # ----------------------------------------

        if vt_result.get("success"):

            reputation = float(
                vt_result.get(
                    "reputation",
                    0
                )
            )

            malicious = int(
                vt_result
                .get(
                    "analysis",
                    {}
                )
                .get(
                    "malicious",
                    0
                )
            )


            score += min(
                max(
                    reputation,
                    0
                ) * 0.10,
                10
            )


            score += min(
                malicious * 1.0,
                10
            )


        # ----------------------------------------
        # AbuseIPDB evidence
        # ----------------------------------------

        if abuse_result.get("success"):

            abuse_score = float(
                abuse_result.get(
                    "abuse_confidence",
                    0
                )
            )


            score += (
                abuse_score * 0.15
            )


        # ----------------------------------------
        # Limit
        # ----------------------------------------

        score = max(
            0,
            min(
                score,
                100
            )
        )


        # ----------------------------------------
        # Priority
        # ----------------------------------------

        if score >= 80:

            priority = "CRITICAL"

        elif score >= 60:

            priority = "HIGH"

        elif score >= 35:

            priority = "MEDIUM"

        else:

            priority = "LOW"


        return {

            "score":
                round(
                    score,
                    2
                ),

            "priority":
                priority

        }
    # ================= CELL 27 =================
    # ============================================================
    # FINAL THREAT PRIORITY SCORE
    # ============================================================

    def calculate_final_priority(
        rf_score,
        nn_score,
        lstm_score,
        severity,
        abuse_result=None,
        attack_count=1
    ):

        # --------------------------------------------------------
        # 1. AI MODEL CONFIDENCE
        # --------------------------------------------------------

        model_confidence = (
            rf_score +
            nn_score +
            lstm_score
        ) / 3


        # --------------------------------------------------------
        # 2. THREAT SEVERITY
        # --------------------------------------------------------

        severity_score = (
            SEVERITY_SCORE.get(
                severity,
                50
            )
        )


        # --------------------------------------------------------
        # 3. ABUSEIPDB
        # --------------------------------------------------------

        if (
            abuse_result
            and
            abuse_result.get("success")
        ):

            abuse_score = float(
                abuse_result.get(
                    "abuse_confidence",
                    0
                )
            )

        else:

            abuse_score = 0


        # --------------------------------------------------------
        # 4. ATTACK BEHAVIOUR
        # --------------------------------------------------------

        # More repeated attacks from the same source
        # increase priority.

        behaviour_score = min(
            attack_count * 10,
            100
        )


        # --------------------------------------------------------
        # 5. FINAL SCORE
        # --------------------------------------------------------

        final_score = (

            model_confidence * 0.40

            +

            severity_score * 0.25

            +

            abuse_score * 0.20

            +

            behaviour_score * 0.15

        )


        # --------------------------------------------------------
        # LIMIT
        # --------------------------------------------------------

        final_score = max(
            0,
            min(
                final_score,
                100
            )
        )


        # --------------------------------------------------------
        # PRIORITY
        # --------------------------------------------------------

        if final_score >= 80:

            priority = "CRITICAL"

        elif final_score >= 60:

            priority = "HIGH"

        elif final_score >= 35:

            priority = "MEDIUM"

        else:

            priority = "LOW"


        return (
            round(
                final_score,
                2
            ),
            priority
        )

# ================= SAVE TRAINED MODELS =================
    # ============================================================
    # CELL 27 — SAVE MODELS
    # ============================================================

    joblib.dump(
        rf_model,
        "random_forest.pkl"
    )

    joblib.dump(
        nn_model,
        "neural_network.pkl"
    )

    joblib.dump(
        nn_scaler,
        "neural_network_scaler.pkl"
    )

    joblib.dump(
        lstm_encoder,
        "lstm_label_encoder.pkl"
    )

    joblib.dump(
        lstm_scaler,
        "lstm_scaler.pkl"
    )

    joblib.dump(
        kmeans,
        "kmeans.pkl"
    )

    joblib.dump(
        cluster_scaler,
        "cluster_scaler.pkl"
    )


    lstm_model.save(
        "lstm_model.keras"
    )


    print(
        "All trained models saved."
    )


    print("All trained models saved.")
    PREPARATION_LOG.write_text(
        "Full dataset processing and model training completed successfully.\n",
        encoding="utf-8",
    )
    print("Full pipeline completed successfully.")


# Import Streamlit after the installation check and pipeline definitions.
import streamlit as st
import pydeck as pdk
from tensorflow.keras.models import load_model


# Streamlit page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Automated Threat Prioritization",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner="Preparing datasets and AI models. First run can take time...")
def run_preparation_once():
    prepare_pipeline()
    return True


# ------------------------------------------------------------
# RUN DATA + MODEL PIPELINE FIRST
# ------------------------------------------------------------
try:
    run_preparation_once()
except Exception as exc:
    st.error("The dataset/model preparation step failed.")
    st.exception(exc)
    st.stop()

# ============================================================
# CELL 28 — FINAL WORKING WEBSITE
# ============================================================

# ============================================================
# SETTINGS
# ============================================================

TIMESTEPS = 8

CLOUDS = [
    "AWS",
    "Azure",
    "GCP"
]

SEVERITY_SCORE = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100
}

SEVERITY_VALUE = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
    "Critical": 1.00
}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #07090d;
    }

    .main-title {
        font-size: 36px;
        font-weight: 900;
        color: #ff1744;
        letter-spacing: 1px;
    }

    .subtitle {
        color: #8d96a6;
        margin-bottom: 25px;
    }

    .login-title {
        font-size: 38px;
        font-weight: 900;
        color: #ff1744;
        text-align: center;
        margin-top: 70px;
    }

    .login-subtitle {
        color: #8d96a6;
        text-align: center;
        margin-bottom: 30px;
    }

    .focus-box {
        background-color: #12161e;
        border-left: 6px solid #ff1744;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .focus-title {
        color: #ff1744;
        font-size: 18px;
        font-weight: 900;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# USER SYSTEM
# ============================================================

USER_FILE = str(BASE_DIR / "users.json")


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# ============================================================
# LOGIN STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "events" not in st.session_state:
    st.session_state.events = []

if "sequence" not in st.session_state:
    st.session_state.sequence = []

if "cloud_index" not in st.session_state:
    st.session_state.cloud_index = 0

if "location_cache" not in st.session_state:
    st.session_state.location_cache = {}

if "manual_result" not in st.session_state:
    st.session_state.manual_result = None


# ============================================================
# LOGIN / SIGNUP
# ============================================================

if not st.session_state.authenticated:

    st.markdown(
        '<div class="login-title">'
        'AUTOMATED THREAT PRIORITIZATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-subtitle">'
        'AI-Powered Multi-Cloud Cybersecurity Platform'
        '</div>',
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 2, 1])

    with center:
        login_tab, signup_tab = st.tabs(["LOGIN", "CREATE ACCOUNT"])

        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("LOGIN", use_container_width=True):
                users = load_users()
                if username not in users:
                    st.error("Account not found.")
                elif users[username] != hash_password(password):
                    st.error("Incorrect username or password.")
                else:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()

        with signup_tab:
            new_username = st.text_input("Choose username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")

            if st.button("CREATE ACCOUNT", use_container_width=True):
                users = load_users()

                if not new_username:
                    st.error("Enter a username.")
                elif len(new_password) < 6:
                    st.error("Password must contain at least 6 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif new_username in users:
                    st.error("Username already exists.")
                else:
                    users[new_username] = hash_password(new_password)
                    save_users(users)
                    st.success("Account created successfully. Please log in.")

    st.stop()


# ============================================================
# LOAD CACHED LOGS
# ============================================================

@st.cache_data(show_spinner="Loading security logs...")
def load_data():
    filename = BASE_DIR / "combined_threats.csv"

    if not filename.exists():
        raise FileNotFoundError("combined_threats.csv was not found.")

    df = pd.read_csv(filename, low_memory=False)

    required = [
        "src",
        "payload_text",
        "type",
        "severity",
        "target"
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    for column in ["latitude", "longitude"]:
        if column not in df.columns:
            df[column] = np.nan

    if "city" not in df.columns:
        df["city"] = "Unknown"

    if "country_code" not in df.columns:
        df["country_code"] = "Unknown"

    return df.copy()


data = load_data()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    required_model_files = [
        "random_forest.pkl",
        "neural_network.pkl",
        "neural_network_scaler.pkl",
        "lstm_model.keras",
        "lstm_label_encoder.pkl",
        "lstm_scaler.pkl",
    ]

    missing = [
        name for name in required_model_files
        if not (BASE_DIR / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
        )

    return {
        "rf": joblib.load(BASE_DIR / "random_forest.pkl"),
        "nn": joblib.load(BASE_DIR / "neural_network.pkl"),
        "nn_scaler": joblib.load(BASE_DIR / "neural_network_scaler.pkl"),
        "lstm": load_model(BASE_DIR / "lstm_model.keras"),
        "lstm_encoder": joblib.load(BASE_DIR / "lstm_label_encoder.pkl"),
        "lstm_scaler": joblib.load(BASE_DIR / "lstm_scaler.pkl"),
    }


models = load_models()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def ip_features(ip):
    try:
        parts = str(ip).strip().split(".")
        if len(parts) != 4:
            return [0, 0, 0, 0]
        values = [int(x) for x in parts]
        if not all(0 <= x <= 255 for x in values):
            return [0, 0, 0, 0]
        return values
    except Exception:
        return [0, 0, 0, 0]


def payload_features(payload):
    text = str(payload).lower()
    return [
        int("<script" in text),
        int("jndi" in text),
        int("union select" in text),
        int("login" in text or "wp-login" in text),
        int("mstshash" in text),
        int("wget" in text or "curl" in text),
        int("jsonrpc" in text or "eth_blocknumber" in text),
        int("select " in text and " from " in text),
        int("network port" in text),
    ]


def make_features(ip, payload, cloud):
    values = (
        ip_features(ip)
        + payload_features(payload)
        + [
            int(cloud == "AWS"),
            int(cloud == "Azure"),
            int(cloud == "GCP"),
        ]
    )

    if len(values) != 16:
        raise ValueError(
            f"Model requires 16 features, but {len(values)} were created."
        )

    return np.asarray([values], dtype=np.float64)


# ============================================================
# RANDOM FOREST
# ============================================================

def calculate_rf(X, threat):
    probabilities = models["rf"].predict_proba(X)[0]
    classes = models["rf"].classes_
    match = np.where(classes == threat)[0]
    if len(match) == 0:
        return 0.0
    return float(probabilities[match[0]])


# ============================================================
# NEURAL NETWORK
# ============================================================

def calculate_nn(X, threat):
    scaled = models["nn_scaler"].transform(X)
    probabilities = models["nn"].predict_proba(scaled)[0]
    classes = models["nn"].classes_
    match = np.where(classes == threat)[0]
    if len(match) == 0:
        return 0.0
    return float(probabilities[match[0]])


# ============================================================
# LSTM
# ============================================================

def make_lstm_row(threat, severity, cloud):
    encoder = models["lstm_encoder"]

    if threat in encoder.classes_:
        threat_value = int(encoder.transform([threat])[0])
    else:
        threat_value = 0

    cloud_value = {
        "AWS": 0,
        "Azure": 1,
        "GCP": 2,
    }.get(cloud, 0)

    severity_value = SEVERITY_VALUE.get(severity, 0.5)

    return [threat_value, severity_value, cloud_value]


def calculate_lstm(threat, severity, cloud):
    current = make_lstm_row(threat, severity, cloud)

    history = st.session_state.sequence + [current]
    history = history[-TIMESTEPS:]

    while len(history) < TIMESTEPS:
        history.insert(0, current)

    array = np.asarray(history, dtype=np.float32)

    scaled = models["lstm_scaler"].transform(array)

    X = scaled.reshape(1, TIMESTEPS, 3)

    probabilities = models["lstm"].predict(X, verbose=0)[0]

    encoder = models["lstm_encoder"]

    if threat not in encoder.classes_:
        return 0.0

    index = int(encoder.transform([threat])[0])

    return float(probabilities[index])


# ============================================================
# LOCATION LOOKUP
# ============================================================

def is_public_ip(ip):
    try:
        address = ipaddress.ip_address(str(ip).strip())
        return (
            not address.is_private
            and not address.is_loopback
            and not address.is_reserved
            and not address.is_multicast
        )
    except Exception:
        return False


# Keep the API helper name used by the original notebook available.
valid_public_ip = is_public_ip


def lookup_ip_location(ip):
    ip = str(ip).strip()

    if ip in st.session_state.location_cache:
        return st.session_state.location_cache[ip]

    if not is_public_ip(ip):
        result = {
            "success": False,
            "latitude": None,
            "longitude": None,
            "city": "Private Network",
            "country": "Local"
        }
        st.session_state.location_cache[ip] = result
        return result

    try:
        response = requests.get(
            f"https://ipapi.co/{ip}/json/",
            timeout=6
        )

        if response.status_code == 200:
            info = response.json()
            lat = info.get("latitude")
            lon = info.get("longitude")

            if lat is not None and lon is not None:
                result = {
                    "success": True,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "city": info.get("city", "Unknown"),
                    "country": info.get("country_name", "Unknown")
                }
                st.session_state.location_cache[ip] = result
                return result
    except Exception:
        pass

    try:
        response = requests.get(
            f"https://ipwho.is/{ip}",
            timeout=6
        )

        if response.status_code == 200:
            info = response.json()

            if (
                info.get("success")
                and info.get("latitude") is not None
                and info.get("longitude") is not None
            ):
                result = {
                    "success": True,
                    "latitude": float(info["latitude"]),
                    "longitude": float(info["longitude"]),
                    "city": info.get("city", "Unknown"),
                    "country": info.get("country", "Unknown")
                }
                st.session_state.location_cache[ip] = result
                return result
    except Exception:
        pass

    result = {
        "success": False,
        "latitude": None,
        "longitude": None,
        "city": "Location unavailable",
        "country": "Unknown"
    }
    st.session_state.location_cache[ip] = result
    return result


# ============================================================
# ABUSEIPDB API — kept from the notebook for future/API use
# ============================================================

def abuseipdb_lookup(ip):
    api_key = os.getenv("ABUSEIPDB_KEY")

    if not api_key:
        return {"success": False, "error": "AbuseIPDB API key missing."}

    if not valid_public_ip(ip):
        return {"success": False, "error": "Not a valid public IP."}

    url = "https://api.abuseipdb.com/api/v2/check"

    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json", "Key": api_key},
            params={"ipAddress": ip, "maxAgeInDays": 90},
            timeout=15
        )

        if response.status_code != 200:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:500]
            }

        payload = response.json().get("data", {})

        return {
            "success": True,
            "ip": payload.get("ipAddress", ip),
            "abuse_confidence": payload.get("abuseConfidenceScore", 0),
            "country": payload.get("countryCode", "Unknown"),
            "isp": payload.get("isp", "Unknown"),
            "domain": payload.get("domain", "Unknown"),
            "total_reports": payload.get("totalReports", 0),
            "last_reported": payload.get("lastReportedAt")
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ============================================================
# FINAL THREAT PRIORITY SCORE — from the notebook
# ============================================================

def calculate_final_priority(
    rf_score,
    nn_score,
    lstm_score,
    severity,
    abuse_result=None,
    attack_count=1
):
    model_confidence = (
        rf_score + nn_score + lstm_score
    ) / 3

    severity_score = SEVERITY_SCORE.get(severity, 50)

    if abuse_result and abuse_result.get("success"):
        abuse_score = float(abuse_result.get("abuse_confidence", 0))
    else:
        abuse_score = 0

    behaviour_score = min(attack_count * 10, 100)

    final_score = (
        model_confidence * 0.40
        + severity_score * 0.25
        + abuse_score * 0.20
        + behaviour_score * 0.15
    )

    final_score = max(0, min(final_score, 100))

    if final_score >= 80:
        priority = "CRITICAL"
    elif final_score >= 60:
        priority = "HIGH"
    elif final_score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return round(final_score, 2), priority


# ============================================================
# FAST LIVE EVENT
# NO API NETWORK CALLS DURING THE ONE-SECOND LIVE FEED
# ============================================================

def process_live_event(row):
    ip = str(row["src"]).strip()
    threat = str(row["type"])
    severity = str(row["severity"])
    cloud = str(row["target"])
    payload = str(row.get("payload_text", ""))

    X = make_features(ip, payload, cloud)

    rf = calculate_rf(X, threat)
    nn = calculate_nn(X, threat)
    lstm = calculate_lstm(threat, severity, cloud)

    ai_score = (
        rf * 0.35
        + nn * 0.30
        + lstm * 0.35
    ) * 100

    severity_score = SEVERITY_SCORE.get(severity, 50)

    repeated = sum(
        e["source"] == ip
        for e in st.session_state.events
    )

    behaviour_score = min(repeated * 10, 100)

    final_score = (
        ai_score * 0.60
        + severity_score * 0.30
        + behaviour_score * 0.10
    )

    final_score = max(0, min(100, final_score))

    if final_score >= 80:
        priority = "CRITICAL"
    elif final_score >= 60:
        priority = "HIGH"
    elif final_score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    latitude = row.get("latitude", np.nan)
    longitude = row.get("longitude", np.nan)

    if pd.notna(latitude) and pd.notna(longitude):
        latitude = float(latitude)
        longitude = float(longitude)
        city = str(row.get("city", "Dataset location"))
        country = str(row.get("country_code", "Unknown"))
        location_source = "Dataset"
    else:
        latitude = np.nan
        longitude = np.nan
        city = "IP lookup required"
        country = str(row.get("country_code", "Unknown"))
        location_source = "Not geolocated during live stream"

    st.session_state.sequence.append(
        make_lstm_row(threat, severity, cloud)
    )
    st.session_state.sequence = st.session_state.sequence[-TIMESTEPS:]

    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": ip,
        "threat": threat,
        "severity": severity,
        "cloud": cloud,
        "rf": round(rf * 100, 2),
        "nn": round(nn * 100, 2),
        "lstm": round(lstm * 100, 2),
        "ai_score": round(ai_score, 2),
        "final_score": round(final_score, 2),
        "priority": priority,
        "city": city,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "location_source": location_source
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">AUTOMATED THREAT PRIORITIZATION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Multi-Cloud Cybersecurity Dashboard</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.username}**")
    st.divider()

    live_feed = st.toggle("LIVE THREAT FEED", value=True)

    st.divider()
    st.write("### DATASET")
    st.write(f"Total: {len(data):,}")

    for cloud in CLOUDS:
        count = int((data["target"] == cloud).sum())
        st.write(f"{cloud}: **{count:,}**")

    st.divider()
    st.write("### API STATUS")
    st.write(
        "VirusTotal: **configured**" if os.getenv("VIRUSTOTAL_KEY")
        else "VirusTotal: **not configured**"
    )
    st.write(
        "AbuseIPDB: **configured**" if os.getenv("ABUSEIPDB_KEY")
        else "AbuseIPDB: **not configured**"
    )
    st.caption("Keys are read locally and are not hard-coded in app.py.")

    st.divider()

    if st.button("CLEAR LIVE FEED", use_container_width=True):
        st.session_state.events = []
        st.session_state.sequence = []
        st.session_state.cloud_index = 0
        st.rerun()

    if st.button("LOG OUT", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.events = []
        st.session_state.sequence = []
        st.rerun()


# ============================================================
# LIVE ENGINE
# AWS → AZURE → GCP → AWS...
# One new event every second.
# ============================================================

def live_engine():
    available_clouds = [
        cloud
        for cloud in CLOUDS
        if "target" in data.columns and (data["target"] == cloud).any()
    ]

    if live_feed and available_clouds:
        current_index = (
            st.session_state.cloud_index
            % len(available_clouds)
        )
        cloud = available_clouds[current_index]
        st.session_state.cloud_index += 1

        cloud_data = data[data["target"] == cloud]
        row = cloud_data.sample(1).iloc[0]

        event = process_live_event(row)

        st.session_state.events.insert(0, event)
        st.session_state.events = st.session_state.events[:60]

    events = st.session_state.events

    if not events:
        st.info("Waiting for live attacks...")
        return

    df = pd.DataFrame(events)

    critical = sum(x["priority"] == "CRITICAL" for x in events)
    high = sum(x["priority"] == "HIGH" for x in events)
    medium = sum(x["priority"] == "MEDIUM" for x in events)
    low = sum(x["priority"] == "LOW" for x in events)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LIVE THREATS", len(events))
    c2.metric("CRITICAL", critical)
    c3.metric("HIGH", high)
    c4.metric("MEDIUM", medium)
    c5.metric("LOW", low)

    st.divider()

    focus = max(events, key=lambda x: x["final_score"])

    st.markdown(
        '<div class="focus-box">'
        '<div class="focus-title">🚨 FOCUS ON THIS THREAT FIRST</div>'
        f'<h2>{focus["threat"]}</h2>'
        f'<b>Priority:</b> {focus["priority"]}<br>'
        f'<b>Final Score:</b> {focus["final_score"]}%<br>'
        f'<b>AI Score:</b> {focus["ai_score"]}%<br>'
        f'<b>Source IP:</b> {focus["source"]}<br>'
        f'<b>Location:</b> {focus["city"]}, {focus["country"]}<br>'
        f'<b>Cloud:</b> {focus["cloud"]}<br>'
        f'<b>Severity:</b> {focus["severity"]}'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("☁️ Multi-Cloud Threat Status")
    cloud_columns = st.columns(3)

    for column, cloud in zip(cloud_columns, CLOUDS):
        with column:
            st.markdown(f"## {cloud}")

            cloud_events = [
                event for event in events
                if event["cloud"] == cloud
            ]

            st.metric("Live Threats", len(cloud_events))

            dataset_count = int((data["target"] == cloud).sum())
            st.caption(f"Dataset records: {dataset_count:,}")

            if cloud_events:
                highest = max(cloud_events, key=lambda x: x["final_score"])
                st.metric("Highest Risk", f'{highest["final_score"]}%')
                st.write(f'**Threat:** {highest["threat"]}')
                st.write(f'**Priority:** {highest["priority"]}')
                st.write(f'**Severity:** {highest["severity"]}')
            else:
                st.info("Waiting for event.")

    st.divider()

    latest = events[0]
    st.subheader("🤖 AI Threat Assessment")

    a, b, c, d = st.columns(4)
    a.metric("AI SCORE", f'{latest["ai_score"]}%')
    b.metric("Random Forest", f'{latest["rf"]}%')
    c.metric("Neural Network", f'{latest["nn"]}%')
    d.metric("LSTM", f'{latest["lstm"]}%')

    st.subheader("🌍 Live Attack Map")

    map_df = df.dropna(subset=["latitude", "longitude"]).copy()

    if len(map_df) > 0:
        map_df["radius"] = map_df["final_score"].clip(lower=10) * 1500

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_radius="radius",
            get_fill_color=[255, 40, 60, 190],
            pickable=True,
            auto_highlight=True
        )

        view = pdk.ViewState(
            latitude=float(map_df["latitude"].mean()),
            longitude=float(map_df["longitude"].mean()),
            zoom=1.1
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={
                "html": """
                <b>{threat}</b><br/>
                IP: {source}<br/>
                Cloud: {cloud}<br/>
                Severity: {severity}<br/>
                Priority: {priority}<br/>
                Score: {final_score}%<br/>
                Location: {city}, {country}
                """
            }
        )

        st.pydeck_chart(deck, use_container_width=True)

        st.caption(
            "Dataset coordinates are used when available. "
            "IP-derived coordinates are approximate."
        )
    else:
        st.warning("The current live events do not have coordinates yet.")

    st.subheader("🎯 Threat Priority Queue")

    queue = df[
        [
            "time",
            "source",
            "threat",
            "severity",
            "cloud",
            "city",
            "country",
            "ai_score",
            "final_score",
            "priority"
        ]
    ].sort_values("final_score", ascending=False)

    st.dataframe(queue, use_container_width=True, hide_index=True)


# Use st.fragment when available; fall back to a normal function on older Streamlit.
if hasattr(st, "fragment"):
    live_engine_fragment = st.fragment(run_every="1s")(live_engine)
    live_engine_fragment()
else:
    live_engine()


# ============================================================
# MANUAL IP LOCATION
# ============================================================

st.divider()
st.subheader("🔎 Investigate Public IP Location")

manual_ip = st.text_input(
    "Enter public IP address",
    placeholder="Example: 8.8.8.8"
)

if st.button("LOOK UP LOCATION"):
    manual_ip = manual_ip.strip()

    if not is_public_ip(manual_ip):
        st.error("Enter a valid public IP address.")
    else:
        with st.spinner("Looking up IP location..."):
            result = lookup_ip_location(manual_ip)
        st.session_state.manual_result = result

result = st.session_state.manual_result

if result:
    if result.get("success"):
        st.subheader("IP Location")
        a, b, c = st.columns(3)
        a.metric("City", result.get("city", "Unknown"))
        b.metric("Country", result.get("country", "Unknown"))
        c.metric("Latitude", result.get("latitude"))
        st.write("Longitude:", result.get("longitude"))
        st.info(
            "IP geolocation is an approximate network location, "
            "not an exact physical/GPS location."
        )
    else:
        st.warning("Location could not be determined.")
