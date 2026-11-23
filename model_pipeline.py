"""Pipeline ML : prédiction de réadmission (NO / READMITTED) des patients diabétiques."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

MEDICAMENTS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
COLONNES_RARES = [
    "acarbose",
    "chlorpropamide",
    "miglitol",
    "tolazamide",
    "glipizide-metformin",
    "tolbutamide",
    "troglitazone",
    "metformin-rosiglitazone",
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-pioglitazone",
]
AGE_MAPPING = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def regrouper_diagnostic(code):
    """Regroupe un code de diagnostic (str) en grande catégorie."""
    if code == "Unknown":
        return "Unknown"
    if str(code).startswith(("V", "E")):
        return "Other"
    code = int(float(code))
    if 250 <= code < 251:
        return "Diabetes"
    if (390 <= code <= 459) or code == 785:
        return "Circulatory"
    if (460 <= code <= 519) or code == 786:
        return "Respiratory"
    if (520 <= code <= 579) or code == 787:
        return "Digestive"
    if 800 <= code <= 999:
        return "Injury"
    if 710 <= code <= 739:
        return "Musculoskeletal"
    if (580 <= code <= 629) or code == 788:
        return "Genitourinary"
    if 140 <= code <= 239:
        return "Neoplasms"
    return "Other"


def prepare_data(csv_path):
    """Charge et prétraite les données, puis sépare train / validation / test.

    Args:
        csv_path (str): chemin vers diabetic_data.csv.
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
        (y = 0 pour NO, 1 pour READMITTED ; séparation par patient).
    """
    df = pd.read_csv(csv_path, keep_default_na=False)

    # --- Nettoyage ---
    d = df.copy()
    d = d.replace("?", np.nan)
    d = d.drop(columns=["weight", "examide", "citoglipton"])
    for col in ["max_glu_serum", "A1Cresult"]:
        d[col] = d[col].replace("None", "Not_measured")
    for col in [
        "medical_specialty",
        "payer_code",
        "race",
        "diag_1",
        "diag_2",
        "diag_3",
    ]:
        d[col] = d[col].fillna("Unknown")
    d["gender"] = d["gender"].replace("Unknown/Invalid", "Unknown")
    d["admission_type_id"] = d["admission_type_id"].replace([5, 6, 8], "Unknown")
    d["discharge_disposition_id"] = d["discharge_disposition_id"].replace(
        [18, 25, 26], "Unknown"
    )
    d["admission_source_id"] = d["admission_source_id"].replace(
        [9, 15, 17, 20, 21], "Unknown"
    )

    # --- Feature engineering ---
    d["age"] = d["age"].map(AGE_MAPPING)
    for i in (1, 2, 3):
        d[f"diag_{i}_group"] = d[f"diag_{i}"].apply(regrouper_diagnostic)
    d = d.drop(columns=["diag_1", "diag_2", "diag_3"])
    d["num_active_diabetes_meds"] = (df[MEDICAMENTS] != "No").sum(axis=1)
    d["num_med_changes"] = df[MEDICAMENTS].isin(["Up", "Down"]).sum(axis=1)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        d[col] = d[col].astype(str)  # codes = catégories

    # --- Retirer les décès, créer la cible binaire ---
    codes = pd.to_numeric(d["discharge_disposition_id"], errors="coerce")
    d = d.loc[~codes.isin([11, 19, 20, 21])].reset_index(drop=True)
    y = (d["readmitted"] != "NO").astype(int)
    patients = d["patient_nbr"]
    X = d.drop(
        columns=["encounter_id", "patient_nbr", "readmitted", "diabetesMed"]
        + COLONNES_RARES
    )

    # --- Séparation par patient (un patient reste dans un seul ensemble) ---
    sep = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    idx_dev, idx_test = next(sep.split(X, y, groups=patients))
    sep_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    idx_tr, idx_va = next(
        sep_val.split(X.iloc[idx_dev], y.iloc[idx_dev], groups=patients.iloc[idx_dev])
    )
    idx_train, idx_val = idx_dev[idx_tr], idx_dev[idx_va]

    return (
        X.iloc[idx_train],
        X.iloc[idx_val],
        X.iloc[idx_test],
        y.iloc[idx_train],
        y.iloc[idx_val],
        y.iloc[idx_test],
    )


def train_model(X_train, y_train):
    """Entraîne l'arbre de décision retenu (profondeur 5, entropy, balanced).

    Returns:
        Pipeline: préparation + arbre, entraîné.
    """
    preparation = ColumnTransformer(
        [
            ("num", "passthrough", make_column_selector(dtype_include="number")),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    model = Pipeline(
        [
            ("preparation", preparation),
            (
                "arbre",
                DecisionTreeClassifier(
                    max_depth=5,
                    criterion="entropy",
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """Évalue le modèle et affiche les métriques.

    Returns:
        dict: accuracy, macro_f1, precision, recall (classe READMITTED).
    """
    pred = model.predict(X)
    resultats = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, pred, pos_label=1, zero_division=0),
    }
    for nom, valeur in resultats.items():
        print(f"{nom:10s}: {valeur:.4f}")
    print("Matrice de confusion :\n", confusion_matrix(y, pred, labels=[0, 1]))
    print(
        classification_report(
            y, pred, target_names=["NO", "READMITTED"], digits=4, zero_division=0
        )
    )
    return resultats


def save_model(model, path="model.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé."""
    return joblib.load(path)


def predict_model(model, X):
    """Prédit la classe (0 = NO, 1 = READMITTED) pour de nouveaux séjours."""
    return model.predict(X)
