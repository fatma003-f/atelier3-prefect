"""Pipeline Prefect : automatise installation, qualité du code et étapes ML."""

import argparse
import subprocess
import sys

from prefect import flow, get_run_logger, task

from model_pipeline import (
    evaluate_model,
    load_model,
    predict_model,
    prepare_data,
    save_model,
    train_model,
)

DATA = "diabetic_data.csv"
MODEL = "model.joblib"
# On ne cible que nos 3 fichiers (+ le test) pour garder les outils rapides.
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd):
    """Exécute une commande et lève une erreur si elle échoue."""
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- Tasks : code
@task(name="install-dependencies")
def install_dependencies():
    get_run_logger().info("Installation des dépendances...")
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_code():
    get_run_logger().info("Formatage avec black...")
    _run([sys.executable, "-m", "black", *FILES, "test_model_pipeline.py"])


@task(name="quality-code")
def quality_code():
    get_run_logger().info("Qualité avec flake8...")
    _run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=100",
            *FILES,
            "test_model_pipeline.py",
        ]
    )


@task(name="security-code")
def security_code():
    get_run_logger().info("Sécurité avec bandit...")
    _run([sys.executable, "-m", "bandit", "-q", "-ll", *FILES])


@task(name="unit-tests")
def unit_tests():
    get_run_logger().info("Tests unitaires avec pytest...")
    _run([sys.executable, "-m", "pytest", "-q", "test_model_pipeline.py"])


# ------------------------------------------------------------------ Tasks : ML
@task(name="prepare-data")
def prepare_data_task():
    get_run_logger().info("Préparation des données...")
    return prepare_data(DATA)


@task(name="train-model")
def train_model_task(X_train, y_train):
    get_run_logger().info("Entraînement du modèle...")
    return train_model(X_train, y_train)


@task(name="save-model")
def save_model_task(model):
    save_model(model, MODEL)


@task(name="load-model")
def load_model_task():
    return load_model(MODEL)


@task(name="evaluate-model")
def evaluate_model_task(model, X, y):
    get_run_logger().info("Évaluation du modèle...")
    return evaluate_model(model, X, y)


@task(name="predict")
def predict_task(model, X):
    pred = predict_model(model, X.head(10))
    get_run_logger().info(f"Prédictions : {pred.tolist()}")
    return pred


# ----------------------------------------------------------------------- Flows
@flow(name="install", log_prints=True)
def flow_install():
    install_dependencies()


@flow(name="code", log_prints=True)
def flow_code():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    _, X_val, _, _, y_val, _ = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, X_val, y_val)


@flow(name="predict", log_prints=True)
def flow_predict():
    _, _, X_test, _, _, _ = prepare_data_task()
    model = load_model_task()
    predict_task(model, X_test)


@flow(name="all", log_prints=True)
def flow_all():
    flow_code()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, X_val, y_val)


FLOWS = {
    "install": flow_install,
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "all": flow_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", required=True, choices=FLOWS.keys())
    FLOWS[parser.parse_args().flow]()
