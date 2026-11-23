"""Point d'entrée : exécute les étapes du pipeline via la ligne de commande."""

import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - réadmission")
    parser.add_argument(
        "--step",
        required=True,
        choices=["prepare", "train", "evaluate", "predict", "all"],
        help="étape à exécuter",
    )
    parser.add_argument("--data", default=DATA, help="chemin du CSV")
    parser.add_argument("--model", default=MODEL, help="chemin du modèle")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(args.data)

    if args.step == "prepare":
        print(
            "Train :", X_train.shape, "| Val :", X_val.shape, "| Test :", X_test.shape
        )

    elif args.step in ("train", "all"):
        model = train_model(X_train, y_train)
        save_model(model, args.model)
        if args.step == "all":
            print("\n--- Validation ---")
            evaluate_model(model, X_val, y_val)

    elif args.step == "predict":
        model = load_model(args.model)
        pred = predict_model(model, X_test.head(10))
        print("Prédictions (10 premiers séjours du test) :", pred.tolist())

    elif args.step == "evaluate":
        model = load_model(args.model)
        print("--- Validation ---")
        evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()
