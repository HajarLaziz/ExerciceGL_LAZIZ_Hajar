"""Démonstration en ligne de commande sur des patients fictifs."""

import argparse
from pathlib import Path

from virus_diag.app.api import DiagnosisAPI
from virus_diag.core.dataset import Dataset
from virus_diag.core.logistic_regression import LogisticRegression
from virus_diag.core.neural_network import NeuralNetwork
from virus_diag.pipeline.evaluator import Evaluator
from virus_diag.pipeline.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Démonstration pédagogique de ClinicalPredictor")
    parser.add_argument("--model", choices=("logistic", "neural"), default="logistic")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "data" / "patient_data.csv"
    train_data = Dataset.from_csv(path, "train")
    test_data = Dataset.from_csv(path, "test")
    model = LogisticRegression() if args.model == "logistic" else NeuralNetwork()

    # Phase 1 : apprentissage. L'application reçoit ensuite ce modèle entraîné.
    Trainer(model).train(train_data)
    print(f"Modèle : {type(model).__name__}")
    print(f"Entraînement : {len(train_data.labels)} exemples fictifs")
    print(f"Évaluation : {len(test_data.labels)} autres exemples fictifs")
    for name, value in Evaluator().evaluate(model, test_data).items():
        print(f"  {name} = {value:.2f}")

    # Phase 2 : inférence. L'API et ClinicalPredictor n'entraînent rien.
    api = DiagnosisAPI(model)
    for patient in (
        {"temperature": 36.6, "toux": 0, "fatigue": 0},
        {"temperature": 39.1, "toux": 1, "fatigue": 1},
    ):
        print(f"{patient} -> {api.diagnose(patient)['diagnostic']}")
    print("Données fictives : ces scores n'établissent aucune performance clinique.")


if __name__ == "__main__":
    main()
