"""Évaluation sur des exemples distincts de l'entraînement."""

from virus_diag.pipeline.predictor import ClinicalPredictor
from virus_diag.utils.metrics import classification_metrics


class Evaluator:
    def evaluate(self, model, dataset):
        predictor = ClinicalPredictor(model)
        predicted = [int(predictor.diagnose(patient) == "Infecté") for patient in dataset.patients]
        return classification_metrics(dataset.labels, predicted)
