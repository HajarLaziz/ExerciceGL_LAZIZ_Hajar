"""Régression logistique binaire, implémentée sans bibliothèque externe."""

from virus_diag.core.model import Model, sigmoid
from virus_diag.utils.preprocessing import preprocess


class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self._weights = [0.0] * 3
        self._bias = 0.0

    def _forward(self, features):
        return sigmoid(sum(w * x for w, x in zip(self._weights, features)) + self._bias)

    def train(self, dataset, optimizer, epochs):
        # Gradient de l'entropie croisée binaire, un exemple à la fois.
        features = [preprocess(patient) for patient in dataset.patients]
        for _ in range(epochs):
            for values, label in zip(features, dataset.labels):
                error = self._forward(values) - label
                self._weights = [optimizer.update(w, error * x) for w, x in zip(self._weights, values)]
                self._bias = optimizer.update(self._bias, error)
        self._trained = True

    def predict(self, patient_data):
        self._require_trained()
        return self._forward(preprocess(patient_data))
