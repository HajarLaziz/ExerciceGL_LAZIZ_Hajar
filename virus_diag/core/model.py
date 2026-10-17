"""Interface commune aux modèles entraînables de la démonstration."""

from abc import ABC, abstractmethod
import math


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


class Model(ABC):
    def __init__(self):
        self._trained = False

    def _require_trained(self):
        if not self._trained:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")

    @abstractmethod
    def train(self, dataset, optimizer, epochs):
        """Apprend les paramètres du modèle."""

    @abstractmethod
    def predict(self, patient_data):
        """Renvoie un score scalaire entre 0 et 1 pour un seul patient."""
