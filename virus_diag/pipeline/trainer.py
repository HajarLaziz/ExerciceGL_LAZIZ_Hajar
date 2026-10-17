"""Coordination de l'apprentissage, indépendante du diagnostic."""

from virus_diag.core.optimizer import GradientDescent
from virus_diag.utils.preprocessing import preprocess


class Trainer:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer if optimizer is not None else GradientDescent()

    def train(self, dataset, epochs=1500):
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Le nombre d'époques doit être un entier positif.")
        if not dataset.patients or len(dataset.patients) != len(dataset.labels):
            raise ValueError("Les patients et étiquettes doivent être non vides et alignés.")
        for patient, label in zip(dataset.patients, dataset.labels):
            preprocess(patient)
            if label not in (0, 1):
                raise ValueError("L'étiquette doit valoir 0 ou 1.")
        self.model.train(dataset, self.optimizer, epochs)
        return self.model
