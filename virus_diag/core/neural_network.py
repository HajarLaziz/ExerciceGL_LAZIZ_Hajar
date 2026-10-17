"""Petit réseau 3 -> 4 -> 1, pour illustrer le polymorphisme."""

import random

from virus_diag.core.model import Model, sigmoid
from virus_diag.utils.preprocessing import preprocess


class NeuralNetwork(Model):
    def __init__(self, seed=42):
        super().__init__()
        rng = random.Random(seed)
        self._hidden_weights = [[rng.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(4)]
        self._hidden_biases = [0.0] * 4
        self._output_weights = [rng.uniform(-0.5, 0.5) for _ in range(4)]
        self._output_bias = 0.0

    def _forward(self, features):
        hidden = [sigmoid(sum(w * x for w, x in zip(row, features)) + bias)
                  for row, bias in zip(self._hidden_weights, self._hidden_biases)]
        output = sigmoid(sum(w * h for w, h in zip(self._output_weights, hidden)) + self._output_bias)
        return hidden, output

    def train(self, dataset, optimizer, epochs):
        features = [preprocess(patient) for patient in dataset.patients]
        for _ in range(epochs):
            for values, label in zip(features, dataset.labels):
                hidden, output = self._forward(values)
                error = output - label
                # Calculer les gradients cachés avant de modifier les poids de sortie.
                hidden_errors = [error * w * h * (1 - h)
                                 for w, h in zip(self._output_weights, hidden)]
                self._output_weights = [optimizer.update(w, error * h)
                                        for w, h in zip(self._output_weights, hidden)]
                self._output_bias = optimizer.update(self._output_bias, error)
                for j, hidden_error in enumerate(hidden_errors):
                    self._hidden_weights[j] = [optimizer.update(w, hidden_error * x)
                                               for w, x in zip(self._hidden_weights[j], values)]
                    self._hidden_biases[j] = optimizer.update(self._hidden_biases[j], hidden_error)
        self._trained = True

    def predict(self, patient_data):
        self._require_trained()
        return self._forward(preprocess(patient_data))[1]
